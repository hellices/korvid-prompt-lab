from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from korvid_prompt_lab.adapter import KorvidGEPAAdapter
from korvid_prompt_lab.campaign_artifacts import _validate_projected_response_shape
from korvid_prompt_lab.config import load_campaign
from korvid_prompt_lab.contracts import EvalCase, KorvidNativeServing
from korvid_prompt_lab.native import KorvidNativeRunner, initialize_native
from korvid_prompt_lab.native_cli import export_native_rules
from korvid_prompt_lab.native_contract import (
    find_native_case,
    native_cases,
    rules_candidate,
)
from korvid_prompt_lab.native_source import run_native_request
from korvid_prompt_lab.rounds import _parse_response
from korvid_prompt_lab.scoring import result_passed, score_result

pytestmark = pytest.mark.skipif(
    not os.environ.get("KORVID_NATIVE_SOURCE_ROOT"),
    reason="native application tests require the unchanged v0.4.1 source environment",
)


def action_script(case: EvalCase) -> list[list[dict[str, Any]]]:
    authored = find_native_case(case.case_id)
    task = case.case_id.split("-", 1)[1]
    if task in {"pods", "all-pods"}:
        name = "list_resources"
        args = {"kind": "pods", **({"namespace": authored.namespace} if task == "pods" else {})}
    elif task == "helm":
        name, args = "helm_list_releases", {"namespace": authored.namespace}
    elif task == "logs":
        name, args = "open_logs", {"namespace": authored.namespace, "pod": authored.pod, "container": "main"}
    else:
        name, args = "open_describe", {"kind": "pods", "namespace": authored.namespace, "name": authored.pod}
    return [
        [{"type": "tool_call", "id": "native-action", "name": name, "arguments": json.dumps(args)}, {"type": "done"}],
        [{"type": "text_delta", "text": "Opened."}, {"type": "done"}],
    ]


@pytest.mark.parametrize("case_id", [case.case_id for case in native_cases()])
def test_native_low_follow_reaches_each_authored_ui_target(
    case_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    _, path = initialize_native(tmp_path / "setup", model="ollama/qwen3:0.6b")
    campaign = load_campaign(path)
    case = next(case for case in campaign.cases if case.case_id == case_id)
    runner = KorvidNativeRunner(campaign, script_factory=action_script)
    result = runner.run(rules_candidate([]), case, tmp_path / "run")
    assert result.execution_mode == "scripted"
    assert result_passed(score_result(result))
    summary = json.loads((tmp_path / "run" / "native-summary.json").read_text())
    assert summary["policy"]["tier"] == "low"
    assert not {"navigate", "set_filter", "drill_down"} & set(summary["policy"]["tools"])
    assert summary["rules_applied"]
    assert summary["runtime"]["dependencies"]["korvid"] == "0.4.1"


def test_exported_rules_run_through_native_ui_after_actual_config_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    _, path = initialize_native(tmp_path / "setup", model="ollama/qwen3:0.6b")
    campaign = load_campaign(path)
    rules = ["For a display request, open the requested view and keep the reply brief."]
    candidate = rules_candidate(rules)
    exported = export_native_rules(candidate, campaign, tmp_path / "export")
    assert yaml.safe_load(exported.read_text())["agent"]["rules"] == rules
    assert isinstance(campaign.serving, KorvidNativeServing)
    case = next(case for case in campaign.cases if case.case_id == "train-pods")
    response = run_native_request(campaign.serving, {
        "protocol_version": 1, "operation": "evaluate", "case_id": case.case_id,
        "rules": rules, "config_path": str(exported),
        "model": {"reference": "ollama/qwen3:0.6b", "endpoint": "http://127.0.0.1:11434",
                  "options": {"temperature": 0.0, "seed": 0}},
        "script": action_script(case),
    })
    assert response["rules_applied"] is True
    assert response["observed"]["kind"] == "pods"
    assert response["observed"]["scope"] == "shop"
    assert response["missing_postconditions"] == []


def test_native_feedback_contains_real_policy_not_added_mcp_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    _, path = initialize_native(tmp_path / "setup", model="ollama/qwen3:0.6b")
    campaign = load_campaign(path)
    runner = KorvidNativeRunner(campaign, script_factory=action_script)
    adapter = KorvidGEPAAdapter(runner, tmp_path / "trace")
    candidate = rules_candidate([])
    case = next(case for case in campaign.cases if case.case_id == "train-pods")
    batch = adapter.evaluate([case], candidate.components, capture_traces=True)
    record = adapter.make_reflective_dataset(candidate.components, batch, ["rules"])["rules"][0]
    assert "JSON array" in record["Feedback"]["guidance"]
    assert record["Inputs"]["runtime_policy"]["tier"] == "low"
    assert "available_mcp_tools" not in record["Inputs"]
    assert record["Generated Outputs"]["observed_state"]["scope"] == "shop"


def test_actual_native_writer_and_safe_projection_match_consumer_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    _, campaign_path = initialize_native(tmp_path / "setup", model="ollama/qwen3:0.6b")
    campaign = load_campaign(campaign_path)
    root = tmp_path / "run"
    KorvidNativeRunner(campaign, script_factory=action_script).run(
        rules_candidate([]), campaign.cases[0], root,
    )
    stat = root.stat()
    projected = dict(_parse_response(
        root / "response.json", root=root, expected_root_identity=(stat.st_dev, stat.st_ino),
    ).payload)
    assert projected["protocol_version"] == 2
    with pytest.raises(ValueError, match="must be live"):
        _validate_projected_response_shape(projected, "fixture", evaluation_backend="korvid_native")
    # Exercise only the live wire-shape contract; never persist or publish
    # this relabelled scripted fixture as live model evidence.
    live_shape = {
        **projected, "execution_mode": "live",
        "request_identity": {**projected["request_identity"], "seed_applied": True},
    }
    _validate_projected_response_shape(live_shape, "fixture", evaluation_backend="korvid_native")
