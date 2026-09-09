from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from korvid_prompt_lab.config import load_campaign
from korvid_prompt_lab.native import (
    KorvidNativeRunner,
    _validate_worker_result,
    initialize_native,
    validate_native_policy,
)
from korvid_prompt_lab.native_contract import find_native_case, rules_candidate
from korvid_prompt_lab.native_source import validate_native_source

LOW_POLICY = {
    "tier": "low",
    "prompt_pack_id": "low-korvid-operator",
    "max_iterations": 6,
    "max_history_chars": 24000,
    "max_result_chars": 3000,
    "max_tool_calls_per_iteration": 1,
    "tools": [
        "diagnose_pod", "diagnose_pvc", "diagnose_service", "diagnose_workload",
        "get_events", "get_logs", "get_resource", "helm_list_releases",
        "list_operators", "list_resources", "open_describe", "open_logs",
    ],
}


def test_unreviewed_source_is_rejected_before_worker_start(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    with pytest.raises(ValueError, match="revision"):
        validate_native_source(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("rules_applied", False), ("korvid_version", "0.3.0"), ("execution_mode", "scripted"),
    ("rules", ["not the requested rules"]), ("case_id", "holdout-pods"),
])
def test_worker_identity_must_match_the_real_invocation(field: str, value: object) -> None:
    payload = {
        "protocol_version": 1, "rules_applied": True, "korvid_version": "0.4.1",
        "rules": [], "case_id": "train-pods", "execution_mode": "live",
        "ui_follow": True,
        "policy": LOW_POLICY, "prompt_fingerprint": "a" * 64,
        "wall_time_seconds": 1.25, "iterations": 1,
    }
    payload[field] = value
    with pytest.raises(ValueError):
        _validate_worker_result(payload, rules=[], case_id="train-pods", execution_mode="live")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wall_time_seconds", None),
        ("wall_time_seconds", True),
        ("wall_time_seconds", float("nan")),
        ("wall_time_seconds", -0.01),
        ("iterations", None),
        ("iterations", True),
        ("iterations", 1.5),
        ("iterations", 0),
    ],
)
def test_worker_metrics_must_be_real_executed_turn_measurements(
    field: str,
    value: object,
) -> None:
    payload = {
        "protocol_version": 1, "rules_applied": True, "korvid_version": "0.4.1",
        "rules": [], "case_id": "train-pods", "execution_mode": "live",
        "ui_follow": True,
        "policy": LOW_POLICY, "prompt_fingerprint": "a" * 64,
        "wall_time_seconds": 1.25, "iterations": 1,
    }
    payload[field] = value

    with pytest.raises(ValueError):
        _validate_worker_result(
            payload, rules=[], case_id="train-pods", execution_mode="live"
        )


@pytest.mark.parametrize("field", ["wall_time_seconds", "iterations"])
def test_worker_metrics_are_required(field: str) -> None:
    payload = {
        "protocol_version": 1, "rules_applied": True, "korvid_version": "0.4.1",
        "rules": [], "case_id": "train-pods", "execution_mode": "live",
        "ui_follow": True,
        "policy": LOW_POLICY, "prompt_fingerprint": "a" * 64,
        "wall_time_seconds": 1.25, "iterations": 1,
    }
    del payload[field]

    with pytest.raises(ValueError):
        _validate_worker_result(
            payload, rules=[], case_id="train-pods", execution_mode="live"
        )


def test_self_reported_low_tier_cannot_widen_tools() -> None:
    with pytest.raises(ValueError, match="policy"):
        validate_native_policy({"tier": "low", "tools": ["navigate"]})


def test_library_case_cannot_change_the_model_measured_by_the_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "source"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    _, path = initialize_native(tmp_path / "setup", model="ollama/qwen3:0.6b")
    campaign = load_campaign(path)
    foreign_case = replace(campaign.cases[0], models=("ollama/another-model",))

    def unexpected_worker(*args, **kwargs):
        pytest.fail("a foreign case must be rejected before worker invocation")

    monkeypatch.setattr("korvid_prompt_lab.native.run_native_request", unexpected_worker)
    with pytest.raises(ValueError, match="campaign"):
        KorvidNativeRunner(campaign).run(rules_candidate([]), foreign_case, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_runner_publishes_worker_iterations_instead_of_tool_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "source"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    _, path = initialize_native(tmp_path / "setup", model="ollama/qwen3:0.6b")
    campaign = load_campaign(path)
    case = campaign.cases[0]
    expected = dict(find_native_case(case.case_id).expected)
    calls = [
        {
            "name": "list_resources",
            "arguments": {"kind": "pods"},
            "ok": True,
            "result": "shown",
        }
        for _ in range(3)
    ]
    worker_result = {
        "protocol_version": 1,
        "korvid_version": "0.4.1",
        "rules": [],
        "case_id": case.case_id,
        "execution_mode": "live",
        "rules_applied": True,
        "ui_follow": True,
        "policy": LOW_POLICY,
        "prompt_fingerprint": "a" * 64,
        "initial": expected,
        "expected": expected,
        "observed": expected,
        "missing_postconditions": [],
        "calls": calls,
        "errors": [],
        "runtime": {"dependencies": {"korvid": "0.4.1"}, "fingerprint": "b" * 64},
        "wall_time_seconds": 1.25,
        "iterations": 2,
    }
    monkeypatch.setattr(
        "korvid_prompt_lab.native.run_native_request",
        lambda serving, payload: worker_result,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    bridge = KorvidNativeRunner(campaign).run(rules_candidate([]), case, run_dir)

    expected_usage = {
        "tool_calls": 3,
        "iterations": 2,
        "wall_time_seconds": 1.25,
    }
    assert bridge.usage == expected_usage
    response = json.loads((run_dir / "response.json").read_text(encoding="utf-8"))
    assert response["usage"] == expected_usage
