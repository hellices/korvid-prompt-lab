from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from korvid_prompt_lab.contracts import Campaign, EvalCase, KorvidUpstreamServing
from korvid_prompt_lab.runner import BridgeInvocationError
from korvid_prompt_lab.scoring import EvaluationResult
from korvid_prompt_lab.upstream import KorvidUpstreamRunner, inspect_upstream
from korvid_prompt_lab.upstream_contract import load_source_cases, prompt_candidate

SOURCE_ROOT = Path(os.environ.get("KORVID_NATIVE_SOURCE_ROOT", ""))
pytestmark = pytest.mark.skipif(
    not os.environ.get("KORVID_NATIVE_SOURCE_ROOT"),
    reason="set KORVID_NATIVE_SOURCE_ROOT to the Korvid v0.4.1 source checkout",
)
REVISION = "33c483e041006eb20259a024ed85a9323e52c8f0"


def _serving() -> KorvidUpstreamServing:
    return KorvidUpstreamServing(
        backend="korvid_upstream",
        source_root=str(SOURCE_ROOT),
        base_url="http://127.0.0.1:11434",
        korvid_revision=REVISION,
        timeout_seconds=240,
        model_options=MappingProxyType({"temperature": 0.0}),
    )


def _campaign(reference: str = "scenarios/image-pull-typo") -> Campaign:
    model = "ollama/qwen3:0.6b"
    case = load_source_cases(SOURCE_ROOT, (reference,))[0].eval_case(model)
    return Campaign(
        campaign_id="upstream",
        repetitions=2,
        models=(model,),
        cases=(case,),
        serving=_serving(),
    )


def test_inspect_upstream_launches_only_pinned_upstream_worker() -> None:
    references = ("scenarios/image-pull-typo", "journeys/tui-follow")
    snapshot = inspect_upstream(
        replace(_serving(), base_url="http://127.0.0.1:1"),
        "ollama/qwen3:0.6b",
        references,
        seed=3,
    )
    endpoint_snapshot = inspect_upstream(
        _serving(),
        "ollama/qwen3:0.6b",
        references,
        seed=3,
    )

    assert snapshot["prompt"]["baseline_equivalent"] is True
    assert snapshot["prompt"]["text"] != "[]"
    assert [case["reference"] for case in snapshot["cases"]] == [
        "scenarios/image-pull-typo",
        "journeys/tui-follow",
    ]
    assert snapshot["model"]["options"]["seed"] == 3
    assert snapshot["model"]["endpoint"] == "http://127.0.0.1:1"
    assert snapshot["runtime"]["dependencies"]["korvid"] == "0.4.1"
    assert snapshot["prompt"]["text"] == endpoint_snapshot["prompt"]["text"]
    assert snapshot["runtime"] == endpoint_snapshot["runtime"]
    assert snapshot["cases"] == endpoint_snapshot["cases"]


def _worker_result(
    case: EvalCase,
    *,
    success: bool,
    execution_mode: str = "live",
) -> dict[str, Any]:
    reference = (
        f"scenarios/{case.case_id}"
        if case.template_id == "korvid-scenario"
        else f"journeys/{case.case_id}"
    )
    source = SOURCE_ROOT / "src/korvid/evals" / f"{reference}.yaml"
    scenario_upstream: dict[str, Any] = {
        "outcome": "success" if success else "failure",
        "failure_class": None if success else "misdiagnosis",
        "grade": {
            "diagnosis_success": success,
            "evidence_fetched": True,
            "missing_mentions": [],
            "forbidden_mentions": [],
            "missing_evidence": [],
        },
        "iterations": 2,
        "tool_calls": 1,
        "resolvable_tool_calls": 1,
        "on_target_tool_calls": 1,
        "malformed_tool_calls": 0,
        "write_attempts": 0,
        "safety_violations": 0,
        "input_tokens": 10,
        "output_tokens": 4,
        "tokens_estimated": False,
        "wall_time_seconds": 0.25,
        "error_label": None,
    }
    upstream: dict[str, Any]
    source_report: dict[str, Any]
    questions: list[str]
    if case.template_id == "korvid-scenario":
        upstream = scenario_upstream
        questions = [case.prompt]
        source_report = {
            "scenario": case.case_id,
            "root_cause": "Image tag v99 does not exist.",
            "successes": int(success),
            "evidence_hits": 1,
            "interaction": {
                "mode": "list",
                "namespace": "front",
                "resource": "pods",
            },
            "max_tool_calls": 1,
            "runs": [
                {
                    "grade": upstream["grade"],
                    "citations": {"claims": [], "coverage": 0.0},
                    "answer": "ImagePullBackOff: tag v99 does not exist.",
                    "iterations": upstream["iterations"],
                    "tool_calls": upstream["tool_calls"],
                    "resolvable_tool_calls": upstream["resolvable_tool_calls"],
                    "on_target_tool_calls": upstream["on_target_tool_calls"],
                    "malformed_tool_calls": upstream["malformed_tool_calls"],
                    "write_attempts": upstream["write_attempts"],
                    "safety_violations": upstream["safety_violations"],
                    "input_tokens": upstream["input_tokens"],
                    "output_tokens": upstream["output_tokens"],
                    "tokens_estimated": upstream["tokens_estimated"],
                    "wall_time_s": upstream["wall_time_seconds"],
                    "error": None,
                    "outcome": upstream["outcome"],
                    "failure_class": upstream["failure_class"],
                }
            ],
        }
    else:
        questions = json.loads(case.prompt)
        turns = [
            {
                "grade": {
                    "diagnosis_success": success,
                    "evidence_fetched": True,
                    "missing_mentions": [],
                    "forbidden_mentions": [],
                    "missing_evidence": [],
                },
                "citations": {"claims": [], "coverage": 0.0},
                "answer": f"answer-{index}",
                "iterations": 1,
                "tool_calls": 1,
                "resolvable_tool_calls": 1,
                "on_target_tool_calls": 1,
                "malformed_tool_calls": 0,
                "write_attempts": 0,
                "safety_violations": 0,
                "input_tokens": 10,
                "output_tokens": 4,
                "tokens_estimated": False,
                "wall_time_s": 0.25,
                "error": None,
                "outcome": "success" if success else "failure",
                "failure_class": None if success else "misdiagnosis",
                "interaction": {"mode": "list", "turn": index},
                "final_interaction": {"mode": "list", "turn": index},
            }
            for index, _question in enumerate(questions, start=1)
        ]
        upstream = {
            "success": success,
            "failure_class": None if success else "misdiagnosis",
            "turns": turns,
        }
        source_report = {
            "journey": case.case_id,
            "interaction": {"mode": "list", "turn": len(turns)},
            "runs": [{"turns": turns}],
        }
    return {
        "protocol_version": 1,
        "operation": "evaluate",
        "execution_mode": execution_mode,
        "reference": reference,
        "case_id": case.case_id,
        "kind": case.template_id.removeprefix("korvid-"),
        "source_path": f"src/korvid/evals/{reference}.yaml",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "questions": questions,
        "tier_pack_sha256": hashlib.sha256(b"optimized prompt").hexdigest(),
        "prompt_path_verified": False,
        "model": {
            "reference": case.models[0],
            "endpoint": "http://127.0.0.1:11434",
            "options": {"temperature": 0.0, "seed": 5},
        },
        "policy": {"tier": "low", "prompt_pack": "low-korvid-operator", "tools": ["diagnose_pod"]},
        "prompt": {"pack": "low-korvid-operator", "source": "override", "sha256": "a" * 64},
        "runtime": {"dependencies": {"korvid": "0.4.1"}, "fingerprint": "b" * 64},
        "success": success,
        "hard_failures": [],
        "source_report": source_report,
        "upstream": upstream,
        "original_payload": upstream,
        "calls": [{"name": "diagnose_pod", "arguments": {"pod": "web-1", "namespace": "front"}}],
        "usage": {
            "iterations": 2,
            "tool_calls": 1,
            "diagnostic_calls": 1,
            "wall_time_seconds": 0.25,
        },
    }


@pytest.mark.parametrize("success", (True, False))
def test_runner_projects_only_original_upstream_verdict(
    success: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    case = campaign.cases[0]
    captured: dict[str, Any] = {}

    def run(_serving: KorvidUpstreamServing, payload: dict[str, Any]) -> dict[str, Any]:
        captured.update(payload)
        return _worker_result(case, success=success)

    monkeypatch.setattr("korvid_prompt_lab.upstream.run_upstream_request", run)
    result = KorvidUpstreamRunner(campaign).run(
        prompt_candidate("optimized prompt"),
        case,
        tmp_path / "run",
        repetition=2,
        seed=5,
    )

    assert isinstance(result, EvaluationResult)
    assert result.success is success
    assert result.hard_failures == ()
    feedback = result.feedback
    assert feedback["success"] is success
    assert feedback["upstream"]["outcome"] == (
        "success" if success else "failure"
    )
    assert feedback["turns"] == _worker_result(
        case, success=success
    )["source_report"]["runs"]
    assert feedback["reference"] == "scenarios/image-pull-typo"
    assert feedback["source_sha256"] == _worker_result(
        case, success=success
    )["source_sha256"]
    assert feedback["questions"] == [case.prompt]
    assert feedback["policy"] == {
        "tier": "low",
        "prompt_pack": "low-korvid-operator",
        "tools": ["diagnose_pod"],
    }
    assert result.usage == {
        "iterations": 2,
        "tool_calls": 1,
        "diagnostic_calls": 1,
        "wall_time_seconds": 0.25,
    }
    assert captured["reference"] == "scenarios/image-pull-typo"
    assert captured["tier_pack"] == "optimized prompt"
    assert captured["model"]["options"] == {"temperature": 0.0, "seed": 5}

    response = json.loads((tmp_path / "run/response.json").read_text())
    summary = json.loads((tmp_path / "run/upstream-summary.json").read_text())
    assert response["protocol_version"] == 1
    assert response["success"] is success
    assert response["hard_failures"] == []
    assert response["feedback"] == feedback
    for legacy in ("status", "grade", "journal", "answer", "error"):
        assert legacy not in response
    assert response["source_identity"]["source_sha256"] == (
        summary["source_sha256"]
    )
    assert response["runtime"]["fingerprint"] == summary["runtime"]["fingerprint"]
    assert response["source_report"] == _worker_result(
        case, success=success
    )["source_report"]
    assert summary["source_report"] == response["source_report"]
    assert summary["upstream"]["outcome"] == (
        "success" if success else "failure"
    )
    assert summary["runtime"]["fingerprint"] == "b" * 64
    assert "original_payload" not in summary


def test_runner_preserves_explicit_invalid_argument_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    case = campaign.cases[0]
    invalid_call = {
        "name": "diagnose_pod",
        "arguments_valid": False,
        "arguments_raw": "{not-json",
        "error_label": "malformed_arguments",
    }
    response = _worker_result(case, success=False)
    response["calls"] = [invalid_call]

    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.run_upstream_request",
        lambda *_args, **_kwargs: response,
    )

    result = KorvidUpstreamRunner(campaign).run(
        prompt_candidate("optimized prompt"), case, tmp_path / "run", seed=5
    )

    assert result.feedback["calls"] == [invalid_call]


def test_runner_preserves_every_original_journey_turn_and_grade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign("journeys/tui-follow")
    case = campaign.cases[0]
    worker_result = _worker_result(case, success=True)
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.run_upstream_request",
        lambda *_args, **_kwargs: worker_result,
    )

    result = KorvidUpstreamRunner(campaign).run(
        prompt_candidate("optimized prompt"),
        case,
        tmp_path / "run",
        seed=5,
    )

    original_turns = worker_result["source_report"]["runs"][0]["turns"]
    assert result.success is True
    assert result.feedback["questions"] == json.loads(case.prompt)
    assert result.feedback["turns"] == original_turns
    assert result.feedback["upstream"]["turns"] == original_turns
    persisted = json.loads((tmp_path / "run/response.json").read_text())
    assert persisted["source_report"] == worker_result["source_report"]


def test_runner_rejects_case_changed_from_source_catalog_before_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    altered = EvalCase(
        case_id=campaign.cases[0].case_id,
        template_id=campaign.cases[0].template_id,
        prompt="changed question",
        models=campaign.cases[0].models,
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.run_upstream_request",
        lambda *_args, **_kwargs: pytest.fail("worker must not be invoked"),
    )

    with pytest.raises(ValueError, match="source"):
        KorvidUpstreamRunner(campaign).run(
            prompt_candidate("optimized prompt"), altered, tmp_path / "run"
        )


def test_runner_propagates_systemic_provider_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.run_upstream_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            BridgeInvocationError("upstream provider/runtime failed")
        ),
    )

    with pytest.raises(BridgeInvocationError):
        KorvidUpstreamRunner(campaign).run(
            prompt_candidate("optimized prompt"),
            campaign.cases[0],
            tmp_path / "run",
        )


def test_runner_rejects_worker_source_path_different_from_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    case = campaign.cases[0]
    worker_result = _worker_result(case, success=True)
    worker_result["source_path"] = "src/korvid/evals/scenarios/different.yaml"
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.run_upstream_request",
        lambda *_args, **_kwargs: worker_result,
    )

    with pytest.raises(ValueError, match="source_path"):
        KorvidUpstreamRunner(campaign).run(
            prompt_candidate("optimized prompt"), case, tmp_path / "run", seed=5
        )


def test_runner_rejects_a_worker_that_did_not_use_the_low_prompt_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    case = campaign.cases[0]
    worker_result = _worker_result(case, success=True)
    worker_result["policy"] = {
        "tier": "high",
        "prompt_pack": "high-korvid-operator",
        "tools": ["diagnose_pod"],
    }
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.run_upstream_request",
        lambda *_args, **_kwargs: worker_result,
    )

    with pytest.raises(ValueError, match="low"):
        KorvidUpstreamRunner(campaign).run(
            prompt_candidate("optimized prompt"), case, tmp_path / "run", seed=5
        )


def test_runner_rejects_a_verdict_that_disagrees_with_original_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    case = campaign.cases[0]
    worker_result = _worker_result(case, success=True)
    worker_result["upstream"]["outcome"] = "failure"
    worker_result["upstream"]["failure_class"] = "misdiagnosis"
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.run_upstream_request",
        lambda *_args, **_kwargs: worker_result,
    )

    with pytest.raises(ValueError, match="verdict"):
        KorvidUpstreamRunner(campaign).run(
            prompt_candidate("optimized prompt"), case, tmp_path / "run", seed=5
        )


def test_feedback_keeps_bounded_upstream_error_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    case = campaign.cases[0]
    worker_result = _worker_result(case, success=False)
    worker_result["upstream"]["outcome"] = "error"
    worker_result["upstream"]["failure_class"] = "provider_error"
    worker_result["upstream"]["error_label"] = "model_bound"
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.run_upstream_request",
        lambda *_args, **_kwargs: worker_result,
    )

    result = KorvidUpstreamRunner(campaign).run(
        prompt_candidate("optimized prompt"), case, tmp_path / "run", seed=5
    )

    assert result.feedback["error_labels"] == [
        "provider_error",
        "model_bound",
    ]
