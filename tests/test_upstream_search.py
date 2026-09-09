from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from current_helpers import MODEL, campaign, candidate, case, serving

from korvid_prompt_lab.runner import BridgeInvocationError
from korvid_prompt_lab.upstream import KorvidUpstreamRunner, validate_upstream_candidate
from korvid_prompt_lab.upstream_contract import (
    KORVID_REVISION,
    KORVID_VERSION,
    SourceCase,
)


def source_case() -> SourceCase:
    return SourceCase(
        reference="scenarios/image-pull-typo",
        case_id="image-pull-typo",
        kind="scenario",
        prompt="Diagnose image-pull-typo.",
        source_path="src/korvid/evals/scenarios/image-pull-typo.yaml",
        source_sha256="4" * 64,
        questions=("Diagnose image-pull-typo.",),
    )


def raw_response(
    *, success: bool, calls: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    authored = source_case()
    tier_pack = candidate().components["tier_pack"]
    outcome = "success" if success else "failure"
    grade = {
        "diagnosis_success": success,
        "evidence_fetched": True,
        "missing_mentions": [],
        "forbidden_mentions": [],
        "missing_evidence": [],
    }
    source_run = {
        "answer": "The image tag does not exist.",
        "outcome": outcome,
        "failure_class": None if success else "misdiagnosis",
        "grade": grade,
    }
    return {
        "protocol_version": 1,
        "operation": "evaluate",
        "execution_mode": "scripted",
        "reference": authored.reference,
        "case_id": authored.case_id,
        "kind": authored.kind,
        "source_path": authored.source_path,
        "source_sha256": authored.source_sha256,
        "questions": list(authored.questions),
        "tier_pack_sha256": hashlib.sha256(tier_pack.encode()).hexdigest(),
        "prompt_path_verified": False,
        "model": {
            "reference": MODEL,
            "endpoint": "http://127.0.0.1:11434",
            "options": {"temperature": 0.0, "seed": 0},
        },
        "policy": {
            "source": "low-korvid-operator",
            "tier": "low",
            "prompt_pack": "low-korvid-operator",
            "tools": ["diagnose_pod"],
        },
        "prompt": {"source": "override", "pack": "low-korvid-operator"},
        "runtime": {
            "dependencies": {"korvid": KORVID_VERSION},
            "fingerprint": "b" * 64,
        },
        "success": success,
        "hard_failures": [],
        "source_report": {
            "scenario": authored.case_id,
            "interaction": {"mode": "list"},
            "runs": [source_run],
        },
        "upstream": {
            "outcome": outcome,
            "failure_class": source_run["failure_class"],
            "grade": grade,
        },
        "calls": calls if calls is not None else [],
        "usage": {
            "iterations": 1,
            "tool_calls": len(calls or []),
            "wall_time_seconds": 0.1,
        },
    }


def runner(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]
) -> KorvidUpstreamRunner:
    authored = source_case()
    source_eval_case = case(authored.case_id)
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream._source_catalog",
        lambda _campaign: {authored.case_id: authored},
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.run_upstream_request",
        lambda _serving, _payload: response,
    )
    return KorvidUpstreamRunner(campaign([source_eval_case]))


@pytest.mark.parametrize("success", [False, True])
def test_upstream_runner_returns_the_exact_source_verdict_and_direct_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    success: bool,
) -> None:
    source_runner = runner(monkeypatch, raw_response(success=success))
    source_eval_case = source_runner.campaign.cases[0]

    result = source_runner.run(candidate(), source_eval_case, tmp_path)

    assert result.success is success
    assert result.execution_mode == "scripted"
    assert result.candidate_fingerprint == candidate().fingerprint
    assert result.feedback["success"] is success
    assert result.feedback["reference"] == "scenarios/image-pull-typo"
    assert "upstream_feedback" not in result.feedback
    assert result.usage == {
        "iterations": 1,
        "tool_calls": 0,
        "wall_time_seconds": 0.1,
    }


def test_upstream_runner_persists_the_original_source_report_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = raw_response(success=False)
    source_runner = runner(monkeypatch, response)

    result = source_runner.run(
        candidate(),
        source_runner.campaign.cases[0],
        tmp_path,
    )
    artifact = json.loads((tmp_path / "response.json").read_text(encoding="utf-8"))

    assert artifact["source_report"] == response["source_report"]
    assert "source_report" not in result.feedback
    assert artifact["source_identity"] == {
        "korvid_revision": KORVID_REVISION,
        "reference": "scenarios/image-pull-typo",
        "source_path": "src/korvid/evals/scenarios/image-pull-typo.yaml",
        "source_sha256": "4" * 64,
        "contract_sha256": artifact["source_identity"]["contract_sha256"],
    }


def test_upstream_runner_preserves_malformed_call_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = {
        "name": "diagnose_pod",
        "arguments_valid": False,
        "arguments_raw": "{not-json",
        "error_label": "malformed_arguments",
    }
    source_runner = runner(
        monkeypatch,
        raw_response(success=False, calls=[malformed]),
    )

    result = source_runner.run(
        candidate(),
        source_runner.campaign.cases[0],
        tmp_path,
    )

    assert result.success is False
    assert result.feedback["calls"] == [malformed]


def test_upstream_runner_rejects_malformed_call_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = {
        "name": "diagnose_pod",
        "arguments_valid": False,
        "arguments_raw": "{not-json",
        "error_label": "unknown",
    }
    source_runner = runner(
        monkeypatch,
        raw_response(success=False, calls=[invalid]),
    )

    with pytest.raises(ValueError, match="malformed tool call evidence"):
        source_runner.run(
            candidate(),
            source_runner.campaign.cases[0],
            tmp_path,
        )


def test_upstream_runner_refuses_to_overwrite_source_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_runner = runner(monkeypatch, raw_response(success=False))
    source_eval_case = source_runner.campaign.cases[0]
    source_runner.run(candidate(), source_eval_case, tmp_path)

    with pytest.raises(FileExistsError, match="response already exists"):
        source_runner.run(candidate(), source_eval_case, tmp_path)


def test_upstream_runner_propagates_source_runtime_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_runner = runner(monkeypatch, raw_response(success=False))
    failure = BridgeInvocationError("source worker failed")

    def fail(*_args: object) -> dict[str, Any]:
        raise failure

    monkeypatch.setattr("korvid_prompt_lab.upstream.run_upstream_request", fail)

    with pytest.raises(BridgeInvocationError) as error:
        source_runner.run(
            candidate(),
            source_runner.campaign.cases[0],
            tmp_path,
        )

    assert error.value is failure


def validation_response() -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "operation": "validate_candidate",
        "model": {
            "reference": MODEL,
            "endpoint": serving().base_url,
            "options": {"temperature": 0.0, "seed": 0},
        },
        "tier_pack_sha256": hashlib.sha256(
            candidate().components["tier_pack"].encode()
        ).hexdigest(),
        "valid": True,
        "error_label": None,
    }


@pytest.mark.parametrize("label", [None, "static_prompt_too_large"])
def test_upstream_candidate_validation_attests_the_exact_text_without_case_access(
    monkeypatch: pytest.MonkeyPatch, label: str | None,
) -> None:
    def invoke(actual_serving: object, payload: dict[str, Any]) -> dict[str, Any]:
        assert actual_serving == serving()
        assert payload == {
            "protocol_version": 1, "operation": "validate_candidate",
            "model": validation_response()["model"],
            "tier_pack": candidate().components["tier_pack"],
        }
        return {**validation_response(), "valid": label is None, "error_label": label}

    monkeypatch.setattr("korvid_prompt_lab.upstream.run_upstream_request", invoke)
    assert validate_upstream_candidate(serving(), MODEL, candidate()) == label


@pytest.mark.parametrize(
    "changed",
    [
        {"protocol_version": 2},
        {"operation": "evaluate"},
        {"model": {"reference": "different"}},
        {"tier_pack_sha256": "0" * 64},
        {"valid": 1},
        {"valid": None},
        {"valid": False, "error_label": None},
        {"valid": False, "error_label": "unknown_prompt_pack"},
        {"valid": True, "error_label": "static_prompt_too_large"},
    ],
)
def test_upstream_candidate_validation_rejects_unattested_or_malformed_verdicts(
    monkeypatch: pytest.MonkeyPatch, changed: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.run_upstream_request",
        lambda *_args: {**validation_response(), **changed},
    )
    with pytest.raises(ValueError, match="candidate validation"):
        validate_upstream_candidate(serving(), MODEL, candidate())


def test_upstream_candidate_validation_never_reclassifies_bridge_error_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = BridgeInvocationError("StaticPromptTooLargeError: static prompt too large")

    def fail(*_args: Any) -> dict[str, Any]:
        raise failure

    monkeypatch.setattr("korvid_prompt_lab.upstream.run_upstream_request", fail)
    with pytest.raises(BridgeInvocationError) as error:
        validate_upstream_candidate(serving(), MODEL, candidate())
    assert error.value is failure
