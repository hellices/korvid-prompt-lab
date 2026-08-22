from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from korvid_prompt_lab.bridge_worker import EXECUTION_MODE_LIVE, PROTOCOL_VERSION
from korvid_prompt_lab.contracts import Candidate
from korvid_prompt_lab.round_cli import main
from korvid_prompt_lab.rounds import (
    build_round_report,
    render_round_markdown,
    write_safe_evidence,
)

DEFAULT_BEST_CANDIDATE = {
    "schema_version": 1,
    "candidate_id": "candidate-alpha",
    "components": {"system": "Stay grounded."},
    "metadata": {},
}
FINGERPRINT = Candidate.from_mapping(DEFAULT_BEST_CANDIDATE).fingerprint


def test_build_round_report_groups_safe_failures_without_raw_payloads(tmp_path: Path) -> None:
    artifact_root = write_live_fixture(
        tmp_path,
        aggregate_score=0.01,
        pass_at_3=0.0,
        pass_at_5=0.0,
        milestone_passed=False,
        responses=[
            response("completed", completion=0.0, hard_failures=["wrong_target_write"], answer="raw answer"),
            response("model_failure", case_id="case-b", error="turn timeout"),
        ],
    )

    report = build_round_report(artifact_root)
    markdown = render_round_markdown(report)

    assert report.hard_failure_counts == {"wrong_target_write": 1}
    assert report.status_counts == {"completed": 1, "model_failure": 1}
    assert report.promotion_eligible is False
    assert "wrong_target_write" in markdown
    assert "raw answer" not in markdown
    assert "audit" not in markdown.lower()


@pytest.mark.parametrize("forbidden", ["audit.jsonl", "request.json", ".kubeconfig-x.yaml", "gepa_state.bin"])
def test_write_safe_evidence_never_copies_forbidden_files(tmp_path: Path, forbidden: str) -> None:
    artifact_root = write_live_fixture(tmp_path)
    path = artifact_root / "runs" / "case-a-model-a-r01" / forbidden
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("SECRET", encoding="utf-8")

    output = write_safe_evidence(artifact_root, tmp_path / "safe")

    assert not any(item.name == forbidden for item in output.rglob("*"))
    assert "SECRET" not in "\n".join(
        item.read_text(encoding="utf-8") for item in output.rglob("*") if item.is_file()
    )
    assert "raw answer" not in "\n".join(item.read_text(encoding="utf-8") for item in output.rglob("*") if item.is_file())


def test_build_round_report_rejects_response_fingerprint_mismatch(tmp_path: Path) -> None:
    artifact_root = write_live_fixture(
        tmp_path,
        responses=[response("completed", candidate_fingerprint="f" * 64)],
    )

    with pytest.raises(ValueError, match="fingerprint"):
        build_round_report(artifact_root)


def test_build_round_report_rejects_missing_duplicate_and_extra_evidence(tmp_path: Path) -> None:
    artifact_root = write_live_fixture(
        tmp_path,
        responses=[
            response("completed", repetition=1),
            response("completed", repetition=1, run_id="case-a-model-a-r01-duplicate"),
            response("completed", case_id="case-b", run_id="case-b-model-a-r01"),
        ],
        evaluated_case_ids=["case-a"],
        repetitions_per_case=2,
    )

    with pytest.raises(ValueError, match="evidence"):
        build_round_report(artifact_root)


def test_build_round_report_rejects_unknown_evaluation_summary_fields(tmp_path: Path) -> None:
    artifact_root = write_live_fixture(tmp_path)
    summary_path = artifact_root / "evaluation-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["raw_answer"] = "SECRET"
    summary_path.write_text(json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown field"):
        build_round_report(artifact_root)


def test_write_safe_evidence_rejects_nested_response_leaks(tmp_path: Path) -> None:
    artifact_root = write_live_fixture(tmp_path)
    response_path = artifact_root / "runs" / "case-a-model-a-r01" / "response.json"
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    payload["journal"]["audit"] = "SECRET"
    payload["usage"]["stdout"] = "LEAK"
    response_path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="journal|usage"):
        write_safe_evidence(artifact_root, tmp_path / "safe")


def test_write_safe_evidence_rejects_best_candidate_that_does_not_match_summary(tmp_path: Path) -> None:
    artifact_root = write_live_fixture(tmp_path, include_best_candidate=True)
    best_candidate_path = artifact_root / "best-candidate.yaml"
    best_candidate_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "candidate_id": "candidate-alpha",
                "components": {"system": "Different prompt"},
                "metadata": {},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="best-candidate"):
        write_safe_evidence(artifact_root, tmp_path / "safe")


def test_round_cli_writes_safe_package_and_prints_markdown_path_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact_root = write_live_fixture(tmp_path, include_optimization=True, include_best_candidate=True)
    safe_output = tmp_path / "safe-evidence"

    exit_code = main(
        [
            "--artifact-root",
            str(artifact_root),
            "--safe-output",
            str(safe_output),
            "--prompt-lab-revision",
            "1234567",
            "--korvid-revision",
            "89abcde",
            "--workflow-run-url",
            "https://github.example/actions/runs/42",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == f"{safe_output / 'round-summary.md'}\n"
    assert sorted(path.relative_to(safe_output).as_posix() for path in safe_output.rglob("*") if path.is_file()) == [
        "best-candidate.yaml",
        "evaluation-summary.json",
        "optimization-summary.json",
        "responses/case-a-model-a-r01.json",
        "round-summary.json",
        "round-summary.md",
    ]
    summary_payload = json.loads((safe_output / "round-summary.json").read_text(encoding="utf-8"))
    assert summary_payload["prompt_lab_revision"] == "1234567"
    assert summary_payload["korvid_revision"] == "89abcde"
    assert summary_payload["workflow_run_url"] == "https://github.example/actions/runs/42"
    assert "raw answer" not in (safe_output / "round-summary.md").read_text(encoding="utf-8")


def test_round_cli_accepts_a_separate_optimize_artifact_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluation_root = write_live_fixture(tmp_path)
    optimize_root = tmp_path / "optimize" / "invocations" / "opt-run-1"
    optimize_root.mkdir(parents=True, exist_ok=True)
    (optimize_root / "optimization-summary.json").write_text(
        json.dumps(
            {
                "run_id": "opt-run-1",
                "seed": 0,
                "run_identity": {
                    "schema_version": 1,
                    "campaign_id": "campaign-2026-08-22",
                    "candidate_id": "candidate-alpha",
                    "seed_candidate_fingerprint": FINGERPRINT,
                    "train_case_ids": ["case-a"],
                    "validation_case_ids": ["case-a"],
                    "max_metric_calls": 1,
                    "seed": 0,
                    "proposal_source": "none",
                },
                "invocation_dir": str(optimize_root),
                "best_idx": 0,
                "best_validation_score": 1.0,
                "best_candidate_fingerprint": FINGERPRINT,
                "seed_candidate_fingerprint": FINGERPRINT,
                "best_candidate_differs_from_seed": False,
                "execution_modes": ["live"],
                "train_case_ids": ["case-a"],
                "validation_case_ids": ["case-a"],
                "num_candidates": 1,
                "total_metric_calls": 1,
                "num_full_val_evals": 1,
                "run_dir": str(optimize_root / "gepa"),
            },
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (optimize_root / "best-candidate.yaml").write_text(
        yaml.safe_dump(DEFAULT_BEST_CANDIDATE, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    safe_output = tmp_path / "safe-evidence"
    exit_code = main(
        [
            "--artifact-root",
            str(evaluation_root),
            "--optimize-artifact-root",
            str(optimize_root),
            "--safe-output",
            str(safe_output),
            "--prompt-lab-revision",
            "1234567",
            "--korvid-revision",
            "89abcde",
            "--workflow-run-url",
            "https://github.example/actions/runs/42",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == f"{safe_output / 'round-summary.md'}\n"
    assert (safe_output / "optimization-summary.json").is_file()
    assert (safe_output / "best-candidate.yaml").is_file()


def response(
    status: str,
    *,
    case_id: str = "case-a",
    model: str = "model-a",
    repetition: int = 1,
    completion: float = 1.0,
    verification: float = 1.0,
    efficiency: float = 1.0,
    hard_failures: Sequence[str] = (),
    error: str | None = None,
    answer: str = "raw answer",
    candidate_fingerprint: str = FINGERPRINT,
    run_id: str | None = None,
) -> dict[str, Any]:
    resolved_run_id = run_id or f"{case_id}-{model}-r{repetition:02d}"
    payload: dict[str, Any] = {
        "run_id": resolved_run_id,
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "execution_mode": EXECUTION_MODE_LIVE,
        "candidate_fingerprint": candidate_fingerprint,
        "request_identity": {
            "case_id": case_id,
            "template_id": f"{case_id}-template",
            "model": model,
            "repetition": repetition,
            "seed": repetition - 1,
        },
        "grade": None,
        "answer": "" if status == "model_failure" else answer,
        "journal": {"checkpoints": [], "checkpoint_counts": {}},
        "usage": {},
        "error": error,
    }
    if status == "completed":
        payload["grade"] = {
            "completion": completion,
            "verification": verification,
            "efficiency": efficiency,
            "hard_failures": list(hard_failures),
        }
        payload["journal"] = {
            "journey_id": resolved_run_id,
            "checkpoints": ["goal_received", "outcome_reported"],
            "missing_checkpoints": [],
            "checkpoint_counts": {"goal_received": 1, "outcome_reported": 1},
            "journal_event_count": 2,
            "audit_record_count": 0,
            "hard_failure_count": len(hard_failures),
        }
        payload["usage"] = {"tool_calls": 0, "iterations": 1, "wall_time_seconds": 1.25}
        payload["error"] = None
    return payload


def write_live_fixture(
    tmp_path: Path,
    *,
    aggregate_score: float = 1.0,
    pass_at_3: float | None = 1.0,
    pass_at_5: float | None = 1.0,
    milestone_passed: bool = True,
    responses: Sequence[Mapping[str, Any]] | None = None,
    evaluated_case_ids: Sequence[str] | None = None,
    repetitions_per_case: int = 1,
    execution_modes: Sequence[str] = ("live",),
    include_optimization: bool = False,
    include_best_candidate: bool = False,
) -> Path:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    raw_responses = list(responses or [response("completed")])
    case_ids = list(dict.fromkeys(evaluated_case_ids or [str(item["request_identity"]["case_id"]) for item in raw_responses]))
    models = list(dict.fromkeys(str(item["request_identity"]["model"]) for item in raw_responses))
    pairs = list(dict.fromkeys(_pair(item) for item in raw_responses if item["request_identity"]["case_id"] in case_ids))
    run_execution_modes = {pair: "live" for pair in pairs}

    summary = {
        "bundle_kind": "common",
        "candidate_id": "candidate-alpha",
        "candidate_fingerprint": FINGERPRINT,
        "campaign_id": "campaign-2026-08-22",
        "campaign_case_ids": list(case_ids),
        "evaluated_case_ids": list(case_ids),
        "evaluated_models": list(models),
        "campaign_case_model_pairs": list(pairs),
        "evaluated_case_model_pairs": list(pairs),
        "aggregate_score": aggregate_score,
        "model_scores": {model: aggregate_score for model in models},
        "execution_modes": list(execution_modes),
        "run_execution_modes": run_execution_modes,
        "repetitions_per_case": repetitions_per_case,
        "pass_at_3": pass_at_3,
        "pass_at_5": pass_at_5,
        "hard_safety_failures": sum(len((item.get("grade") or {}).get("hard_failures", [])) for item in raw_responses),
        "systemic_failures": 0,
        "milestone_passed": milestone_passed,
        "case_sets": {"train": list(case_ids), "validation": list(case_ids), "milestone": list(case_ids)},
        "artifact_refs": ["evaluation-summary.json"],
        "reproduction_command": ["uv", "run", "--python", "3.12", "korvid-prompt-lab", "evaluate"],
    }
    (artifact_root / "evaluation-summary.json").write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if include_optimization:
        (artifact_root / "optimization-summary.json").write_text(
            json.dumps(
                {
                    "run_id": "opt-run-1",
                    "seed": 0,
                    "run_identity": {
                        "schema_version": 1,
                        "campaign_id": "campaign-2026-08-22",
                        "candidate_id": "candidate-alpha",
                        "seed_candidate_fingerprint": FINGERPRINT,
                        "train_case_ids": list(case_ids),
                        "validation_case_ids": list(case_ids),
                        "max_metric_calls": 1,
                        "seed": 0,
                        "proposal_source": "none",
                    },
                    "invocation_dir": str(artifact_root / "invocations" / "opt-run-1"),
                    "best_idx": 0,
                    "best_validation_score": aggregate_score,
                    "best_candidate_fingerprint": FINGERPRINT,
                    "seed_candidate_fingerprint": FINGERPRINT,
                    "best_candidate_differs_from_seed": False,
                    "execution_modes": ["live"],
                    "train_case_ids": list(case_ids),
                    "validation_case_ids": list(case_ids),
                    "num_candidates": 1,
                    "total_metric_calls": 1,
                    "num_full_val_evals": 1,
                    "run_dir": str(artifact_root / "invocations" / "opt-run-1" / "gepa"),
                },
                sort_keys=True,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if include_best_candidate:
        (artifact_root / "best-candidate.yaml").write_text(
            yaml.safe_dump(DEFAULT_BEST_CANDIDATE, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    for item in raw_responses:
        run_id = str(item["run_id"])
        run_dir = artifact_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {key: value for key, value in item.items() if key != "run_id"}
        (run_dir / "response.json").write_text(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return artifact_root


def _pair(payload: Mapping[str, Any]) -> str:
    identity = payload["request_identity"]
    return f"{identity['case_id']}::{identity['model']}"
