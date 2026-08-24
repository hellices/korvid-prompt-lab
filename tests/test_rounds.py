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

CHANGED_BEST_CANDIDATE = {
    "schema_version": 1,
    "candidate_id": "candidate-alpha",
    "components": {"system": "Stay grounded and verify every target."},
    "metadata": {},
}
CHANGED_FINGERPRINT = Candidate.from_mapping(CHANGED_BEST_CANDIDATE).fingerprint


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


# ---------------------------------------------------------------------------
# Result contract: the design's Job Summary is per-model scores, per-run elapsed
# duration, artifact names, and a reproducible command — and nothing raw
# ---------------------------------------------------------------------------


def test_round_report_carries_per_model_scores_durations_artifacts_and_command(
    tmp_path: Path,
) -> None:
    report = build_round_report(write_realistic_live_fixture(tmp_path))

    assert report.model_scores == {LIVE_MODEL: 0.01}, (
        "the design requires a per-model score beside the aggregate"
    )
    assert {run.run_id: run.elapsed_seconds for run in report.runs} == LIVE_WALL_TIMES, (
        "per-run elapsed duration must come from the safe usage.wall_time_seconds"
    )
    assert report.reproduction_command == LIVE_REPRODUCTION_COMMAND
    assert report.artifact_refs == (
        "evaluation-summary.json",
        "runs/aks-restart-denied-qwen3-0.6b-r01/response.json",
        "runs/aks-restart-denied-qwen3-0.6b-r02/response.json",
        "runs/aks-scale-deployment-up-qwen3-0.6b-r01/response.json",
        "runs/aks-scale-deployment-up-qwen3-0.6b-r02/response.json",
    ), "artifact names come from the evaluation summary, minus forbidden artifacts"
    assert not any("request.json" in ref for ref in report.artifact_refs), (
        "request payload artifacts must never be named in the round report"
    )


def test_render_round_markdown_shows_scores_durations_artifacts_and_safe_command(
    tmp_path: Path,
) -> None:
    report = build_round_report(write_realistic_live_fixture(tmp_path))
    markdown = render_round_markdown(report)

    assert "## Per-model scores" in markdown
    assert f"| `{LIVE_MODEL}` | 0.010 |" in markdown

    assert "Elapsed (s)" in markdown
    for elapsed in LIVE_WALL_TIMES.values():
        assert f"{elapsed:.3f}" in markdown

    assert "## Artifacts" in markdown
    assert "- `evaluation-summary.json`" in markdown
    assert "request.json" not in markdown, "request artifacts are never displayed"

    assert "## Reproduction command" in markdown
    assert "'artifacts/live/round 1'" in markdown, (
        "the reproduction command must be displayed shell-quoted so copying it "
        "cannot silently split or re-interpret an argument"
    )
    assert markdown == render_round_markdown(report), "rendering must be deterministic"
    assert "raw answer" not in markdown


def test_round_report_never_names_forbidden_artifacts(tmp_path: Path) -> None:
    artifact_root = write_realistic_live_fixture(
        tmp_path,
        artifact_refs=[
            "evaluation-summary.json",
            "runs/aks-restart-denied-qwen3-0.6b-r01/request.json",
            "runs/aks-restart-denied-qwen3-0.6b-r01/audit.jsonl",
            "runs/aks-restart-denied-qwen3-0.6b-r01/response.json",
            ".kubeconfig-round.yaml",
            "bridge-stderr.log",
            "gepa_state.bin",
            "reflection-credential.json",
        ],
    )

    report = build_round_report(artifact_root)
    markdown = render_round_markdown(report)

    assert report.artifact_refs == (
        "evaluation-summary.json",
        "runs/aks-restart-denied-qwen3-0.6b-r01/response.json",
    )
    for forbidden in (
        "request.json",
        "audit.jsonl",
        "kubeconfig",
        ".log",
        "gepa_state",
        "credential",
    ):
        assert forbidden not in markdown


@pytest.mark.parametrize(
    "unsafe_ref",
    ["/etc/passwd", "../../escape.json", "runs/../../escape.json", "", "runs/\u0000.json"],
)
def test_round_report_rejects_unsafe_artifact_ref_paths(
    tmp_path: Path, unsafe_ref: str
) -> None:
    artifact_root = write_realistic_live_fixture(
        tmp_path, artifact_refs=["evaluation-summary.json", unsafe_ref]
    )

    with pytest.raises(ValueError, match="artifact_refs"):
        build_round_report(artifact_root)


@pytest.mark.parametrize("unsafe_token", ["kubectl\nrm -rf /", "kubectl\u0000"])
def test_round_report_rejects_unsafe_reproduction_command_tokens(
    tmp_path: Path, unsafe_token: str
) -> None:
    artifact_root = write_realistic_live_fixture(
        tmp_path, reproduction_command=["uv", "run", unsafe_token]
    )

    with pytest.raises(ValueError, match="reproduction_command"):
        build_round_report(artifact_root)


@pytest.mark.parametrize("wall_time", [-1.0, float("inf"), float("nan")])
def test_round_report_rejects_an_unusable_elapsed_duration(
    tmp_path: Path, wall_time: float
) -> None:
    artifact_root = write_live_fixture(tmp_path)
    response_path = artifact_root / "runs" / "case-a-model-a-r01" / "response.json"
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    payload["usage"]["wall_time_seconds"] = wall_time
    response_path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="wall_time_seconds"):
        build_round_report(artifact_root)


def test_round_report_reports_no_duration_for_a_model_failure(tmp_path: Path) -> None:
    artifact_root = write_live_fixture(
        tmp_path,
        responses=[response("model_failure", error="turn timeout")],
        aggregate_score=0.0,
        pass_at_3=0.0,
        pass_at_5=0.0,
        milestone_passed=False,
    )

    report = build_round_report(artifact_root)
    markdown = render_round_markdown(report)

    assert [run.elapsed_seconds for run in report.runs] == [None], (
        "a model failure carries no usage block, so its duration is unknown"
    )
    assert "| n/a |" in markdown


def test_safe_round_summary_json_publishes_the_full_result_contract(
    tmp_path: Path,
) -> None:
    artifact_root = write_realistic_live_fixture(tmp_path)
    safe_output = tmp_path / "safe-evidence"

    write_safe_evidence(
        artifact_root,
        safe_output,
        prompt_lab_revision="0" * 40,
        korvid_revision="1" * 40,
        workflow_run_url="https://github.example/actions/runs/42",
    )

    payload = json.loads((safe_output / "round-summary.json").read_text(encoding="utf-8"))

    assert payload["model_scores"] == {LIVE_MODEL: 0.01}
    assert {run["run_id"]: run["elapsed_seconds"] for run in payload["runs"]} == LIVE_WALL_TIMES
    assert payload["reproduction_command"] == list(LIVE_REPRODUCTION_COMMAND)
    assert payload["evaluation_artifact_refs"] == list(
        build_round_report(artifact_root).artifact_refs
    )
    assert sorted(payload["artifact_refs"]) == sorted(
        path.relative_to(safe_output).as_posix()
        for path in safe_output.rglob("*")
        if path.is_file()
    ), "the package manifest must name exactly the files that were written"

    written = "\n".join(
        path.read_text(encoding="utf-8") for path in safe_output.rglob("*") if path.is_file()
    )
    for forbidden in ("raw answer", "SECRET REQUEST PROMPT", "request.json"):
        assert forbidden not in written


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
    wall_time_seconds: float = 1.25,
    tool_calls: int = 0,
    iterations: int = 1,
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
        payload["usage"] = {
            "tool_calls": tool_calls,
            "iterations": iterations,
            "wall_time_seconds": wall_time_seconds,
        }
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
    model_scores: Mapping[str, float] | None = None,
    artifact_refs: Sequence[str] | None = None,
    reproduction_command: Sequence[str] | None = None,
    write_request_artifacts: bool = False,
    campaign_id: str = "campaign-2026-08-22",
    candidate: Mapping[str, Any] = DEFAULT_BEST_CANDIDATE,
    seed_candidate_fingerprint: str | None = None,
    systemic_failures: int = 0,
) -> Path:
    resolved_candidate = Candidate.from_mapping(candidate)
    resolved_seed_fingerprint = seed_candidate_fingerprint or resolved_candidate.fingerprint

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    raw_responses = list(responses or [response("completed")])
    case_ids = list(dict.fromkeys(evaluated_case_ids or [str(item["request_identity"]["case_id"]) for item in raw_responses]))
    models = list(dict.fromkeys(str(item["request_identity"]["model"]) for item in raw_responses))
    pairs = list(dict.fromkeys(_pair(item) for item in raw_responses if item["request_identity"]["case_id"] in case_ids))
    run_execution_modes = {pair: "live" for pair in pairs}
    run_ids = [str(item["run_id"]) for item in raw_responses]

    #: The evaluator records every JSON/YAML artifact it wrote, request payloads
    #: included; the round report must never surface those names.
    default_artifact_refs = ["evaluation-summary.json"]
    for run_id in sorted(run_ids):
        default_artifact_refs.append(f"runs/{run_id}/request.json")
        default_artifact_refs.append(f"runs/{run_id}/response.json")

    summary = {
        "bundle_kind": "common",
        "candidate_id": resolved_candidate.candidate_id,
        "candidate_fingerprint": resolved_candidate.fingerprint,
        "campaign_id": campaign_id,
        "campaign_case_ids": list(case_ids),
        "evaluated_case_ids": list(case_ids),
        "evaluated_models": list(models),
        "campaign_case_model_pairs": list(pairs),
        "evaluated_case_model_pairs": list(pairs),
        "aggregate_score": aggregate_score,
        "model_scores": dict(model_scores) if model_scores is not None else {model: aggregate_score for model in models},
        "execution_modes": list(execution_modes),
        "run_execution_modes": run_execution_modes,
        "repetitions_per_case": repetitions_per_case,
        "pass_at_3": pass_at_3,
        "pass_at_5": pass_at_5,
        "hard_safety_failures": sum(len((item.get("grade") or {}).get("hard_failures", [])) for item in raw_responses),
        "systemic_failures": systemic_failures,
        "milestone_passed": milestone_passed,
        "case_sets": {"train": list(case_ids), "validation": list(case_ids), "milestone": list(case_ids)},
        "artifact_refs": list(artifact_refs) if artifact_refs is not None else default_artifact_refs,
        "reproduction_command": list(reproduction_command)
        if reproduction_command is not None
        else ["uv", "run", "--python", "3.12", "korvid-prompt-lab", "evaluate"],
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
                        "campaign_id": campaign_id,
                        "candidate_id": resolved_candidate.candidate_id,
                        "seed_candidate_fingerprint": resolved_seed_fingerprint,
                        "train_case_ids": list(case_ids),
                        "validation_case_ids": list(case_ids),
                        "max_metric_calls": 1,
                        "seed": 0,
                        "proposal_source": "none",
                    },
                    "invocation_dir": str(artifact_root / "invocations" / "opt-run-1"),
                    "best_idx": 0,
                    "best_validation_score": aggregate_score,
                    "best_candidate_fingerprint": resolved_candidate.fingerprint,
                    "seed_candidate_fingerprint": resolved_seed_fingerprint,
                    "best_candidate_differs_from_seed": (
                        resolved_candidate.fingerprint != resolved_seed_fingerprint
                    ),
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
            yaml.safe_dump(
                {
                    "schema_version": resolved_candidate.schema_version,
                    "candidate_id": resolved_candidate.candidate_id,
                    "components": resolved_candidate.components,
                    "metadata": resolved_candidate.metadata,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
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
        if write_request_artifacts:
            (run_dir / "request.json").write_text(
                json.dumps(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "request_identity": payload["request_identity"],
                        "prompt": "SECRET REQUEST PROMPT",
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    return artifact_root


#: A realistic live round: one allowlisted model, the two shipped AKS cases, two
#: repetitions each, and the artifact names an actual evaluate run records.
LIVE_MODEL = "qwen3:0.6b"
LIVE_CASES = ("aks-scale-deployment-up", "aks-restart-denied")
LIVE_WALL_TIMES = {
    "aks-scale-deployment-up-qwen3-0.6b-r01": 61.42,
    "aks-scale-deployment-up-qwen3-0.6b-r02": 58.004,
    "aks-restart-denied-qwen3-0.6b-r01": 35.87,
    "aks-restart-denied-qwen3-0.6b-r02": 41.2,
}
LIVE_REPRODUCTION_COMMAND = (
    "uv",
    "run",
    "--python",
    "3.12",
    "korvid-prompt-lab",
    "evaluate",
    "--candidate",
    "examples/candidates/shipped-small.yaml",
    "--campaign",
    "examples/campaigns/aks-shared-runners.yaml",
    "--artifact-root",
    "artifacts/live/round 1",
    "--bundle-kind",
    "common",
)


def write_realistic_live_fixture(tmp_path: Path, **overrides: Any) -> Path:
    responses = [
        response(
            "completed",
            case_id=case_id,
            model=LIVE_MODEL,
            repetition=repetition,
            run_id=f"{case_id}-qwen3-0.6b-r{repetition:02d}",
            completion=0.0,
            verification=0.0,
            efficiency=1.0,
            hard_failures=["write_before_fresh_read"] if case_id == LIVE_CASES[1] else [],
            answer="raw answer the model produced",
            wall_time_seconds=LIVE_WALL_TIMES[f"{case_id}-qwen3-0.6b-r{repetition:02d}"],
            tool_calls=1,
            iterations=2,
        )
        for case_id in LIVE_CASES
        for repetition in (1, 2)
    ]
    kwargs: dict[str, Any] = {
        "responses": responses,
        "repetitions_per_case": 2,
        "aggregate_score": 0.01,
        "pass_at_3": 0.0,
        "pass_at_5": 0.0,
        "milestone_passed": False,
        "model_scores": {LIVE_MODEL: 0.01},
        "reproduction_command": list(LIVE_REPRODUCTION_COMMAND),
        "write_request_artifacts": True,
        "campaign_id": "aks-shared-runners",
    }
    kwargs.update(overrides)
    return write_live_fixture(tmp_path, **kwargs)


def _pair(payload: Mapping[str, Any]) -> str:
    identity = payload["request_identity"]
    return f"{identity['case_id']}::{identity['model']}"


def all_safe_text(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
    )


def test_safe_evidence_renders_comparison_before_collapsed_detail(tmp_path: Path) -> None:
    before_root = write_live_fixture(
        tmp_path / "before",
        aggregate_score=0.1,
        responses=[
            response("completed", hard_failures=["wrong_target_write"], answer="before raw"),
            response(
                "completed",
                run_id="case-a-model-a-r02",
                repetition=2,
                hard_failures=["wrong_target_write"],
                answer="before raw",
            ),
        ],
        repetitions_per_case=2,
    )
    after_root = write_live_fixture(
        tmp_path / "after",
        candidate=CHANGED_BEST_CANDIDATE,
        aggregate_score=0.2,
        responses=[
            response(
                "completed",
                candidate_fingerprint=CHANGED_FINGERPRINT,
                answer="after raw",
            ),
            response(
                "completed",
                run_id="case-a-model-a-r02",
                repetition=2,
                candidate_fingerprint=CHANGED_FINGERPRINT,
                hard_failures=["wrong_target_write"],
                answer="after raw",
            ),
        ],
        repetitions_per_case=2,
        include_optimization=True,
        include_best_candidate=True,
        seed_candidate_fingerprint=FINGERPRINT,
    )

    output = write_safe_evidence(
        after_root,
        tmp_path / "safe",
        before_artifact_root=before_root,
        optimize_artifact_root=after_root,
        prompt_lab_revision="prompt-sha",
        korvid_revision="korvid-sha",
        workflow_run_url="https://github.example/actions/runs/42",
    )

    markdown = (output / "round-summary.md").read_text(encoding="utf-8")
    assert markdown.index("# Grounding Round Outcome") < markdown.index("<details>")
    assert "✅ improved" in markdown
    assert "<summary>Detailed round evidence</summary>" in markdown
    assert (output / "comparison-summary.json").is_file()
    assert (output / "before-evaluation-summary.json").is_file()
    assert len(list((output / "before-responses").glob("*.json"))) == 2
    assert "before raw" not in all_safe_text(output)
    assert "after raw" not in all_safe_text(output)


def test_before_evaluation_summary_projects_only_safe_artifact_refs(tmp_path: Path) -> None:
    """The before summary must publish only in-package safe refs, never the
    seed run's raw request/audit/kubeconfig/credential/GEPA artifact names."""
    forbidden_refs = [
        "runs/case-a-model-a-r01/request.json",
        "runs/case-a-model-a-r01/audit.jsonl",
        "runs/case-a-model-a-r01/response.json",
        ".kubeconfig-round.yaml",
        "reflection-credential.json",
        "gepa_state.bin",
    ]
    before_root = write_live_fixture(
        tmp_path / "before",
        aggregate_score=0.1,
        responses=[
            response("completed", hard_failures=["wrong_target_write"], answer="before raw"),
            response(
                "completed",
                run_id="case-a-model-a-r02",
                repetition=2,
                hard_failures=["wrong_target_write"],
                answer="before raw",
            ),
        ],
        repetitions_per_case=2,
        artifact_refs=["evaluation-summary.json", *forbidden_refs],
    )
    after_root = write_live_fixture(
        tmp_path / "after",
        candidate=CHANGED_BEST_CANDIDATE,
        aggregate_score=0.2,
        responses=[
            response("completed", candidate_fingerprint=CHANGED_FINGERPRINT, answer="after raw"),
            response(
                "completed",
                run_id="case-a-model-a-r02",
                repetition=2,
                candidate_fingerprint=CHANGED_FINGERPRINT,
                hard_failures=["wrong_target_write"],
                answer="after raw",
            ),
        ],
        repetitions_per_case=2,
        include_optimization=True,
        include_best_candidate=True,
        seed_candidate_fingerprint=FINGERPRINT,
    )

    output = write_safe_evidence(
        after_root,
        tmp_path / "safe",
        before_artifact_root=before_root,
        optimize_artifact_root=after_root,
    )

    before_summary = json.loads(
        (output / "before-evaluation-summary.json").read_text(encoding="utf-8")
    )
    refs = before_summary["artifact_refs"]
    assert refs[0] == "before-evaluation-summary.json"
    assert all(
        ref == "before-evaluation-summary.json" or ref.startswith("before-responses/")
        for ref in refs
    ), refs
    assert sorted(ref for ref in refs if ref.startswith("before-responses/")) == [
        "before-responses/case-a-model-a-r01.json",
        "before-responses/case-a-model-a-r02.json",
    ]

    package_text = all_safe_text(output)
    # response.json is a legitimate displayable name the after run records, so it
    # is allowed elsewhere; the raw request/audit/kubeconfig/credential/GEPA
    # names must never appear anywhere in the published package.
    for forbidden in (
        "runs/case-a-model-a-r01/request.json",
        "runs/case-a-model-a-r01/audit.jsonl",
        ".kubeconfig-round.yaml",
        "reflection-credential.json",
        "gepa_state.bin",
        "kubeconfig",
        "credential",
        "gepa_state",
    ):
        assert forbidden not in package_text, forbidden


def test_round_summary_starts_with_outcome_heading_even_with_metadata(
    tmp_path: Path,
) -> None:
    """Even when workflow run URL and both revisions are present (the production
    case), the decision headline must be the first heading on the page."""
    before_root = write_live_fixture(
        tmp_path / "before",
        aggregate_score=0.1,
        responses=[
            response("completed", answer="before raw"),
            response(
                "completed",
                run_id="case-a-model-a-r02",
                repetition=2,
                answer="before raw",
            ),
        ],
        repetitions_per_case=2,
    )
    after_root = write_live_fixture(
        tmp_path / "after",
        candidate=CHANGED_BEST_CANDIDATE,
        aggregate_score=0.2,
        responses=[
            response("completed", candidate_fingerprint=CHANGED_FINGERPRINT, answer="after raw"),
            response(
                "completed",
                run_id="case-a-model-a-r02",
                repetition=2,
                candidate_fingerprint=CHANGED_FINGERPRINT,
                answer="after raw",
            ),
        ],
        repetitions_per_case=2,
        include_optimization=True,
        include_best_candidate=True,
        seed_candidate_fingerprint=FINGERPRINT,
    )

    output = write_safe_evidence(
        after_root,
        tmp_path / "safe",
        before_artifact_root=before_root,
        optimize_artifact_root=after_root,
        prompt_lab_revision="prompt-sha",
        korvid_revision="korvid-sha",
        workflow_run_url="https://github.example/actions/runs/42",
    )

    markdown = (output / "round-summary.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Grounding Round Outcome\n")
    # Metadata is retained, but below the decision surface, inside <details>.
    assert "Workflow run: https://github.example/actions/runs/42" in markdown
    assert markdown.index("<details>") < markdown.index("Workflow run:")
    assert markdown.index("# Grounding Round Outcome") < markdown.index("Prompt Lab revision")


def test_unchanged_candidate_reuses_final_evidence_without_duplication(tmp_path: Path) -> None:
    root = write_live_fixture(
        tmp_path,
        include_optimization=True,
        include_best_candidate=True,
    )

    output = write_safe_evidence(
        root,
        tmp_path / "safe",
        before_artifact_root=root,
        optimize_artifact_root=root,
    )

    payload = json.loads((output / "comparison-summary.json").read_text(encoding="utf-8"))
    assert payload["status"] == "unchanged"
    assert "➖ UNCHANGED" in (output / "round-summary.md").read_text(encoding="utf-8")
    assert not (output / "before-responses").exists()
    assert not (output / "before-evaluation-summary.json").exists()


def test_evaluate_only_summary_has_single_evaluation_headline(tmp_path: Path) -> None:
    root = write_live_fixture(tmp_path)
    output = write_safe_evidence(root, tmp_path / "safe")
    markdown = (output / "round-summary.md").read_text(encoding="utf-8")
    assert "ℹ️ SINGLE EVALUATION — no before/after pair" in markdown
    assert "Before vs after" not in markdown
    assert "| Aggregate score | 1.000 |" in markdown
    assert markdown.index("| Aggregate score | 1.000 |") < markdown.index("<details>")


def test_round_cli_passes_before_artifact_root_to_write_safe_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before_root = write_live_fixture(tmp_path / "before", aggregate_score=0.5)
    after_root = write_live_fixture(
        tmp_path / "after",
        candidate=CHANGED_BEST_CANDIDATE,
        aggregate_score=0.8,
        responses=[response("completed", candidate_fingerprint=CHANGED_FINGERPRINT)],
        include_optimization=True,
        include_best_candidate=True,
        seed_candidate_fingerprint=FINGERPRINT,
    )
    safe_output = tmp_path / "safe-evidence"

    exit_code = main(
        [
            "--artifact-root",
            str(after_root),
            "--before-artifact-root",
            str(before_root),
            "--optimize-artifact-root",
            str(after_root),
            "--safe-output",
            str(safe_output),
            "--prompt-lab-revision",
            "abc1234",
            "--korvid-revision",
            "def5678",
            "--workflow-run-url",
            "https://github.example/actions/runs/99",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert (safe_output / "comparison-summary.json").is_file()
    comparison = json.loads((safe_output / "comparison-summary.json").read_text(encoding="utf-8"))
    assert comparison["status"] == "changed"


def test_publication_bullet_blocked_precedes_details(tmp_path: Path) -> None:
    """Blocked publication bullet appears before <details> in round-summary.md."""
    artifact_root = write_live_fixture(tmp_path / "blocked", milestone_passed=False)
    safe_output = tmp_path / "out"
    write_safe_evidence(artifact_root, safe_output)
    md = (safe_output / "round-summary.md").read_text(encoding="utf-8")
    assert "- Publication: blocked" in md
    assert md.index("- Publication:") < md.index("<details>")


def test_publication_bullet_eligible_precedes_details(tmp_path: Path) -> None:
    """Eligible publication bullet appears before <details> in round-summary.md."""
    artifact_root = write_live_fixture(tmp_path / "eligible", milestone_passed=True)
    safe_output = tmp_path / "out"
    write_safe_evidence(artifact_root, safe_output)
    md = (safe_output / "round-summary.md").read_text(encoding="utf-8")
    assert "- Publication: eligible" in md
    assert md.index("- Publication:") < md.index("<details>")
