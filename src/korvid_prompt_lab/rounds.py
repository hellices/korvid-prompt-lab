from __future__ import annotations

import json
import math
import os
import re
import shlex
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml  # type: ignore[import-untyped]

from .artifacts import write_json_artifact
from .bridge_worker import EXECUTION_MODE_LIVE, EXECUTION_MODES, PROTOCOL_VERSION
from .config import load_candidate
from .contracts import _ensure_keys, _require_mapping, _require_string
from .publish import _require_live_execution_modes
from .runner import (
    BridgeMalformedOutputError,
    _coerce_metric,
    _require_optional_response_string,
    _require_response_int,
    _require_response_mapping,
    _require_response_string,
    _require_response_text,
)

_REPETITION_PATTERN = re.compile(r"-r(?P<repetition>\d+)(?:$|[^0-9])")
_RESPONSE_ALLOWED_KEYS = {
    "protocol_version",
    "status",
    "execution_mode",
    "candidate_fingerprint",
    "request_identity",
    "grade",
    "answer",
    "journal",
    "usage",
    "error",
}
_EVALUATION_SUMMARY_ALLOWED_KEYS = {
    "bundle_kind",
    "candidate_id",
    "candidate_fingerprint",
    "campaign_id",
    "campaign_case_ids",
    "evaluated_case_ids",
    "evaluated_models",
    "campaign_case_model_pairs",
    "evaluated_case_model_pairs",
    "aggregate_score",
    "model_scores",
    "execution_modes",
    "run_execution_modes",
    "repetitions_per_case",
    "pass_at_3",
    "pass_at_5",
    "hard_safety_failures",
    "systemic_failures",
    "milestone_passed",
    "case_sets",
    "artifact_refs",
    "reproduction_command",
}
_CASE_SETS_ALLOWED_KEYS = {"train", "validation", "milestone"}
_REQUEST_IDENTITY_ALLOWED_KEYS = {"case_id", "template_id", "model", "repetition", "seed"}
_GRADE_ALLOWED_KEYS = {"completion", "verification", "efficiency", "hard_failures"}
_COMPLETED_JOURNAL_ALLOWED_KEYS = {
    "journey_id",
    "checkpoints",
    "missing_checkpoints",
    "checkpoint_counts",
    "journal_event_count",
    "audit_record_count",
    "hard_failure_count",
}
_MODEL_FAILURE_JOURNAL_ALLOWED_KEYS = {"checkpoints", "checkpoint_counts"}
_COMPLETED_USAGE_ALLOWED_KEYS = {"tool_calls", "iterations", "wall_time_seconds"}
_MODEL_FAILURE_USAGE_ALLOWED_KEYS: set[str] = set()
_OPTIMIZATION_SUMMARY_ALLOWED_KEYS = {
    "run_id",
    "seed",
    "run_identity",
    "invocation_dir",
    "best_idx",
    "best_validation_score",
    "best_candidate_fingerprint",
    "seed_candidate_fingerprint",
    "best_candidate_differs_from_seed",
    "train_case_ids",
    "validation_case_ids",
    "execution_modes",
    "num_candidates",
    "total_metric_calls",
    "num_full_val_evals",
    "run_dir",
}
_RUN_IDENTITY_ALLOWED_KEYS = {
    "schema_version",
    "campaign_id",
    "candidate_id",
    "seed_candidate_fingerprint",
    "train_case_ids",
    "validation_case_ids",
    "max_metric_calls",
    "seed",
    "proposal_source",
}
_PROMOTION_BLOCKER_HARD_SAFETY = "hard_safety_failures"
_PROMOTION_BLOCKER_SYSTEMIC = "systemic_failures"
_PROMOTION_BLOCKER_MILESTONE = "milestone_failed"
_PROMOTION_BLOCKER_PASS_AT_3 = "pass_at_3_below_1_0"
_PROMOTION_BLOCKER_PASS_AT_5 = "pass_at_5_below_1_0"

#: An artifact may only be *named* in a round report when its own suffix says it
#: is a structured summary the safe projection already covers.  Anything else —
#: raw logs, serialized optimizer state, sockets — is not nameable evidence.
_ARTIFACT_REF_ALLOWED_SUFFIXES = {".json", ".yaml", ".yml", ".md"}
#: Substrings that mark an artifact as unsafe to name, let alone copy: request
#: payloads, audit records, kubeconfigs, credentials, and optimizer state.
_FORBIDDEN_ARTIFACT_TOKENS = (
    "kubeconfig",
    "credential",
    "secret",
    "token",
    "password",
    "audit",
    "gepa_state",
    "answer",
    ".env",
)
_FORBIDDEN_ARTIFACT_NAMES = {"request.json"}


@dataclass(frozen=True, slots=True)
class CaseRunSummary:
    run_id: str
    case_id: str
    model: str
    repetition: int
    status: str
    completion: float | None
    verification: float | None
    efficiency: float | None
    hard_failures: tuple[str, ...]
    execution_mode: str
    elapsed_seconds: float | None


@dataclass(frozen=True, slots=True)
class RoundReport:
    campaign_id: str
    candidate_id: str
    candidate_fingerprint: str
    models: tuple[str, ...]
    aggregate_score: float
    model_scores: Mapping[str, float]
    pass_at_3: float | None
    pass_at_5: float | None
    systemic_failures: int
    promotion_eligible: bool
    promotion_blockers: tuple[str, ...]
    status_counts: Mapping[str, int]
    hard_failure_counts: Mapping[str, int]
    runs: tuple[CaseRunSummary, ...]
    artifact_refs: tuple[str, ...]
    reproduction_command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ParsedResponse:
    run_id: str
    case_id: str
    model: str
    repetition: int
    status: str
    execution_mode: str
    candidate_fingerprint: str
    completion: float | None
    verification: float | None
    efficiency: float | None
    hard_failures: tuple[str, ...]
    elapsed_seconds: float | None
    payload: Mapping[str, Any]


def build_round_report(artifact_root: Path | str) -> RoundReport:
    artifact_root_path = _resolve_existing_directory(artifact_root, "artifact_root")
    evaluation_summary = _load_json_mapping(_resolve_source_path(artifact_root_path, artifact_root_path / "evaluation-summary.json"))
    summary = _normalize_evaluation_summary(evaluation_summary)
    runs = _load_run_summaries(artifact_root_path, summary)

    status_counts = Counter(run.status for run in runs)
    hard_failure_counts = Counter(failure for run in runs for failure in run.hard_failures)
    hard_failure_total = sum(hard_failure_counts.values())
    if hard_failure_total != summary["hard_safety_failures"]:
        raise ValueError("evaluation summary hard_safety_failures does not match the response evidence")

    promotion_blockers = _promotion_blockers(
        hard_failure_total=hard_failure_total,
        systemic_failures=summary["systemic_failures"],
        milestone_passed=summary["milestone_passed"],
        pass_at_3=summary["pass_at_3"],
        pass_at_5=summary["pass_at_5"],
    )
    ordered_runs = tuple(sorted(runs, key=_run_sort_key))
    models = tuple(sorted({run.model for run in ordered_runs}))
    model_scores = dict(summary["model_scores"])
    if set(models) != set(model_scores):
        raise ValueError("model_scores must cover exactly the models the evidence names")
    return RoundReport(
        campaign_id=summary["campaign_id"],
        candidate_id=summary["candidate_id"],
        candidate_fingerprint=summary["candidate_fingerprint"],
        models=models,
        aggregate_score=summary["aggregate_score"],
        model_scores={model: model_scores[model] for model in models},
        pass_at_3=summary["pass_at_3"],
        pass_at_5=summary["pass_at_5"],
        systemic_failures=summary["systemic_failures"],
        promotion_eligible=not promotion_blockers,
        promotion_blockers=promotion_blockers,
        status_counts={status: status_counts[status] for status in sorted(status_counts)},
        hard_failure_counts={name: hard_failure_counts[name] for name in sorted(hard_failure_counts)},
        runs=tuple(
            CaseRunSummary(
                run_id=run.run_id,
                case_id=run.case_id,
                model=run.model,
                repetition=run.repetition,
                status=run.status,
                completion=run.completion,
                verification=run.verification,
                efficiency=run.efficiency,
                hard_failures=run.hard_failures,
                execution_mode=run.execution_mode,
                elapsed_seconds=run.elapsed_seconds,
            )
            for run in ordered_runs
        ),
        artifact_refs=_safe_artifact_refs(summary["artifact_refs"]),
        reproduction_command=tuple(summary["reproduction_command"]),
    )


def render_round_markdown(report: RoundReport) -> str:
    blockers = ", ".join(report.promotion_blockers) if report.promotion_blockers else "none"
    lines = [
        "# Grounding Round Summary",
        "",
        "## Overview",
        "",
        f"- Campaign: `{report.campaign_id}`",
        f"- Candidate: `{report.candidate_id}`",
        f"- Candidate fingerprint: `{report.candidate_fingerprint}`",
        f"- Models: {', '.join(report.models)}",
        f"- Promotion eligible: {'yes' if report.promotion_eligible else 'no'}",
        f"- Promotion blockers: {blockers}",
        "",
        "## Aggregate",
        "",
        "| Aggregate score | pass^3 | pass^5 |",
        "| ---: | ---: | ---: |",
        f"| {report.aggregate_score:.3f} | {_format_metric(report.pass_at_3)} | {_format_metric(report.pass_at_5)} |",
        "",
        "## Per-model scores",
        "",
        "| Model | Score |",
        "| --- | ---: |",
    ]
    for model in sorted(report.model_scores):
        lines.append(f"| `{model}` | {report.model_scores[model]:.3f} |")

    lines.extend(
        [
            "",
            "## Status counts",
            "",
            "| Status | Count |",
            "| --- | ---: |",
        ]
    )
    for status, count in report.status_counts.items():
        lines.append(f"| {status} | {count} |")

    lines.extend(["", "## Hard failure counts", "", "| Failure | Count |", "| --- | ---: |"])
    if report.hard_failure_counts:
        for failure, count in report.hard_failure_counts.items():
            lines.append(f"| {failure} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Runs",
            "",
            (
                "| Model | Case | Run ID | Status | Completion | Verification |"
                " Efficiency | Elapsed (s) | Hard failures | Execution mode |"
            ),
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for run in sorted(report.runs, key=_run_sort_key):
        failures = ", ".join(run.hard_failures) if run.hard_failures else "none"
        lines.append(
            f"| {run.model} | {run.case_id} | {run.run_id} | {run.status} | {_format_metric(run.completion)} | "
            f"{_format_metric(run.verification)} | {_format_metric(run.efficiency)} | "
            f"{_format_metric(run.elapsed_seconds)} | {failures} | {run.execution_mode} |"
        )

    lines.extend(["", "## Artifacts", "", "Artifact names recorded by the evaluation run:", ""])
    if report.artifact_refs:
        lines.extend(f"- `{ref}`" for ref in report.artifact_refs)
    else:
        lines.append("- none")

    lines.extend(["", "## Reproduction command", ""])
    if report.reproduction_command:
        # Rendered through shlex.join so a copied command can never re-split or
        # re-interpret an argument that contains spaces or shell metacharacters.
        lines.extend(["```shell", shlex.join(report.reproduction_command), "```"])
    else:
        lines.append("- not recorded")
    return "\n".join(lines) + "\n"


def write_safe_evidence(
    artifact_root: Path | str,
    safe_output: Path | str,
    *,
    before_artifact_root: Path | str | None = None,
    optimize_artifact_root: Path | str | None = None,
    prompt_lab_revision: str | None = None,
    korvid_revision: str | None = None,
    workflow_run_url: str | None = None,
) -> Path:
    artifact_root_path = _resolve_existing_directory(artifact_root, "artifact_root")
    optimize_artifact_root_path = (
        artifact_root_path
        if optimize_artifact_root is None
        else _resolve_existing_directory(optimize_artifact_root, "optimize_artifact_root")
    )
    before_artifact_root_path = (
        None
        if before_artifact_root is None
        else _resolve_existing_directory(before_artifact_root, "before_artifact_root")
    )
    evaluation_summary_path = _resolve_source_path(artifact_root_path, artifact_root_path / "evaluation-summary.json")
    evaluation_summary = _normalize_evaluation_summary(_load_json_mapping(evaluation_summary_path))
    optimization_summary = _load_optional_optimization_summary(optimize_artifact_root_path)
    best_candidate_yaml = _load_optional_best_candidate(
        optimize_artifact_root_path,
        evaluation_summary,
        optimization_summary,
    )
    safe_output_path = Path(safe_output).expanduser().resolve(strict=False)
    if safe_output_path.exists():
        raise FileExistsError(f"safe output already exists: {safe_output_path}")
    safe_output_path.mkdir(parents=True, exist_ok=False)

    report = build_round_report(artifact_root_path)

    safe_response_paths: list[str] = _write_safe_responses(
        report=report,
        source_root=artifact_root_path,
        safe_output=safe_output_path,
        destination_dir="responses",
    )

    # Build comparison when a before root is provided
    from .comparison import (  # noqa: PLC0415
        build_round_comparison,
        comparison_payload,
        render_comparison_markdown,
        render_single_evaluation_markdown,
    )
    comparison = None
    extra_copied_artifacts: list[str] = []
    if before_artifact_root_path is not None:
        if optimization_summary is None:
            raise ValueError("before_artifact_root requires an optimization summary")
        if best_candidate_yaml is None:
            raise ValueError("before_artifact_root requires a best-candidate.yaml")
        seed_fingerprint = optimization_summary["seed_candidate_fingerprint"]
        best_fingerprint = optimization_summary["best_candidate_fingerprint"]
        before_report = build_round_report(before_artifact_root_path)
        comparison = build_round_comparison(
            before_report,
            report,
            seed_fingerprint=seed_fingerprint,
            best_fingerprint=best_fingerprint,
        )
        _write_json(
            _resolve_destination_path(safe_output_path, safe_output_path / "comparison-summary.json"),
            comparison_payload(comparison),
        )
        extra_copied_artifacts.append("comparison-summary.json")

        if comparison.status == "changed":
            # Write safe before-evaluation-summary
            before_evaluation_summary = _normalize_evaluation_summary(
                _load_json_mapping(
                    _resolve_source_path(
                        before_artifact_root_path, before_artifact_root_path / "evaluation-summary.json"
                    )
                )
            )
            safe_before_summary = _safe_evaluation_summary_payload(before_evaluation_summary)
            _write_json(
                _resolve_destination_path(safe_output_path, safe_output_path / "before-evaluation-summary.json"),
                safe_before_summary,
            )
            extra_copied_artifacts.append("before-evaluation-summary.json")

            # Write safe before-responses
            before_response_paths = _write_safe_responses(
                report=before_report,
                source_root=before_artifact_root_path,
                safe_output=safe_output_path,
                destination_dir="before-responses",
            )
            extra_copied_artifacts.extend(before_response_paths)
        else:
            # unchanged: roots must be the same directory, no duplication
            if before_artifact_root_path.resolve() != artifact_root_path.resolve():
                raise ValueError(
                    "unchanged comparison status requires before_artifact_root and artifact_root to be the same directory"
                )

    copied_artifacts = ["evaluation-summary.json"]
    safe_evaluation_summary = _safe_evaluation_summary_payload(evaluation_summary)
    safe_evaluation_summary["artifact_refs"] = [
        "evaluation-summary.json",
        *(["optimization-summary.json"] if optimization_summary is not None else []),
        *(["best-candidate.yaml"] if best_candidate_yaml is not None else []),
        *extra_copied_artifacts,
        *safe_response_paths,
        "round-summary.json",
        "round-summary.md",
    ]
    _write_json(_resolve_destination_path(safe_output_path, safe_output_path / "evaluation-summary.json"), safe_evaluation_summary)

    if optimization_summary is not None:
        copied_artifacts.append("optimization-summary.json")
        _write_json(
            _resolve_destination_path(safe_output_path, safe_output_path / "optimization-summary.json"),
            optimization_summary,
        )
    if best_candidate_yaml is not None:
        copied_artifacts.append("best-candidate.yaml")
        _write_text(
            _resolve_destination_path(safe_output_path, safe_output_path / "best-candidate.yaml"),
            best_candidate_yaml,
        )

    summary_payload = {
        "schema_version": 1,
        "campaign_id": report.campaign_id,
        "candidate_id": report.candidate_id,
        "candidate_fingerprint": report.candidate_fingerprint,
        "models": list(report.models),
        "aggregate_score": report.aggregate_score,
        "model_scores": dict(report.model_scores),
        "pass_at_3": report.pass_at_3,
        "pass_at_5": report.pass_at_5,
        "systemic_failures": report.systemic_failures,
        "promotion_eligible": report.promotion_eligible,
        "promotion_blockers": list(report.promotion_blockers),
        "status_counts": dict(report.status_counts),
        "hard_failure_counts": dict(report.hard_failure_counts),
        "runs": [
            {
                "run_id": run.run_id,
                "case_id": run.case_id,
                "model": run.model,
                "repetition": run.repetition,
                "status": run.status,
                "completion": run.completion,
                "verification": run.verification,
                "efficiency": run.efficiency,
                "elapsed_seconds": run.elapsed_seconds,
                "hard_failures": list(run.hard_failures),
                "execution_mode": run.execution_mode,
            }
            for run in report.runs
        ],
        # What this package contains…
        "artifact_refs": [
            "round-summary.json",
            "round-summary.md",
            *copied_artifacts,
            *extra_copied_artifacts,
            *safe_response_paths,
        ],
        # …and the safe artifact names the evaluation run itself recorded.
        "evaluation_artifact_refs": list(report.artifact_refs),
        "prompt_lab_revision": prompt_lab_revision,
        "korvid_revision": korvid_revision,
        "workflow_run_url": workflow_run_url,
        "reproduction_command": list(report.reproduction_command),
    }
    _write_json(_resolve_destination_path(safe_output_path, safe_output_path / "round-summary.json"), summary_payload)

    # Build markdown: comparison/single-evaluation headline first, then detailed
    # round evidence collapsed inside a <details> block.
    headline = (
        render_comparison_markdown(comparison, report)
        if comparison is not None
        else render_single_evaluation_markdown(report)
    )
    details = render_round_markdown(report).rstrip()
    markdown_lines: list[str] = []
    if workflow_run_url or prompt_lab_revision or korvid_revision:
        markdown_lines.extend(["# Round Metadata", ""])
        if workflow_run_url:
            markdown_lines.append(f"- Workflow run: {workflow_run_url}")
        if prompt_lab_revision:
            markdown_lines.append(f"- Prompt Lab revision: `{prompt_lab_revision}`")
        if korvid_revision:
            markdown_lines.append(f"- Korvid revision: `{korvid_revision}`")
        markdown_lines.extend(["", ""])
    markdown_lines.extend(
        [
            headline.rstrip(),
            "",
            "<details>",
            "<summary>Detailed round evidence</summary>",
            "",
            details,
            "",
            "</details>",
        ]
    )
    markdown_path = _resolve_destination_path(safe_output_path, safe_output_path / "round-summary.md")
    _write_text(markdown_path, "\n".join(markdown_lines).rstrip() + "\n")
    return safe_output_path


def _write_safe_responses(
    *,
    report: RoundReport,
    source_root: Path,
    safe_output: Path,
    destination_dir: str,
) -> list[str]:
    references: list[str] = []
    for run in report.runs:
        source_path = _resolve_source_path(
            source_root,
            source_root / "runs" / run.run_id / "response.json",
        )
        parsed = _parse_response(source_path)
        if parsed.candidate_fingerprint != report.candidate_fingerprint:
            raise ValueError("response fingerprint does not match the report fingerprint")
        destination_path = _resolve_destination_path(
            safe_output,
            safe_output / destination_dir / f"{run.run_id}.json",
        )
        _write_json(destination_path, parsed.payload)
        references.append(destination_path.relative_to(safe_output).as_posix())
    return references


def _normalize_evaluation_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    _ensure_keys(summary, _EVALUATION_SUMMARY_ALLOWED_KEYS, "evaluation summary")
    candidate_id = _require_string(summary.get("candidate_id"), "candidate_id")
    candidate_fingerprint = _require_string(summary.get("candidate_fingerprint"), "candidate_fingerprint")
    campaign_id = _require_string(summary.get("campaign_id"), "campaign_id")
    bundle_kind = _require_string(summary.get("bundle_kind"), "bundle_kind")
    aggregate_score = _require_numeric(summary.get("aggregate_score"), "aggregate_score")
    campaign_case_ids = _require_unique_string_list(summary.get("campaign_case_ids"), "campaign_case_ids")
    pass_at_3 = _require_optional_metric(summary.get("pass_at_3"), "pass_at_3")
    pass_at_5 = _require_optional_metric(summary.get("pass_at_5"), "pass_at_5")
    hard_safety_failures = _require_non_negative_int(summary.get("hard_safety_failures"), "hard_safety_failures")
    systemic_failures = _require_non_negative_int(summary.get("systemic_failures"), "systemic_failures")
    milestone_passed = summary.get("milestone_passed")
    if not isinstance(milestone_passed, bool):
        raise ValueError("milestone_passed must be a boolean")  # noqa: TRY004
    execution_modes = _require_live_execution_modes(summary.get("execution_modes"))
    evaluated_case_ids = _require_unique_string_list(summary.get("evaluated_case_ids"), "evaluated_case_ids")
    evaluated_models = _require_unique_string_list(summary.get("evaluated_models"), "evaluated_models")
    campaign_pairs = _require_unique_string_list(summary.get("campaign_case_model_pairs"), "campaign_case_model_pairs")
    evaluated_pairs = _require_unique_string_list(summary.get("evaluated_case_model_pairs"), "evaluated_case_model_pairs")
    repetitions_per_case = _require_positive_int(summary.get("repetitions_per_case"), "repetitions_per_case")
    run_execution_modes = _normalize_run_execution_modes(summary.get("run_execution_modes"), evaluated_pairs)
    model_scores = _normalize_model_scores(summary.get("model_scores"))
    case_sets = _normalize_case_sets(summary.get("case_sets"))
    artifact_refs = _require_artifact_refs(summary.get("artifact_refs"))
    reproduction_command = _require_reproduction_command(summary.get("reproduction_command"))

    case_ids_from_pairs = {case_id for case_id, _ in (_decode_case_model_pair(pair) for pair in evaluated_pairs)}
    models_from_pairs = {model for _, model in (_decode_case_model_pair(pair) for pair in evaluated_pairs)}
    if set(evaluated_case_ids) != case_ids_from_pairs:
        raise ValueError("evaluated_case_ids must match evaluated_case_model_pairs")
    if set(evaluated_models) != models_from_pairs:
        raise ValueError("evaluated_models must match evaluated_case_model_pairs")
    if set(campaign_case_ids) != {case_id for case_id, _ in (_decode_case_model_pair(pair) for pair in campaign_pairs)}:
        raise ValueError("campaign_case_ids must match campaign_case_model_pairs")
    if set(run_execution_modes) != set(evaluated_pairs):
        raise ValueError("run_execution_modes must match evaluated_case_model_pairs")
    if execution_modes != [EXECUTION_MODE_LIVE]:
        raise ValueError("execution_modes must be exactly ['live']")
    if set(model_scores) != set(evaluated_models):
        raise ValueError("model_scores must match evaluated_models")

    return {
        "bundle_kind": bundle_kind,
        "candidate_id": candidate_id,
        "candidate_fingerprint": candidate_fingerprint,
        "campaign_id": campaign_id,
        "campaign_case_ids": tuple(campaign_case_ids),
        "campaign_case_model_pairs": tuple(campaign_pairs),
        "evaluated_case_ids": tuple(evaluated_case_ids),
        "evaluated_models": tuple(evaluated_models),
        "aggregate_score": aggregate_score,
        "model_scores": model_scores,
        "pass_at_3": pass_at_3,
        "pass_at_5": pass_at_5,
        "hard_safety_failures": hard_safety_failures,
        "systemic_failures": systemic_failures,
        "milestone_passed": milestone_passed,
        "evaluated_case_model_pairs": tuple(sorted(evaluated_pairs)),
        "repetitions_per_case": repetitions_per_case,
        "run_execution_modes": run_execution_modes,
        "execution_modes": tuple(execution_modes),
        "case_sets": case_sets,
        "artifact_refs": tuple(artifact_refs),
        "reproduction_command": tuple(reproduction_command),
    }


def _load_run_summaries(artifact_root: Path, summary: Mapping[str, Any]) -> list[_ParsedResponse]:
    runs_root = _resolve_existing_directory(artifact_root / "runs", "artifact_root/runs")
    responses = sorted(runs_root.glob("*/response.json"))
    if not responses:
        raise ValueError("response evidence is required")

    expected = {
        (case_id, model, repetition)
        for case_id, model in (_decode_case_model_pair(pair) for pair in summary["evaluated_case_model_pairs"])
        for repetition in range(1, summary["repetitions_per_case"] + 1)
    }
    observed: dict[tuple[str, str, int], _ParsedResponse] = {}
    duplicates: set[tuple[str, str, int]] = set()
    extras: set[tuple[str, str, int]] = set()

    for path in responses:
        parsed = _parse_response(_resolve_source_path(artifact_root, path))
        if parsed.candidate_fingerprint != summary["candidate_fingerprint"]:
            raise ValueError("response fingerprint does not match evaluation summary candidate_fingerprint")
        if parsed.execution_mode != EXECUTION_MODE_LIVE:
            raise ValueError("response execution_mode must be live")
        pair = f"{parsed.case_id}::{parsed.model}"
        if pair not in summary["run_execution_modes"]:
            extras.add((parsed.case_id, parsed.model, parsed.repetition))
            continue
        if summary["run_execution_modes"][pair] != parsed.execution_mode:
            raise ValueError("response execution_mode does not match run_execution_modes")
        key = (parsed.case_id, parsed.model, parsed.repetition)
        if key in observed:
            duplicates.add(key)
            continue
        observed[key] = parsed
        if key not in expected:
            extras.add(key)

    missing = expected - set(observed)
    if missing or duplicates or extras:
        details: list[str] = []
        if missing:
            details.append(f"missing={_format_evidence_keys(missing)}")
        if duplicates:
            details.append(f"duplicate={_format_evidence_keys(duplicates)}")
        if extras:
            details.append(f"extra={_format_evidence_keys(extras)}")
        raise ValueError(f"response evidence is inconsistent: {', '.join(details)}")
    return list(observed.values())


def _parse_response(path: Path) -> _ParsedResponse:
    payload = _load_json_mapping(path)
    try:
        _ensure_keys(payload, _RESPONSE_ALLOWED_KEYS, "bridge response")
        protocol_version = _require_response_int(payload, "protocol_version")
        if protocol_version != PROTOCOL_VERSION:
            raise BridgeMalformedOutputError(f"bridge response protocol_version must be {PROTOCOL_VERSION}")
        status = _require_response_string(payload, "status")
        if status not in {"completed", "model_failure"}:
            raise BridgeMalformedOutputError(f"bridge returned systemic status: {status}")
        execution_mode = _require_response_string(payload, "execution_mode")
        if execution_mode not in EXECUTION_MODES:
            raise BridgeMalformedOutputError(
                f"bridge response execution_mode must be one of {', '.join(sorted(EXECUTION_MODES))}"
            )
        candidate_fingerprint = _require_response_string(payload, "candidate_fingerprint")
        identity = _require_response_mapping(payload, "request_identity")
        _ensure_keys(identity, _REQUEST_IDENTITY_ALLOWED_KEYS, "request_identity")
        case_id = _require_response_string(identity, "case_id")
        template_id = _require_response_string(identity, "template_id")
        model = _require_response_string(identity, "model")
        repetition = _require_response_int(identity, "repetition")
        seed = _require_response_int(identity, "seed")
        _require_response_text(payload, "answer")
        journal = _parse_journal(payload, status)
        usage = _parse_usage(payload, status)
        error = _require_optional_response_string(payload, "error")
        completion, verification, efficiency, hard_failures = _parse_grade(payload, status)
    except (BridgeMalformedOutputError, ValueError) as exc:
        raise ValueError(str(exc)) from exc

    elapsed_seconds = usage.get("wall_time_seconds")
    return _ParsedResponse(
        run_id=path.parent.name,
        case_id=case_id,
        model=model,
        repetition=repetition,
        status=status,
        execution_mode=execution_mode,
        candidate_fingerprint=candidate_fingerprint,
        completion=completion,
        verification=verification,
        efficiency=efficiency,
        hard_failures=hard_failures,
        elapsed_seconds=None if elapsed_seconds is None else float(elapsed_seconds),
        payload={
            "protocol_version": protocol_version,
            "status": status,
            "execution_mode": execution_mode,
            "candidate_fingerprint": candidate_fingerprint,
            "request_identity": {
                "case_id": case_id,
                "template_id": template_id,
                "model": model,
                "repetition": repetition,
                "seed": seed,
            },
            "grade": None
            if completion is None
            else {
                "completion": completion,
                "verification": verification,
                "efficiency": efficiency,
                "hard_failures": list(hard_failures),
            },
            "answer": "",
            "journal": journal,
            "usage": usage,
            "error": error,
        },
    )


def _parse_grade(payload: Mapping[str, Any], status: str) -> tuple[float | None, float | None, float | None, tuple[str, ...]]:
    if "grade" not in payload:
        raise BridgeMalformedOutputError("bridge response missing grade")
    grade_payload = payload["grade"]
    if grade_payload is None:
        if status == "completed":
            raise BridgeMalformedOutputError("completed bridge responses must include a grade")
        return None, None, None, ()
    if status != "completed":
        raise BridgeMalformedOutputError("only completed bridge responses may include a grade")

    grade = _require_mapping(grade_payload, "grade")
    _ensure_keys(grade, _GRADE_ALLOWED_KEYS, "grade")
    hard_failures_value = grade.get("hard_failures")
    if not isinstance(hard_failures_value, list):
        raise BridgeMalformedOutputError("bridge grade is malformed")
    return (
        _coerce_metric(grade.get("completion"), "completion"),
        _coerce_metric(grade.get("verification"), "verification"),
        _coerce_metric(grade.get("efficiency"), "efficiency"),
        tuple(_require_string(item, "hard_failure") for item in hard_failures_value),
    )


def _parse_journal(payload: Mapping[str, Any], status: str) -> dict[str, Any]:
    journal = _require_response_mapping(payload, "journal")
    if status == "completed":
        _ensure_keys(journal, _COMPLETED_JOURNAL_ALLOWED_KEYS, "journal")
        return {
            "journey_id": _require_string_or_empty(journal.get("journey_id"), "journal.journey_id"),
            "checkpoints": _require_string_list_allow_empty(journal.get("checkpoints"), "journal.checkpoints"),
            "missing_checkpoints": _require_string_list_allow_empty(
                journal.get("missing_checkpoints"), "journal.missing_checkpoints"
            ),
            "checkpoint_counts": _require_count_mapping(journal.get("checkpoint_counts"), "journal.checkpoint_counts"),
            "journal_event_count": _require_non_negative_int(journal.get("journal_event_count"), "journal.journal_event_count"),
            "audit_record_count": _require_non_negative_int(journal.get("audit_record_count"), "journal.audit_record_count"),
            "hard_failure_count": _require_non_negative_int(journal.get("hard_failure_count"), "journal.hard_failure_count"),
        }
    _ensure_keys(journal, _MODEL_FAILURE_JOURNAL_ALLOWED_KEYS, "journal")
    return {
        "checkpoints": _require_string_list_allow_empty(journal.get("checkpoints"), "journal.checkpoints"),
        "checkpoint_counts": _require_count_mapping(journal.get("checkpoint_counts"), "journal.checkpoint_counts"),
    }


def _parse_usage(payload: Mapping[str, Any], status: str) -> dict[str, Any]:
    usage = _require_response_mapping(payload, "usage")
    if status == "completed":
        _ensure_keys(usage, _COMPLETED_USAGE_ALLOWED_KEYS, "usage")
        return {
            "tool_calls": _require_non_negative_int(usage.get("tool_calls"), "usage.tool_calls"),
            "iterations": _require_non_negative_int(usage.get("iterations"), "usage.iterations"),
            "wall_time_seconds": _require_duration(usage.get("wall_time_seconds"), "usage.wall_time_seconds"),
        }
    _ensure_keys(usage, _MODEL_FAILURE_USAGE_ALLOWED_KEYS, "usage")
    return {}


def _promotion_blockers(
    *,
    hard_failure_total: int,
    systemic_failures: int,
    milestone_passed: bool,
    pass_at_3: float | None,
    pass_at_5: float | None,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if hard_failure_total:
        blockers.append(_PROMOTION_BLOCKER_HARD_SAFETY)
    if systemic_failures:
        blockers.append(_PROMOTION_BLOCKER_SYSTEMIC)
    if not milestone_passed:
        blockers.append(_PROMOTION_BLOCKER_MILESTONE)
    if pass_at_3 is None or pass_at_3 < 1.0:
        blockers.append(_PROMOTION_BLOCKER_PASS_AT_3)
    if pass_at_5 is None or pass_at_5 < 1.0:
        blockers.append(_PROMOTION_BLOCKER_PASS_AT_5)
    return tuple(blockers)


def _safe_evaluation_summary_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bundle_kind": summary["bundle_kind"],
        "candidate_id": summary["candidate_id"],
        "candidate_fingerprint": summary["candidate_fingerprint"],
        "campaign_id": summary["campaign_id"],
        "campaign_case_ids": list(summary["campaign_case_ids"]),
        "evaluated_case_ids": list(summary["evaluated_case_ids"]),
        "evaluated_models": list(summary["evaluated_models"]),
        "campaign_case_model_pairs": list(summary["campaign_case_model_pairs"]),
        "evaluated_case_model_pairs": list(summary["evaluated_case_model_pairs"]),
        "aggregate_score": summary["aggregate_score"],
        "model_scores": dict(summary["model_scores"]),
        "execution_modes": list(summary["execution_modes"]),
        "run_execution_modes": dict(summary["run_execution_modes"]),
        "repetitions_per_case": summary["repetitions_per_case"],
        "pass_at_3": summary["pass_at_3"],
        "pass_at_5": summary["pass_at_5"],
        "hard_safety_failures": summary["hard_safety_failures"],
        "systemic_failures": summary["systemic_failures"],
        "milestone_passed": summary["milestone_passed"],
        "case_sets": {name: list(values) for name, values in dict(summary["case_sets"]).items()},
        "artifact_refs": list(summary["artifact_refs"]),
        "reproduction_command": list(summary["reproduction_command"]),
    }


def _load_optional_optimization_summary(artifact_root: Path) -> dict[str, Any] | None:
    source_path = artifact_root / "optimization-summary.json"
    if not source_path.is_file():
        return None
    payload = _load_json_mapping(_resolve_source_path(artifact_root, source_path))
    return _normalize_optimization_summary(payload)


def _load_optional_best_candidate(
    artifact_root: Path,
    evaluation_summary: Mapping[str, Any],
    optimization_summary: Mapping[str, Any] | None,
) -> str | None:
    source_path = artifact_root / "best-candidate.yaml"
    if not source_path.is_file():
        return None
    candidate = load_candidate(_resolve_source_path(artifact_root, source_path))
    if candidate.candidate_id != evaluation_summary["candidate_id"]:
        raise ValueError("best-candidate candidate_id does not match evaluation summary")
    if candidate.fingerprint != evaluation_summary["candidate_fingerprint"]:
        raise ValueError("best-candidate fingerprint does not match evaluation summary")
    if (
        optimization_summary is not None
        and candidate.fingerprint != optimization_summary["best_candidate_fingerprint"]
    ):
        raise ValueError("best-candidate fingerprint does not match optimization summary")
    payload = {
        "schema_version": candidate.schema_version,
        "candidate_id": candidate.candidate_id,
        "components": candidate.components,
        "metadata": candidate.metadata,
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def _normalize_optimization_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    _ensure_keys(summary, _OPTIMIZATION_SUMMARY_ALLOWED_KEYS, "optimization summary")
    run_identity = _require_mapping(summary.get("run_identity"), "optimization summary.run_identity")
    _ensure_keys(run_identity, _RUN_IDENTITY_ALLOWED_KEYS, "optimization summary.run_identity")
    execution_modes = _require_live_execution_modes(summary.get("execution_modes"))
    return {
        "run_id": _require_string(summary.get("run_id"), "optimization summary.run_id"),
        "seed": _require_non_negative_int(summary.get("seed"), "optimization summary.seed"),
        "run_identity": {
            "schema_version": _require_positive_int(run_identity.get("schema_version"), "run_identity.schema_version"),
            "campaign_id": _require_string(run_identity.get("campaign_id"), "run_identity.campaign_id"),
            "candidate_id": _require_string(run_identity.get("candidate_id"), "run_identity.candidate_id"),
            "seed_candidate_fingerprint": _require_string(
                run_identity.get("seed_candidate_fingerprint"), "run_identity.seed_candidate_fingerprint"
            ),
            "train_case_ids": _require_unique_string_list(run_identity.get("train_case_ids"), "run_identity.train_case_ids"),
            "validation_case_ids": _require_unique_string_list(
                run_identity.get("validation_case_ids"), "run_identity.validation_case_ids"
            ),
            "max_metric_calls": _require_positive_int(run_identity.get("max_metric_calls"), "run_identity.max_metric_calls"),
            "seed": _require_non_negative_int(run_identity.get("seed"), "run_identity.seed"),
            "proposal_source": _require_string(run_identity.get("proposal_source"), "run_identity.proposal_source"),
        },
        "invocation_dir": _require_string_or_empty(summary.get("invocation_dir"), "optimization summary.invocation_dir"),
        "best_idx": _require_non_negative_int(summary.get("best_idx"), "optimization summary.best_idx"),
        "best_validation_score": _require_numeric(
            summary.get("best_validation_score"), "optimization summary.best_validation_score"
        ),
        "best_candidate_fingerprint": _require_string(
            summary.get("best_candidate_fingerprint"), "optimization summary.best_candidate_fingerprint"
        ),
        "seed_candidate_fingerprint": _require_string(
            summary.get("seed_candidate_fingerprint"), "optimization summary.seed_candidate_fingerprint"
        ),
        "best_candidate_differs_from_seed": _require_bool(
            summary.get("best_candidate_differs_from_seed"), "optimization summary.best_candidate_differs_from_seed"
        ),
        "train_case_ids": _require_unique_string_list(summary.get("train_case_ids"), "optimization summary.train_case_ids"),
        "validation_case_ids": _require_unique_string_list(
            summary.get("validation_case_ids"), "optimization summary.validation_case_ids"
        ),
        "execution_modes": execution_modes,
        "num_candidates": _require_positive_int(summary.get("num_candidates"), "optimization summary.num_candidates"),
        "total_metric_calls": _require_positive_int(
            summary.get("total_metric_calls"), "optimization summary.total_metric_calls"
        ),
        "num_full_val_evals": _require_non_negative_int(
            summary.get("num_full_val_evals"), "optimization summary.num_full_val_evals"
        ),
        "run_dir": _require_string_or_empty(summary.get("run_dir"), "optimization summary.run_dir"),
    }


def _resolve_existing_directory(path: Path | str, context: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{context} must be an existing directory")
    return resolved


def _resolve_source_path(root: Path, path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source path escapes artifact_root: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"source path is not a file: {path}")
    return resolved


def _resolve_destination_path(root: Path, path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"destination path escapes safe_output: {path}") from exc
    return resolved


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read JSON artifact: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"artifact is not valid UTF-8 JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"artifact is not valid JSON: {path}") from exc
    return _require_mapping(payload, path.name)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(path, payload)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def _require_numeric(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")  # noqa: TRY004
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{context} must be a finite number")
    return numeric


def _require_duration(value: Any, context: str) -> float:
    duration = _require_numeric(value, context)
    if duration < 0.0:
        raise ValueError(f"{context} must not be negative")
    return duration


def _require_artifact_refs(value: Any) -> list[str]:
    """Validate every recorded artifact name as a safe, relative, in-tree path."""
    refs = _require_string_list_allow_empty(value, "artifact_refs")
    validated: list[str] = []
    for index, ref in enumerate(refs):
        validated.append(_require_relative_path(ref, f"artifact_refs[{index}]"))
    return list(dict.fromkeys(validated))


def _require_relative_path(value: str, context: str) -> str:
    if value != value.strip():
        raise ValueError(f"{context} must not be padded with whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{context} must not contain control characters")
    if "\\" in value or value.startswith(("/", "~")):
        raise ValueError(f"{context} must be a relative POSIX path inside the artifact root")
    parts = PurePosixPath(value).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{context} must not traverse outside the artifact root")
    return value


def _safe_artifact_refs(refs: Sequence[str]) -> tuple[str, ...]:
    """Keep only the artifact names a round report is allowed to display."""
    return tuple(ref for ref in refs if _is_displayable_artifact_ref(ref))


def _is_displayable_artifact_ref(ref: str) -> bool:
    path = PurePosixPath(ref)
    if path.suffix.lower() not in _ARTIFACT_REF_ALLOWED_SUFFIXES:
        return False
    if path.name.lower() in _FORBIDDEN_ARTIFACT_NAMES:
        return False
    lowered = ref.lower()
    return not any(token in lowered for token in _FORBIDDEN_ARTIFACT_TOKENS)


def _require_reproduction_command(value: Any) -> list[str]:
    tokens = _require_string_list_allow_empty(value, "reproduction_command")
    for index, token in enumerate(tokens):
        if any(ord(character) < 32 or ord(character) == 127 for character in token):
            raise ValueError(f"reproduction_command[{index}] must not contain control characters")
    return tokens


def _require_optional_metric(value: Any, context: str) -> float | None:
    if value is None:
        return None
    metric = _require_numeric(value, context)
    if not 0.0 <= metric <= 1.0:
        raise ValueError(f"{context} must be between 0.0 and 1.0")
    return metric


def _require_non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _require_unique_string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list of strings")
    items: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        text = _require_string(item, f"{context}[{index}]")
        if text in seen:
            raise ValueError(f"{context} contains duplicate values")
        seen.add(text)
        items.append(text)
    return items


def _require_string_list_allow_empty(value: Any, context: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of strings")  # noqa: TRY004
    items: list[str] = []
    for index, item in enumerate(value):
        items.append(_require_string(item, f"{context}[{index}]"))
    return items


def _normalize_model_scores(value: Any) -> dict[str, float]:
    mapping = _require_mapping(value, "model_scores")
    if not mapping:
        raise ValueError("model_scores must not be empty")
    return {
        _require_string(model_name, "model_scores key"): _require_numeric(score, "model_scores value")
        for model_name, score in sorted(mapping.items())
    }


def _normalize_case_sets(value: Any) -> dict[str, tuple[str, ...]]:
    mapping = _require_mapping(value, "case_sets")
    _ensure_keys(mapping, _CASE_SETS_ALLOWED_KEYS, "case_sets")
    return {
        name: tuple(_require_string_list_allow_empty(mapping.get(name), f"case_sets.{name}"))
        for name in sorted(_CASE_SETS_ALLOWED_KEYS)
    }


def _require_count_mapping(value: Any, context: str) -> dict[str, int]:
    mapping = _require_mapping(value, context)
    counts: dict[str, int] = {}
    for key, raw_count in sorted(mapping.items()):
        counts[_require_string(key, f"{context} key")] = _require_non_negative_int(raw_count, f"{context}[{key}]")
    return counts


def _require_string_or_empty(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")  # noqa: TRY004
    return value


def _require_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")  # noqa: TRY004
    return value


def _normalize_run_execution_modes(value: Any, expected_pairs: Sequence[str]) -> dict[str, str]:
    mapping = _require_mapping(value, "run_execution_modes")
    normalized: dict[str, str] = {}
    for pair, mode in sorted(mapping.items()):
        pair_name = _require_string(pair, "run_execution_modes key")
        if pair_name not in expected_pairs:
            raise ValueError("run_execution_modes contains unexpected case/model pairs")
        mode_name = _require_string(mode, "run_execution_modes value")
        if mode_name != EXECUTION_MODE_LIVE:
            raise ValueError("run_execution_modes must contain only live evidence")
        normalized[pair_name] = mode_name
    return normalized


def _decode_case_model_pair(value: str) -> tuple[str, str]:
    case_id, separator, model = value.partition("::")
    if not separator or not case_id or not model:
        raise ValueError("evaluated_case_model_pairs entries must use '<case_id>::<model>'")
    return case_id, model


def _run_sort_key(run: CaseRunSummary | _ParsedResponse) -> tuple[str, str, int, str]:
    return (run.model, run.case_id, _repetition_from_run_id(run.run_id), run.run_id)


def _repetition_from_run_id(run_id: str) -> int:
    match = _REPETITION_PATTERN.search(run_id)
    if match is None:
        return 0
    return int(match.group("repetition"))


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_evidence_keys(values: Sequence[tuple[str, str, int]] | set[tuple[str, str, int]]) -> str:
    return ", ".join(f"{case_id}::{model}#{repetition}" for case_id, model, repetition in sorted(values))
