"""Safe campaign evidence ingestion and artifact rendering.

This module forms the trust boundary between Grounding safe-evidence packages
and the pure campaign state machine. It reads allowlisted top-level summaries
and only the identity/provenance fields of explicitly referenced, redacted
responses. It never traverses symlinks or touches raw answers, requests, audit
journals, optimizer state, or reflection transcripts.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import stat
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml  # type: ignore[import-untyped]

from .campaigns import (
    ActionKind,
    CampaignAction,
    CampaignScore,
    CampaignState,
    CampaignStatus,
    OptimizationCampaign,
    _has_unmeasured_incumbent,
    max_search_metric_calls,
    next_action,
    state_hash,
)
from .contracts import Candidate
from .scoring import RepetitionOutcome, pass_hat_k

#: Files the safe ingestion layer is allowed to read from a round evidence package.
_ALLOWED_FILES = frozenset({
    "round-summary.json",
    "evaluation-summary.json",
    "comparison-summary.json",
    "optimization-summary.json",
    "best-candidate.yaml",
    "before-evaluation-summary.json",
})

#: Every top-level file `write_safe_evidence` is allowed to emit.
SAFE_ROUND_PACKAGE_FILES = frozenset({
    "round-summary.json",
    "round-summary.md",
    "evaluation-summary.json",
    "optimization-summary.json",
    "best-candidate.yaml",
    "comparison-summary.json",
    "before-evaluation-summary.json",
})

#: Files that must always be present in a safe round projection.
SAFE_ROUND_PACKAGE_REQUIRED_FILES = frozenset({
    "round-summary.json",
    "round-summary.md",
    "evaluation-summary.json",
})

#: The two sanitized response projection directories, and nothing else.
SAFE_ROUND_PACKAGE_DIRECTORIES = frozenset({"responses", "before-responses"})

_SAFE_RESPONSE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json$")


def validate_safe_round_package(root: Path | str) -> None:
    """Validate a safe round evidence projection against an explicit allowlist.

    Accepts exactly what :func:`korvid_prompt_lab.rounds.write_safe_evidence`
    produces — the allowlisted summary files plus the sanitized ``responses/``
    and ``before-responses/`` projections — and rejects everything else:
    raw artifact roots, transcripts, audit journals, kubeconfig, credentials,
    GEPA state, unexpected paths, symlinks and non-regular files.
    """
    package = Path(root)
    if package.is_symlink():
        raise ValueError(f"safe round package must not be a symlink: {package}")
    if not package.is_dir():
        raise ValueError(f"safe round package is not a directory: {package}")

    files: set[str] = set()
    directories: set[str] = set()
    for entry in sorted(package.iterdir()):
        if entry.is_symlink():
            raise ValueError(
                f"safe round package contains a symlink: {entry.name}"
            )
        if entry.is_dir():
            if entry.name not in SAFE_ROUND_PACKAGE_DIRECTORIES:
                raise ValueError(
                    f"safe round package contains an unexpected directory: {entry.name}"
                )
            directories.add(entry.name)
            _validate_safe_response_directory(entry)
            continue
        if not entry.is_file():
            raise ValueError(
                f"safe round package contains a non-regular entry: {entry.name}"
            )
        if entry.name not in SAFE_ROUND_PACKAGE_FILES:
            raise ValueError(
                f"safe round package contains an unexpected file: {entry.name}"
            )
        files.add(entry.name)

    missing = sorted(SAFE_ROUND_PACKAGE_REQUIRED_FILES - files)
    if missing:
        raise ValueError(
            f"safe round package is missing required file(s): {', '.join(missing)}"
        )
    if "responses" not in directories:
        raise ValueError(
            "safe round package is missing the responses projection directory"
        )
    if "before-responses" in directories and not {
        "comparison-summary.json",
        "before-evaluation-summary.json",
    } <= files:
        raise ValueError(
            "safe round package has before-responses without a comparison projection"
        )


def _validate_safe_response_directory(directory: Path) -> None:
    """Only sanitized per-run JSON projections may live in a response directory."""
    entries = sorted(directory.iterdir())
    if not entries:
        raise ValueError(
            f"safe round package has an empty projection directory: {directory.name}"
        )
    for entry in entries:
        label = f"{directory.name}/{entry.name}"
        if entry.is_symlink():
            raise ValueError(f"safe round package contains a symlink: {label}")
        if not entry.is_file():
            raise ValueError(
                f"safe round package contains a non-regular entry: {label}"
            )
        if not _SAFE_RESPONSE_NAME_RE.match(entry.name):
            raise ValueError(
                f"safe round package contains an unexpected projection: {label}"
            )


_OPTIMIZATION_SUMMARY_REQUIRED_KEYS = frozenset({
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
})

_RUN_IDENTITY_REQUIRED_KEYS = frozenset({
    "schema_version",
    "campaign_id",
    "candidate_id",
    "seed_candidate_fingerprint",
    "train_case_ids",
    "validation_case_ids",
    "max_metric_calls",
    "seed",
    "proposal_source",
})

_ROUND_SUMMARY_V1_KEYS = frozenset({
    "schema_version",
    "campaign_id",
    "candidate_id",
    "candidate_fingerprint",
    "models",
    "aggregate_score",
    "model_scores",
    "pass_at_3",
    "pass_at_5",
    "systemic_failures",
    "promotion_eligible",
    "promotion_blockers",
    "status_counts",
    "hard_failure_counts",
    "runs",
    "artifact_refs",
    "evaluation_artifact_refs",
    "prompt_lab_revision",
    "korvid_revision",
    "workflow_run_url",
    "reproduction_command",
    "campaign_action_id",
})
_ROUND_SUMMARY_V2_KEYS = _ROUND_SUMMARY_V1_KEYS | {
    "evaluation_backend",
    "evidence_sources",
}

_EVAL_SUMMARY_REQUIRED_KEYS = frozenset({
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
})

_COMPARISON_SUMMARY_REQUIRED_KEYS = frozenset({
    "schema_version",
    "status",
    "outcome",
    "seed_candidate_fingerprint",
    "best_candidate_fingerprint",
    "contract",
    "metrics",
    "improved_count",
    "unchanged_count",
    "regressed_count",
    "not_comparable_count",
})

_COMPARISON_CONTRACT_V1_KEYS = frozenset({
    "campaign_id",
    "models",
    "case_repetitions",
    "execution_modes",
})
_COMPARISON_CONTRACT_V2_KEYS = _COMPARISON_CONTRACT_V1_KEYS | {
    "evidence_sources"
}

_COMPARISON_METRIC_REQUIRED_KEYS = frozenset({
    "key",
    "label",
    "before",
    "after",
    "delta",
    "result",
    "integer",
    "core",
})
_COMPARISON_CORE_METRICS = {
    "aggregate_score": (False, False),
    "pass_at_3": (False, False),
    "pass_at_5": (False, False),
    "hard_safety_failures": (True, True),
    "systemic_failures": (True, True),
}
_MAX_SAFE_RESPONSE_BYTES = 64 * 1024
_MAX_SAFE_TOP_LEVEL_BYTES = 1024 * 1024
_PROJECTED_RESPONSE_KEYS = frozenset({
    "protocol_version",
    "status",
    "execution_mode",
    "candidate_fingerprint",
    "request_identity",
    "evidence_source",
    "grade",
    "answer",
    "journal",
    "usage",
    "error",
})
_PROJECTED_PROCESS_RESPONSE_KEYS = _PROJECTED_RESPONSE_KEYS - {"evidence_source"}
_PROJECTED_IDENTITY_KEYS = frozenset({
    "case_id",
    "template_id",
    "model",
    "repetition",
    "seed",
    "seed_applied",
})
_PROJECTED_PROCESS_IDENTITY_KEYS = _PROJECTED_IDENTITY_KEYS - {"seed_applied"}
_PROJECTED_SOURCE_KEYS = frozenset({
    "kind",
    "korvid_version",
    "scenario_sha256",
})
_PROJECTED_GRADE_KEYS = frozenset({
    "completion",
    "verification",
    "efficiency",
    "hard_failures",
})
_PROJECTED_COMPLETED_JOURNAL_KEYS = frozenset({
    "journey_id",
    "checkpoints",
    "missing_checkpoints",
    "checkpoint_counts",
    "journal_event_count",
    "audit_record_count",
    "hard_failure_count",
})
_PROJECTED_FAILURE_JOURNAL_KEYS = frozenset({
    "checkpoints",
    "checkpoint_counts",
})
_PROJECTED_COMPLETED_USAGE_KEYS = frozenset({
    "tool_calls",
    "iterations",
    "wall_time_seconds",
})
_ROUND_RUN_KEYS = frozenset({
    "run_id",
    "case_id",
    "model",
    "repetition",
    "status",
    "completion",
    "verification",
    "efficiency",
    "elapsed_seconds",
    "hard_failures",
    "execution_mode",
})


# ---------------------------------------------------------------------------
# Strict field validators (no coercions)
# ---------------------------------------------------------------------------


def _require_str(value: Any, context: str) -> str:
    """Require a non-empty string. No coercion."""
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string, got {type(value).__name__}")  # noqa: TRY004
    if not value:
        raise ValueError(f"{context} must not be empty")
    return value


def _require_str_or_empty(value: Any, context: str) -> str:
    """Require a string (may be empty). No coercion."""
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string, got {type(value).__name__}")  # noqa: TRY004
    return value


def _require_int(value: Any, context: str) -> int:
    """Require an integer (not bool). No coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer, got {type(value).__name__}")  # noqa: TRY004
    return value


def _require_non_negative_int(value: Any, context: str) -> int:
    """Require a non-negative integer. No coercion."""
    v = _require_int(value, context)
    if v < 0:
        raise ValueError(f"{context} must be non-negative, got {v}")
    return v


def _require_positive_int(value: Any, context: str) -> int:
    """Require a positive integer. No coercion."""
    v = _require_int(value, context)
    if v <= 0:
        raise ValueError(f"{context} must be positive, got {v}")
    return v


def _require_finite_float(value: Any, context: str) -> float:
    """Require a finite float/int numeric. No coercion from strings."""
    if isinstance(value, bool):
        raise ValueError(f"{context} must be numeric, got bool")  # noqa: TRY004
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise ValueError(f"{context} must be numeric, got {type(value).__name__}")  # noqa: TRY004
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite, got {value}")
    return value


def _require_bool(value: Any, context: str) -> bool:
    """Require a boolean. No coercion."""
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean, got {type(value).__name__}")  # noqa: TRY004
    return value


def _require_string_list(value: Any, context: str) -> list[str]:
    """Require a list of non-empty strings. No coercion."""
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")  # noqa: TRY004
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{context}[{i}] must be a non-empty string")
    return value


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    """Require a dict/mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")  # noqa: TRY004
    return value


def _ensure_exact_keys(
    mapping: dict[str, Any], required: frozenset[str], context: str,
) -> None:
    """Reject unknown or missing keys."""
    keys = set(mapping)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{context} missing key(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} unknown key(s): {', '.join(unknown)}")


def _forbidden_file_present(root: Path, filename: str) -> bool:
    path = root / filename
    return path.exists() or path.is_symlink()


def _validate_artifact_refs(refs: Any, root: Path, *, context: str) -> list[str]:
    """Validate a list of artifact references without opening the files."""
    if not isinstance(refs, list) or not refs:
        raise ValueError(f"{context} must be a non-empty list of strings")

    root_resolved = root.resolve()
    seen: set[str] = set()
    validated: list[str] = []
    for index, ref in enumerate(refs):
        item_context = f"{context}[{index}]"
        if not isinstance(ref, str) or not ref:
            raise ValueError(f"{item_context} must be a non-empty string")
        if ref in seen:
            raise ValueError(f"{item_context} duplicates {ref!r}")
        seen.add(ref)

        ref_path = PurePosixPath(ref)
        if ref_path.is_absolute() or ref.startswith("/"):
            raise ValueError(f"{item_context} must be a relative path")
        if any(part == ".." for part in ref_path.parts):
            raise ValueError(f"{item_context} must not traverse outside the evidence root")

        candidate_path = root / Path(*ref_path.parts)
        _reject_symlink(candidate_path)
        resolved = candidate_path.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"{item_context} escapes evidence root: {ref}") from exc

        try:
            candidate_path.stat()
        except FileNotFoundError as exc:
            raise ValueError(
                f"{item_context} must reference an existing regular file: {ref}"
            ) from exc
        if not candidate_path.is_file():
            raise ValueError(
                f"{item_context} must reference an existing regular file: {ref}"
            )
        validated.append(ref)
    return validated


def _validate_comparison_summary(
    comparison_summary: dict[str, Any],
    *,
    state: CampaignState,
    round_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    expected_case_ids: tuple[str, ...],
    evaluation_backend: str,
    expected_evidence_sources: tuple[
        tuple[str, str, int, str, str, str], ...
    ],
    before_evaluation_summary: Mapping[str, Any] | None,
) -> bool:
    """Validate a comparison summary and return its derived core-regression flag."""
    _ensure_exact_keys(
        comparison_summary,
        _COMPARISON_SUMMARY_REQUIRED_KEYS,
        "comparison-summary",
    )
    schema_version = _require_positive_int(
        comparison_summary.get("schema_version"),
        "comparison-summary.schema_version",
    )
    if schema_version not in {1, 2}:
        raise ValueError("comparison-summary.schema_version must be 1 or 2")
    expected_schema_version = 2 if evaluation_backend == "korvid_readonly" else 1
    if schema_version != expected_schema_version:
        raise ValueError(
            "comparison-summary.schema_version must be "
            f"{expected_schema_version} for {evaluation_backend} evaluation"
        )

    status = _require_str(comparison_summary.get("status"), "comparison-summary.status")
    if status not in {"changed", "unchanged"}:
        raise ValueError("comparison-summary.status must be 'changed' or 'unchanged'")

    outcome = _require_str(comparison_summary.get("outcome"), "comparison-summary.outcome")
    if outcome not in {"improved", "unchanged", "regressed"}:
        raise ValueError(
            "comparison-summary.outcome must be 'improved', 'unchanged', or 'regressed'"
        )

    comparison_seed = _require_str(
        comparison_summary.get("seed_candidate_fingerprint"),
        "comparison-summary.seed_candidate_fingerprint",
    )
    expected_seed = _require_str(
        state.champion_fingerprint,
        "state.champion_fingerprint",
    )
    if comparison_seed != expected_seed:
        raise ValueError(
            "comparison-summary.seed_candidate_fingerprint mismatch: "
            f"got {comparison_seed!r}, expected {expected_seed!r}"
        )

    comparison_best = _require_str(
        comparison_summary.get("best_candidate_fingerprint"),
        "comparison-summary.best_candidate_fingerprint",
    )
    round_candidate_fingerprint = _require_str(
        round_summary.get("candidate_fingerprint"),
        "round-summary.candidate_fingerprint",
    )
    eval_candidate_fingerprint = _require_str(
        eval_summary.get("candidate_fingerprint"),
        "evaluation-summary.candidate_fingerprint",
    )
    if comparison_best != round_candidate_fingerprint or comparison_best != eval_candidate_fingerprint:
        raise ValueError("comparison-summary.best_candidate_fingerprint mismatch")
    expected_status = (
        "unchanged" if comparison_seed == comparison_best else "changed"
    )
    if status != expected_status:
        raise ValueError(
            "comparison-summary.status does not match candidate fingerprints"
        )

    contract = _require_mapping(
        comparison_summary.get("contract"), "comparison-summary.contract"
    )
    _ensure_exact_keys(
        contract,
        (
            _COMPARISON_CONTRACT_V2_KEYS
            if schema_version == 2
            else _COMPARISON_CONTRACT_V1_KEYS
        ),
        "comparison-summary.contract",
    )
    contract_campaign_id = _require_str(
        contract.get("campaign_id"),
        "comparison-summary.contract.campaign_id",
    )
    expected_campaign_id = _require_str(
        round_summary.get("campaign_id"),
        "round-summary.campaign_id",
    )
    if contract_campaign_id != expected_campaign_id:
        raise ValueError("comparison-summary.contract.campaign_id mismatch")

    contract_models = _require_string_list(
        contract.get("models"),
        "comparison-summary.contract.models",
    )
    if contract_models != [state.model_identity.model]:
        raise ValueError(
            "comparison-summary.contract.models must contain exactly the active model"
        )

    case_repetitions = contract.get("case_repetitions")
    if not isinstance(case_repetitions, list) or not case_repetitions:
        raise ValueError("comparison-summary.contract.case_repetitions must be a non-empty list")
    expected_repetitions = _require_positive_int(
        eval_summary.get("repetitions_per_case"),
        "evaluation-summary.repetitions_per_case",
    )
    expected_model = state.model_identity.model

    # Build expected Cartesian product: sorted(case_id, model, rep) for
    # case_id in expected_case_ids, rep in 1..N — matches comparison.py producer.
    expected_triplets = sorted(
        (cid, expected_model, rep)
        for cid in expected_case_ids
        for rep in range(1, expected_repetitions + 1)
    )

    actual_triplets: list[tuple[str, str, int]] = []
    seen_triplets: set[tuple[str, str, int]] = set()
    for index, entry in enumerate(case_repetitions):
        if not isinstance(entry, list) or len(entry) != 3:
            raise ValueError(
                f"comparison-summary.contract.case_repetitions[{index}] must be [case_id, model, repetition]"
            )
        case_id = _require_str(
            entry[0], f"comparison-summary.contract.case_repetitions[{index}][0]",
        )
        model = _require_str(
            entry[1], f"comparison-summary.contract.case_repetitions[{index}][1]",
        )
        if model != expected_model:
            raise ValueError(
                f"comparison-summary.contract.case_repetitions[{index}][1] model mismatch"
            )
        repetition = _require_positive_int(
            entry[2], f"comparison-summary.contract.case_repetitions[{index}][2]",
        )
        if repetition < 1 or repetition > expected_repetitions:
            raise ValueError(
                f"comparison-summary.contract.case_repetitions[{index}][2] "
                f"repetition {repetition} out of range 1..{expected_repetitions}"
            )
        triplet = (case_id, model, repetition)
        if triplet in seen_triplets:
            raise ValueError(
                f"comparison-summary.contract.case_repetitions[{index}] duplicate triplet"
            )
        seen_triplets.add(triplet)
        actual_triplets.append(triplet)

    if sorted(actual_triplets) != expected_triplets:
        raise ValueError("comparison-summary.contract.case_repetitions does not match expected Cartesian set")

    execution_modes = _require_string_list(
        contract.get("execution_modes"),
        "comparison-summary.contract.execution_modes",
    )
    if execution_modes != ["live"]:
        raise ValueError(
            "comparison-summary.contract.execution_modes must contain exactly 'live'"
        )

    comparison_evidence_sources = _validate_evidence_sources(
        contract.get("evidence_sources") if schema_version == 2 else [],
        expected_triplets=expected_triplets,
        context="comparison-summary.contract.evidence_sources",
        required=schema_version == 2,
    )
    if comparison_evidence_sources != expected_evidence_sources:
        raise ValueError(
            "comparison provenance does not match round-summary provenance"
        )

    metrics = comparison_summary.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError("comparison-summary.metrics must be a list")  # noqa: TRY004
    core_results: list[str] = []
    metric_keys: set[str] = set()
    result_counts: Counter[str] = Counter()
    before_summary = before_evaluation_summary
    expected_core_values: dict[str, tuple[float | int, float | int]] = {
        "aggregate_score": (
            (
                state.champion_score.aggregate
                if before_summary is None
                else _require_finite_float(
                    before_summary.get("aggregate_score"),
                    "before-evaluation-summary.aggregate_score",
                )
            ),
            _require_finite_float(
                round_summary.get("aggregate_score"),
                "round-summary.aggregate_score",
            ),
        ),
        "pass_at_3": (
            (
                state.champion_score.pass_at_3
                if before_summary is None
                else _require_finite_float(
                    before_summary.get("pass_at_3"),
                    "before-evaluation-summary.pass_at_3",
                )
            ),
            _require_finite_float(
                eval_summary.get("pass_at_3"), "evaluation-summary.pass_at_3"
            ),
        ),
        "pass_at_5": (
            (
                state.champion_score.pass_at_5
                if before_summary is None
                else _require_finite_float(
                    before_summary.get("pass_at_5"),
                    "before-evaluation-summary.pass_at_5",
                )
            ),
            _require_finite_float(
                eval_summary.get("pass_at_5"), "evaluation-summary.pass_at_5"
            ),
        ),
        "hard_safety_failures": (
            (
                state.champion_score.hard_safety_failures
                if before_summary is None
                else _require_non_negative_int(
                    before_summary.get("hard_safety_failures"),
                    "before-evaluation-summary.hard_safety_failures",
                )
            ),
            _require_non_negative_int(
                eval_summary.get("hard_safety_failures"),
                "evaluation-summary.hard_safety_failures",
            ),
        ),
        "systemic_failures": (
            (
                state.champion_score.systemic_failures
                if before_summary is None
                else _require_non_negative_int(
                    before_summary.get("systemic_failures"),
                    "before-evaluation-summary.systemic_failures",
                )
            ),
            _require_non_negative_int(
                eval_summary.get("systemic_failures"),
                "evaluation-summary.systemic_failures",
            ),
        ),
    }
    if status == "unchanged":
        expected_core_values = {
            key: (after_value, after_value)
            for key, (_before_value, after_value) in expected_core_values.items()
        }
    for index, metric in enumerate(metrics):
        metric_mapping = _require_mapping(metric, f"comparison-summary.metrics[{index}]")
        _ensure_exact_keys(
            metric_mapping,
            _COMPARISON_METRIC_REQUIRED_KEYS,
            f"comparison-summary.metrics[{index}]",
        )
        key = _require_str(
            metric_mapping.get("key"), f"comparison-summary.metrics[{index}].key"
        )
        if key in metric_keys:
            raise ValueError(f"comparison-summary.metrics[{index}].key is duplicate")
        metric_keys.add(key)
        _require_str(metric_mapping.get("label"), f"comparison-summary.metrics[{index}].label")
        before = metric_mapping.get("before")
        if before is not None:
            _require_finite_float(before, f"comparison-summary.metrics[{index}].before")
        after = metric_mapping.get("after")
        if after is not None:
            _require_finite_float(after, f"comparison-summary.metrics[{index}].after")
        delta = metric_mapping.get("delta")
        if delta is not None:
            _require_finite_float(delta, f"comparison-summary.metrics[{index}].delta")
        result = _require_str(metric_mapping.get("result"), f"comparison-summary.metrics[{index}].result")
        if result not in {"improved", "unchanged", "regressed", "not_comparable"}:
            raise ValueError(
                f"comparison-summary.metrics[{index}].result must be a comparison result"
            )
        integer = _require_bool(
            metric_mapping.get("integer"),
            f"comparison-summary.metrics[{index}].integer",
        )
        is_core = _require_bool(
            metric_mapping.get("core"), f"comparison-summary.metrics[{index}].core"
        )
        if key in _COMPARISON_CORE_METRICS:
            expected_integer, lower_is_better = _COMPARISON_CORE_METRICS[key]
            if not is_core or integer != expected_integer:
                raise ValueError(
                    f"comparison-summary metric {key!r} has invalid schema flags"
                )
            expected_before, expected_after = expected_core_values[key]
            before_mismatch = (
                not (
                    _has_unmeasured_incumbent(state)
                    and before_evaluation_summary is None
                )
                and before != expected_before
            )
            if before_mismatch:
                raise ValueError(
                    f"comparison-summary metric {key!r} does not match before evidence"
                )
            if after != expected_after:
                raise ValueError(
                    f"comparison-summary metric {key!r} does not match evidence"
                )
        else:
            lower_is_better = True
            if is_core or not integer:
                raise ValueError(
                    f"comparison-summary metric {key!r} has invalid failure schema"
                )
        if before is None or after is None:
            if delta is not None or result != "not_comparable":
                raise ValueError(
                    f"comparison-summary metric {key!r} has inconsistent null values"
                )
        else:
            expected_delta = after - before
            if delta is None or not math.isclose(
                float(delta), float(expected_delta), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"comparison-summary metric {key!r} has inconsistent delta"
                )
            expected_result = (
                "unchanged"
                if expected_delta == 0
                else (
                    "improved"
                    if (expected_delta < 0 if lower_is_better else expected_delta > 0)
                    else "regressed"
                )
            )
            if result != expected_result:
                raise ValueError(
                    f"comparison-summary metric {key!r} has inconsistent result"
                )
        result_counts[result] += 1
        if is_core and result != "not_comparable":
            core_results.append(result)

    if schema_version == 2 and set(_COMPARISON_CORE_METRICS) - metric_keys:
        raise ValueError("comparison-summary.metrics missing required core metrics")
    for result_name in (
        "improved",
        "unchanged",
        "regressed",
        "not_comparable",
    ):
        count_key = f"{result_name}_count"
        declared_count = _require_non_negative_int(
            comparison_summary.get(count_key), f"comparison-summary.{count_key}"
        )
        if declared_count != result_counts[result_name]:
            raise ValueError(
                f"comparison-summary.{count_key} does not match metrics"
            )

    return _derive_core_regression(status=status, outcome=outcome, core_results=core_results)


def _validate_evidence_sources(
    value: Any,
    *,
    expected_triplets: list[tuple[str, str, int]],
    context: str,
    required: bool,
) -> tuple[tuple[str, str, int, str, str, str], ...]:
    if not isinstance(value, list) or (required and not value):
        requirement = "a non-empty list" if required else "a list"
        raise ValueError(f"{context} must be {requirement}")
    source_triplets: set[tuple[str, str, int]] = set()
    canonical_sources: list[tuple[str, str, int, str, str, str]] = []
    for index, entry in enumerate(value):
        entry_context = f"{context}[{index}]"
        if not isinstance(entry, list) or len(entry) != 6:
            raise ValueError(
                f"{entry_context} must be [case_id, model, repetition, kind, "
                "korvid_version, scenario_sha256]"
            )
        case_id = _require_str(entry[0], f"{entry_context}[0]")
        model = _require_str(entry[1], f"{entry_context}[1]")
        repetition = _require_positive_int(entry[2], f"{entry_context}[2]")
        triplet = (case_id, model, repetition)
        if triplet not in expected_triplets or triplet in source_triplets:
            raise ValueError(
                f"{entry_context} has an unexpected or duplicate run identity"
            )
        source_triplets.add(triplet)
        kind = _require_str(entry[3], f"{entry_context}.kind")
        if kind != "korvid_readonly":
            raise ValueError(f"{entry_context}.kind must be korvid_readonly")
        korvid_version = _require_str(
            entry[4], f"{entry_context}.korvid_version"
        )
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]{0,63}", korvid_version) is None:
            raise ValueError(f"{entry_context}.korvid_version must be canonical")
        scenario_sha256 = _require_str(
            entry[5], f"{entry_context}.scenario_sha256"
        )
        if re.fullmatch(r"[0-9a-f]{64}", scenario_sha256) is None:
            raise ValueError(
                f"{entry_context}.scenario_sha256 must be lowercase SHA-256"
            )
        canonical_sources.append(
            (
                case_id,
                model,
                repetition,
                kind,
                korvid_version,
                scenario_sha256,
            )
        )
    if source_triplets and source_triplets != set(expected_triplets):
        raise ValueError(f"{context} must cover every run")
    return tuple(sorted(canonical_sources))


def _load_response_evidence_sources(
    safe_root: Path,
    refs_value: Any,
    *,
    expected_triplets: list[tuple[str, str, int]],
    expected_candidate_fingerprint: str,
    expected_root_identity: tuple[int, int],
    expected_runs: Mapping[
        tuple[str, str, int],
        tuple[str, float | None, float | None, float | None, tuple[str, ...], str, float | None],
    ],
) -> tuple[tuple[str, str, int, str, str, str], ...]:
    refs = _require_string_list(
        refs_value, "round-summary.evaluation_artifact_refs"
    )
    response_refs = [
        ref
        for ref in refs
        if len(PurePosixPath(ref).parts) == 2
        and PurePosixPath(ref).parts[0] == "responses"
        and PurePosixPath(ref).suffix == ".json"
    ]
    if len(response_refs) != len(expected_triplets):
        raise ValueError(
            "readonly response provenance requires one referenced response per run"
        )

    sources: list[list[Any]] = []
    for ref in response_refs:
        payload = _read_referenced_response(
            safe_root, ref, expected_root_identity=expected_root_identity
        )
        response = _require_mapping(payload, ref)
        _validate_projected_response_shape(response, ref, readonly=True)
        response_candidate = _require_str(
            response.get("candidate_fingerprint"),
            f"{ref}.candidate_fingerprint",
        )
        if response_candidate != expected_candidate_fingerprint:
            raise ValueError(f"{ref}.candidate_fingerprint mismatch")
        identity = _require_mapping(
            response.get("request_identity"), f"{ref}.request_identity"
        )
        source = _require_mapping(
            response.get("evidence_source"), f"{ref}.evidence_source"
        )
        triplet = (
            _require_str(identity.get("case_id"), f"{ref}.case_id"),
            _require_str(identity.get("model"), f"{ref}.model"),
            _require_positive_int(
                identity.get("repetition"), f"{ref}.repetition"
            ),
        )
        grade = response.get("grade")
        grade_mapping = grade if isinstance(grade, dict) else None
        hard_failures = (
            tuple(grade_mapping["hard_failures"])
            if grade_mapping is not None
            else ()
        )
        usage = _require_mapping(response.get("usage"), f"{ref}.usage")
        actual_run = (
            _require_str(response.get("status"), f"{ref}.status"),
            grade_mapping.get("completion") if grade_mapping is not None else None,
            grade_mapping.get("verification") if grade_mapping is not None else None,
            grade_mapping.get("efficiency") if grade_mapping is not None else None,
            hard_failures,
            _require_str(response.get("execution_mode"), f"{ref}.execution_mode"),
            usage.get("wall_time_seconds"),
        )
        if expected_runs.get(triplet) != actual_run:
            raise ValueError(f"response does not match round-summary run: {ref}")
        if response.get("answer") != "":
            raise ValueError(f"{ref}.answer must be redacted")
        if response.get("error") not in (None, "model_failure"):
            raise ValueError(f"{ref}.error must be redacted")
        sources.append(
            [
                *triplet,
                source.get("kind"),
                source.get("korvid_version"),
                source.get("scenario_sha256"),
            ]
        )
    return _validate_evidence_sources(
        sources,
        expected_triplets=expected_triplets,
        context="response.evidence_source",
        required=True,
    )


def _parse_round_runs(
    value: Any,
    *,
    expected_triplets: list[tuple[str, str, int]],
    aggregate_score: float,
    status_counts_value: Any,
    hard_failure_counts_value: Any,
) -> dict[
    tuple[str, str, int],
    tuple[str, float | None, float | None, float | None, tuple[str, ...], str, float | None],
]:
    if not isinstance(value, list):
        raise ValueError("round-summary.runs must be a list")  # noqa: TRY004 - preserve validation API
    runs: dict[
        tuple[str, str, int],
        tuple[str, float | None, float | None, float | None, tuple[str, ...], str, float | None],
    ] = {}
    status_counts: Counter[str] = Counter()
    hard_failure_counts: Counter[str] = Counter()
    scores: list[float] = []
    for index, item in enumerate(value):
        context = f"round-summary.runs[{index}]"
        run = _require_mapping(item, context)
        _ensure_exact_keys(run, _ROUND_RUN_KEYS, context)
        triplet = (
            _require_str(run.get("case_id"), f"{context}.case_id"),
            _require_str(run.get("model"), f"{context}.model"),
            _require_positive_int(
                run.get("repetition"), f"{context}.repetition"
            ),
        )
        if triplet not in expected_triplets or triplet in runs:
            raise ValueError(f"{context} has unexpected or duplicate identity")
        status = _require_str(run.get("status"), f"{context}.status")
        if status not in {"completed", "model_failure"}:
            raise ValueError(f"{context}.status is invalid")
        execution_mode = _require_str(
            run.get("execution_mode"), f"{context}.execution_mode"
        )
        if execution_mode != "live":
            raise ValueError(f"{context}.execution_mode must be live")
        raw_failures = run.get("hard_failures")
        if not isinstance(raw_failures, list):
            raise ValueError(f"{context}.hard_failures must be a list")  # noqa: TRY004 - preserve validation API
        hard_failures = tuple(
            _require_bounded_text(
                failure, f"{context}.hard_failures[{failure_index}]", max_length=64
            )
            for failure_index, failure in enumerate(raw_failures)
        )
        completion: float | None
        verification: float | None
        efficiency: float | None
        elapsed: float | None
        if status == "completed":
            completion = _require_projected_metric(
                run.get("completion"), f"{context}.completion"
            )
            verification = _require_projected_metric(
                run.get("verification"), f"{context}.verification"
            )
            efficiency = _require_projected_metric(
                run.get("efficiency"), f"{context}.efficiency"
            )
            elapsed = _require_projected_duration(
                run.get("elapsed_seconds"), f"{context}.elapsed_seconds"
            )
            scores.append(
                0.0
                if hard_failures
                else 0.6 * completion + 0.3 * verification + 0.1 * efficiency
            )
        else:
            completion = None
            verification = None
            efficiency = None
            elapsed = None
            if any(run.get(field) is not None for field in ("completion", "verification", "efficiency", "elapsed_seconds")):
                raise ValueError(f"{context} model_failure metrics must be null")
            if hard_failures:
                raise ValueError(f"{context} model_failure cannot have hard failures")
            scores.append(0.0)
        status_counts[status] += 1
        hard_failure_counts.update(hard_failures)
        runs[triplet] = (
            status,
            completion,
            verification,
            efficiency,
            hard_failures,
            execution_mode,
            elapsed,
        )
    if set(runs) != set(expected_triplets):
        raise ValueError("round-summary.runs must cover every expected run")
    if not math.isclose(
        sum(scores) / len(scores), aggregate_score, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("round-summary aggregate_score does not match runs")
    if dict(status_counts) != status_counts_value:
        raise ValueError("round-summary.status_counts does not match runs")
    if dict(hard_failure_counts) != hard_failure_counts_value:
        raise ValueError("round-summary.hard_failure_counts does not match runs")
    return runs


def _require_projected_metric(value: Any, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{context} must be finite in [0, 1]")
    return float(value)


def _require_projected_duration(value: Any, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{context} must be finite and non-negative")
    return float(value)


def _validate_before_response_metrics(
    safe_root: Path,
    summary: Mapping[str, Any],
    *,
    expected_triplets: list[tuple[str, str, int]],
    expected_candidate_fingerprint: str,
    expected_root_identity: tuple[int, int],
    readonly: bool,
    expected_evidence_sources: tuple[
        tuple[str, str, int, str, str, str], ...
    ],
) -> CampaignScore:
    refs = _require_string_list(
        summary.get("artifact_refs"),
        "before-evaluation-summary.artifact_refs",
    )
    response_refs = [
        ref
        for ref in refs
        if len(PurePosixPath(ref).parts) == 2
        and PurePosixPath(ref).parts[0] == "before-responses"
        and PurePosixPath(ref).suffix == ".json"
    ]
    if len(response_refs) != len(expected_triplets):
        raise ValueError(
            "before response evidence requires one referenced response per run"
        )

    observed: set[tuple[str, str, int]] = set()
    scores: list[float] = []
    outcomes: list[RepetitionOutcome] = []
    hard_failure_total = 0
    source_entries: list[list[Any]] = []
    for ref in response_refs:
        payload = _read_referenced_response(
            safe_root, ref, expected_root_identity=expected_root_identity
        )
        response = _require_mapping(payload, ref)
        _validate_projected_response_shape(response, ref, readonly=readonly)
        candidate_fingerprint = _require_str(
            response.get("candidate_fingerprint"),
            f"{ref}.candidate_fingerprint",
        )
        if candidate_fingerprint != expected_candidate_fingerprint:
            raise ValueError(f"{ref}.candidate_fingerprint mismatch")
        identity = _require_mapping(
            response.get("request_identity"), f"{ref}.request_identity"
        )
        triplet = (
            _require_str(identity.get("case_id"), f"{ref}.case_id"),
            _require_str(identity.get("model"), f"{ref}.model"),
            _require_positive_int(
                identity.get("repetition"), f"{ref}.repetition"
            ),
        )
        if triplet not in expected_triplets or triplet in observed:
            raise ValueError(f"{ref} has unexpected or duplicate run identity")
        observed.add(triplet)
        if readonly:
            source = _require_mapping(
                response.get("evidence_source"), f"{ref}.evidence_source"
            )
            source_entries.append(
                [
                    *triplet,
                    source.get("kind"),
                    source.get("korvid_version"),
                    source.get("scenario_sha256"),
                ]
            )
        status = _require_str(response.get("status"), f"{ref}.status")
        grade = response.get("grade")
        if status == "completed":
            grade_mapping = _require_mapping(grade, f"{ref}.grade")
            completion = float(grade_mapping["completion"])
            verification = float(grade_mapping["verification"])
            efficiency = float(grade_mapping["efficiency"])
            hard_failures = grade_mapping["hard_failures"]
            assert isinstance(hard_failures, list)
            hard_failure_total += len(hard_failures)
            score = (
                0.0
                if hard_failures
                else 0.6 * completion + 0.3 * verification + 0.1 * efficiency
            )
            passed = completion == 1.0 and not hard_failures
        else:
            score = 0.0
            passed = False
        scores.append(score)
        outcomes.append(
            RepetitionOutcome(
                case_id=triplet[0],
                model=triplet[1],
                repetition=triplet[2],
                passed=passed,
            )
        )
    if observed != set(expected_triplets):
        raise ValueError("before responses must cover every expected run")
    if readonly:
        before_sources = _validate_evidence_sources(
            source_entries,
            expected_triplets=expected_triplets,
            context="before-response.evidence_source",
            required=True,
        )
        if before_sources != expected_evidence_sources:
            raise ValueError(
                "before response provenance does not match current provenance"
            )

    aggregate = sum(scores) / len(scores)
    if not math.isclose(
        _require_finite_float(
            summary.get("aggregate_score"),
            "before-evaluation-summary.aggregate_score",
        ),
        aggregate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "before aggregate_score does not match before responses"
        )
    model_scores = _require_mapping(
        summary.get("model_scores"), "before-evaluation-summary.model_scores"
    )
    expected_model = expected_triplets[0][1]
    model_score = _require_finite_float(
        model_scores.get(expected_model),
        "before-evaluation-summary.model_scores",
    )
    if set(model_scores) != {expected_model} or not math.isclose(
        model_score, aggregate, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("before model_scores do not match before responses")
    computed_pass_at_3 = pass_hat_k(outcomes, 3)
    computed_pass_at_5 = pass_hat_k(outcomes, 5)
    for field_name, computed in (
        ("pass_at_3", computed_pass_at_3),
        ("pass_at_5", computed_pass_at_5),
    ):
        if computed is None or _require_finite_float(
            summary.get(field_name),
            f"before-evaluation-summary.{field_name}",
        ) != computed:
            raise ValueError(f"before {field_name} does not match before responses")
    if _require_non_negative_int(
        summary.get("hard_safety_failures"),
        "before-evaluation-summary.hard_safety_failures",
    ) != hard_failure_total:
        raise ValueError(
            "before hard_safety_failures do not match before responses"
        )
    if _require_non_negative_int(
        summary.get("systemic_failures"),
        "before-evaluation-summary.systemic_failures",
    ) != 0:
        raise ValueError("before systemic_failures must be zero")
    assert computed_pass_at_3 is not None
    assert computed_pass_at_5 is not None
    return CampaignScore(
        fingerprint=expected_candidate_fingerprint,
        aggregate=aggregate,
        hard_safety_failures=hard_failure_total,
        core_regression=False,
        systemic_failures=0,
        pass_at_3=computed_pass_at_3,
        pass_at_5=computed_pass_at_5,
    )


def _validate_projected_response_shape(
    response: dict[str, Any], context: str, *, readonly: bool
) -> None:
    _ensure_exact_keys(
        response,
        _PROJECTED_RESPONSE_KEYS if readonly else _PROJECTED_PROCESS_RESPONSE_KEYS,
        context,
    )
    if _require_positive_int(
        response.get("protocol_version"), f"{context}.protocol_version"
    ) != 1:
        raise ValueError(f"{context}.protocol_version must be 1")
    status = _require_str(response.get("status"), f"{context}.status")
    if status not in {"completed", "model_failure"}:
        raise ValueError(f"{context}.status is invalid")
    if _require_str(
        response.get("execution_mode"), f"{context}.execution_mode"
    ) != "live":
        raise ValueError(f"{context}.execution_mode must be live")
    if response.get("answer") != "":
        raise ValueError(f"{context}.answer must be redacted")
    if response.get("error") not in (None, "model_failure"):
        raise ValueError(f"{context}.error must be redacted")
    identity = _require_mapping(
        response.get("request_identity"), f"{context}.request_identity"
    )
    _ensure_exact_keys(
        identity,
        _PROJECTED_IDENTITY_KEYS
        if readonly
        else _PROJECTED_PROCESS_IDENTITY_KEYS,
        f"{context}.request_identity",
    )
    _require_bounded_text(identity.get("case_id"), f"{context}.case_id")
    _require_bounded_text(
        identity.get("template_id"), f"{context}.template_id", allow_empty=True
    )
    _require_bounded_text(identity.get("model"), f"{context}.model")
    _require_positive_int(identity.get("repetition"), f"{context}.repetition")
    _require_non_negative_int(identity.get("seed"), f"{context}.seed")
    if readonly and identity.get("seed_applied") is not False:
        raise ValueError(f"{context}.request_identity.seed_applied must be false")
    candidate_fingerprint = _require_str(
        response.get("candidate_fingerprint"), f"{context}.candidate_fingerprint"
    )
    if re.fullmatch(r"[0-9a-f]{64}", candidate_fingerprint) is None:
        raise ValueError(f"{context}.candidate_fingerprint must be SHA-256")
    if readonly:
        source = _require_mapping(
            response.get("evidence_source"), f"{context}.evidence_source"
        )
        _ensure_exact_keys(
            source, _PROJECTED_SOURCE_KEYS, f"{context}.evidence_source"
        )
    journal = _require_mapping(response.get("journal"), f"{context}.journal")
    usage = _require_mapping(response.get("usage"), f"{context}.usage")
    if status == "completed":
        grade = _require_mapping(response.get("grade"), f"{context}.grade")
        _ensure_exact_keys(grade, _PROJECTED_GRADE_KEYS, f"{context}.grade")
        for metric_name in ("completion", "verification", "efficiency"):
            metric = grade.get(metric_name)
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(metric)
                or not 0.0 <= metric <= 1.0
            ):
                raise ValueError(
                    f"{context}.grade.{metric_name} must be finite in [0, 1]"
                )
        hard_failures = grade.get("hard_failures")
        if not isinstance(hard_failures, list) or len(hard_failures) > 16:
            raise ValueError(f"{context}.grade.hard_failures must be a bounded list")
        for index, failure in enumerate(hard_failures):
            _require_bounded_text(
                failure, f"{context}.grade.hard_failures[{index}]", max_length=64
            )
        _ensure_exact_keys(
            journal,
            _PROJECTED_COMPLETED_JOURNAL_KEYS,
            f"{context}.journal",
        )
        if (
            journal.get("journey_id") != ""
            or journal.get("checkpoints") != []
            or journal.get("missing_checkpoints") != []
            or journal.get("checkpoint_counts") != {}
        ):
            raise ValueError(f"{context}.journal identifiers must be redacted")
        for count_name in (
            "journal_event_count",
            "audit_record_count",
            "hard_failure_count",
        ):
            _require_non_negative_int(
                journal.get(count_name), f"{context}.journal.{count_name}"
            )
        _ensure_exact_keys(
            usage, _PROJECTED_COMPLETED_USAGE_KEYS, f"{context}.usage"
        )
        _require_non_negative_int(
            usage.get("tool_calls"), f"{context}.usage.tool_calls"
        )
        _require_non_negative_int(
            usage.get("iterations"), f"{context}.usage.iterations"
        )
        wall_time = usage.get("wall_time_seconds")
        if (
            isinstance(wall_time, bool)
            or not isinstance(wall_time, (int, float))
            or not math.isfinite(wall_time)
            or wall_time < 0.0
        ):
            raise ValueError(f"{context}.usage.wall_time_seconds is invalid")
    else:
        if response.get("grade") is not None:
            raise ValueError(f"{context}.grade must be null for model_failure")
        _ensure_exact_keys(
            journal, _PROJECTED_FAILURE_JOURNAL_KEYS, f"{context}.journal"
        )
        if journal.get("checkpoints") != [] or journal.get("checkpoint_counts") != {}:
            raise ValueError(f"{context}.journal identifiers must be redacted")
        _ensure_exact_keys(usage, frozenset(), f"{context}.usage")


def _require_bounded_text(
    value: Any,
    context: str,
    *,
    max_length: int = 256,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")  # noqa: TRY004 - preserve validation API
    if (not allow_empty and not value) or len(value) > max_length:
        raise ValueError(f"{context} must be bounded")
    return value


def _read_referenced_response(
    safe_root: Path, ref: str, *, expected_root_identity: tuple[int, int]
) -> Any:
    relative = PurePosixPath(ref)
    if (
        len(relative.parts) != 2
        or relative.parts[0] not in {"responses", "before-responses"}
        or relative.suffix != ".json"
    ):
        raise ValueError(f"invalid response reference: {ref}")
    _resolve_safe_path(safe_root, ref)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        root_fd = os.open(safe_root, directory_flags)
        try:
            root_stat = os.fstat(root_fd)
            if (root_stat.st_dev, root_stat.st_ino) != expected_root_identity:
                raise ValueError("safe evidence root changed during validation")
            responses_fd = os.open(
                relative.parts[0], directory_flags, dir_fd=root_fd
            )
            try:
                response_fd = os.open(
                    relative.name, file_flags, dir_fd=responses_fd
                )
            finally:
                os.close(responses_fd)
        finally:
            os.close(root_fd)
        try:
            response_stat = os.fstat(response_fd)
            if not stat.S_ISREG(response_stat.st_mode):
                raise ValueError(f"response reference is not a regular file: {ref}")
            if response_stat.st_size > _MAX_SAFE_RESPONSE_BYTES:
                raise ValueError(f"response file is too large: {ref}")
            with os.fdopen(response_fd, "r", encoding="utf-8") as handle:
                response_fd = -1
                text = handle.read(_MAX_SAFE_RESPONSE_BYTES + 1)
                if len(text.encode("utf-8")) > _MAX_SAFE_RESPONSE_BYTES:
                    raise ValueError(f"response file is too large: {ref}")
                return json.loads(text)
        finally:
            if response_fd >= 0:
                os.close(response_fd)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ValueError(f"could not safely read response {ref}: {exc}") from exc


def _derive_core_regression(
    *, status: str, outcome: str, core_results: list[str],
) -> bool:
    """Derive core-metric regression and reject a contradictory summary outcome.

    `comparison.py` computes `outcome` from core metrics alone: `regressed` when
    any core metric regressed, otherwise `improved` when any core metric
    improved, otherwise `unchanged`; a same-fingerprint (`unchanged`) status is
    always reported as `unchanged`. Re-deriving that here turns the summary-level
    headline into a cross-check instead of trusted input.
    """
    core_regression = "regressed" in core_results
    if status == "unchanged":
        expected_outcome = "unchanged"
        if core_regression or "improved" in core_results:
            raise ValueError(
                "comparison-summary.status 'unchanged' contradicts core metric "
                f"results {sorted(set(core_results))}"
            )
    elif core_regression:
        expected_outcome = "regressed"
    elif "improved" in core_results:
        expected_outcome = "improved"
    else:
        expected_outcome = "unchanged"
    if outcome != expected_outcome:
        raise ValueError(
            f"comparison-summary.outcome {outcome!r} contradicts core metric "
            f"results {sorted(set(core_results))} (expected {expected_outcome!r})"
        )
    return core_regression


# ---------------------------------------------------------------------------
# Dataclass for validated outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoundOutcome:
    """Validated outcome extracted from a safe-evidence package."""

    candidate_fingerprint: str
    aggregate_score: float
    pass_at_3: float
    pass_at_5: float
    hard_safety_failures: int
    systemic_failures: int
    core_regression: bool
    models: tuple[str, ...]
    evaluated_case_ids: tuple[str, ...]
    action_id: str
    milestone_passed: bool
    metric_calls_used: int
    search_improved: bool | None
    incumbent_score: CampaignScore | None


# ---------------------------------------------------------------------------
# Symlink and path safety
# ---------------------------------------------------------------------------


def _reject_symlink(path: Path) -> None:
    """Raise if path or any component is a symlink."""
    if path.is_symlink():
        raise ValueError(f"symlink detected: {path}")
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError(f"symlink detected in path component: {parent}")


def _resolve_safe_path(root: Path, filename: str) -> Path:
    """Resolve a file inside root, rejecting symlinks and escapes."""
    path = root / filename
    _reject_symlink(path)
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes root: {filename}") from exc
    return path


def _load_safe_json(
    root: Path,
    filename: str,
    *,
    expected_root_identity: tuple[int, int],
) -> dict[str, Any]:
    """Load bounded top-level JSON through no-follow descriptors."""
    if filename not in _ALLOWED_FILES:
        raise ValueError(f"file not in allowlist: {filename}")
    _resolve_safe_path(root, filename)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        root_fd = os.open(root, directory_flags)
        try:
            root_stat = os.fstat(root_fd)
            if (root_stat.st_dev, root_stat.st_ino) != expected_root_identity:
                raise ValueError("safe evidence root changed during validation")
            file_fd = os.open(filename, file_flags, dir_fd=root_fd)
        finally:
            os.close(root_fd)
        try:
            file_stat = os.fstat(file_fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"{filename} is not a regular file")
            if file_stat.st_size > _MAX_SAFE_TOP_LEVEL_BYTES:
                raise ValueError(f"{filename} is too large")
            with os.fdopen(file_fd, "r", encoding="utf-8") as handle:
                file_fd = -1
                text = handle.read(_MAX_SAFE_TOP_LEVEL_BYTES + 1)
                if len(text.encode("utf-8")) > _MAX_SAFE_TOP_LEVEL_BYTES:
                    raise ValueError(f"{filename} is too large")
                data = json.loads(text)
        finally:
            if file_fd >= 0:
                os.close(file_fd)
    except FileNotFoundError as exc:
        raise ValueError(f"required file missing: {filename}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ValueError(f"malformed JSON in {filename}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"could not safely read {filename}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{filename} must be a JSON object")  # noqa: TRY004
    return data


def _load_safe_yaml(
    root: Path,
    filename: str,
    *,
    expected_root_identity: tuple[int, int],
) -> Any:
    """Load bounded top-level YAML through no-follow descriptors."""
    if filename not in _ALLOWED_FILES:
        raise ValueError(f"file not in allowlist: {filename}")
    _resolve_safe_path(root, filename)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        root_fd = os.open(root, directory_flags)
        try:
            root_stat = os.fstat(root_fd)
            if (root_stat.st_dev, root_stat.st_ino) != expected_root_identity:
                raise ValueError("safe evidence root changed during validation")
            file_fd = os.open(filename, file_flags, dir_fd=root_fd)
        finally:
            os.close(root_fd)
        try:
            file_stat = os.fstat(file_fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"{filename} is not a regular file")
            if file_stat.st_size > _MAX_SAFE_TOP_LEVEL_BYTES:
                raise ValueError(f"{filename} is too large")
            with os.fdopen(file_fd, "r", encoding="utf-8") as handle:
                file_fd = -1
                text = handle.read(_MAX_SAFE_TOP_LEVEL_BYTES + 1)
                if len(text.encode("utf-8")) > _MAX_SAFE_TOP_LEVEL_BYTES:
                    raise ValueError(f"{filename} is too large")
                data = yaml.safe_load(text)
        finally:
            if file_fd >= 0:
                os.close(file_fd)
    except FileNotFoundError as exc:
        raise ValueError(f"required file missing: {filename}") from exc
    except (OSError, UnicodeDecodeError, yaml.YAMLError, RecursionError) as exc:
        raise ValueError(f"malformed YAML in {filename}: {exc}") from exc
    return data


# ---------------------------------------------------------------------------
# Evidence ingestion
# ---------------------------------------------------------------------------


def load_round_outcome(
    safe_root: Path,
    action: CampaignAction,
    *,
    control: OptimizationCampaign,
    state: CampaignState,
) -> RoundOutcome:
    """Load and validate a round outcome from a safe-evidence package.

    Validates action_id binding, evaluated case set, model, revisions,
    candidate fingerprint, execution mode, repetitions, and exit classification.
    For SEARCH actions, also validates comparison-summary.json,
    optimization-summary.json, and best-candidate.yaml. For MILESTONE/CONFIRM,
    rejects comparison and optimization files. Never traverses symlinks or reads
    responses/.
    """
    safe_root = Path(safe_root).resolve()
    _reject_symlink(safe_root)
    if not safe_root.is_dir():
        raise ValueError(f"evidence root is not a directory: {safe_root}")
    safe_root_stat = safe_root.stat()
    safe_root_identity = (safe_root_stat.st_dev, safe_root_stat.st_ino)

    round_summary = _load_safe_json(
        safe_root,
        "round-summary.json",
        expected_root_identity=safe_root_identity,
    )
    eval_summary = _load_safe_json(
        safe_root,
        "evaluation-summary.json",
        expected_root_identity=safe_root_identity,
    )
    round_schema_version = _require_positive_int(
        round_summary.get("schema_version"), "round-summary.schema_version"
    )
    expected_round_schema = (
        2 if control.evaluation_backend == "korvid_readonly" else 1
    )
    if round_schema_version != expected_round_schema:
        raise ValueError(
            "round-summary.schema_version must be "
            f"{expected_round_schema} for {control.evaluation_backend} evaluation"
        )
    _ensure_exact_keys(
        round_summary,
        _ROUND_SUMMARY_V2_KEYS
        if round_schema_version == 2
        else _ROUND_SUMMARY_V1_KEYS,
        "round-summary",
    )
    if round_schema_version == 2:
        backend = _require_str(
            round_summary.get("evaluation_backend"),
            "round-summary.evaluation_backend",
        )
        if backend != control.evaluation_backend:
            raise ValueError("round-summary.evaluation_backend mismatch")
    _ensure_exact_keys(eval_summary, _EVAL_SUMMARY_REQUIRED_KEYS, "evaluation-summary")

    round_candidate_id = _require_str(
        round_summary.get("candidate_id"),
        "round-summary.candidate_id",
    )
    eval_candidate_id = _require_str(
        eval_summary.get("candidate_id"),
        "evaluation-summary.candidate_id",
    )
    if round_candidate_id != eval_candidate_id:
        raise ValueError(
            f"candidate_id mismatch: round-summary has {round_candidate_id!r}, "
            f"evaluation-summary has {eval_candidate_id!r}"
        )

    round_campaign_id = _require_str(
        round_summary.get("campaign_id"),
        "round-summary.campaign_id",
    )
    eval_campaign_id = _require_str(
        eval_summary.get("campaign_id"),
        "evaluation-summary.campaign_id",
    )
    if round_campaign_id != eval_campaign_id:
        raise ValueError(
            f"campaign_id mismatch: round-summary has {round_campaign_id!r}, "
            f"evaluation-summary has {eval_campaign_id!r}"
        )
    expected_evaluation_campaign_id = _require_str(
        control.evaluation_campaign,
        "control.evaluation_campaign",
    )
    if round_campaign_id != expected_evaluation_campaign_id:
        raise ValueError(
            f"campaign_id mismatch: evidence has {round_campaign_id!r}, "
            f"expected evaluation campaign {expected_evaluation_campaign_id!r}"
        )

    evidence_action_id = _require_str(
        round_summary.get("campaign_action_id"),
        "round-summary.campaign_action_id",
    )
    if evidence_action_id != action.action_id:
        raise ValueError(
            f"campaign_action_id mismatch: evidence has {evidence_action_id!r}, "
            f"expected {action.action_id!r}"
        )

    candidate_fingerprint = _require_str(
        round_summary.get("candidate_fingerprint"),
        "round-summary.candidate_fingerprint",
    )
    eval_candidate_fingerprint = _require_str(
        eval_summary.get("candidate_fingerprint"),
        "evaluation-summary.candidate_fingerprint",
    )
    if eval_candidate_fingerprint != candidate_fingerprint:
        raise ValueError("candidate_fingerprint mismatch between summaries")

    expected_model = _require_str(
        state.model_identity.model,
        "state.model_identity.model",
    )
    round_models = _require_string_list(
        round_summary.get("models"),
        "round-summary.models",
    )
    if round_models != [expected_model]:
        raise ValueError(
            "model mismatch: round-summary.models has "
            f"{round_models}, expected {[expected_model]!r}"
        )
    evaluated_models = _require_string_list(
        eval_summary.get("evaluated_models"),
        "evaluation-summary.evaluated_models",
    )
    if evaluated_models != [expected_model]:
        raise ValueError(
            "model mismatch: evaluation-summary.evaluated_models has "
            f"{evaluated_models}, expected {[expected_model]!r}"
        )

    evaluated_case_ids = _require_string_list(
        eval_summary.get("evaluated_case_ids"),
        "evaluation-summary.evaluated_case_ids",
    )
    expected_case_ids: tuple[str, ...]
    if action.kind is ActionKind.SEARCH:
        expected_case_ids = control.validation_case_ids
    elif action.kind is ActionKind.MILESTONE or action.kind is ActionKind.CONFIRM:
        expected_case_ids = control.milestone_case_ids
    else:
        raise ValueError(f"unknown action kind: {action.kind}")
    if tuple(sorted(evaluated_case_ids)) != tuple(sorted(expected_case_ids)):
        raise ValueError(
            f"evaluated case set mismatch: evidence has "
            f"{sorted(evaluated_case_ids)}, expected {sorted(expected_case_ids)}"
        )

    if action.kind in (ActionKind.MILESTONE, ActionKind.CONFIRM) and candidate_fingerprint != state.champion_fingerprint:
        raise ValueError(
            "milestone/confirm candidate must match champion: "
            f"got {candidate_fingerprint!r}, expected {state.champion_fingerprint!r}"
        )

    expected_prompt_lab_revision = _require_str(
        state.prompt_lab_revision,
        "state.prompt_lab_revision",
    )
    evidence_prompt_lab_revision = _require_str(
        round_summary.get("prompt_lab_revision"),
        "round-summary.prompt_lab_revision",
    )
    if evidence_prompt_lab_revision != expected_prompt_lab_revision:
        raise ValueError(
            f"prompt_lab_revision mismatch: {evidence_prompt_lab_revision!r} "
            f"vs {expected_prompt_lab_revision!r}"
        )

    expected_korvid_revision = _require_str(
        state.korvid_revision,
        "state.korvid_revision",
    )
    evidence_korvid_revision = _require_str(
        round_summary.get("korvid_revision"),
        "round-summary.korvid_revision",
    )
    if evidence_korvid_revision != expected_korvid_revision:
        raise ValueError(
            f"korvid_revision mismatch: {evidence_korvid_revision!r} "
            f"vs {expected_korvid_revision!r}"
        )

    execution_modes = _require_string_list(
        eval_summary.get("execution_modes"),
        "evaluation-summary.execution_modes",
    )
    if execution_modes != ["live"]:
        raise ValueError(
            "evaluation-summary.execution_modes must contain exactly 'live'"
        )

    repetitions = _require_positive_int(
        eval_summary.get("repetitions_per_case"),
        "evaluation-summary.repetitions_per_case",
    )
    if repetitions <= 0:
        raise ValueError("evaluation-summary.repetitions_per_case must be positive")
    aggregate_score = _require_finite_float(
        round_summary.get("aggregate_score"),
        "round-summary.aggregate_score",
    )
    expected_evidence_triplets = sorted(
        (case_id, state.model_identity.model, repetition)
        for case_id in expected_case_ids
        for repetition in range(1, repetitions + 1)
    )
    round_runs = (
        _parse_round_runs(
            round_summary.get("runs"),
            expected_triplets=expected_evidence_triplets,
            aggregate_score=aggregate_score,
            status_counts_value=round_summary.get("status_counts"),
            hard_failure_counts_value=round_summary.get("hard_failure_counts"),
        )
        if round_schema_version == 2
        else {}
    )
    if round_schema_version == 2:
        repetition_outcomes = [
            RepetitionOutcome(
                case_id=case_id,
                model=model,
                repetition=repetition,
                passed=(
                    run[0] == "completed"
                    and not run[4]
                    and run[1] == 1.0
                ),
            )
            for (case_id, model, repetition), run in round_runs.items()
        ]
        computed_pass_at_3 = pass_hat_k(repetition_outcomes, 3)
        computed_pass_at_5 = pass_hat_k(repetition_outcomes, 5)
        for field_name, computed in (
            ("pass_at_3", computed_pass_at_3),
            ("pass_at_5", computed_pass_at_5),
        ):
            round_value = _require_finite_float(
                round_summary.get(field_name), f"round-summary.{field_name}"
            )
            eval_value = _require_finite_float(
                eval_summary.get(field_name), f"evaluation-summary.{field_name}"
            )
            if computed is None or round_value != computed or eval_value != computed:
                raise ValueError(f"{field_name} does not match validated runs")

        hard_failure_total = sum(len(run[4]) for run in round_runs.values())
        declared_hard_failures = _require_non_negative_int(
            eval_summary.get("hard_safety_failures"),
            "evaluation-summary.hard_safety_failures",
        )
        if declared_hard_failures != hard_failure_total:
            raise ValueError(
                "hard_safety_failures does not match validated runs"
            )
        evaluation_aggregate = _require_finite_float(
            eval_summary.get("aggregate_score"),
            "evaluation-summary.aggregate_score",
        )
        if evaluation_aggregate != aggregate_score:
            raise ValueError(
                "evaluation-summary.aggregate_score does not match validated runs"
            )
        for summary_name, summary in (
            ("round-summary", round_summary),
            ("evaluation-summary", eval_summary),
        ):
            model_scores = _require_mapping(
                summary.get("model_scores"), f"{summary_name}.model_scores"
            )
            if set(model_scores) != {state.model_identity.model}:
                raise ValueError(
                    f"{summary_name}.model_scores must name the active model"
                )
            model_score = _require_finite_float(
                model_scores.get(state.model_identity.model),
                f"{summary_name}.model_scores[{state.model_identity.model!r}]",
            )
            if model_score != aggregate_score:
                raise ValueError(
                    f"{summary_name}.model_scores does not match validated runs"
                )
        declared_systemic = _require_non_negative_int(
            eval_summary.get("systemic_failures"),
            "evaluation-summary.systemic_failures",
        )
        declared_milestone = _require_bool(
            eval_summary.get("milestone_passed"),
            "evaluation-summary.milestone_passed",
        )
        expected_blockers: list[str] = []
        if hard_failure_total:
            expected_blockers.append("hard_safety_failures")
        if declared_systemic:
            expected_blockers.append("systemic_failures")
        if not declared_milestone:
            expected_blockers.append("milestone_failed")
        if computed_pass_at_3 is None or computed_pass_at_3 < 1.0:
            expected_blockers.append("pass_at_3_below_1_0")
        if computed_pass_at_5 is None or computed_pass_at_5 < 1.0:
            expected_blockers.append("pass_at_5_below_1_0")
        blockers = round_summary.get("promotion_blockers")
        if not isinstance(blockers, list) or any(
            not isinstance(blocker, str) for blocker in blockers
        ):
            raise ValueError("round-summary.promotion_blockers must be a string list")
        if blockers != expected_blockers:
            raise ValueError(
                "round-summary.promotion_blockers do not match validated runs"
            )
        promotion_eligible = _require_bool(
            round_summary.get("promotion_eligible"),
            "round-summary.promotion_eligible",
        )
        if promotion_eligible != (not expected_blockers):
            raise ValueError(
                "round-summary.promotion_eligible does not match validated runs"
            )
    round_evidence_sources = _validate_evidence_sources(
        round_summary.get("evidence_sources")
        if round_schema_version == 2
        else [],
        expected_triplets=expected_evidence_triplets,
        context="round-summary.evidence_sources",
        required=round_schema_version == 2,
    )
    if round_schema_version == 2:
        response_evidence_sources = _load_response_evidence_sources(
            safe_root,
            round_summary.get("evaluation_artifact_refs"),
            expected_triplets=expected_evidence_triplets,
            expected_candidate_fingerprint=candidate_fingerprint,
            expected_root_identity=safe_root_identity,
            expected_runs=round_runs,
        )
        if response_evidence_sources != round_evidence_sources:
            raise ValueError(
                "response provenance does not match round-summary provenance"
            )

    pass_at_3 = _require_finite_float(
        eval_summary.get("pass_at_3"),
        "evaluation-summary.pass_at_3",
    )
    pass_at_5 = _require_finite_float(
        eval_summary.get("pass_at_5"),
        "evaluation-summary.pass_at_5",
    )
    hard_safety_failures = _require_non_negative_int(
        eval_summary.get("hard_safety_failures"),
        "evaluation-summary.hard_safety_failures",
    )
    systemic_failures = _require_non_negative_int(
        eval_summary.get("systemic_failures"),
        "evaluation-summary.systemic_failures",
    )
    milestone_passed = _require_bool(
        eval_summary.get("milestone_passed"),
        "evaluation-summary.milestone_passed",
    )
    expected_milestone_passed = (
        action.kind in (ActionKind.MILESTONE, ActionKind.CONFIRM)
        and hard_safety_failures == 0
        and systemic_failures == 0
    )
    if milestone_passed != expected_milestone_passed:
        raise ValueError(
            "evaluation-summary.milestone_passed does not match action evidence"
        )

    core_regression = False
    metric_calls_used = 0
    search_improved: bool | None = None
    incumbent_score: CampaignScore | None = None
    if action.kind is ActionKind.SEARCH:
        comparison_summary = _load_safe_json(
            safe_root,
            "comparison-summary.json",
            expected_root_identity=safe_root_identity,
        )
        before_evaluation_summary: Mapping[str, Any] | None = None
        if comparison_summary.get("status") == "changed":
            before_evaluation_summary = _load_safe_json(
                safe_root,
                "before-evaluation-summary.json",
                expected_root_identity=safe_root_identity,
            )
            _ensure_exact_keys(
                before_evaluation_summary,
                _EVAL_SUMMARY_REQUIRED_KEYS,
                "before-evaluation-summary",
            )
            if _require_str(
                before_evaluation_summary.get("candidate_fingerprint"),
                "before-evaluation-summary.candidate_fingerprint",
            ) != state.champion_fingerprint:
                raise ValueError(
                    "before-evaluation-summary candidate fingerprint mismatch"
                )
            if _require_str(
                before_evaluation_summary.get("campaign_id"),
                "before-evaluation-summary.campaign_id",
            ) != eval_campaign_id:
                raise ValueError("before-evaluation-summary campaign_id mismatch")
            if tuple(
                sorted(
                    _require_string_list(
                        before_evaluation_summary.get("evaluated_case_ids"),
                        "before-evaluation-summary.evaluated_case_ids",
                    )
                )
            ) != tuple(sorted(expected_case_ids)):
                raise ValueError(
                    "before-evaluation-summary evaluated_case_ids mismatch"
                )
            if _require_string_list(
                before_evaluation_summary.get("evaluated_models"),
                "before-evaluation-summary.evaluated_models",
            ) != [state.model_identity.model]:
                raise ValueError(
                    "before-evaluation-summary evaluated_models mismatch"
                )
            if _require_positive_int(
                before_evaluation_summary.get("repetitions_per_case"),
                "before-evaluation-summary.repetitions_per_case",
            ) != repetitions:
                raise ValueError(
                    "before-evaluation-summary repetitions_per_case mismatch"
                )
            if _require_string_list(
                before_evaluation_summary.get("execution_modes"),
                "before-evaluation-summary.execution_modes",
            ) != ["live"]:
                raise ValueError(
                    "before-evaluation-summary execution_modes must be live"
                )
            incumbent_score = _validate_before_response_metrics(
                safe_root,
                before_evaluation_summary,
                expected_triplets=expected_evidence_triplets,
                expected_candidate_fingerprint=state.champion_fingerprint,
                expected_root_identity=safe_root_identity,
                readonly=control.evaluation_backend == "korvid_readonly",
                expected_evidence_sources=round_evidence_sources,
            )
        core_regression = _validate_comparison_summary(
            comparison_summary,
            state=state,
            round_summary=round_summary,
            eval_summary=eval_summary,
            expected_case_ids=expected_case_ids,
            evaluation_backend=control.evaluation_backend,
            expected_evidence_sources=round_evidence_sources,
            before_evaluation_summary=before_evaluation_summary,
        )
        search_improved = comparison_summary["outcome"] == "improved"
        metric_calls_used = _validate_search_optimization_evidence(
            safe_root,
            action,
            control,
            state,
            safe_root_identity=safe_root_identity,
            round_summary=round_summary,
            eval_summary=eval_summary,
            comparison_summary=comparison_summary,
        )

    if action.kind in (ActionKind.MILESTONE, ActionKind.CONFIRM):
        forbidden = [
            filename
            for filename in (
                "comparison-summary.json",
                "optimization-summary.json",
                "best-candidate.yaml",
            )
            if _forbidden_file_present(safe_root, filename)
        ]
        if forbidden:
            joined = " or ".join(forbidden)
            raise ValueError(
                "milestone/confirm evidence must not contain "
                f"{joined}"
            )

    _validate_artifact_refs(
        round_summary.get("artifact_refs"),
        safe_root,
        context="round-summary.artifact_refs",
    )
    _validate_artifact_refs(
        round_summary.get("evaluation_artifact_refs"),
        safe_root,
        context="round-summary.evaluation_artifact_refs",
    )
    _validate_artifact_refs(
        eval_summary.get("artifact_refs"),
        safe_root,
        context="evaluation-summary.artifact_refs",
    )

    return RoundOutcome(
        candidate_fingerprint=candidate_fingerprint,
        aggregate_score=aggregate_score,
        pass_at_3=pass_at_3,
        pass_at_5=pass_at_5,
        hard_safety_failures=hard_safety_failures,
        systemic_failures=systemic_failures,
        core_regression=core_regression,
        models=tuple(evaluated_models),
        evaluated_case_ids=tuple(evaluated_case_ids),
        action_id=action.action_id,
        milestone_passed=milestone_passed,
        metric_calls_used=metric_calls_used,
        search_improved=search_improved,
        incumbent_score=incumbent_score,
    )

def _validate_search_optimization_evidence(
    safe_root: Path,
    action: CampaignAction,
    control: OptimizationCampaign,
    state: CampaignState,
    *,
    safe_root_identity: tuple[int, int],
    round_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
) -> int:
    """Validate optimization-summary.json and best-candidate.yaml for SEARCH."""
    opt_summary = _load_safe_json(
        safe_root,
        "optimization-summary.json",
        expected_root_identity=safe_root_identity,
    )
    _ensure_exact_keys(opt_summary, _OPTIMIZATION_SUMMARY_REQUIRED_KEYS, "optimization-summary")

    run_identity = _require_mapping(
        opt_summary.get("run_identity"),
        "optimization-summary.run_identity",
    )
    _ensure_exact_keys(run_identity, _RUN_IDENTITY_REQUIRED_KEYS, "run_identity")

    stage = control.stages[action.stage_index]
    expected_seed = stage.seeds[action.seed_index]
    opt_seed = _require_non_negative_int(
        opt_summary.get("seed"),
        "optimization-summary.seed",
    )
    if opt_seed != expected_seed:
        raise ValueError(
            f"optimization-summary.seed mismatch: got {opt_seed}, expected {expected_seed}"
        )

    total_metric_calls = _require_positive_int(
        opt_summary.get("total_metric_calls"),
        "optimization-summary.total_metric_calls",
    )
    maximum_metric_calls = max_search_metric_calls(control, action.metric_calls)
    if total_metric_calls > maximum_metric_calls:
        raise ValueError(
            f"optimization-summary.total_metric_calls ({total_metric_calls}) "
            f"exceeds bounded GEPA maximum ({maximum_metric_calls})"
        )

    seed_fp = _require_str(
        opt_summary.get("seed_candidate_fingerprint"),
        "optimization-summary.seed_candidate_fingerprint",
    )
    expected_seed_fp = _require_str(
        state.champion_fingerprint,
        "state.champion_fingerprint",
    )
    if seed_fp != expected_seed_fp:
        raise ValueError(
            "optimization-summary.seed_candidate_fingerprint mismatch: "
            f"got {seed_fp!r}, expected {expected_seed_fp!r}"
        )

    comparison_seed_fp = _require_str(
        comparison_summary.get("seed_candidate_fingerprint"),
        "comparison-summary.seed_candidate_fingerprint",
    )
    if comparison_seed_fp != seed_fp:
        raise ValueError("comparison-summary.seed_candidate_fingerprint mismatch")

    opt_best_candidate_fingerprint = _require_str(
        opt_summary.get("best_candidate_fingerprint"),
        "optimization-summary.best_candidate_fingerprint",
    )
    best_candidate_differs_from_seed = _require_bool(
        opt_summary.get("best_candidate_differs_from_seed"),
        "optimization-summary.best_candidate_differs_from_seed",
    )

    opt_train = _require_string_list(
        opt_summary.get("train_case_ids"),
        "optimization-summary.train_case_ids",
    )
    opt_val = _require_string_list(
        opt_summary.get("validation_case_ids"),
        "optimization-summary.validation_case_ids",
    )
    if tuple(sorted(opt_train)) != tuple(sorted(control.train_case_ids)):
        raise ValueError("optimization-summary.train_case_ids mismatch with control")
    if tuple(sorted(opt_val)) != tuple(sorted(control.validation_case_ids)):
        raise ValueError("optimization-summary.validation_case_ids mismatch with control")
    num_candidates = _require_positive_int(
        opt_summary.get("num_candidates"),
        "optimization-summary.num_candidates",
    )
    num_full_val_evals = _require_non_negative_int(
        opt_summary.get("num_full_val_evals"),
        "optimization-summary.num_full_val_evals",
    )
    minimum_metric_calls = max(
        num_candidates,
        num_full_val_evals * len(opt_val),
    )
    if total_metric_calls < minimum_metric_calls:
        raise ValueError(
            "optimization-summary.total_metric_calls is below the verifiable "
            f"minimum ({minimum_metric_calls})"
        )

    ri_seed = _require_non_negative_int(run_identity.get("seed"), "run_identity.seed")
    if ri_seed != expected_seed:
        raise ValueError(f"run_identity.seed mismatch: got {ri_seed}, expected {expected_seed}")
    ri_max = _require_positive_int(
        run_identity.get("max_metric_calls"),
        "run_identity.max_metric_calls",
    )
    if ri_max != action.metric_calls:
        raise ValueError(
            f"run_identity.max_metric_calls ({ri_max}) != action.metric_calls ({action.metric_calls})"
        )
    if (
        _require_str(run_identity.get("campaign_id"), "run_identity.campaign_id")
        != control.evaluation_campaign
    ):
        raise ValueError("run_identity.campaign_id mismatch")
    run_identity_candidate_id = _require_str(
        run_identity.get("candidate_id"),
        "run_identity.candidate_id",
    )
    if _require_str(
        run_identity.get("seed_candidate_fingerprint"),
        "run_identity.seed_candidate_fingerprint",
    ) != seed_fp:
        raise ValueError("run_identity.seed_candidate_fingerprint mismatch")
    _require_str(run_identity.get("proposal_source"), "run_identity.proposal_source")
    if _require_positive_int(run_identity.get("schema_version"), "run_identity.schema_version") != 1:
        raise ValueError("run_identity.schema_version must be 1")
    if tuple(sorted(_require_string_list(run_identity.get("train_case_ids"), "run_identity.train_case_ids"))) != tuple(sorted(control.train_case_ids)):
        raise ValueError("run_identity.train_case_ids mismatch")
    if tuple(sorted(_require_string_list(run_identity.get("validation_case_ids"), "run_identity.validation_case_ids"))) != tuple(sorted(control.validation_case_ids)):
        raise ValueError("run_identity.validation_case_ids mismatch")

    bc_data = _load_safe_yaml(
        safe_root,
        "best-candidate.yaml",
        expected_root_identity=safe_root_identity,
    )
    if not isinstance(bc_data, dict):
        raise ValueError("best-candidate.yaml must be a YAML mapping")  # noqa: TRY004
    candidate = Candidate.from_mapping(bc_data)
    computed_fingerprint = candidate.fingerprint

    round_candidate_fingerprint = _require_str(
        round_summary.get("candidate_fingerprint"),
        "round-summary.candidate_fingerprint",
    )
    eval_candidate_fingerprint = _require_str(
        eval_summary.get("candidate_fingerprint"),
        "evaluation-summary.candidate_fingerprint",
    )
    comparison_best_fingerprint = _require_str(
        comparison_summary.get("best_candidate_fingerprint"),
        "comparison-summary.best_candidate_fingerprint",
    )

    for observed in (
        opt_best_candidate_fingerprint,
        round_candidate_fingerprint,
        eval_candidate_fingerprint,
        comparison_best_fingerprint,
    ):
        if observed != computed_fingerprint:
            raise ValueError(
                "best-candidate fingerprint mismatch: "
                f"computed {computed_fingerprint!r}, observed {observed!r}"
            )

    round_candidate_id = _require_str(
        round_summary.get("candidate_id"),
        "round-summary.candidate_id",
    )
    eval_candidate_id = _require_str(
        eval_summary.get("candidate_id"),
        "evaluation-summary.candidate_id",
    )
    if candidate.candidate_id != round_candidate_id:
        raise ValueError("best-candidate candidate_id mismatch with round-summary")
    if candidate.candidate_id != eval_candidate_id:
        raise ValueError("best-candidate candidate_id mismatch with evaluation-summary")
    if candidate.candidate_id != run_identity_candidate_id:
        raise ValueError("best-candidate candidate_id mismatch with run_identity")

    differs_from_seed = computed_fingerprint != seed_fp
    if best_candidate_differs_from_seed != differs_from_seed:
        raise ValueError(
            "optimization-summary.best_candidate_differs_from_seed mismatch with "
            "seed_candidate_fingerprint"
        )
    return maximum_metric_calls


# ---------------------------------------------------------------------------
# Campaign Artifact Rendering
# ---------------------------------------------------------------------------

_STATUS_ICONS: dict[CampaignStatus, str] = {
    CampaignStatus.RUNNING: "🔄",
    CampaignStatus.QUALIFIED: "✅",
    CampaignStatus.NOT_CONVERGED: "❌",
    CampaignStatus.SYSTEM_ERROR: "⚠️",
}


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_campaign_summary(
    state: CampaignState,
    control: OptimizationCampaign,
) -> str:
    """Render a human-readable campaign decision surface as Markdown.

    Uses next_action() to derive the exact next step text.
    """
    from datetime import UTC, datetime

    icon = _STATUS_ICONS.get(state.status, "❓")
    status_upper = state.status.value.upper()
    stages_count = len(control.stages)

    now = datetime.now(tz=UTC)
    planned = next_action(control, state, now)

    if state.status is CampaignStatus.RUNNING and 0 <= state.stage_index < stages_count:
        stage_name = control.stages[state.stage_index].name
        headline = f"## {icon} {status_upper} — {stage_name} stage"
    else:
        headline = f"## {icon} {status_upper}"

    if planned is not None:
        planned_stage_name = control.stages[planned.stage_index].name
        next_line = (
            f"- Next: {planned_stage_name} seed {planned.seed_index} "
            f"with {planned.metric_calls} metric calls"
        )
        if planned.kind is ActionKind.MILESTONE:
            next_line = "- Next: milestone evaluation"
        elif planned.kind is ActionKind.CONFIRM:
            next_line = (
                f"- Next: confirmation run "
                f"{state.confirmations_passed + 1}/{control.confirmation_runs}"
            )
    else:
        next_line = f"- Next: none (terminal: {state.status.value})"

    lines: list[str] = [
        "# Optimization Campaign Outcome",
        "",
        headline,
        "",
        f"- Model: `{state.model_identity.model}` (`{state.model_identity.digest}`)",
        f"- Champion: `{state.champion_fingerprint}`",
        (
            f"- Budget: {state.metric_calls_used} / "
            f"{control.total_metric_call_limit} metric calls; "
            f"{_format_duration(state.elapsed_seconds)} / "
            f"{_format_duration(control.wall_clock_limit_seconds)}"
        ),
        (
            f"- Progress: stage {state.stage_index + 1}/{stages_count}, "
            f"attempt {state.stagnation_attempts + 1}"
        ),
        f"- Milestone: {'passed' if state.milestone_passed else 'not run'}",
        (
            f"- Confirmation: {state.confirmations_passed} passed"
            if state.confirmations_passed > 0
            else "- Confirmation: not run"
        ),
    ]

    if state.status == CampaignStatus.QUALIFIED:
        lines.append("- Publication: ready")
    elif state.status == CampaignStatus.NOT_CONVERGED:
        lines.append(
            f"- Publication: blocked (`{state.stop_reason or 'not_converged'}`)"
        )
    elif state.status == CampaignStatus.SYSTEM_ERROR:
        lines.append(
            f"- Publication: blocked (`{state.stop_reason or 'system_error'}`)"
        )
    else:
        lines.append("- Publication: blocked (`campaign_not_qualified`)")

    lines.append(next_line)
    lines.extend([
        "",
        "## Candidate leaderboard",
        "",
        "| Rank | Candidate | Aggregate | pass@3 | pass@5 | Hard safety | Result |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        (
            f"| 1 | `{state.champion_fingerprint[:10]}...` | "
            f"{state.champion_score.aggregate:.3f} | "
            f"{state.champion_score.pass_at_3:.3f} | "
            f"{state.champion_score.pass_at_5:.3f} | "
            f"{state.champion_score.hard_safety_failures} | champion |"
        ),
        "",
        "## Failure movement",
        "",
        "No failure data available",
    ])

    return "\n".join(lines)


def write_campaign_artifacts(
    state: CampaignState,
    output_root: Path,
    control: OptimizationCampaign,
) -> Path:
    """Write campaign decision artifacts to output_root.

    Raises FileExistsError if output_root already exists.
    Atomic: cleans up on failure.
    """
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.mkdir(parents=True)

    md_path = output_root / "campaign-summary.md"
    state_path = output_root / "campaign-state.json"
    try:
        markdown = render_campaign_summary(state, control)
        md_path.write_text(markdown, encoding="utf-8")

        state_data = _serialize_state(state)
        state_path.write_text(
            json.dumps(state_data, indent=2), encoding="utf-8"
        )
    except BaseException:
        # Clean up partial output
        for f in (md_path, state_path):
            if f.exists():
                f.unlink()
        if output_root.exists():
            try:
                output_root.rmdir()
            except OSError:
                pass
        raise

    return output_root


def _serialize_state(state: CampaignState) -> dict[str, Any]:
    """Serialize CampaignState to a JSON-safe dict."""
    return {
        "schema_version": state.schema_version,
        "campaign_id": state.campaign_id,
        "prompt_lab_revision": state.prompt_lab_revision,
        "korvid_revision": state.korvid_revision,
        "status": state.status.value,
        "tier_index": state.tier_index,
        "stage_index": state.stage_index,
        "seed_index": state.seed_index,
        "champion_fingerprint": state.champion_fingerprint,
        "seed_candidate_fingerprint": state.seed_candidate_fingerprint,
        "champion_score": {
            "fingerprint": state.champion_score.fingerprint,
            "aggregate": state.champion_score.aggregate,
            "hard_safety_failures": state.champion_score.hard_safety_failures,
            "core_regression": state.champion_score.core_regression,
            "systemic_failures": state.champion_score.systemic_failures,
            "pass_at_3": state.champion_score.pass_at_3,
            "pass_at_5": state.champion_score.pass_at_5,
        },
        "model_identity": {
            "name": state.model_identity.name,
            "model": state.model_identity.model,
            "digest": state.model_identity.digest,
        },
        "metric_calls_used": state.metric_calls_used,
        "elapsed_seconds": state.elapsed_seconds,
        "stagnation_attempts": state.stagnation_attempts,
        "retries_used": state.retries_used,
        "started_at": state.started_at,
        "pending_action_id": state.pending_action_id,
        "milestone_passed": state.milestone_passed,
        "confirmations_passed": state.confirmations_passed,
        "stop_reason": state.stop_reason,
        "tier_results": [
            {
                "tier_index": tr.tier_index,
                "champion_fingerprint": tr.champion_fingerprint,
                "status": tr.status.value,
                "champion_score": {
                    "fingerprint": tr.champion_score.fingerprint,
                    "aggregate": tr.champion_score.aggregate,
                    "hard_safety_failures": tr.champion_score.hard_safety_failures,
                    "core_regression": tr.champion_score.core_regression,
                    "systemic_failures": tr.champion_score.systemic_failures,
                    "pass_at_3": tr.champion_score.pass_at_3,
                    "pass_at_5": tr.champion_score.pass_at_5,
                },
            }
            for tr in state.tier_results
        ],
        "state_hash": state_hash(state),
    }


# ---------------------------------------------------------------------------
# Compare-and-Swap State Persistence
# ---------------------------------------------------------------------------


def _fsync_directory(directory: Path) -> None:
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_campaign_state(
    state: CampaignState,
    path: Path,
    *,
    expected_prior_hash: str,
    state_root: Path | None = None,
) -> None:
    """Atomically write campaign state with cross-process CAS semantics."""
    path = Path(path).resolve()
    allowed_root = Path(state_root).resolve() if state_root else path.parent
    try:
        path.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"state path escapes allowed root: {path}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    try:
        lock_path.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"lock path escapes allowed root: {lock_path}") from exc

    data = _serialize_state(state)
    payload = json.dumps(data, indent=2).encode("utf-8")
    temp_path: Path | None = None

    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"existing state file is malformed: {path}") from exc
            if not isinstance(existing, dict):
                raise ValueError(f"existing state file must be a JSON object: {path}")
            existing_hash = _require_str(existing.get("state_hash"), "existing state_hash")
            if existing_hash != expected_prior_hash:
                raise ValueError(
                    f"stale state: expected prior hash {expected_prior_hash}, "
                    f"got {existing_hash}"
                )

        temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        temp_fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(temp_fd, "wb") as temp_file:
                temp_file.write(payload)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, path)
            _fsync_directory(path.parent)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    finally:
        os.close(lock_fd)
