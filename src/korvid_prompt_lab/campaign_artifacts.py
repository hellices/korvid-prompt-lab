"""Safe campaign evidence ingestion and artifact rendering.

This module forms the trust boundary between Grounding safe-evidence packages
and the pure campaign state machine. It reads only allowlisted top-level
summary files, never traverses symlinks or touches responses/, raw answers,
requests, audit journals, optimizer state, or reflection transcripts.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml  # type: ignore[import-untyped]

from .campaigns import (
    ActionKind,
    CampaignAction,
    CampaignState,
    CampaignStatus,
    OptimizationCampaign,
    next_action,
    state_hash,
)
from .contracts import Candidate

#: Files the safe ingestion layer is allowed to read from a round evidence package.
_ALLOWED_FILES = frozenset({
    "round-summary.json",
    "evaluation-summary.json",
    "comparison-summary.json",
    "optimization-summary.json",
    "best-candidate.yaml",
})

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

_ROUND_SUMMARY_REQUIRED_KEYS = frozenset({
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

_COMPARISON_CONTRACT_REQUIRED_KEYS = frozenset({
    "campaign_id",
    "models",
    "case_repetitions",
    "execution_modes",
})

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
) -> None:
    _ensure_exact_keys(
        comparison_summary,
        _COMPARISON_SUMMARY_REQUIRED_KEYS,
        "comparison-summary",
    )
    if _require_positive_int(
        comparison_summary.get("schema_version"),
        "comparison-summary.schema_version",
    ) != 1:
        raise ValueError("comparison-summary.schema_version must be 1")

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

    contract = _require_mapping(comparison_summary.get("contract"), "comparison-summary.contract")
    _ensure_exact_keys(contract, _COMPARISON_CONTRACT_REQUIRED_KEYS, "comparison-summary.contract")
    contract_campaign_id = _require_str(
        contract.get("campaign_id"),
        "comparison-summary.contract.campaign_id",
    )
    expected_campaign_id = _require_str(
        round_summary.get("campaign_id"),
        "round-summary.campaign_id",
    )
    if contract_campaign_id != expected_campaign_id or contract_campaign_id != state.campaign_id:
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
    comparison_case_ids: list[str] = []
    expected_repetitions = _require_positive_int(
        eval_summary.get("repetitions_per_case"),
        "evaluation-summary.repetitions_per_case",
    )
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
        if model != state.model_identity.model:
            raise ValueError(
                f"comparison-summary.contract.case_repetitions[{index}][1] model mismatch"
            )
        repetition = _require_positive_int(
            entry[2], f"comparison-summary.contract.case_repetitions[{index}][2]",
        )
        if repetition != expected_repetitions:
            raise ValueError("comparison-summary.contract.case_repetitions repetition mismatch")
        comparison_case_ids.append(case_id)
    if tuple(sorted(comparison_case_ids)) != tuple(sorted(expected_case_ids)):
        raise ValueError("comparison-summary.contract.case_repetitions case set mismatch")

    execution_modes = _require_string_list(
        contract.get("execution_modes"),
        "comparison-summary.contract.execution_modes",
    )
    if execution_modes != ["live"]:
        raise ValueError(
            "comparison-summary.contract.execution_modes must contain exactly 'live'"
        )

    metrics = comparison_summary.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError("comparison-summary.metrics must be a list")  # noqa: TRY004
    for index, metric in enumerate(metrics):
        metric_mapping = _require_mapping(metric, f"comparison-summary.metrics[{index}]")
        _ensure_exact_keys(
            metric_mapping,
            _COMPARISON_METRIC_REQUIRED_KEYS,
            f"comparison-summary.metrics[{index}]",
        )
        _require_str(metric_mapping.get("key"), f"comparison-summary.metrics[{index}].key")
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
        _require_bool(metric_mapping.get("integer"), f"comparison-summary.metrics[{index}].integer")
        _require_bool(metric_mapping.get("core"), f"comparison-summary.metrics[{index}].core")

    for key in ("improved_count", "unchanged_count", "regressed_count", "not_comparable_count"):
        _require_non_negative_int(comparison_summary.get(key), f"comparison-summary.{key}")


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


def _load_safe_json(root: Path, filename: str) -> dict[str, Any]:
    """Load JSON from a safe-evidence package, rejecting symlinks."""
    if filename not in _ALLOWED_FILES:
        raise ValueError(f"file not in allowlist: {filename}")
    path = _resolve_safe_path(root, filename)
    if not path.is_file():
        raise ValueError(f"required file missing: {filename}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"malformed JSON in {filename}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{filename} must be a JSON object")  # noqa: TRY004
    return data


def _load_safe_yaml(root: Path, filename: str) -> Any:
    """Load YAML from a safe-evidence package, rejecting symlinks."""
    if filename not in _ALLOWED_FILES:
        raise ValueError(f"file not in allowlist: {filename}")
    path = _resolve_safe_path(root, filename)
    if not path.is_file():
        raise ValueError(f"required file missing: {filename}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
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

    round_summary = _load_safe_json(safe_root, "round-summary.json")
    eval_summary = _load_safe_json(safe_root, "evaluation-summary.json")
    _ensure_exact_keys(round_summary, _ROUND_SUMMARY_REQUIRED_KEYS, "round-summary")
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
    expected_campaign_id = _require_str(state.campaign_id, "state.campaign_id")
    if round_campaign_id != eval_campaign_id:
        raise ValueError(
            f"campaign_id mismatch: round-summary has {round_campaign_id!r}, "
            f"evaluation-summary has {eval_campaign_id!r}"
        )
    if round_campaign_id != control.campaign_id or round_campaign_id != expected_campaign_id:
        raise ValueError(
            f"campaign_id mismatch: evidence has {round_campaign_id!r}, "
            f"expected {control.campaign_id!r}"
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

    if action.kind is ActionKind.SEARCH:
        comparison_summary = _load_safe_json(safe_root, "comparison-summary.json")
        _validate_comparison_summary(
            comparison_summary,
            state=state,
            round_summary=round_summary,
            eval_summary=eval_summary,
            expected_case_ids=expected_case_ids,
        )
        _validate_search_optimization_evidence(
            safe_root,
            action,
            control,
            state,
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
        core_regression=False,
        models=tuple(evaluated_models),
        evaluated_case_ids=tuple(evaluated_case_ids),
        action_id=action.action_id,
        milestone_passed=milestone_passed,
    )

def _validate_search_optimization_evidence(
    safe_root: Path,
    action: CampaignAction,
    control: OptimizationCampaign,
    state: CampaignState,
    *,
    round_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
) -> None:
    """Validate optimization-summary.json and best-candidate.yaml for SEARCH."""
    opt_summary = _load_safe_json(safe_root, "optimization-summary.json")
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
    if total_metric_calls > action.metric_calls:
        raise ValueError(
            f"optimization-summary.total_metric_calls ({total_metric_calls}) "
            f"exceeds action budget ({action.metric_calls})"
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
    if _require_str(run_identity.get("campaign_id"), "run_identity.campaign_id") != control.campaign_id:
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

    bc_data = _load_safe_yaml(safe_root, "best-candidate.yaml")
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
