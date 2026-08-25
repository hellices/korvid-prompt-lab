"""Safe campaign evidence ingestion and artifact rendering.

This module forms the trust boundary between Grounding safe-evidence packages
and the pure campaign state machine. It reads only allowlisted top-level
summary files, never traverses symlinks or touches responses/, raw answers,
requests, audit journals, optimizer state, or reflection transcripts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
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
    For SEARCH actions, also validates optimization-summary.json and
    best-candidate.yaml. For MILESTONE/CONFIRM, rejects optimization files.
    Never traverses symlinks or reads responses/.
    """
    safe_root = Path(safe_root).resolve()
    _reject_symlink(safe_root)
    if not safe_root.is_dir():
        raise ValueError(f"evidence root is not a directory: {safe_root}")

    round_summary = _load_safe_json(safe_root, "round-summary.json")
    eval_summary = _load_safe_json(safe_root, "evaluation-summary.json")

    # --- action_id binding (required, strict) ---
    evidence_action_id = round_summary.get("action_id")
    _require_str(evidence_action_id, "round-summary.action_id")
    if evidence_action_id != action.action_id:
        raise ValueError(
            f"action_id mismatch: evidence has {evidence_action_id!r}, "
            f"expected {action.action_id!r}"
        )

    # --- candidate fingerprint ---
    candidate_fingerprint = _require_str(
        round_summary.get("candidate_fingerprint"),
        "round-summary.candidate_fingerprint",
    )

    # --- model identity ---
    evaluated_models = _require_string_list(
        eval_summary.get("evaluated_models"),
        "evaluation-summary.evaluated_models",
    )
    expected_model = state.model_identity.model
    if expected_model not in evaluated_models:
        raise ValueError(
            f"model mismatch: evidence evaluated {evaluated_models}, "
            f"expected {expected_model!r}"
        )

    # --- evaluated case set ---
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
            f"{sorted(evaluated_case_ids)}, "
            f"expected {sorted(expected_case_ids)}"
        )

    # --- MILESTONE/CONFIRM: champion fingerprint binding ---
    if action.kind in (ActionKind.MILESTONE, ActionKind.CONFIRM) and candidate_fingerprint != state.champion_fingerprint:
        raise ValueError(
            f"milestone/confirm candidate must match champion: "
            f"got {candidate_fingerprint!r}, "
            f"expected {state.champion_fingerprint!r}"
        )

    # --- revisions ---
    evidence_pl_rev = round_summary.get("prompt_lab_revision")
    if evidence_pl_rev is not None:
        _require_str(evidence_pl_rev, "round-summary.prompt_lab_revision")
        if evidence_pl_rev != state.prompt_lab_revision:
            raise ValueError(
                f"prompt_lab_revision mismatch: {evidence_pl_rev!r} "
                f"vs {state.prompt_lab_revision!r}"
            )
    evidence_k_rev = round_summary.get("korvid_revision")
    if evidence_k_rev is not None:
        _require_str(evidence_k_rev, "round-summary.korvid_revision")
        if evidence_k_rev != state.korvid_revision:
            raise ValueError(
                f"korvid_revision mismatch: {evidence_k_rev!r} "
                f"vs {state.korvid_revision!r}"
            )

    # --- execution mode ---
    execution_modes = eval_summary.get("execution_modes")
    if execution_modes is not None:
        _require_string_list(execution_modes, "evaluation-summary.execution_modes")
        if "live" not in execution_modes:
            raise ValueError("execution_modes must include 'live'")

    # --- repetitions ---
    repetitions = eval_summary.get("repetitions_per_case")
    if repetitions is not None:
        _require_positive_int(repetitions, "evaluation-summary.repetitions_per_case")

    # --- scores (strict, no coercion) ---
    aggregate_score = _require_finite_float(
        round_summary.get("aggregate_score"),
        "round-summary.aggregate_score",
    )
    pass_at_3 = _require_finite_float(
        eval_summary.get("pass_at_3") if eval_summary.get("pass_at_3") is not None else round_summary.get("pass_at_3", 0.0),
        "pass_at_3",
    )
    pass_at_5 = _require_finite_float(
        eval_summary.get("pass_at_5") if eval_summary.get("pass_at_5") is not None else round_summary.get("pass_at_5", 0.0),
        "pass_at_5",
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

    # --- SEARCH: require and validate optimization-summary.json + best-candidate.yaml ---
    if action.kind is ActionKind.SEARCH:
        _validate_search_optimization_evidence(safe_root, action, control, state)

    # --- MILESTONE/CONFIRM: reject optimization files ---
    if action.kind in (ActionKind.MILESTONE, ActionKind.CONFIRM):
        opt_path = safe_root / "optimization-summary.json"
        bc_path = safe_root / "best-candidate.yaml"
        if opt_path.exists() or bc_path.exists():
            raise ValueError(
                "milestone/confirm evidence must not contain "
                "optimization-summary.json or best-candidate.yaml"
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
) -> None:
    """Validate optimization-summary.json and best-candidate.yaml for SEARCH."""
    opt_summary = _load_safe_json(safe_root, "optimization-summary.json")
    _ensure_exact_keys(opt_summary, _OPTIMIZATION_SUMMARY_REQUIRED_KEYS, "optimization-summary")

    # Validate run_identity
    run_identity = _require_mapping(
        opt_summary.get("run_identity"), "optimization-summary.run_identity",
    )
    _ensure_exact_keys(run_identity, _RUN_IDENTITY_REQUIRED_KEYS, "run_identity")

    # Seed must match action
    stage = control.stages[action.stage_index]
    expected_seed = stage.seeds[action.seed_index]
    opt_seed = _require_non_negative_int(opt_summary.get("seed"), "optimization-summary.seed")
    if opt_seed != expected_seed:
        raise ValueError(
            f"optimization-summary.seed mismatch: got {opt_seed}, expected {expected_seed}"
        )

    # Metric calls within budget
    total_metric_calls = _require_positive_int(
        opt_summary.get("total_metric_calls"), "optimization-summary.total_metric_calls",
    )
    if total_metric_calls > action.metric_calls:
        raise ValueError(
            f"optimization-summary.total_metric_calls ({total_metric_calls}) "
            f"exceeds action budget ({action.metric_calls})"
        )

    # Seed fingerprint must match champion
    seed_fp = _require_str(
        opt_summary.get("seed_candidate_fingerprint"),
        "optimization-summary.seed_candidate_fingerprint",
    )
    if seed_fp != state.champion_fingerprint:
        raise ValueError(
            f"optimization-summary.seed_candidate_fingerprint mismatch: "
            f"got {seed_fp!r}, expected {state.champion_fingerprint!r}"
        )

    # best_candidate_fingerprint
    _require_str(
        opt_summary.get("best_candidate_fingerprint"),
        "optimization-summary.best_candidate_fingerprint",
    )
    _require_bool(
        opt_summary.get("best_candidate_differs_from_seed"),
        "optimization-summary.best_candidate_differs_from_seed",
    )

    # Validate train/validation case IDs match control
    opt_train = _require_string_list(
        opt_summary.get("train_case_ids"), "optimization-summary.train_case_ids",
    )
    opt_val = _require_string_list(
        opt_summary.get("validation_case_ids"), "optimization-summary.validation_case_ids",
    )
    if tuple(sorted(opt_train)) != tuple(sorted(control.train_case_ids)):
        raise ValueError("optimization-summary.train_case_ids mismatch with control")
    if tuple(sorted(opt_val)) != tuple(sorted(control.validation_case_ids)):
        raise ValueError("optimization-summary.validation_case_ids mismatch with control")

    # run_identity checks
    ri_seed = _require_non_negative_int(run_identity.get("seed"), "run_identity.seed")
    if ri_seed != expected_seed:
        raise ValueError(f"run_identity.seed mismatch: got {ri_seed}, expected {expected_seed}")
    ri_max = _require_positive_int(run_identity.get("max_metric_calls"), "run_identity.max_metric_calls")
    if ri_max > action.metric_calls:
        raise ValueError("run_identity.max_metric_calls exceeds action budget")
    _require_str(run_identity.get("campaign_id"), "run_identity.campaign_id")
    _require_str(run_identity.get("candidate_id"), "run_identity.candidate_id")
    _require_str(run_identity.get("seed_candidate_fingerprint"), "run_identity.seed_candidate_fingerprint")
    _require_str(run_identity.get("proposal_source"), "run_identity.proposal_source")
    _require_positive_int(run_identity.get("schema_version"), "run_identity.schema_version")
    _require_string_list(run_identity.get("train_case_ids"), "run_identity.train_case_ids")
    _require_string_list(run_identity.get("validation_case_ids"), "run_identity.validation_case_ids")

    # best-candidate.yaml must exist
    bc_data = _load_safe_yaml(safe_root, "best-candidate.yaml")
    if not isinstance(bc_data, dict):
        raise ValueError("best-candidate.yaml must be a YAML mapping")  # noqa: TRY004


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

    # Derive next action for exact rendering
    now = datetime.now(tz=UTC)
    planned = next_action(control, state, now)

    # Build next line from planned action
    if planned is not None:
        stage_name = control.stages[planned.stage_index].name
        next_line = (
            f"- Next: {stage_name} seed {planned.seed_index} "
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
        f"## {icon} {status_upper}",
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

    # Publication status
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
    lines.append("")

    # Champion score table
    lines.extend([
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


def write_campaign_state(
    state: CampaignState,
    path: Path,
    *,
    expected_prior_hash: str,
    state_root: Path | None = None,
) -> None:
    """Atomically write campaign state with CAS semantics.

    expected_prior_hash is required. For initial writes where no prior state
    exists, pass the hash of the initial state being written.

    The path must stay within state_root (if provided) or its own parent.
    Cleans up temp files on any failure.
    """
    path = Path(path).resolve()

    # Validate path stays within allowed root
    allowed_root = Path(state_root).resolve() if state_root else path.parent
    try:
        path.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            f"state path escapes allowed root: {path}"
        ) from exc

    # CAS check: compare expected_prior_hash to existing file
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_hash = existing.get("state_hash", "")
        if existing_hash != expected_prior_hash:
            raise ValueError(
                f"stale state: expected prior hash {expected_prior_hash}, "
                f"got {existing_hash}"
            )

    data = _serialize_state(state)
    tmp_path = path.with_suffix(".cas_tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
