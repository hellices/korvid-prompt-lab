"""Safe campaign evidence ingestion and artifact rendering.

This module forms the trust boundary between Grounding safe-evidence packages
and the pure campaign state machine. It reads only allowlisted top-level
summary files, never traverses symlinks or touches responses/, raw answers,
requests, audit journals, optimizer state, or reflection transcripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .campaigns import (
    CampaignAction,
    CampaignState,
    CampaignStatus,
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


@dataclass(frozen=True, slots=True)
class RoundOutcome:
    """Validated outcome extracted from a safe-evidence package."""

    candidate_fingerprint: str
    aggregate_score: float
    pass_at_3: float | None
    pass_at_5: float | None
    hard_safety_failures: int
    systemic_failures: int
    core_regression: bool
    models: tuple[str, ...]
    evaluated_case_ids: tuple[str, ...]
    action_id: str


def _reject_symlink(path: Path) -> None:
    """Raise if path or any component is a symlink."""
    if path.is_symlink():
        raise ValueError(f"symlink detected: {path}")
    # Check parents up to root
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError(f"symlink detected in path component: {parent}")


def _load_safe_json(root: Path, filename: str) -> dict[str, Any]:
    """Load JSON from a safe-evidence package, rejecting symlinks."""
    if filename not in _ALLOWED_FILES:
        raise ValueError(f"file not in allowlist: {filename}")
    path = root / filename
    _reject_symlink(path)
    if not path.is_file():
        raise ValueError(f"required file missing: {filename}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"malformed JSON in {filename}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{filename} must be a JSON object")  # noqa: TRY004
    return data


def load_round_outcome(
    safe_root: Path,
    action: CampaignAction,
    *,
    expected_case_ids: tuple[str, ...] | None = None,
    expected_model: str | None = None,
) -> RoundOutcome:
    """Load and validate a round outcome from a safe-evidence package.

    Validates action_id binding, evaluated case set, and model identity.
    Never traverses symlinks or reads responses/.
    """
    safe_root = Path(safe_root)
    _reject_symlink(safe_root)

    round_summary = _load_safe_json(safe_root, "round-summary.json")
    eval_summary = _load_safe_json(safe_root, "evaluation-summary.json")

    # Validate action_id binding
    evidence_action_id = round_summary.get("action_id")
    if evidence_action_id is not None and evidence_action_id != action.action_id:
        raise ValueError(
            f"action_id mismatch: evidence has {evidence_action_id!r}, "
            f"expected {action.action_id!r}"
        )

    # Validate evaluated case set
    evaluated_case_ids = eval_summary.get("evaluated_case_ids", [])
    if not isinstance(evaluated_case_ids, list):
        raise ValueError("evaluated_case_ids must be a list")  # noqa: TRY004
    if expected_case_ids is not None and tuple(sorted(evaluated_case_ids)) != tuple(sorted(expected_case_ids)):
        raise ValueError(
            f"evaluated case set mismatch: evidence has "
            f"{sorted(evaluated_case_ids)}, expected {sorted(expected_case_ids)}"
        )

    # Validate model
    evaluated_models = eval_summary.get("evaluated_models", [])
    if expected_model is not None and (not isinstance(evaluated_models, list) or expected_model not in evaluated_models):
        raise ValueError(
            f"model mismatch: evidence evaluated "
            f"{evaluated_models}, expected {expected_model!r}"
        )

    candidate_fingerprint = round_summary.get("candidate_fingerprint", "")
    aggregate_score = round_summary.get("aggregate_score", 0.0)
    pass_at_3 = round_summary.get("pass_at_3")
    pass_at_5 = round_summary.get("pass_at_5")
    hard_safety_failures = eval_summary.get("hard_safety_failures", 0)
    systemic_failures = eval_summary.get("systemic_failures", 0)

    return RoundOutcome(
        candidate_fingerprint=candidate_fingerprint,
        aggregate_score=float(aggregate_score),
        pass_at_3=pass_at_3,
        pass_at_5=pass_at_5,
        hard_safety_failures=int(hard_safety_failures),
        systemic_failures=int(systemic_failures),
        core_regression=False,
        models=tuple(evaluated_models) if isinstance(evaluated_models, list) else (),
        evaluated_case_ids=tuple(evaluated_case_ids),
        action_id=action.action_id,
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
    *,
    total_metric_call_limit: int,
    wall_clock_limit_seconds: int,
    stages_count: int,
) -> str:
    """Render a human-readable campaign decision surface as Markdown."""
    icon = _STATUS_ICONS.get(state.status, "❓")
    status_upper = state.status.value.upper()

    lines: list[str] = [
        "# Optimization Campaign Outcome",
        "",
        f"## {icon} {status_upper}",
        "",
        f"- Model: `{state.model_identity.model}` (`{state.model_identity.digest}`)",
        f"- Champion: `{state.champion_fingerprint}`",
        (
            f"- Budget: {state.metric_calls_used} / {total_metric_call_limit} metric calls; "
            f"{_format_duration(state.elapsed_seconds)} / {_format_duration(wall_clock_limit_seconds)}"
        ),
        f"- Progress: stage {state.stage_index + 1}/{stages_count}, attempt {state.stagnation_attempts + 1}",
        f"- Milestone: {'passed' if state.milestone_passed else 'not run'}",
        f"- Confirmation: {state.confirmations_passed} passed" if state.confirmations_passed > 0 else "- Confirmation: not run",
    ]

    # Publication status
    if state.status == CampaignStatus.QUALIFIED:
        lines.append("- Publication: ready")
    elif state.status == CampaignStatus.NOT_CONVERGED:
        lines.append(f"- Publication: blocked (`{state.stop_reason or 'not_converged'}`)")
    elif state.status == CampaignStatus.SYSTEM_ERROR:
        lines.append(f"- Publication: blocked (`{state.stop_reason or 'system_error'}`)")
    else:
        lines.append("- Publication: blocked (`campaign_not_qualified`)")

    # Next action hint
    if state.status == CampaignStatus.RUNNING:
        lines.append(f"- Next: explore seed {state.seed_index} with {state.metric_calls_used or 12} metric calls")
    else:
        lines.append(f"- Next: none (terminal: {state.status.value})")

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
    *,
    total_metric_call_limit: int,
    wall_clock_limit_seconds: int,
    stages_count: int,
) -> Path:
    """Write campaign decision artifacts to output_root.

    Raises FileExistsError if output_root already exists.
    """
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.mkdir(parents=True)

    # Write campaign-summary.md
    markdown = render_campaign_summary(
        state,
        total_metric_call_limit=total_metric_call_limit,
        wall_clock_limit_seconds=wall_clock_limit_seconds,
        stages_count=stages_count,
    )
    (output_root / "campaign-summary.md").write_text(markdown, encoding="utf-8")

    # Write campaign-state.json
    state_data = _serialize_state(state)
    (output_root / "campaign-state.json").write_text(
        json.dumps(state_data, indent=2), encoding="utf-8"
    )

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
    expected_prior_hash: str | None,
) -> None:
    """Atomically write campaign state with CAS semantics.

    If expected_prior_hash is not None, the file must exist and its stored
    state_hash must match exactly; otherwise the write is rejected as stale.
    """
    path = Path(path)

    if expected_prior_hash is not None:
        if not path.is_file():
            raise ValueError("expected prior state file does not exist")
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_hash = existing.get("state_hash", "")
        if existing_hash != expected_prior_hash:
            raise ValueError(
                f"stale state: expected hash {expected_prior_hash}, "
                f"got {existing_hash}"
            )

    data = _serialize_state(state)
    # Atomic write: write to temp then rename
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(path)
