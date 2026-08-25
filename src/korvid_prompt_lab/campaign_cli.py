"""Campaign CLI: plan, advance, render subcommands.

The CLI writes GitHub outputs only to a caller-provided --github-output file;
it never trusts an environment-provided output path implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .campaign_artifacts import (
    load_round_outcome,
    write_campaign_artifacts,
    write_campaign_state,
)
from .campaigns import (
    ActionKind,
    AttemptOutcome,
    CampaignAction,
    CampaignScore,
    CampaignState,
    CampaignStatus,
    ModelIdentity,
    OptimizationCampaign,
    TierResult,
    advance_state,
    load_optimization_campaign,
    next_action,
    state_hash,
    validate_seed_candidate_fingerprint,
    validate_state_binding,
)
from .config import load_campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="korvid-campaign")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_binding_arguments(target: argparse.ArgumentParser) -> None:
        """Optional identity bindings enforced before any planning decision."""
        target.add_argument("--expected-manifest-sha256", type=str, default=None)
        target.add_argument("--expected-prompt-lab-revision", type=str, default=None)
        target.add_argument("--expected-korvid-revision", type=str, default=None)
        target.add_argument("--expected-seed-fingerprint", type=str, default=None)

    # plan
    plan_p = sub.add_parser("plan", help="Emit next action as JSON.")
    plan_p.add_argument("--control", type=Path, required=True)
    plan_p.add_argument("--state", type=Path, required=True)
    plan_p.add_argument("--output", type=Path, required=True)
    plan_p.add_argument("--github-output", type=Path, default=None)
    _add_binding_arguments(plan_p)

    # advance
    adv_p = sub.add_parser("advance", help="Apply evidence to campaign state.")
    adv_p.add_argument("--control", type=Path, required=True)
    adv_p.add_argument("--state", type=Path, required=True)
    adv_p.add_argument("--action", type=Path, required=True)
    adv_p.add_argument("--evidence", type=Path)
    adv_p.add_argument(
        "--outcome-kind",
        choices=("evidence", "system_error", "config_error"),
        default="evidence",
    )
    adv_p.add_argument("--error-message", type=str)
    adv_p.add_argument("--output-state", type=Path, required=True)
    adv_p.add_argument("--expected-prior-hash", type=str, required=True)
    adv_p.add_argument("--github-output", type=Path, default=None)
    _add_binding_arguments(adv_p)

    # validate-evidence (read-only preflight for unambiguous wrapper fallback)
    val_p = sub.add_parser(
        "validate-evidence",
        help="Validate evidence and the resulting pure transition without persistence.",
    )
    val_p.add_argument("--control", type=Path, required=True)
    val_p.add_argument("--state", type=Path, required=True)
    val_p.add_argument("--action", type=Path, required=True)
    val_p.add_argument("--evidence", type=Path, required=True)
    val_p.add_argument("--expected-prior-hash", type=str, required=True)
    _add_binding_arguments(val_p)

    # render
    rend_p = sub.add_parser("render", help="Render campaign summary.")
    rend_p.add_argument("--control", type=Path, required=True)
    rend_p.add_argument("--state", type=Path, required=True)
    rend_p.add_argument("--output-dir", type=Path, required=True)
    rend_p.add_argument("--github-output", type=Path, default=None)

    return parser


def _load_state(path: Path) -> CampaignState:
    """Load campaign state from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    score_data = data["champion_score"]
    mi_data = data["model_identity"]
    tier_results_raw = data.get("tier_results", [])
    tier_results = tuple(
        TierResult(
            tier_index=tr["tier_index"],
            champion_fingerprint=tr["champion_fingerprint"],
            champion_score=CampaignScore(
                fingerprint=tr["champion_score"]["fingerprint"],
                aggregate=tr["champion_score"]["aggregate"],
                hard_safety_failures=tr["champion_score"]["hard_safety_failures"],
                core_regression=tr["champion_score"]["core_regression"],
                systemic_failures=tr["champion_score"]["systemic_failures"],
                pass_at_3=tr["champion_score"].get("pass_at_3", 1.0),
                pass_at_5=tr["champion_score"].get("pass_at_5", 1.0),
            ),
            status=CampaignStatus(tr["status"]),
        )
        for tr in tier_results_raw
    )
    return CampaignState(
        schema_version=data["schema_version"],
        campaign_id=data["campaign_id"],
        prompt_lab_revision=data["prompt_lab_revision"],
        korvid_revision=data["korvid_revision"],
        status=CampaignStatus(data["status"]),
        tier_index=data["tier_index"],
        stage_index=data["stage_index"],
        seed_index=data["seed_index"],
        champion_fingerprint=data["champion_fingerprint"],
        seed_candidate_fingerprint=data["seed_candidate_fingerprint"],
        champion_score=CampaignScore(
            fingerprint=score_data["fingerprint"],
            aggregate=score_data["aggregate"],
            hard_safety_failures=score_data["hard_safety_failures"],
            core_regression=score_data["core_regression"],
            systemic_failures=score_data["systemic_failures"],
            pass_at_3=score_data.get("pass_at_3", 1.0),
            pass_at_5=score_data.get("pass_at_5", 1.0),
        ),
        model_identity=ModelIdentity(
            name=mi_data["name"],
            model=mi_data["model"],
            digest=mi_data["digest"],
        ),
        metric_calls_used=data["metric_calls_used"],
        elapsed_seconds=data["elapsed_seconds"],
        stagnation_attempts=data["stagnation_attempts"],
        retries_used=data["retries_used"],
        started_at=data["started_at"],
        pending_action_id=data.get("pending_action_id"),
        milestone_passed=data.get("milestone_passed", False),
        confirmations_passed=data.get("confirmations_passed", 0),
        stop_reason=data.get("stop_reason"),
        tier_results=tier_results,
    )


_EVALUATION_CAMPAIGN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


def _safe_existing_file(path: Path, label: str) -> Path:
    """Resolve an existing regular file, rejecting symlinks anywhere in it."""
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=False)
    probe = resolved
    while True:
        if probe.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {probe}")
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {candidate}")
    return resolved


def _resolve_evaluation_campaign_path(control_path: Path, campaign_id: str) -> Path:
    """Locate the evaluation campaign declared by a control manifest.

    The identifier is a bare campaign id, never a path: traversal, separators
    and unknown shapes are rejected before any filesystem lookup.
    """
    if not _EVALUATION_CAMPAIGN_ID_RE.match(campaign_id):
        raise ValueError(
            f"evaluation_campaign must be a bare campaign id, got {campaign_id!r}"
        )
    directory = control_path.parent
    candidates = [directory / f"{campaign_id}.yaml"]
    candidates.extend(
        ancestor / "examples" / "campaigns" / f"{campaign_id}.yaml"
        for ancestor in [directory, *directory.parents]
    )
    for candidate in candidates:
        if candidate.exists():
            return _safe_existing_file(candidate, "evaluation campaign")
    raise ValueError(
        f"evaluation campaign {campaign_id!r} was not found next to {control_path}"
    )


def _load_control(path: Path) -> OptimizationCampaign:
    """Load and fully validate an optimization campaign control manifest.

    Uses the strict loader together with the evaluation campaign the manifest
    declares, so disjoint/covering case sets, canonical digests, positive
    limits and unique seeds are all enforced here rather than only in the
    workflow wrapper.
    """
    control_path = _safe_existing_file(path, "control manifest")
    data = yaml.safe_load(control_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("optimization campaign must be a mapping")  # noqa: TRY004 - preserve validation API
    raw_evaluation_campaign = data.get("evaluation_campaign")
    if not isinstance(raw_evaluation_campaign, str):
        raise ValueError("evaluation_campaign must be a string")  # noqa: TRY004 - preserve validation API
    evaluation_path = _resolve_evaluation_campaign_path(
        control_path, raw_evaluation_campaign
    )
    return load_optimization_campaign(control_path, load_campaign(evaluation_path))


def _write_github_output(
    gh_output_path: Path | None, entries: dict[str, str],
) -> None:
    """Write GitHub Actions output entries to explicit path only."""
    if gh_output_path is None:
        return
    lines = [f"{k}={v}" for k, v in entries.items()]
    gh_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _enforce_identity_bindings(
    args: argparse.Namespace,
    control: OptimizationCampaign,
    state: CampaignState,
) -> None:
    """Bind manifest and revision identity to the loaded control and state.

    These bindings are optional flags so that pure planning stays usable
    offline, but when the caller supplies them they are enforced here, inside
    the CLI that performs planning and advancement, not only in the workflow
    wrapper preflight.
    """
    expected_manifest = getattr(args, "expected_manifest_sha256", None)
    if expected_manifest:
        raw = _safe_existing_file(args.control, "control manifest").read_bytes()
        actual = "sha256:" + hashlib.sha256(raw).hexdigest()
        if actual != expected_manifest:
            raise ValueError(
                f"manifest identity mismatch: expected {expected_manifest}, got {actual}"
            )
    expected_prompt_lab = getattr(args, "expected_prompt_lab_revision", None)
    if expected_prompt_lab and state.prompt_lab_revision != expected_prompt_lab:
        raise ValueError(
            "prompt_lab_revision mismatch: state has "
            f"{state.prompt_lab_revision!r}, expected {expected_prompt_lab!r}"
        )
    expected_korvid = getattr(args, "expected_korvid_revision", None)
    if expected_korvid and state.korvid_revision != expected_korvid:
        raise ValueError(
            "korvid_revision mismatch: state has "
            f"{state.korvid_revision!r}, expected {expected_korvid!r}"
        )
    expected_seed = getattr(args, "expected_seed_fingerprint", None)
    if expected_seed:
        validate_seed_candidate_fingerprint(expected_seed)
        if state.seed_candidate_fingerprint != expected_seed:
            raise ValueError(
                "seed_candidate_fingerprint mismatch: state has "
                f"{state.seed_candidate_fingerprint!r}, expected {expected_seed!r}"
            )
    validate_state_binding(control, state)


def _cmd_plan(args: argparse.Namespace) -> int:
    control = _load_control(args.control)
    state = _load_state(args.state)
    _enforce_identity_bindings(args, control, state)
    now = datetime.now(tz=UTC)

    action = next_action(control, state, now)
    if action is None:
        result: dict[str, Any] = {"terminal": True, "status": state.status.value}
        args.output.write_text(
            json.dumps(result, indent=2), encoding="utf-8",
        )
        _write_github_output(
            args.github_output,
            {"terminal": "true", "status": state.status.value},
        )
        return 0

    result = {
        "action_id": action.action_id,
        "kind": action.kind.value,
        "expected_state_hash": action.expected_state_hash,
        "stage_index": action.stage_index,
        "seed_index": action.seed_index,
        "tier_index": action.tier_index,
        "metric_calls": action.metric_calls,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_github_output(args.github_output, {
        "action_id": action.action_id,
        "kind": action.kind.value,
        "terminal": "false",
    })
    return 0


def _cmd_advance(args: argparse.Namespace) -> int:
    control = _load_control(args.control)
    state = _load_state(args.state)
    _enforce_identity_bindings(args, control, state)

    # CAS: validate expected-prior-hash against loaded state
    current_hash = state_hash(state)
    expected_prior = args.expected_prior_hash
    if expected_prior != current_hash:
        print(
            f"Error: --expected-prior-hash mismatch: "
            f"got {expected_prior!r}, state has {current_hash!r}",
            file=sys.stderr,
        )
        return 1

    action_data = json.loads(args.action.read_text(encoding="utf-8"))

    action = CampaignAction(
        action_id=action_data["action_id"],
        kind=ActionKind(action_data["kind"]),
        expected_state_hash=action_data["expected_state_hash"],
        stage_index=action_data.get("stage_index", 0),
        seed_index=action_data.get("seed_index", 0),
        tier_index=action_data.get("tier_index", 0),
        metric_calls=action_data.get("metric_calls", 0),
    )

    if args.outcome_kind == "evidence":
        if args.evidence is None:
            print("Error: --evidence is required for an evidence outcome", file=sys.stderr)
            return 1
        try:
            outcome_data = load_round_outcome(
                args.evidence, action, control=control, state=state,
            )
        except ValueError as exc:
            print(f"Error loading evidence: {exc}", file=sys.stderr)
            return 1

        score = CampaignScore(
            fingerprint=outcome_data.candidate_fingerprint,
            aggregate=outcome_data.aggregate_score,
            hard_safety_failures=outcome_data.hard_safety_failures,
            core_regression=outcome_data.core_regression,
            systemic_failures=outcome_data.systemic_failures,
            pass_at_3=outcome_data.pass_at_3,
            pass_at_5=outcome_data.pass_at_5,
        )
        attempt = AttemptOutcome(
            kind="evidence",
            score=score,
            metric_calls_used=outcome_data.metric_calls_used,
        )
    else:
        if args.evidence is not None:
            print(
                f"Error: --evidence is forbidden for {args.outcome_kind}",
                file=sys.stderr,
            )
            return 1
        attempt = AttemptOutcome(
            kind=args.outcome_kind,
            error_message=args.error_message,
        )

    now = datetime.now(tz=UTC)
    try:
        new_state = advance_state(control, state, action, attempt, now)
    except ValueError as exc:
        print(f"Error advancing state: {exc}", file=sys.stderr)
        return 1

    # Write new state with CAS against the loaded state's hash
    try:
        write_campaign_state(
            new_state,
            args.output_state,
            expected_prior_hash=current_hash,
            state_root=args.output_state.parent,
        )
    except ValueError as exc:
        print(f"Error writing state (CAS): {exc}", file=sys.stderr)
        return 1

    _write_github_output(args.github_output, {
        "status": new_state.status.value,
        "champion_fingerprint": new_state.champion_fingerprint,
    })
    return 0


def _cmd_validate_evidence(args: argparse.Namespace) -> int:
    """Validate evidence and its transition without writing campaign state."""
    control = _load_control(args.control)
    state = _load_state(args.state)
    _enforce_identity_bindings(args, control, state)
    current_hash = state_hash(state)
    if args.expected_prior_hash != current_hash:
        print("Error: expected prior hash mismatch", file=sys.stderr)
        return 1

    try:
        action_data = json.loads(args.action.read_text(encoding="utf-8"))
        action = CampaignAction(
            action_id=action_data["action_id"],
            kind=ActionKind(action_data["kind"]),
            expected_state_hash=action_data["expected_state_hash"],
            stage_index=action_data.get("stage_index", 0),
            seed_index=action_data.get("seed_index", 0),
            tier_index=action_data.get("tier_index", 0),
            metric_calls=action_data.get("metric_calls", 0),
        )
        outcome_data = load_round_outcome(
            args.evidence,
            action,
            control=control,
            state=state,
        )
        score = CampaignScore(
            fingerprint=outcome_data.candidate_fingerprint,
            aggregate=outcome_data.aggregate_score,
            hard_safety_failures=outcome_data.hard_safety_failures,
            core_regression=outcome_data.core_regression,
            systemic_failures=outcome_data.systemic_failures,
            pass_at_3=outcome_data.pass_at_3,
            pass_at_5=outcome_data.pass_at_5,
        )
        advance_state(
            control,
            state,
            action,
            AttemptOutcome(
                kind="evidence",
                score=score,
                metric_calls_used=outcome_data.metric_calls_used,
            ),
            datetime.now(tz=UTC),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error validating evidence: {str(exc)[:240]}", file=sys.stderr)
        return 1
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    control = _load_control(args.control)
    state = _load_state(args.state)
    _enforce_identity_bindings(args, control, state)
    write_campaign_artifacts(state, args.output_dir, control)
    _write_github_output(args.github_output, {"rendered": "true"})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    commands = {
        "plan": _cmd_plan,
        "advance": _cmd_advance,
        "validate-evidence": _cmd_validate_evidence,
        "render": _cmd_render,
    }
    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
