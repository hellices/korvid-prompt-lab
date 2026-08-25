"""Campaign CLI: plan, advance, render subcommands.

The CLI writes GitHub outputs only to a caller-provided --github-output file;
it never trusts an environment-provided output path implicitly.
"""

from __future__ import annotations

import argparse
import json
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
    next_action,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="korvid-campaign")
    sub = parser.add_subparsers(dest="command", required=True)

    # plan
    plan_p = sub.add_parser("plan", help="Emit next action as JSON.")
    plan_p.add_argument("--control", type=Path, required=True)
    plan_p.add_argument("--state", type=Path, required=True)
    plan_p.add_argument("--output", type=Path, required=True)
    plan_p.add_argument("--github-output", type=Path, default=None)

    # advance
    adv_p = sub.add_parser("advance", help="Apply evidence to campaign state.")
    adv_p.add_argument("--control", type=Path, required=True)
    adv_p.add_argument("--state", type=Path, required=True)
    adv_p.add_argument("--action", type=Path, required=True)
    adv_p.add_argument("--evidence", type=Path, required=True)
    adv_p.add_argument("--output-state", type=Path, required=True)
    adv_p.add_argument("--github-output", type=Path, default=None)

    # render
    rend_p = sub.add_parser("render", help="Render campaign summary.")
    rend_p.add_argument("--state", type=Path, required=True)
    rend_p.add_argument("--output-dir", type=Path, required=True)
    rend_p.add_argument("--total-metric-call-limit", type=int, required=True)
    rend_p.add_argument("--wall-clock-limit-seconds", type=int, required=True)
    rend_p.add_argument("--stages-count", type=int, required=True)
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


def _load_control(path: Path) -> OptimizationCampaign:
    """Load optimization campaign control from YAML (lightweight, no eval campaign validation)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    from .campaigns import ModelTier, SearchStage

    stages = tuple(
        SearchStage(name=s["name"], metric_calls=s["metric_calls"], seeds=tuple(s["seeds"]))
        for s in data["stages"]
    )
    model_tiers = tuple(
        ModelTier(name=t["name"], model=t["model"], digest=t["digest"])
        for t in data["model_tiers"]
    )
    return OptimizationCampaign(
        schema_version=data["schema_version"],
        campaign_id=data["campaign_id"],
        evaluation_campaign=data["evaluation_campaign"],
        initial_candidate=data["initial_candidate"],
        train_case_ids=tuple(data["train_case_ids"]),
        validation_case_ids=tuple(data["validation_case_ids"]),
        milestone_case_ids=tuple(data["milestone_case_ids"]),
        stages=stages,
        model_tiers=model_tiers,
        total_metric_call_limit=data["total_metric_call_limit"],
        wall_clock_limit_seconds=data["wall_clock_limit_seconds"],
        infrastructure_retry_limit=data["infrastructure_retry_limit"],
        stagnation_attempt_limit=data["stagnation_attempt_limit"],
        confirmation_runs=data["confirmation_runs"],
    )


def _write_github_output(gh_output_path: Path | None, entries: dict[str, str]) -> None:
    """Write GitHub Actions output entries to explicit path only."""
    if gh_output_path is None:
        return
    lines = [f"{k}={v}" for k, v in entries.items()]
    gh_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cmd_plan(args: argparse.Namespace) -> int:
    control = _load_control(args.control)
    state = _load_state(args.state)
    now = datetime.now(tz=UTC)

    action = next_action(control, state, now)
    if action is None:
        result: dict[str, Any] = {"terminal": True, "status": state.status.value}
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        _write_github_output(args.github_output, {"terminal": "true", "status": state.status.value})
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

    # Load evidence
    evidence_root = args.evidence
    try:
        outcome_data = load_round_outcome(evidence_root, action)
    except ValueError as exc:
        print(f"Error loading evidence: {exc}", file=sys.stderr)
        return 1

    # Build AttemptOutcome
    score = CampaignScore(
        fingerprint=outcome_data.candidate_fingerprint,
        aggregate=outcome_data.aggregate_score,
        hard_safety_failures=outcome_data.hard_safety_failures,
        core_regression=outcome_data.core_regression,
        systemic_failures=outcome_data.systemic_failures,
        pass_at_3=outcome_data.pass_at_3 if outcome_data.pass_at_3 is not None else 0.0,
        pass_at_5=outcome_data.pass_at_5 if outcome_data.pass_at_5 is not None else 0.0,
    )
    attempt = AttemptOutcome(kind="evidence", score=score)

    now = datetime.now(tz=UTC)
    try:
        new_state = advance_state(control, state, action, attempt, now)
    except ValueError as exc:
        print(f"Error advancing state: {exc}", file=sys.stderr)
        return 1

    # Write new state with CAS
    write_campaign_state(new_state, args.output_state, expected_prior_hash=None)

    _write_github_output(args.github_output, {
        "status": new_state.status.value,
        "champion_fingerprint": new_state.champion_fingerprint,
    })
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    state = _load_state(args.state)
    write_campaign_artifacts(
        state,
        args.output_dir,
        total_metric_call_limit=args.total_metric_call_limit,
        wall_clock_limit_seconds=args.wall_clock_limit_seconds,
        stages_count=args.stages_count,
    )
    _write_github_output(args.github_output, {"rendered": "true"})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    commands = {
        "plan": _cmd_plan,
        "advance": _cmd_advance,
        "render": _cmd_render,
    }
    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
