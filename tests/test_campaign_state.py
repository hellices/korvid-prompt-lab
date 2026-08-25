"""Tests for bounded campaign state machine (Task 3)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.campaigns import (
    ActionKind,
    AttemptOutcome,
    CampaignAction,
    CampaignScore,
    CampaignState,
    CampaignStatus,
    ModelTier,
    OptimizationCampaign,
    SearchStage,
    advance_state,
    initial_state,
    next_action,
    state_hash,
)

NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
LATER = datetime(2026, 1, 15, 12, 5, 0, tzinfo=UTC)
MUCH_LATER = datetime(2026, 1, 15, 18, 1, 0, tzinfo=UTC)

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def _control(
    *,
    stages: tuple[SearchStage, ...] | None = None,
    model_tiers: tuple[ModelTier, ...] | None = None,
    total_metric_call_limit: int = 240,
    wall_clock_limit_seconds: int = 21600,
    infrastructure_retry_limit: int = 3,
    stagnation_attempt_limit: int = 3,
    confirmation_runs: int = 1,
) -> OptimizationCampaign:
    if stages is None:
        stages = (
            SearchStage(name="explore", metric_calls=12, seeds=(0, 1, 2)),
            SearchStage(name="refine", metric_calls=24, seeds=(3, 4)),
        )
    if model_tiers is None:
        model_tiers = (ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),)
    return OptimizationCampaign(
        schema_version=1,
        campaign_id="test-campaign",
        evaluation_campaign="eval-campaign",
        initial_candidate="seed.yaml",
        train_case_ids=("case-a", "case-b"),
        validation_case_ids=("case-c",),
        milestone_case_ids=("case-d",),
        stages=stages,
        model_tiers=model_tiers,
        total_metric_call_limit=total_metric_call_limit,
        wall_clock_limit_seconds=wall_clock_limit_seconds,
        infrastructure_retry_limit=infrastructure_retry_limit,
        stagnation_attempt_limit=stagnation_attempt_limit,
        confirmation_runs=confirmation_runs,
    )


def _score(
    fingerprint: str,
    *,
    aggregate: float = 0.5,
    hard_safety_failures: int = 0,
    core_regression: bool = False,
    pass_at_3: float = 1.0,
    pass_at_5: float = 1.0,
) -> CampaignScore:
    return CampaignScore(
        fingerprint=fingerprint,
        aggregate=aggregate,
        hard_safety_failures=hard_safety_failures,
        core_regression=core_regression,
        pass_at_3=pass_at_3,
        pass_at_5=pass_at_5,
    )


def _init(control: OptimizationCampaign | None = None) -> CampaignState:
    c = control or _control()
    return initial_state(c, prompt_lab_revision="abc123", korvid_revision="def456", started_at=NOW)


# --- Promotion tests ---


def test_promotes_only_changed_non_regressing_candidate() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    assert action.kind is ActionKind.SEARCH

    outcome = AttemptOutcome(
        kind="evidence",
        score=_score("better", aggregate=0.9, hard_safety_failures=0),
    )
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    assert advanced.champion_fingerprint == "better"
    assert advanced.stagnation_attempts == 0


def test_rejects_flat_score_no_promotion() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    # Same fingerprint as initial - no promotion
    outcome = AttemptOutcome(
        kind="evidence",
        score=_score(state.champion_fingerprint, aggregate=0.5),
    )
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    assert advanced.champion_fingerprint == state.champion_fingerprint
    assert advanced.stagnation_attempts == 1


def test_rejects_hard_safety_regression() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(
        kind="evidence",
        score=_score("regressor", aggregate=0.99, hard_safety_failures=1),
    )
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    # Champion stays: hard safety failure means no promotion over zero-failure champion
    assert advanced.champion_fingerprint == state.champion_fingerprint
    assert advanced.stagnation_attempts == 1


def test_rejects_core_regression() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(
        kind="evidence",
        score=_score("regressor", aggregate=0.99, core_regression=True),
    )
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    assert advanced.champion_fingerprint == state.champion_fingerprint
    assert advanced.stagnation_attempts == 1


def test_deterministic_tie_break_uses_fingerprint() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None

    # Provide candidate with identical score dimensions as initial
    initial_fp = state.champion_fingerprint
    tied_fp = "aaa-tied-candidate"  # lexicographically less than "seed.yaml"
    outcome = AttemptOutcome(
        kind="evidence",
        score=_score(tied_fp, aggregate=0.0, pass_at_3=0.0, pass_at_5=0.0),
    )
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    # Tie-break: smaller fingerprint wins
    expected = min(initial_fp, tied_fp)
    assert advanced.champion_fingerprint == expected


# --- System error ---


def test_system_error_does_not_consume_experiment_budget() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="system_error", error_message="agent-chat race")
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    assert advanced.metric_calls_used == 0
    assert advanced.retries_used == 1
    assert advanced.elapsed_seconds > state.elapsed_seconds
    assert advanced.status is CampaignStatus.RUNNING


def test_retry_exhaustion_terminates() -> None:
    ctrl = _control(infrastructure_retry_limit=1)
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="system_error", error_message="oops")
    s1 = advance_state(ctrl, state, action, outcome, LATER)
    assert s1.status is CampaignStatus.SYSTEM_ERROR


# --- Budget limits ---


def test_total_call_limit_terminates() -> None:
    ctrl = _control(total_metric_call_limit=1)
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("c1", aggregate=0.1))
    s1 = advance_state(ctrl, state, action, outcome, LATER)
    # Used 1 call, limit is 1, should not have more actions
    assert s1.metric_calls_used == 1
    assert next_action(ctrl, s1, LATER) is None
    assert s1.status is CampaignStatus.NOT_CONVERGED


def test_wall_clock_limit_terminates() -> None:
    ctrl = _control(wall_clock_limit_seconds=21600)
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("c1", aggregate=0.5))
    s1 = advance_state(ctrl, state, action, outcome, MUCH_LATER)
    # Elapsed > wall_clock_limit
    assert s1.elapsed_seconds > ctrl.wall_clock_limit_seconds
    assert next_action(ctrl, s1, MUCH_LATER) is None
    assert s1.status is CampaignStatus.NOT_CONVERGED


def test_stagnation_terminates() -> None:
    ctrl = _control(stagnation_attempt_limit=2)
    state = _init(ctrl)

    for i in range(2):
        action = next_action(ctrl, state, NOW)
        assert action is not None
        outcome = AttemptOutcome(
            kind="evidence",
            score=_score(state.champion_fingerprint, aggregate=0.0),
        )
        state = advance_state(ctrl, state, action, outcome, LATER)

    assert state.stagnation_attempts == 2
    assert next_action(ctrl, state, LATER) is None
    assert state.status is CampaignStatus.NOT_CONVERGED


# --- Stale action ID ---


def test_stale_action_id_fails_closed() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("c1", aggregate=0.5))
    s1 = advance_state(ctrl, state, action, outcome, LATER)

    # Replaying the same action on the new state fails
    with pytest.raises(ValueError, match="state_hash|stale"):
        advance_state(ctrl, s1, action, outcome, LATER)


# --- Qualification / Milestone / Confirmation ---


def test_qualification_requires_confirmation() -> None:
    ctrl = _control(confirmation_runs=1)
    state = _init(ctrl)

    # Exhaust all stage seeds to reach milestone
    stage_seeds = sum(len(s.seeds) for s in ctrl.stages)
    for i in range(stage_seeds):
        action = next_action(ctrl, state, NOW)
        assert action is not None, f"No action at step {i}"
        outcome = AttemptOutcome(
            kind="evidence",
            score=_score(f"candidate-{i}", aggregate=0.1 * (i + 1)),
        )
        state = advance_state(ctrl, state, action, outcome, LATER)

    # After all stages, next action should be MILESTONE
    action = next_action(ctrl, state, LATER)
    assert action is not None
    assert action.kind is ActionKind.MILESTONE

    # Complete milestone
    outcome = AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint, aggregate=0.9))
    state = advance_state(ctrl, state, action, outcome, LATER)

    # After milestone, should get CONFIRM
    action = next_action(ctrl, state, LATER)
    assert action is not None
    assert action.kind is ActionKind.CONFIRM

    # Complete confirmation
    outcome = AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint, aggregate=0.9))
    state = advance_state(ctrl, state, action, outcome, LATER)
    assert state.status is CampaignStatus.QUALIFIED


def test_confirmation_failure_not_converged() -> None:
    ctrl = _control(confirmation_runs=1, stagnation_attempt_limit=100, total_metric_call_limit=1000)
    state = _init(ctrl)

    # Exhaust stages
    stage_seeds = sum(len(s.seeds) for s in ctrl.stages)
    for i in range(stage_seeds):
        action = next_action(ctrl, state, NOW)
        assert action is not None
        outcome = AttemptOutcome(kind="evidence", score=_score(f"c-{i}", aggregate=0.1 * (i + 1)))
        state = advance_state(ctrl, state, action, outcome, LATER)

    # Milestone
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.MILESTONE
    outcome = AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint, aggregate=0.9))
    state = advance_state(ctrl, state, action, outcome, LATER)

    # Confirmation fails
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.CONFIRM
    outcome = AttemptOutcome(kind="evidence", score=_score("failed-confirm", aggregate=0.01))
    state = advance_state(ctrl, state, action, outcome, LATER)

    # After confirmation failure with no remaining budget/stages => NOT_CONVERGED
    assert state.status is CampaignStatus.NOT_CONVERGED


# --- Model tier transition ---


def test_model_tier_transition_independent() -> None:
    ctrl = _control(
        model_tiers=(
            ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            ModelTier(name="large", model="qwen3:14b", digest=DIGEST_B),
        ),
        confirmation_runs=1,
        stagnation_attempt_limit=100,
        total_metric_call_limit=1000,
    )
    state = _init(ctrl)

    # Complete first tier
    stage_seeds = sum(len(s.seeds) for s in ctrl.stages)
    for i in range(stage_seeds):
        action = next_action(ctrl, state, NOW)
        assert action is not None
        outcome = AttemptOutcome(kind="evidence", score=_score(f"c-{i}", aggregate=0.1 * (i + 1)))
        state = advance_state(ctrl, state, action, outcome, LATER)

    # Milestone + Confirm
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.MILESTONE
    outcome = AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint, aggregate=0.9))
    state = advance_state(ctrl, state, action, outcome, LATER)

    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.CONFIRM
    outcome = AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint, aggregate=0.9))
    state = advance_state(ctrl, state, action, outcome, LATER)

    # Should be QUALIFIED at tier 0, transition to tier 1
    assert state.status is CampaignStatus.QUALIFIED
    assert state.tier_index == 0


# --- State hash ---


def test_state_hash_deterministic() -> None:
    ctrl = _control()
    state = _init(ctrl)
    h1 = state_hash(state)
    h2 = state_hash(state)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_state_hash_changes_on_advance() -> None:
    ctrl = _control()
    state = _init(ctrl)
    h1 = state_hash(state)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("new", aggregate=0.9))
    s2 = advance_state(ctrl, state, action, outcome, LATER)
    assert state_hash(s2) != h1


# --- Staged seeds ---


def test_staged_seeds_advance_correctly() -> None:
    ctrl = _control(
        stages=(
            SearchStage(name="explore", metric_calls=12, seeds=(0, 1)),
            SearchStage(name="refine", metric_calls=24, seeds=(2,)),
        ),
    )
    state = _init(ctrl)

    actions_seen: list[CampaignAction] = []
    for _ in range(3):
        action = next_action(ctrl, state, NOW)
        assert action is not None
        actions_seen.append(action)
        outcome = AttemptOutcome(kind="evidence", score=_score(f"c{len(actions_seen)}", aggregate=0.1))
        state = advance_state(ctrl, state, action, outcome, LATER)

    # After 3 advances: seeds (0,1) from stage 0 and seed (2) from stage 1
    # Stage 1 had 1 seed, so after consuming it, cursor advances past it
    assert state.stage_index == 2
    assert state.seed_index == 0
