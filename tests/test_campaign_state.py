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
    total_metric_call_limit: int = 2400,
    wall_clock_limit_seconds: int = 21600,
    infrastructure_retry_limit: int = 3,
    stagnation_attempt_limit: int = 30,
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


def _run_seeds(ctrl: OptimizationCampaign, state: CampaignState, count: int) -> CampaignState:
    """Run `count` SEARCH steps with improving candidates."""
    for i in range(count):
        action = next_action(ctrl, state, NOW)
        assert action is not None, f"No action at step {i}"
        outcome = AttemptOutcome(
            kind="evidence",
            score=_score(f"candidate-{state.metric_calls_used + i}", aggregate=0.1 * (i + 1)),
        )
        state = advance_state(ctrl, state, action, outcome, LATER)
    return state


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


def test_rejects_same_fingerprint_no_promotion() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(
        kind="evidence",
        score=_score(state.champion_fingerprint, aggregate=0.99),
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


def test_equal_core_scores_never_promote() -> None:
    """Finding 5: Equal core scores must not promote based on fingerprint."""
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    # Candidate with identical core dimensions but different fingerprint
    outcome = AttemptOutcome(
        kind="evidence",
        score=_score("aaa-better-fp", aggregate=0.0, pass_at_3=0.0, pass_at_5=0.0),
    )
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    # Must NOT promote even though "aaa-better-fp" < "seed.yaml" lexicographically
    assert advanced.champion_fingerprint == state.champion_fingerprint
    assert advanced.stagnation_attempts == 1


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
    assert s1.stop_reason == "infrastructure_retry_limit_exhausted"


def test_system_error_wall_clock_crossing_terminates() -> None:
    """Finding 7: SYSTEM_ERROR crossing wall-clock must be terminal."""
    ctrl = _control(wall_clock_limit_seconds=21600)
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="system_error", error_message="timeout")
    # MUCH_LATER exceeds wall clock
    s1 = advance_state(ctrl, state, action, outcome, MUCH_LATER)
    assert s1.status is CampaignStatus.SYSTEM_ERROR
    assert s1.stop_reason == "wall_clock_limit_exceeded"


# --- CONFIG_ERROR ---


def test_config_error_terminates_as_system_error() -> None:
    """Finding 4: CONFIG_ERROR must deterministically terminate."""
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="config_error", error_message="bad manifest")
    s1 = advance_state(ctrl, state, action, outcome, LATER)
    assert s1.status is CampaignStatus.SYSTEM_ERROR
    assert "config_error" in (s1.stop_reason or "")


# --- Budget limits ---


def test_total_call_limit_terminates() -> None:
    """Finding 6: metric_calls consumed = action.metric_calls, not +1."""
    ctrl = _control(
        stages=(SearchStage(name="explore", metric_calls=12, seeds=(0,)),),
        total_metric_call_limit=12,
    )
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    assert action.metric_calls == 12
    outcome = AttemptOutcome(kind="evidence", score=_score("c1", aggregate=0.1))
    s1 = advance_state(ctrl, state, action, outcome, LATER)
    assert s1.metric_calls_used == 12
    assert s1.status is CampaignStatus.NOT_CONVERGED


def test_exact_metric_call_accounting() -> None:
    """Finding 6: Two searches with different metric_calls."""
    ctrl = _control(
        stages=(
            SearchStage(name="explore", metric_calls=5, seeds=(0,)),
            SearchStage(name="refine", metric_calls=10, seeds=(1,)),
        ),
        total_metric_call_limit=100,
    )
    state = _init(ctrl)
    # First search: 5 calls
    action = next_action(ctrl, state, NOW)
    assert action is not None and action.metric_calls == 5
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score("a", aggregate=0.2)), LATER)
    assert state.metric_calls_used == 5
    # Second search: 10 calls
    action = next_action(ctrl, state, NOW)
    assert action is not None and action.metric_calls == 10
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score("b", aggregate=0.3)), LATER)
    assert state.metric_calls_used == 15


def test_wall_clock_limit_terminates() -> None:
    ctrl = _control(wall_clock_limit_seconds=21600)
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("c1", aggregate=0.5))
    s1 = advance_state(ctrl, state, action, outcome, MUCH_LATER)
    assert s1.elapsed_seconds > ctrl.wall_clock_limit_seconds
    assert s1.status is CampaignStatus.NOT_CONVERGED


def test_stagnation_terminates() -> None:
    ctrl = _control(stagnation_attempt_limit=2)
    state = _init(ctrl)
    for _ in range(2):
        action = next_action(ctrl, state, NOW)
        assert action is not None
        outcome = AttemptOutcome(
            kind="evidence",
            score=_score(state.champion_fingerprint, aggregate=0.0),
        )
        state = advance_state(ctrl, state, action, outcome, LATER)
    assert state.stagnation_attempts == 2
    assert state.status is CampaignStatus.NOT_CONVERGED


# --- Action validation (Finding 1) ---


def test_stale_action_id_fails_closed() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("c1", aggregate=0.5))
    s1 = advance_state(ctrl, state, action, outcome, LATER)
    # Replaying the same action on new state fails
    with pytest.raises(ValueError, match="stale|state_hash"):
        advance_state(ctrl, s1, action, outcome, LATER)


def test_forged_action_kind_rejected() -> None:
    """Finding 1: Forged kind fails."""
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    # Forge a CONFIRM action with correct state_hash but wrong kind
    forged = CampaignAction(
        action_id=action.action_id,
        kind=ActionKind.CONFIRM,
        expected_state_hash=action.expected_state_hash,
        tier_index=action.tier_index,
    )
    outcome = AttemptOutcome(kind="evidence", score=_score("x"))
    with pytest.raises(ValueError, match="kind"):
        advance_state(ctrl, state, forged, outcome, LATER)


def test_forged_action_id_rejected() -> None:
    """Finding 1: Forged action_id fails."""
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    forged = CampaignAction(
        action_id="forged-id",
        kind=action.kind,
        expected_state_hash=action.expected_state_hash,
        stage_index=action.stage_index,
        seed_index=action.seed_index,
        tier_index=action.tier_index,
        metric_calls=action.metric_calls,
    )
    outcome = AttemptOutcome(kind="evidence", score=_score("x"))
    with pytest.raises(ValueError, match="action_id"):
        advance_state(ctrl, state, forged, outcome, LATER)


def test_forged_cursor_rejected() -> None:
    """Finding 1: Wrong stage/seed cursor fails."""
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    forged = CampaignAction(
        action_id=action.action_id,
        kind=action.kind,
        expected_state_hash=action.expected_state_hash,
        stage_index=99,
        seed_index=action.seed_index,
        tier_index=action.tier_index,
        metric_calls=action.metric_calls,
    )
    outcome = AttemptOutcome(kind="evidence", score=_score("x"))
    with pytest.raises(ValueError, match="stage_index"):
        advance_state(ctrl, state, forged, outcome, LATER)


# --- Qualification / Milestone / Confirmation ---


def _exhaust_stages(ctrl: OptimizationCampaign, state: CampaignState) -> CampaignState:
    """Run all stage seeds."""
    total = sum(len(s.seeds) for s in ctrl.stages)
    return _run_seeds(ctrl, state, total)


def test_qualification_requires_confirmation() -> None:
    ctrl = _control(confirmation_runs=1)
    state = _exhaust_stages(ctrl, _init(ctrl))

    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.MILESTONE
    outcome = AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint, aggregate=0.9))
    state = advance_state(ctrl, state, action, outcome, LATER)

    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.CONFIRM
    outcome = AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint, aggregate=0.9))
    state = advance_state(ctrl, state, action, outcome, LATER)
    assert state.status is CampaignStatus.QUALIFIED


def test_confirmation_runs_2() -> None:
    """Finding 2: confirmation_runs=2 requires two successful confirmations."""
    ctrl = _control(confirmation_runs=2)
    state = _exhaust_stages(ctrl, _init(ctrl))

    # Milestone
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.MILESTONE
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint)), LATER)

    # First confirm
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.CONFIRM
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint)), LATER)
    assert state.status is CampaignStatus.RUNNING  # not yet qualified
    assert state.confirmations_passed == 1

    # Second confirm
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.CONFIRM
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint)), LATER)
    assert state.status is CampaignStatus.QUALIFIED
    assert state.confirmations_passed == 2


def test_confirmation_failure_not_converged() -> None:
    ctrl = _control(confirmation_runs=1)
    state = _exhaust_stages(ctrl, _init(ctrl))

    # Milestone
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.MILESTONE
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint)), LATER)

    # Confirmation fails
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.CONFIRM
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score("different")), LATER)
    assert state.status is CampaignStatus.NOT_CONVERGED
    assert state.stop_reason == "confirmation_failed"


# --- Model tier rollover (Finding 3) ---


def test_tier_rollover_resets_state() -> None:
    """Finding 3: non-final tier qualification rolls over to fresh tier state."""
    ctrl = _control(
        model_tiers=(
            ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            ModelTier(name="large", model="qwen3:14b", digest=DIGEST_B),
        ),
        confirmation_runs=1,
    )
    state = _exhaust_stages(ctrl, _init(ctrl))

    # Milestone + Confirm tier 0
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.MILESTONE
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint)), LATER)

    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.CONFIRM
    metric_before = state.metric_calls_used
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint)), LATER)

    # Should roll over to tier 1 RUNNING, not QUALIFIED
    assert state.status is CampaignStatus.RUNNING
    assert state.tier_index == 1
    assert state.stage_index == 0
    assert state.seed_index == 0
    assert state.champion_fingerprint == ctrl.initial_candidate
    assert state.stagnation_attempts == 0
    assert state.retries_used == 0
    assert state.milestone_passed is False
    assert state.confirmations_passed == 0
    # Campaign-wide accounting preserved
    assert state.metric_calls_used == metric_before
    assert state.elapsed_seconds > 0
    # Prior tier result recorded
    assert len(state.tier_results) == 1
    assert state.tier_results[0].tier_index == 0
    assert state.tier_results[0].status is CampaignStatus.QUALIFIED


def test_final_tier_qualifies() -> None:
    """Final tier confirmation yields QUALIFIED status."""
    ctrl = _control(
        model_tiers=(ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),),
        confirmation_runs=1,
    )
    state = _exhaust_stages(ctrl, _init(ctrl))
    action = next_action(ctrl, state, LATER)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint)), LATER)
    action = next_action(ctrl, state, LATER)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint)), LATER)
    assert state.status is CampaignStatus.QUALIFIED


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
    state = _run_seeds(ctrl, state, 3)
    # After 3 seeds: (0,1) from stage 0, (2) from stage 1 => all exhausted
    assert state.stage_index == 2
    assert state.seed_index == 0
