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


def _qualifying_score(fingerprint: str = "champion") -> CampaignScore:
    """Score that passes the qualification gate."""
    return _score(fingerprint, aggregate=0.9, pass_at_3=1.0, pass_at_5=1.0)


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
            score=_score(f"candidate-{state.metric_calls_used}-{i}", aggregate=0.1 * (i + 1)),
        )
        state = advance_state(ctrl, state, action, outcome, LATER)
    return state


def _exhaust_stages(ctrl: OptimizationCampaign, state: CampaignState) -> CampaignState:
    total = sum(len(s.seeds) for s in ctrl.stages)
    return _run_seeds(ctrl, state, total)


# --- Promotion tests ---


def test_promotes_only_changed_non_regressing_candidate() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("better", aggregate=0.9))
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    assert advanced.champion_fingerprint == "better"
    assert advanced.stagnation_attempts == 0


def test_rejects_same_fingerprint_no_promotion() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint, aggregate=0.99))
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    assert advanced.champion_fingerprint == state.champion_fingerprint
    assert advanced.stagnation_attempts == 1


def test_rejects_hard_safety_regression() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("regressor", aggregate=0.99, hard_safety_failures=1))
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    assert advanced.champion_fingerprint == state.champion_fingerprint


def test_rejects_core_regression() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("regressor", aggregate=0.99, core_regression=True))
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    assert advanced.champion_fingerprint == state.champion_fingerprint


def test_equal_core_scores_never_promote() -> None:
    """Equal core scores must not promote based on fingerprint."""
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("aaa-better-fp", aggregate=0.0, pass_at_3=0.0, pass_at_5=0.0))
    advanced = advance_state(ctrl, state, action, outcome, LATER)
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


def test_system_error_wall_clock_crossing_terminates() -> None:
    ctrl = _control(wall_clock_limit_seconds=21600)
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="system_error", error_message="timeout")
    s1 = advance_state(ctrl, state, action, outcome, MUCH_LATER)
    assert s1.status is CampaignStatus.SYSTEM_ERROR
    assert s1.stop_reason == "wall_clock_limit_exceeded"


# --- CONFIG_ERROR ---


def test_config_error_terminates() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="config_error", error_message="bad manifest")
    s1 = advance_state(ctrl, state, action, outcome, LATER)
    assert s1.status is CampaignStatus.SYSTEM_ERROR
    assert "config_error" in (s1.stop_reason or "")


# --- Budget limits ---


def test_exact_metric_call_accounting() -> None:
    """metric_calls consumed = action.metric_calls per stage, uniformly."""
    ctrl = _control(
        stages=(
            SearchStage(name="explore", metric_calls=5, seeds=(0,)),
            SearchStage(name="refine", metric_calls=10, seeds=(1,)),
        ),
        total_metric_call_limit=100,
    )
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None and action.metric_calls == 5
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score("a", aggregate=0.2)), LATER)
    assert state.metric_calls_used == 5
    action = next_action(ctrl, state, NOW)
    assert action is not None and action.metric_calls == 10
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score("b", aggregate=0.3)), LATER)
    assert state.metric_calls_used == 15


def test_total_call_limit_terminates() -> None:
    ctrl = _control(
        stages=(SearchStage(name="explore", metric_calls=12, seeds=(0,)),),
        total_metric_call_limit=12,
    )
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("c1", aggregate=0.1))
    s1 = advance_state(ctrl, state, action, outcome, LATER)
    assert s1.metric_calls_used == 12
    assert s1.status is CampaignStatus.NOT_CONVERGED


def test_wall_clock_limit_terminates() -> None:
    ctrl = _control(wall_clock_limit_seconds=21600)
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("c1", aggregate=0.5))
    s1 = advance_state(ctrl, state, action, outcome, MUCH_LATER)
    assert s1.status is CampaignStatus.NOT_CONVERGED


def test_stagnation_terminates() -> None:
    ctrl = _control(stagnation_attempt_limit=2)
    state = _init(ctrl)
    for _ in range(2):
        action = next_action(ctrl, state, NOW)
        assert action is not None
        outcome = AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint, aggregate=0.0))
        state = advance_state(ctrl, state, action, outcome, LATER)
    assert state.stagnation_attempts == 2
    assert state.status is CampaignStatus.NOT_CONVERGED


def test_wall_clock_crossing_during_milestone_terminates() -> None:
    """Wall-clock crossing during milestone evidence → NOT_CONVERGED."""
    ctrl = _control(wall_clock_limit_seconds=21600)
    state = _exhaust_stages(ctrl, _init(ctrl))
    action = next_action(ctrl, state, NOW)
    assert action is not None and action.kind is ActionKind.MILESTONE
    outcome = AttemptOutcome(kind="evidence", score=_qualifying_score(state.champion_fingerprint))
    s1 = advance_state(ctrl, state, action, outcome, MUCH_LATER)
    assert s1.status is CampaignStatus.NOT_CONVERGED


def test_wall_clock_crossing_during_confirm_terminates() -> None:
    """Wall-clock crossing during confirm evidence → NOT_CONVERGED."""
    ctrl = _control(wall_clock_limit_seconds=21600)
    state = _exhaust_stages(ctrl, _init(ctrl))
    # Pass milestone
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.MILESTONE
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(state.champion_fingerprint)), LATER)
    # Confirm crosses wall clock
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.CONFIRM
    s1 = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(state.champion_fingerprint)), MUCH_LATER)
    assert s1.status is CampaignStatus.NOT_CONVERGED


# --- Action validation ---


def test_stale_action_replay_against_advanced_state() -> None:
    """CAS semantics: replaying action against persisted advanced state fails on hash."""
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("c1", aggregate=0.5))
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    # Two workers computed same transition; only one persisted state hash matches
    with pytest.raises(ValueError, match="stale|state_hash"):
        advance_state(ctrl, advanced, action, outcome, LATER)


def test_forged_action_kind_rejected() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    forged = CampaignAction(
        action_id=action.action_id,
        kind=ActionKind.CONFIRM,
        expected_state_hash=action.expected_state_hash,
        tier_index=action.tier_index,
    )
    with pytest.raises(ValueError, match="kind"):
        advance_state(ctrl, state, forged, AttemptOutcome(kind="evidence", score=_score("x")), LATER)


def test_forged_action_id_rejected() -> None:
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
    with pytest.raises(ValueError, match="action_id"):
        advance_state(ctrl, state, forged, AttemptOutcome(kind="evidence", score=_score("x")), LATER)


def test_forged_cursor_rejected() -> None:
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
    with pytest.raises(ValueError, match="stage_index"):
        advance_state(ctrl, state, forged, AttemptOutcome(kind="evidence", score=_score("x")), LATER)


# --- Qualification / Milestone / Confirmation ---


def test_milestone_gate_pass_at_3_failure_rejects() -> None:
    """Milestone fails if pass@3 < 1.0."""
    ctrl = _control()
    state = _exhaust_stages(ctrl, _init(ctrl))
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.MILESTONE
    bad_score = _score(state.champion_fingerprint, aggregate=0.9, pass_at_3=0.8)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=bad_score), LATER)
    # Single tier → NOT_CONVERGED
    assert state.status is CampaignStatus.NOT_CONVERGED


def test_milestone_gate_hard_safety_rejects() -> None:
    """Milestone fails if hard_safety_failures > 0."""
    ctrl = _control()
    state = _exhaust_stages(ctrl, _init(ctrl))
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.MILESTONE
    bad_score = _score(state.champion_fingerprint, aggregate=0.9, hard_safety_failures=1)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=bad_score), LATER)
    assert state.status is CampaignStatus.NOT_CONVERGED


def test_qualification_tier0_success_qualifies_campaign() -> None:
    """Success on tier 0 → QUALIFIED (never rolls to larger model)."""
    ctrl = _control(
        model_tiers=(
            ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            ModelTier(name="large", model="qwen3:14b", digest=DIGEST_B),
        ),
        confirmation_runs=1,
    )
    state = _exhaust_stages(ctrl, _init(ctrl))
    champ = state.champion_fingerprint

    # Milestone passes gate
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.MILESTONE
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    assert state.milestone_passed is True

    # Confirmation passes gate
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.CONFIRM
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)

    assert state.status is CampaignStatus.QUALIFIED
    assert state.tier_index == 0  # never rolled to tier 1


def test_confirmation_runs_2() -> None:
    """confirmation_runs=2 requires two successful confirmations."""
    ctrl = _control(confirmation_runs=2)
    state = _exhaust_stages(ctrl, _init(ctrl))
    champ = state.champion_fingerprint

    # Milestone
    action = next_action(ctrl, state, LATER)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)

    # First confirm
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.CONFIRM
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    assert state.status is CampaignStatus.RUNNING
    assert state.confirmations_passed == 1

    # Second confirm
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.CONFIRM
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    assert state.status is CampaignStatus.QUALIFIED


def test_confirmation_failure_rolls_to_next_tier() -> None:
    """Confirmation fail with non-final tier → roll to next tier."""
    ctrl = _control(
        model_tiers=(
            ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            ModelTier(name="large", model="qwen3:14b", digest=DIGEST_B),
        ),
        confirmation_runs=1,
    )
    state = _exhaust_stages(ctrl, _init(ctrl))
    champ = state.champion_fingerprint

    # Milestone passes
    action = next_action(ctrl, state, LATER)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)

    # Confirmation fails gate (pass@3 < 1.0)
    action = next_action(ctrl, state, LATER)
    bad_score = _score(champ, aggregate=0.9, pass_at_3=0.9)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=bad_score), LATER)

    # Rolled to tier 1
    assert state.status is CampaignStatus.RUNNING
    assert state.tier_index == 1
    assert state.champion_fingerprint == ctrl.initial_candidate
    assert state.stagnation_attempts == 0
    assert state.milestone_passed is False
    assert state.confirmations_passed == 0
    assert len(state.tier_results) == 1
    assert state.tier_results[0].status is CampaignStatus.NOT_CONVERGED


def test_milestone_failure_rolls_to_next_tier() -> None:
    """Milestone fail with non-final tier → roll to next tier."""
    ctrl = _control(
        model_tiers=(
            ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            ModelTier(name="large", model="qwen3:14b", digest=DIGEST_B),
        ),
    )
    state = _exhaust_stages(ctrl, _init(ctrl))
    # Milestone fails gate
    action = next_action(ctrl, state, LATER)
    bad_score = _score(state.champion_fingerprint, aggregate=0.9, hard_safety_failures=1)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=bad_score), LATER)

    assert state.status is CampaignStatus.RUNNING
    assert state.tier_index == 1
    assert len(state.tier_results) == 1


def test_final_tier_failure_not_converged() -> None:
    """Final tier milestone failure → campaign NOT_CONVERGED."""
    ctrl = _control(
        model_tiers=(ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),),
    )
    state = _exhaust_stages(ctrl, _init(ctrl))
    action = next_action(ctrl, state, LATER)
    bad_score = _score(state.champion_fingerprint, aggregate=0.9, pass_at_5=0.8)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=bad_score), LATER)
    assert state.status is CampaignStatus.NOT_CONVERGED
    assert len(state.tier_results) == 1


# --- Uniform metric accounting for milestone/confirm ---


def test_milestone_confirm_metric_accounting_uniform() -> None:
    """Milestone/confirm with metric_calls=0 does not increase metric_calls_used."""
    ctrl = _control()
    state = _exhaust_stages(ctrl, _init(ctrl))
    metric_before = state.metric_calls_used

    # Milestone (metric_calls=0)
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.metric_calls == 0
    champ = state.champion_fingerprint
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    assert state.metric_calls_used == metric_before

    # Confirm (metric_calls=0)
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.metric_calls == 0
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    assert state.metric_calls_used == metric_before


# --- State hash ---


def test_state_hash_deterministic() -> None:
    state = _init()
    assert state_hash(state) == state_hash(state)
    assert state_hash(state).startswith("sha256:")


def test_state_hash_changes_on_advance() -> None:
    ctrl = _control()
    state = _init(ctrl)
    h1 = state_hash(state)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    s2 = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score("new", aggregate=0.9)), LATER)
    assert state_hash(s2) != h1


# --- Staged seeds ---


def test_staged_seeds_advance_correctly() -> None:
    ctrl = _control(
        stages=(
            SearchStage(name="explore", metric_calls=12, seeds=(0, 1)),
            SearchStage(name="refine", metric_calls=24, seeds=(2,)),
        ),
    )
    state = _run_seeds(ctrl, _init(ctrl), 3)
    assert state.stage_index == 2
    assert state.seed_index == 0
