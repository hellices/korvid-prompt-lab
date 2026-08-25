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
    ModelIdentity,
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
    systemic_failures: int = 0,
    pass_at_3: float = 1.0,
    pass_at_5: float = 1.0,
) -> CampaignScore:
    return CampaignScore(
        fingerprint=fingerprint,
        aggregate=aggregate,
        hard_safety_failures=hard_safety_failures,
        core_regression=core_regression,
        systemic_failures=systemic_failures,
        pass_at_3=pass_at_3,
        pass_at_5=pass_at_5,
    )


def _qualifying_score(fingerprint: str) -> CampaignScore:
    return _score(fingerprint, aggregate=0.9, systemic_failures=0, hard_safety_failures=0, pass_at_3=1.0, pass_at_5=1.0)


def _init(control: OptimizationCampaign | None = None) -> CampaignState:
    c = control or _control()
    return initial_state(c, prompt_lab_revision="abc123", korvid_revision="def456", started_at=NOW)


def _run_seeds(ctrl: OptimizationCampaign, state: CampaignState, count: int) -> CampaignState:
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


# --- CampaignScore validation ---


def test_campaign_score_rejects_bool_systemic() -> None:
    with pytest.raises(ValueError, match="systemic_failures"):
        CampaignScore(fingerprint="x", aggregate=0.5, hard_safety_failures=0, core_regression=False, systemic_failures=True)  # type: ignore[arg-type]


def test_campaign_score_rejects_negative_systemic() -> None:
    with pytest.raises(ValueError, match="systemic_failures"):
        CampaignScore(fingerprint="x", aggregate=0.5, hard_safety_failures=0, core_regression=False, systemic_failures=-1)


def test_campaign_score_rejects_bool_hard_safety() -> None:
    with pytest.raises(ValueError, match="hard_safety_failures"):
        CampaignScore(fingerprint="x", aggregate=0.5, hard_safety_failures=True, core_regression=False, systemic_failures=0)  # type: ignore[arg-type]


def test_campaign_score_rejects_negative_hard_safety() -> None:
    with pytest.raises(ValueError, match="hard_safety_failures"):
        CampaignScore(fingerprint="x", aggregate=0.5, hard_safety_failures=-1, core_regression=False, systemic_failures=0)


# --- Promotion tests ---


def test_promotes_strictly_better_candidate() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("better", aggregate=0.9))
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    assert advanced.champion_fingerprint == "better"
    assert advanced.stagnation_attempts == 0


def test_systemic_search_never_promoted() -> None:
    """systemic_failures > 0 is never promotable."""
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("systemic-candidate", aggregate=0.99, systemic_failures=1))
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    assert advanced.champion_fingerprint == state.champion_fingerprint
    assert advanced.stagnation_attempts == 1


def test_equal_core_scores_never_promote() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score("aaa", aggregate=0.0, pass_at_3=0.0, pass_at_5=0.0, systemic_failures=0))
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    assert advanced.champion_fingerprint == state.champion_fingerprint


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


# --- System error ---


def test_system_error_does_not_consume_budget() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="system_error", error_message="race")
    advanced = advance_state(ctrl, state, action, outcome, LATER)
    assert advanced.metric_calls_used == 0
    assert advanced.retries_used == 1
    assert advanced.status is CampaignStatus.RUNNING


def test_retry_exhaustion_terminates() -> None:
    ctrl = _control(infrastructure_retry_limit=1)
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    s1 = advance_state(ctrl, state, action, AttemptOutcome(kind="system_error", error_message="x"), LATER)
    assert s1.status is CampaignStatus.SYSTEM_ERROR


def test_system_error_wall_clock_crossing_terminates() -> None:
    ctrl = _control(wall_clock_limit_seconds=21600)
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    s1 = advance_state(ctrl, state, action, AttemptOutcome(kind="system_error", error_message="t"), MUCH_LATER)
    assert s1.status is CampaignStatus.SYSTEM_ERROR


def test_config_error_terminates() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    s1 = advance_state(ctrl, state, action, AttemptOutcome(kind="config_error", error_message="bad"), LATER)
    assert s1.status is CampaignStatus.SYSTEM_ERROR


# --- Budget ---


def test_exact_metric_accounting() -> None:
    ctrl = _control(
        stages=(SearchStage(name="a", metric_calls=5, seeds=(0,)), SearchStage(name="b", metric_calls=10, seeds=(1,))),
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
    ctrl = _control(stages=(SearchStage(name="x", metric_calls=12, seeds=(0,)),), total_metric_call_limit=12)
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    s1 = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score("c1", aggregate=0.1)), LATER)
    assert s1.status is CampaignStatus.NOT_CONVERGED


def test_wall_clock_crossing_during_confirm_terminates() -> None:
    ctrl = _control(wall_clock_limit_seconds=21600)
    state = _exhaust_stages(ctrl, _init(ctrl))
    champ = state.champion_fingerprint
    action = next_action(ctrl, state, LATER)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    action = next_action(ctrl, state, LATER)
    s1 = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), MUCH_LATER)
    assert s1.status is CampaignStatus.NOT_CONVERGED


# --- Stagnation with tier rollover ---


def test_stagnation_non_final_tier_rolls_over() -> None:
    """Non-final tier stagnation → roll to next tier."""
    ctrl = _control(
        model_tiers=(
            ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            ModelTier(name="large", model="qwen3:14b", digest=DIGEST_B),
        ),
        stagnation_attempt_limit=2,
    )
    state = _init(ctrl)
    for _ in range(2):
        action = next_action(ctrl, state, NOW)
        assert action is not None
        outcome = AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint, aggregate=0.0))
        state = advance_state(ctrl, state, action, outcome, LATER)
    # Rolled to tier 1
    assert state.status is CampaignStatus.RUNNING
    assert state.tier_index == 1
    assert state.stagnation_attempts == 0
    assert len(state.tier_results) == 1
    assert state.tier_results[0].status is CampaignStatus.NOT_CONVERGED


def test_stagnation_final_tier_terminates() -> None:
    """Final tier stagnation → campaign NOT_CONVERGED."""
    ctrl = _control(stagnation_attempt_limit=2)
    state = _init(ctrl)
    for _ in range(2):
        action = next_action(ctrl, state, NOW)
        assert action is not None
        outcome = AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint, aggregate=0.0))
        state = advance_state(ctrl, state, action, outcome, LATER)
    assert state.status is CampaignStatus.NOT_CONVERGED


# --- Action validation ---


def test_stale_action_replay() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    advanced = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_score("c1", aggregate=0.5)), LATER)
    with pytest.raises(ValueError, match="stale|state_hash"):
        advance_state(ctrl, advanced, action, AttemptOutcome(kind="evidence", score=_score("c1", aggregate=0.5)), LATER)


def test_forged_kind_rejected() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    forged = CampaignAction(action_id=action.action_id, kind=ActionKind.CONFIRM, expected_state_hash=action.expected_state_hash, tier_index=action.tier_index)
    with pytest.raises(ValueError, match="kind"):
        advance_state(ctrl, state, forged, AttemptOutcome(kind="evidence", score=_score("x")), LATER)


def test_forged_id_rejected() -> None:
    ctrl = _control()
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    forged = CampaignAction(action_id="bad", kind=action.kind, expected_state_hash=action.expected_state_hash, stage_index=action.stage_index, seed_index=action.seed_index, tier_index=action.tier_index, metric_calls=action.metric_calls)
    with pytest.raises(ValueError, match="action_id"):
        advance_state(ctrl, state, forged, AttemptOutcome(kind="evidence", score=_score("x")), LATER)


# --- Evidence fingerprint binding ---


def test_milestone_wrong_fingerprint_rejected() -> None:
    """Milestone outcome with different fingerprint → ValueError."""
    ctrl = _control()
    state = _exhaust_stages(ctrl, _init(ctrl))
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.MILESTONE
    wrong_score = _qualifying_score("wrong-candidate")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=wrong_score), LATER)


def test_confirm_wrong_fingerprint_rejected() -> None:
    """Confirm outcome with different fingerprint → ValueError."""
    ctrl = _control()
    state = _exhaust_stages(ctrl, _init(ctrl))
    champ = state.champion_fingerprint
    action = next_action(ctrl, state, LATER)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    action = next_action(ctrl, state, LATER)
    assert action is not None and action.kind is ActionKind.CONFIRM
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score("wrong")), LATER)


# --- Qualification gate with systemic ---


def test_systemic_milestone_not_qualified() -> None:
    """Milestone with systemic_failures > 0 fails gate."""
    ctrl = _control(
        model_tiers=(ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),),
    )
    state = _exhaust_stages(ctrl, _init(ctrl))
    champ = state.champion_fingerprint
    action = next_action(ctrl, state, LATER)
    bad_score = _score(champ, aggregate=0.9, systemic_failures=1)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=bad_score), LATER)
    assert state.status is CampaignStatus.NOT_CONVERGED


def test_systemic_confirm_not_qualified() -> None:
    """Confirm with systemic_failures > 0 fails gate."""
    ctrl = _control(
        model_tiers=(
            ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            ModelTier(name="large", model="qwen3:14b", digest=DIGEST_B),
        ),
    )
    state = _exhaust_stages(ctrl, _init(ctrl))
    champ = state.champion_fingerprint
    action = next_action(ctrl, state, LATER)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    action = next_action(ctrl, state, LATER)
    bad_score = _score(champ, aggregate=0.9, systemic_failures=1)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=bad_score), LATER)
    # Rolls to tier 1 (non-final)
    assert state.status is CampaignStatus.RUNNING
    assert state.tier_index == 1


# --- Qualification ---


def test_tier0_success_qualifies_campaign() -> None:
    ctrl = _control(
        model_tiers=(
            ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            ModelTier(name="large", model="qwen3:14b", digest=DIGEST_B),
        ),
        confirmation_runs=1,
    )
    state = _exhaust_stages(ctrl, _init(ctrl))
    champ = state.champion_fingerprint
    action = next_action(ctrl, state, LATER)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    action = next_action(ctrl, state, LATER)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    assert state.status is CampaignStatus.QUALIFIED
    assert state.tier_index == 0


def test_confirmation_runs_2() -> None:
    ctrl = _control(confirmation_runs=2)
    state = _exhaust_stages(ctrl, _init(ctrl))
    champ = state.champion_fingerprint
    action = next_action(ctrl, state, LATER)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    action = next_action(ctrl, state, LATER)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    assert state.status is CampaignStatus.RUNNING and state.confirmations_passed == 1
    action = next_action(ctrl, state, LATER)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=_qualifying_score(champ)), LATER)
    assert state.status is CampaignStatus.QUALIFIED


# --- Tier rollover ---


def test_milestone_failure_rolls_to_next_tier() -> None:
    ctrl = _control(
        model_tiers=(
            ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            ModelTier(name="large", model="qwen3:14b", digest=DIGEST_B),
        ),
    )
    state = _exhaust_stages(ctrl, _init(ctrl))
    champ = state.champion_fingerprint
    action = next_action(ctrl, state, LATER)
    bad_score = _score(champ, aggregate=0.9, hard_safety_failures=1)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=bad_score), LATER)
    assert state.status is CampaignStatus.RUNNING
    assert state.tier_index == 1


def test_final_tier_failure_not_converged() -> None:
    ctrl = _control(model_tiers=(ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),))
    state = _exhaust_stages(ctrl, _init(ctrl))
    champ = state.champion_fingerprint
    action = next_action(ctrl, state, LATER)
    bad_score = _score(champ, aggregate=0.9, pass_at_5=0.8)
    state = advance_state(ctrl, state, action, AttemptOutcome(kind="evidence", score=bad_score), LATER)
    assert state.status is CampaignStatus.NOT_CONVERGED


# --- Model identity ---


def test_model_identity_initial() -> None:
    ctrl = _control()
    state = _init(ctrl)
    assert state.model_identity.name == "small"
    assert state.model_identity.model == "qwen3:0.6b"
    assert state.model_identity.digest == DIGEST_A


def test_model_identity_rollover() -> None:
    """Tier rollover updates model_identity to next tier."""
    ctrl = _control(
        model_tiers=(
            ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            ModelTier(name="large", model="qwen3:14b", digest=DIGEST_B),
        ),
        stagnation_attempt_limit=1,
    )
    state = _init(ctrl)
    action = next_action(ctrl, state, NOW)
    assert action is not None
    outcome = AttemptOutcome(kind="evidence", score=_score(state.champion_fingerprint))
    state = advance_state(ctrl, state, action, outcome, LATER)
    # Rolled to tier 1
    assert state.tier_index == 1
    assert state.model_identity.name == "large"
    assert state.model_identity.model == "qwen3:14b"
    assert state.model_identity.digest == DIGEST_B


def test_model_identity_tamper_rejected() -> None:
    """Tampered model_identity fails validation on next_action/advance."""
    ctrl = _control()
    state = _init(ctrl)
    # Tamper model_identity
    tampered = CampaignState(
        schema_version=state.schema_version,
        campaign_id=state.campaign_id,
        prompt_lab_revision=state.prompt_lab_revision,
        korvid_revision=state.korvid_revision,
        status=state.status,
        tier_index=state.tier_index,
        stage_index=state.stage_index,
        seed_index=state.seed_index,
        champion_fingerprint=state.champion_fingerprint,
        champion_score=state.champion_score,
        model_identity=ModelIdentity(name="tampered", model="bad", digest=DIGEST_B),
        metric_calls_used=state.metric_calls_used,
        elapsed_seconds=state.elapsed_seconds,
        stagnation_attempts=state.stagnation_attempts,
        retries_used=state.retries_used,
        started_at=state.started_at,
    )
    with pytest.raises(ValueError, match="model_identity"):
        next_action(ctrl, tampered, NOW)


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
