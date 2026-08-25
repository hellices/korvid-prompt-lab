# Task 3 Report: Deterministic Campaign State Transitions

## Status: COMPLETE ✅

## Commits
- `ee923fc` — feat(campaigns): add bounded campaign state machine

## Files Modified
- `src/korvid_prompt_lab/campaigns.py` — Added state machine types and functions
- `tests/test_campaign_state.py` — 17 focused tests (new file)

## RED/GREEN Evidence

**RED phase:** Tests initially failed with `ImportError` for missing `CampaignStatus`, `ActionKind`, etc.

**GREEN phase:**
```
$ uv run --python 3.12 pytest tests/test_campaign_state.py tests/test_campaigns.py -q
54 passed in 0.26s

$ uv run --python 3.12 ruff check src/korvid_prompt_lab/campaigns.py tests/test_campaign_state.py
All checks passed!

$ uv run --python 3.12 mypy src/korvid_prompt_lab/campaigns.py
Success: no issues found in 1 source file
```

## State Invariants
1. `state_hash(state)` is deterministic SHA-256 over sorted-key compact JSON
2. `advance_state` rejects any action whose `expected_state_hash` doesn't match current state
3. SYSTEM_ERROR increments `retries_used` and `elapsed_seconds` only; never `metric_calls_used`
4. Promotion requires: different fingerprint, no hard-safety regression, no core regression, strictly better rank
5. Tie-break is full lexicographic fingerprint comparison (embedded in rank key tuple)
6. Milestone/confirmation evidence does not affect GEPA candidate ranking
7. Confirmation failure → NOT_CONVERGED (never publishes)
8. Terminal states: QUALIFIED, NOT_CONVERGED, SYSTEM_ERROR — `next_action` returns None

## Self-Review
- All brief requirements implemented
- Immutable frozen+slotted dataclasses throughout
- No network/filesystem/GitHub side effects in state machine

## Concerns
- Multi-tier re-entry (creating fresh state for tier 1+) left to orchestrator layer
- `confirmation_runs > 1` not explicitly iterated in tests

---

## Review Fix (2026-08-26)

### Status: DONE

### Commit
- `65d204c` — fix(campaigns): address review findings for state machine

### Findings Addressed
1. **Action validation**: `_validate_action()` verifies action_id, kind, tier/stage/seed cursors, metric_calls against `next_action()` output. Forged/replayed actions raise ValueError.
2. **confirmation_runs>1**: Loop until `confirmations_passed >= control.confirmation_runs`. Tested with confirmation_runs=2.
3. **Tier rollover**: Non-final tier qualification resets champion, stage/seed, stagnation, retries, milestone/confirmation to initial. Preserves metric_calls_used, elapsed. Records TierResult.
4. **CONFIG_ERROR**: Terminates as SYSTEM_ERROR with `stop_reason="config_error: ..."`.
5. **Equal-score no-promote**: `_is_strictly_better()` uses core dimensions only (no fingerprint). Equal scores → stagnation.
6. **metric_calls accounting**: Uses `action.metric_calls` (from stage definition), not +1. Tested exact increments.
7. **Wall-clock crossing on SYSTEM_ERROR**: Checked in system_error branch → terminal SYSTEM_ERROR.
8. **Tests replaced**: Removed equal-score promotion test, added forged-kind/id/cursor, stale replay, confirmation_runs=2, tier rollover, config_error, exact metric increments, wall-clock-crossing system error.

### RED/GREEN Evidence
```
$ uv run --python 3.12 pytest tests/test_campaign_state.py tests/test_campaigns.py -q
62 passed in 0.21s

$ uv run --python 3.12 ruff check src/korvid_prompt_lab/campaigns.py tests/test_campaign_state.py
All checks passed!

$ uv run --python 3.12 mypy src/korvid_prompt_lab/campaigns.py
Success: no issues found in 1 source file
```

### Self-Review
- All 8 findings resolved with test coverage
- Pure function boundary maintained (no global state, no side effects)
- Task 2 manifest APIs preserved (OptimizationCampaign, SearchStage, ModelTier, load_optimization_campaign unchanged)
- Immutable frozen+slotted dataclasses throughout

### Concerns
- None blocking. Milestone/confirm metric_calls set to 0 (free operations per the action's declared accounting contract). If the orchestrator needs non-zero cost for these, the action.metric_calls field is ready.

---

## Review Fix Wave 2 (2026-08-26)

### Status: DONE

### Commit
- `f416c32` — fix(campaigns): correct qualification gate, tier rollover, and budget semantics

### Corrected Semantics
1. **Qualification/tier**: Success on ANY tier → campaign QUALIFIED. Never rolls to larger model after success.
2. **Qualification gate**: `_passes_qualification_gate()` requires `hard_safety_failures=0`, `core_regression=False`, `pass_at_3=1.0`, `pass_at_5=1.0`. Milestone only sets `milestone_passed=True` when gate passes.
3. **Tier exhaustion**: Milestone fail or confirmation fail → `_handle_tier_failure()` records `TierResult(NOT_CONVERGED)`, rolls to next tier (fresh champion/stage/seed/stagnation/retry/milestone/confirm), preserves campaign-wide metric_calls/elapsed. Final tier → campaign NOT_CONVERGED.
4. **Budget accounting**: Uniform `new_metric_calls = state.metric_calls_used + action.metric_calls` before status check for ALL evidence kinds. Milestone/confirm with `metric_calls=0` add zero. Budget exceeded during milestone/confirm → NOT_CONVERGED.
5. **Replay semantics**: Pure deterministic `_validate_action` with CAS on `expected_state_hash`. No global mutable state. Test named `test_stale_action_replay_against_advanced_state` — two workers compute same transition, only one persisted state hash matches.
6. **New tests**: `test_milestone_gate_pass_at_3_failure_rejects`, `test_milestone_gate_hard_safety_rejects`, `test_qualification_tier0_success_qualifies_campaign`, `test_confirmation_failure_rolls_to_next_tier`, `test_milestone_failure_rolls_to_next_tier`, `test_final_tier_failure_not_converged`, `test_milestone_confirm_metric_accounting_uniform`, `test_wall_clock_crossing_during_milestone_terminates`, `test_wall_clock_crossing_during_confirm_terminates`.

### RED/GREEN Evidence
```
$ uv run --python 3.12 pytest tests/test_campaign_state.py tests/test_campaigns.py -q
67 passed in 0.19s

$ uv run --python 3.12 ruff check src/korvid_prompt_lab/campaigns.py tests/test_campaign_state.py
All checks passed!

$ uv run --python 3.12 mypy src/korvid_prompt_lab/campaigns.py
Success: no issues found in 1 source file
```

### Self-Review
- All 6 review findings addressed with exact test coverage
- Pure function boundary preserved — no global state, no side effects
- Task 2 manifest APIs unchanged (OptimizationCampaign/SearchStage/ModelTier/load_optimization_campaign)
- Tier rollover resets all tier-local state while preserving campaign-wide budget/time

### Concerns
- None blocking.

---

## Review Fix Wave 3 (2026-08-26)

### Status: DONE

### Commit
- `62d92d6` — fix(campaigns): add systemic_failures, evidence binding, stagnation rollover, model identity

### Changes
1. **systemic_failures**: New validated non-negative int field on CampaignScore. `__post_init__` rejects bool/negative. `_is_strictly_better()` blocks promotion when systemic>0. `_passes_qualification_gate()` requires systemic==0.
2. **Evidence fingerprint binding**: MILESTONE/CONFIRM `advance_state` raises ValueError if `outcome.score.fingerprint != state.champion_fingerprint`. Fail-closed, can never set milestone_passed/QUALIFIED.
3. **Stagnation rollover**: Stagnation at limit uses `_handle_tier_exhaustion()` (same helper as milestone/confirm failure). Non-final tier records TierResult(NOT_CONVERGED) and rolls; final tier ends campaign NOT_CONVERGED.
4. **ModelIdentity**: New frozen dataclass (name/model/digest). `initial_state` derives from `control.model_tiers[0]`. Rollover replaces from next tier. Included in state_hash. `_validate_model_identity` called in `next_action`/`_validate_action` — tampered identity raises ValueError.
5. **Tests**: 34 state-machine tests covering all new behaviors.

### RED/GREEN Evidence
```
$ uv run --python 3.12 pytest tests/test_campaign_state.py tests/test_campaigns.py -q
71 passed in 0.50s

$ uv run --python 3.12 ruff check src/korvid_prompt_lab/campaigns.py tests/test_campaign_state.py
All checks passed!

$ uv run --python 3.12 mypy src/korvid_prompt_lab/campaigns.py
Success: no issues found in 1 source file
```

### Self-Review
- All 5 findings resolved with exact test coverage
- Pure function boundary preserved
- Task 2 manifest APIs unchanged

### Concerns
- None blocking.
