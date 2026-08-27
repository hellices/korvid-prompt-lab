# Persisted Search Incumbent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the first measured and subsequently improved search candidate across campaign actions without weakening qualification or publication.

**Architecture:** Recognize the first SEARCH result from immutable state cursor/accounting fields and use its validated comparison outcome to establish an improved incumbent or record an unchanged seed baseline. Reuse existing score ordering thereafter; keep milestone, confirmation, and publication gates unchanged.

**Tech Stack:** Python 3.12, pytest, GitHub Actions, YAML

## Global Constraints

- Systemic and core-regression outcomes never become search incumbents.
- Unsafe incumbents may seed search but may never qualify or publish.
- Existing immutable state hashing and evidence validation remain fail-closed.
- No state schema migration is introduced.

---

### Task 1: Establish the First Measured Incumbent

**Files:**
- Modify: `src/korvid_prompt_lab/campaigns.py`
- Test: `tests/test_campaign_state.py`

**Interfaces:**
- Consumes: `advance_state(..., AttemptOutcome(kind="evidence", score=...))`
- Produces: the first non-systemic, non-regressed SEARCH result as `CampaignState.champion_score`

- [ ] **Step 1: Write failing tests**

Add tests proving that the first improved unsafe candidate replaces the
synthetic seed score and that an unchanged same-fingerprint seed records its
real score while counting stagnation.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_campaign_state.py -k "first_search" -q
```

Expected: both tests fail under the synthetic zero-safety baseline.

- [ ] **Step 3: Implement the first-search predicate**

Add a private predicate over tier, stage, seed, metric calls, stagnation, and
seed identity. Carry the validated comparison outcome through `RoundOutcome`
and `AttemptOutcome`. Establish a new first incumbent only for `improved`, and
record an unchanged seed baseline without resetting stagnation. Keep
`_is_strictly_better` unchanged for every subsequent result.

- [ ] **Step 4: Verify search ordering and strict gates**

Run:

```bash
uv run pytest tests/test_campaign_state.py tests/test_campaigns.py tests/test_publish.py -q
```

Expected: all tests pass, including systemic/core regression rejection and
strict qualification tests.

### Task 2: Verify Workflow Carry-Forward

**Files:**
- Test: `tests/test_optimization_campaign_workflow.py`
- Test: `tests/test_safe_round_package.py`

**Interfaces:**
- Consumes: `CampaignState.champion_fingerprint`
- Produces: matching persisted `champion-candidate.yaml` for the next action

- [ ] **Step 1: Add or strengthen contract assertions**

Assert that the workflow selects the round best candidate when its fingerprint
matches the new incumbent and that the next action validates the candidate
against the state incumbent.

- [ ] **Step 2: Run focused and full verification**

Run:

```bash
uv run pytest tests/test_campaign_state.py tests/test_campaigns.py tests/test_optimization_campaign_workflow.py tests/test_safe_round_package.py tests/test_publish.py -q
uv run pytest
uv run ruff check .
uv run mypy src
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 3: Review, commit, and merge**

Request independent review focused on unsafe artifact containment and strict
publication. Commit and merge only after approval.

### Task 3: Fresh Live Validation

**Files:**
- Modify: `examples/optimization-campaigns/qwen3-small-operator.yaml`
- Modify: manifest-backed fixture expectations

**Interfaces:**
- Consumes: merged Prompt Lab SHA and pinned Korvid revision
- Produces: immutable evidence showing the improved incumbent seeds action two

- [ ] **Step 1: Create a new immutable campaign lineage**

Version only the campaign ID and its exact fixture/marker expectations.

- [ ] **Step 2: Execute two bounded actions**

Approve the first action, inspect its improved incumbent, then approve the
second action only if its input candidate fingerprint equals the first action's
persisted incumbent.

- [ ] **Step 3: Stop before obsolete-eval expansion**

Use this campaign only to prove carry-forward. Do not spend the full budget
against the old Korvid harness; proceed next to the versioned Eval Protocol
work.
