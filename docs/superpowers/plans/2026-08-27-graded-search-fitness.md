# Graded Search Fitness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve per-case GEPA fitness signals while keeping hard-safety qualification and publication strictly fail-closed.

**Architecture:** Add a GEPA-only lexicographic search score that prioritizes the number of safe cases and uses weighted grade quality to distinguish candidates at the same safety level. Retain hard-failure labels in reflection traces and leave strict scoring, full evaluation, campaign qualification, and publication decisions unchanged.

**Tech Stack:** Python 3.12, GEPA, pytest, Ruff, mypy

## Global Constraints

- Any hard-safety failure must continue to block qualification and publication.
- Systemic bridge failures remain errors rather than scores.
- Evidence schemas and immutable lineage rules remain unchanged.
- The live conclusion is accepted only if GEPA compares more than the seed or demonstrates a non-flat candidate score.

---

### Task 1: Preserve Per-Case Search Scores

**Files:**
- Modify: `src/korvid_prompt_lab/adapter.py:73-105`
- Test: `tests/test_adapter.py`

**Interfaces:**
- Consumes: `score_result(result: BridgeResult) -> ScoreResult`
- Produces: `KorvidGEPAAdapter.evaluate(...) -> EvaluationBatch` with bounded lexicographic search scores

- [ ] **Step 1: Write the failing regression test**

Add tests that supply unsafe results with different weighted grade quality and
assert they receive different positive GEPA search scores. Add a boundary test
proving one safe case with minimum quality outranks an all-unsafe batch with
maximum quality. Assert unsafe traces keep their hard-failure labels.

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
uv run pytest tests/test_adapter.py -k unsafe -q
```

Expected: the new tests fail because all unsafe scores are overwritten to zero.

- [ ] **Step 3: Implement the minimal fix**

Calculate weighted grade quality without changing `score_result`. Return
`0.75 + 0.25 * quality` for safe completed cases,
`2 ** (-len(hard_failures)) * (0.75 + 0.25 * quality)` for unsafe completed
cases, and zero for model failures. Delete the batch-wide zeroing block. Do not
change strict scoring, publication, or error handling.

- [ ] **Step 4: Run focused verification**

Run:

```bash
uv run pytest tests/test_adapter.py tests/test_scoring.py tests/test_publish.py tests/test_campaigns.py -q
uv run ruff check src/korvid_prompt_lab/adapter.py tests/test_adapter.py
uv run mypy src
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 5: Run full verification and review**

Run:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
git diff --check
```

Expected: all commands exit zero. Request an independent code review focused
on safety-gate preservation and GEPA search semantics.

- [ ] **Step 6: Commit and merge**

Commit the implementation and tests using the repository's conventional
commit style, create a PR, and merge only after review.

### Task 2: Validate With a Fresh Live Lineage

**Files:**
- Modify: `examples/optimization-campaigns/qwen3-small-operator.yaml`
- Modify: manifest-backed campaign fixture expectations

**Interfaces:**
- Consumes: merged Prompt Lab SHA and pinned Korvid SHA
- Produces: immutable safe campaign and round evidence

- [ ] **Step 1: Version the campaign identity**

Change only the production campaign ID and corresponding fixture/lineage
expectations from v5 to v6.

- [ ] **Step 2: Verify, review, commit, and merge**

Run the focused campaign tests and full repository verification, obtain an
independent review, then commit and merge the lineage-only PR.

- [ ] **Step 3: Execute the bounded campaign**

Dispatch from `main` using the exact merged Prompt Lab SHA and pinned Korvid
SHA `62bd3cbee2e27369bb81abc0957dae341c2aa434`. Approve each protected
`aks-grounding` action while the persisted state remains `RUNNING`.

- [ ] **Step 4: Validate the search signal**

Inspect safe round evidence. Require `optimization-summary.json` to report
either `num_candidates > 1` or a candidate comparison with a non-zero score
delta. Confirm hard-safety failures still block campaign qualification.

- [ ] **Step 5: Finish and clean up**

Continue until `QUALIFIED` or bounded `NOT_CONVERGED`, audit `modeleval` at
count zero with successful provisioning, preserve safe evidence, and remove
temporary environments.
