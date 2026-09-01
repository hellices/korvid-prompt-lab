# Task 4 Report: Stable Search Orchestrator

## Status

Complete.

## Commit

- `feat(search): orchestrate stable prompt qualification`

## RED / GREEN Evidence

1. Wrote `tests/test_stable_search.py` before the module existed.
2. RED: `uv run --python 3.12 pytest -q tests/test_stable_search.py`
   failed with `ModuleNotFoundError: No module named 'korvid_prompt_lab.stable_search'`.
3. Implemented `src/korvid_prompt_lab/stable_search.py` with staged search orchestration.
4. Re-ran the focused suite and fixed two real integration defects discovered by the tests:
   - import-time `NameError` from constructing the default config before helper definitions were available;
   - request artifacts persisted under stage runs, violating the normalized-artifacts-only requirement.
5. GREEN: `uv run --python 3.12 pytest -q tests/test_stable_search.py`
   finished with `3 passed`.

## What landed

- Added immutable `StableSearchConfig` and `StableSearchArtifacts`.
- Added `run_stable_search(...)` over the existing `KorvidRunner`,
  `StructuredCandidate`, `ScenarioManifest`, and stable-ranking interfaces.
- Implemented Stage A screening (`1` repetition, top `3` survivors), Stage B
  repeated validation (`3` repetitions, top `2` finalists), and Stage C
  qualification (`5` repetitions on validation + milestone for baseline and
  finalists).
- Enforced early stop on hard-safety/systemic candidate evidence.
- Refused to reuse an existing artifact root.
- Removed per-run `request.json` artifacts after each runner invocation so the
  campaign persists only normalized `response.json` evidence plus normalized
  summary/manifests.

## Verification

```bash
uv run --python 3.12 pytest -q tests/test_stable_candidates.py tests/test_stable_scenarios.py tests/test_stable_ranking.py tests/test_stable_search.py
uv run --python 3.12 ruff check src/korvid_prompt_lab/stable_candidates.py src/korvid_prompt_lab/stable_scenarios.py src/korvid_prompt_lab/stable_ranking.py src/korvid_prompt_lab/stable_search.py tests/test_stable_candidates.py tests/test_stable_scenarios.py tests/test_stable_ranking.py tests/test_stable_search.py
uv run --python 3.12 mypy --python-version 3.12 src/korvid_prompt_lab/stable_candidates.py src/korvid_prompt_lab/stable_scenarios.py src/korvid_prompt_lab/stable_ranking.py src/korvid_prompt_lab/stable_search.py tests/test_stable_candidates.py tests/test_stable_scenarios.py tests/test_stable_ranking.py tests/test_stable_search.py
```

Latest results:

- `pytest`: `29 passed`
- `ruff`: `All checks passed!`
- `mypy`: `Success: no issues found in 8 source files`

## Self-review

- Verified the fake-runner winner path promotes `evidence-first` and the flat
  path records `no_stable_winner`.
- Verified Stage A safety/systemic candidates run once, stop early, and never
  reach later stages.
- Verified Stage B and Stage C repetition counts match the required `1/3/5`
  progression.
- Verified no `request.json` artifacts survive anywhere under the campaign root.

## Concerns

- None.

---

## Review Fix (2026-09-01)

### Status

Complete.

### Findings addressed

1. Systemic runner failures now abort the current stage/search instead of being
   recorded as candidate rejections.
2. Stable search now persists only orchestrator-owned normalized/redacted run
   artifacts; runner request/raw response artifacts are removed.
3. Candidate and case IDs are slugged before filesystem use so traversal and
   metacharacters cannot escape the artifact root.

### RED → GREEN evidence

- Added a real-shaped `BridgeProcessExitError` regression that proves the error
  propagates and no stage/search success summary is written.
- Added a raw-response regression using traversal-heavy candidate/case IDs and a
  fake process-style response containing known raw answer/error strings.
- RED: `uv run --python 3.12 pytest -q tests/test_stable_search.py` -> `1 failed, 4 passed`
  (`LEAKED_RAW_ANSWER` persisted, and unsanitized run paths escaped the campaign root).
- GREEN: `uv run --python 3.12 pytest -q tests/test_stable_search.py` -> `5 passed`.

### Implementation notes

- Runner calls now execute in a private `_runner/` tree using sanitized path
  segments.
- After every successful run, stable search writes its own redacted
  `response.json` containing blank `answer`, generic model-failure labeling,
  normalized record data, and bounded grade/journal/usage projections only.
- After every runner invocation, private runner artifacts are removed and empty
  `_runner/` directories are pruned.
- Any systemic `BridgeResult.status` is treated as a `BridgeStatusError`, which
  aborts the stage/search instead of feeding ranking.

### Verification

```bash
uv run --python 3.12 pytest -q tests/test_stable_candidates.py tests/test_stable_scenarios.py tests/test_stable_ranking.py tests/test_stable_search.py
uv run --python 3.12 ruff check src/korvid_prompt_lab/stable_candidates.py src/korvid_prompt_lab/stable_scenarios.py src/korvid_prompt_lab/stable_ranking.py src/korvid_prompt_lab/stable_search.py tests/test_stable_candidates.py tests/test_stable_scenarios.py tests/test_stable_ranking.py tests/test_stable_search.py
uv run --python 3.12 mypy --python-version 3.12 src/korvid_prompt_lab/stable_candidates.py src/korvid_prompt_lab/stable_scenarios.py src/korvid_prompt_lab/stable_ranking.py src/korvid_prompt_lab/stable_search.py tests/test_stable_candidates.py tests/test_stable_scenarios.py tests/test_stable_ranking.py tests/test_stable_search.py
```

Latest results at report update:

- `pytest`: `31 passed`
- `ruff`: `All checks passed!`
- `mypy`: `Success: no issues found in 8 source files`

### Self-review

- Verified the winner-path test no longer models systemic failure as a candidate
  rejection.
- Verified redacted run artifacts exclude the planted raw answer, raw error,
  raw request prompt, and sensitive journal field.
- Verified all persisted `response.json` paths stay under the campaign root even
  with traversal-heavy IDs.

### Concerns

- None.

---

## Live Collapse Fix (2026-09-01)

### Status

Complete.

### Live symptom

- Stage A had completed signal, but a later serving collapse turned Stage B
  baseline + all candidates into `model_failure` with zero completed runs.
- The orchestrator incorrectly returned `no_stable_winner` instead of treating
  the stage/campaign as systemic failure.

### RED → GREEN evidence

- Added a regression proving Stage B all-model-failure collapse raises a typed
  `StableSearchSystemError`, writes bounded failure summaries, and never writes a
  success decision.
- Added a mixed-stage regression proving genuine completed + `model_failure`
  evidence still ranks normally and can still promote a winner.
- RED: `uv run --python 3.12 pytest -q tests/test_stable_search.py -k 'stage_b_collapses or mixed_completed'`
  -> `2 failed` before the fix.
- GREEN: the same focused regression command -> `2 passed`.

### Implementation notes

- Added `StableSearchSystemError` as a typed `BridgeSystemError` subclass for
  systemic stable-search collapse reporting.
- Stable search now detects the stage-level condition “baseline and every
  evaluated candidate have zero completed runs” before ranking.
- On collapse, the stage summary is persisted as a bounded `system_error`
  artifact and the campaign root summary is also written as `system_error`
  without any raw runner error text.
- Mixed `completed` / `model_failure` stages still flow through existing ranking
  logic unchanged.

### Verification

```bash
uv run --python 3.12 pytest -q tests/test_stable_candidates.py tests/test_stable_scenarios.py tests/test_stable_ranking.py tests/test_stable_search.py
uv run --python 3.12 ruff check src/korvid_prompt_lab/stable_candidates.py src/korvid_prompt_lab/stable_scenarios.py src/korvid_prompt_lab/stable_ranking.py src/korvid_prompt_lab/stable_search.py tests/test_stable_candidates.py tests/test_stable_scenarios.py tests/test_stable_ranking.py tests/test_stable_search.py
uv run --python 3.12 mypy --python-version 3.12 src/korvid_prompt_lab/stable_candidates.py src/korvid_prompt_lab/stable_scenarios.py src/korvid_prompt_lab/stable_ranking.py src/korvid_prompt_lab/stable_search.py tests/test_stable_candidates.py tests/test_stable_scenarios.py tests/test_stable_ranking.py tests/test_stable_search.py
```

Latest results at report update:

- `pytest`: `39 passed`
- `ruff`: `All checks passed!`
- `mypy`: `Success: no issues found in 8 source files`

### Self-review

- Verified Stage B collapse now raises `StableSearchSystemError` after Stage A
  success, instead of degrading to `no_stable_winner`.
- Verified both stage and root summaries record only bounded
  `serving_collapse_all_model_failure` status and omit raw runner error/answer
  content.
- Verified mixed completed/model-failure validation evidence still promotes the
  stronger candidate under the existing ranking rules.

### Concerns

- None.
