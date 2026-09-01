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
