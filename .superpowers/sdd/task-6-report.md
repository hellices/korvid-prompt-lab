# Task 6 Report

## Status
- implemented `korvid-prompt-lab stable-search`
- wired installed `small` baseline materialization, scenario-manifest discovery, structured candidate matrix, `KorvidReadonlyRunner`, staged search orchestration, and bounded proposer extension reporting
- added example stable-search config and README commands/documentation, including the `306` run upper bound and `promote` / `no_stable_winner` semantics

## Changed Files
- `src/korvid_prompt_lab/cli.py`
- `src/korvid_prompt_lab/stable_search.py`
- `tests/test_stable_search_cli.py`
- `tests/test_stable_search.py`
- `tests/test_stable_proposer.py`
- `examples/stable-search/korvid-small.yaml`
- `README.md`

## Verification
- `uv run --python 3.12 pytest -q tests/test_stable_candidates.py tests/test_stable_scenarios.py tests/test_stable_ranking.py tests/test_stable_proposer.py tests/test_stable_search.py tests/test_stable_search_cli.py tests/test_korvid_readonly.py tests/test_cli.py`
  - result: `213 passed`
- `uv run --python 3.12 ruff check .`
  - result: passed
- `uv run --python 3.12 mypy src tests`
  - result: passed
- `git diff --check`
  - result: passed

## Self-Review
- confirmed the CLI refuses an existing artifact root before any env-dependent runtime work
- confirmed the readonly campaign is built from installed Korvid scenarios and env-only `KORVID_READONLY_BASE_URL`
- confirmed optional proposer wiring is explicitly represented in orchestrator `extension` artifacts after a Stage B structured signal, without changing the structured winner decision path

## Commit
- HEAD feat(search): expose stable prompt campaign

## Concerns
- bounded proposer output is recorded as an extension artifact only; it is not yet replayed through additional target-model Stage B/C runs, which preserves the documented `306` upper bound and matches the explicit extension-path requirement

## Review Fix Round (2026-09-01)
- caught propagated proposer provider/configuration errors at the stable-search orchestrator boundary and record sanitized `error_label` values in `extension`
- when the optional proposer returns a revised append, Prompt Lab now materializes an append-only candidate from the baseline system/metadata, replays the same Stage B validation gate, and only then runs Stage C qualification
- proposer-enabled documentation now distinguishes the structured `306`-run bound from the `384`-run upper bound when one bounded replay candidate is attempted

### Additional Verification
- `uv run --python 3.12 pytest -q tests/test_stable_candidates.py tests/test_stable_scenarios.py tests/test_stable_ranking.py tests/test_stable_proposer.py tests/test_stable_search.py tests/test_stable_search_cli.py tests/test_korvid_readonly.py tests/test_cli.py`
  - result: `219 passed`
- `uv run --python 3.12 ruff check .`
  - result: passed
- `uv run --python 3.12 mypy src tests`
  - result: passed

### Additional Self-Review
- verified proposer authentication / bad-request / API errors no longer abort a completed structured evaluation
- verified a proposer candidate can be rejected at replayed Stage B without entering Stage C
- verified a proposer candidate can become the final promoted winner only by passing replayed Stage B and Stage C gates

### Updated Concerns
- no additional functional concerns; proposer-enabled runs now have a higher documented target-model upper bound than structured-only runs

## Rework Round (2026-09-01 23:56 JST)
- reworked proposer integration so proposer-enabled searches still cap at the exact `306` target-model calls
- Stage B now keeps the normal baseline measurement, evaluates only the top two structured candidates up front, and can add at most one proposed candidate on the same validation split before selecting the final top two finalists
- Stage C is now a single integrated qualification flow under `stage-c`; the separate proposer qualification branch and duplicate baseline replay were removed
- restored README/example/help text to the exact `306` upper bound and clarified that the proposer attempts at most one proposal total

### Rework Verification
- `uv run --python 3.12 pytest -q tests/test_stable_search.py tests/test_stable_search_cli.py`
  - result: `17 passed`

### Rework Self-Review
- verified the maximum-path proposer regression covers 8 structured candidates, 18 cases, and exactly `306` runner calls
- verified proposer validation runs land under the integrated `stage-b` directory and no `stage-c-proposer` directory is created
