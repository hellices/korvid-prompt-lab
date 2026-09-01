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
