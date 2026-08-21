# Task 1 Report

## Status
DONE

## Files
- `pyproject.toml`
- `src/korvid_prompt_lab/__init__.py`
- `src/korvid_prompt_lab/contracts.py`
- `src/korvid_prompt_lab/config.py`
- `tests/test_contracts.py`
- `examples/candidates/shipped-small.yaml`
- `examples/campaigns/local-smoke.yaml`
- `examples/campaigns/aks-shared-runners.yaml`

## Commands and results
- `pytest tests/test_contracts.py -q`
  - RED: failed with `ModuleNotFoundError: No module named 'korvid_prompt_lab'`
- `pytest tests/test_contracts.py -q`
  - GREEN: `9 passed in 0.06s`

## Self-review
- Verified strict unknown-field rejection at candidate, case, and serving boundaries.
- Verified deterministic candidate fingerprints across mapping order.
- Verified duplicate case IDs, invalid repetitions, and incomplete model coverage fail.
- Verified the AKS example resolves namespace/service/model from environment variables.

## Commit
- `023ca40` — `feat: add prompt lab contracts`

## Concerns
- The local interpreter is Python 3.9, so I could not use `dataclass(slots=True)` literally; the implementation stays frozen/immutable and passes the required tests.
- The declared console script points to the future `korvid_prompt_lab.cli:main` module, which is not part of Task 1.

## Post-fix validation
- `uv run --python /Users/hwang-inhwan/.local/bin/python3.12 pytest tests/test_contracts.py -q -k slots`
  - RED: failed with `assert not hasattr(candidate, "__dict__")`
- `uv sync --python /Users/hwang-inhwan/.local/bin/python3.12 --extra dev`
  - installed `pytest`, `mypy`, and `ruff` into the Python 3.12 environment
- `uv run --python /Users/hwang-inhwan/.local/bin/python3.12 python -m pytest tests/test_contracts.py -q`
  - GREEN: `10 passed in 0.11s`

## Fixed commit
- `da034c4` — `fix: add dataclass slots`

## Whitespace env fix
- Commit: `5e708d6568c122156e12aec9654cbca5d65a049d`
- Test: `uv run --python /Users/hwang-inhwan/.local/bin/python3.12 python -m pytest tests/test_contracts.py -q`
  - Result: `11 passed in 0.07s`
