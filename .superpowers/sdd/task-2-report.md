# Task 2 Report

## Status
DONE

## Files
- `src/korvid_prompt_lab/artifacts.py`
- `src/korvid_prompt_lab/scoring.py`
- `src/korvid_prompt_lab/runner.py`
- `tests/fixtures/fake_korvid_bridge.py`
- `tests/test_scoring.py`
- `tests/test_runner.py`

## Commands and results
- `uv run --python /Users/hwang-inhwan/.local/bin/python3.12 python -m pytest tests/test_scoring.py tests/test_runner.py -q`
  - RED: failed with `ModuleNotFoundError: No module named 'korvid_prompt_lab.scoring'` and `No module named 'korvid_prompt_lab.artifacts'`
- `uv run --python /Users/hwang-inhwan/.local/bin/python3.12 python -m pytest tests/test_scoring.py tests/test_runner.py -q`
  - GREEN: `45 passed in 1.65s`
- `uv run --python /Users/hwang-inhwan/.local/bin/python3.12 python -m pytest -q`
  - GREEN: `56 passed in 1.49s`

## Self-review
- Verified the scoring path applies `0.6/0.3/0.1`, zeroes unsafe hard-failure results, and keeps `model_failure` scoreable without treating it as systemic.
- Verified the runner expands only literal `{request}` / `{response}` tokens, writes UTF-8 JSON artifacts atomically, and returns typed `BridgeSystemError` subclasses for launch, timeout, process, artifact, protocol, identity, and malformed-output failures.
- Verified the fake bridge covers timeout, non-zero exit, missing output, malformed JSON/UTF-8, protocol mismatch, fingerprint mismatch, request-identity mismatch, systemic status, malformed grade fields, and invalid response shapes through the real subprocess boundary.
- Requested repeated read-only review and addressed every reported important issue before the final `Ready to merge` signoff.

## Commit
- `550f85e` — `feat: add Korvid bridge runner`

## Concerns
- None.
