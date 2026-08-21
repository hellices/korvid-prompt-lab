# Task 3 Report

## Status
DONE

## Files
- `src/korvid_prompt_lab/adapter.py`
- `src/korvid_prompt_lab/reflection.py`
- `src/korvid_prompt_lab/optimize.py`
- `tests/test_adapter.py`
- `tests/test_reflection.py`
- `tests/test_optimize.py`
- `.superpowers/sdd/task-3-report.md`

## Confirmed external APIs
- `uv run --python 3.12 python` confirmed `gepa.optimize(...)` is installed and accepts `adapter`, `seed_candidate`, `trainset`, `valset`, `custom_candidate_proposer`, `max_metric_calls`, and `run_dir`.
- `uv run --python 3.12 python` confirmed `dspy` 3.3.0 exposes `Signature`, `Predict`, `InputField`, and `OutputField`.
- Implementation uses `KorvidProcessRunner.run(...)` for runtime execution and reserves DSPy for proposal generation only.

## Commands and results
- `uv run --python 3.12 pytest tests/test_adapter.py tests/test_reflection.py tests/test_optimize.py -q`
  - RED: failed with `ModuleNotFoundError` for `korvid_prompt_lab.adapter`, `korvid_prompt_lab.reflection`, and `korvid_prompt_lab.optimize`.
- `uv run --python 3.12 pytest tests/test_adapter.py tests/test_reflection.py tests/test_optimize.py -q`
  - GREEN: `8 passed in 2.63s`
- `uv run --python 3.12 pytest -q`
  - GREEN: `65 passed in 4.57s`

## Implementation summary
- Added `KorvidGEPAAdapter.evaluate(...)` to materialize strict candidates from component maps, invoke the real process runner once per case, derive GEPA scores through existing scoring rules, and emit safe typed traces only when requested.
- Added `KorvidGEPAAdapter.make_reflective_dataset(...)` to serialize compact JSON-safe reflection records containing case identity, answer, checkpoints, tool-call counts, outcome, missing checkpoints, hard failures, and score while excluding raw journal payloads and sensitive tool results.
- Added a lazy DSPy proposer that builds a `dspy.Predict` instance only on first reflection use, serializes per-component reflection datasets to JSON, and rejects unknown component requests and blank rewritten text.
- Added `optimize_campaign(...)` orchestration that wires the adapter into `gepa.optimize`, optionally attaches the DSPy proposer, persists the best strict candidate YAML, and writes an optimization summary JSON.

## Self-review
- Verified unsafe bridge results score `0.0`, retain hard-failure feedback for reflection, and never leak raw tool outputs or audit payloads into reflection datasets.
- Verified systemic runner failures still raise their typed exceptions instead of being flattened into bad-example scores.
- Verified candidate component maps are copied before validation/evaluation so GEPA-owned dictionaries are not mutated in place.
- Verified optimization persistence revalidates the best candidate through the strict `Candidate` contract before writing YAML.

## Commit
- `7b65272` — `feat: add GEPA optimization adapter` (with required Co-authored-by trailer).

## Concerns
- None.

## Task 3 regression fix
- Fixed `KorvidGEPAAdapter.evaluate(...)` so any unsafe case zeros the entire returned `scores` vector, while preserving one output and one trajectory per input case.
- Regression test added: `test_adapter_zeroes_all_scores_when_any_case_is_unsafe`.

## Verification
- `uv run --python 3.12 pytest tests/test_adapter.py -q -k 'zeroes_all_scores_when_any_case_is_unsafe'`
  - PASS: `1 passed, 3 deselected in 0.38s`
- `uv run --python 3.12 pytest tests/test_adapter.py tests/test_reflection.py tests/test_optimize.py -q`
  - PASS: `9 passed in 3.42s`
- `uv run --python 3.12 pytest -q`
  - PASS: `66 passed in 3.00s`

## Fix commit
- `648705bb6b9417a54bc0d035ba629981da7afe24` — `fix: zero unsafe batch scores`
