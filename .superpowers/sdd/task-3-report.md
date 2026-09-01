# Task 3 Report: Stable Ranking and Qualification

## Status

Complete.

## Commits

- `de4912d` — `feat(search): add stable winner qualification`

## Summary

Implemented the pure stable-ranking layer in
`src/korvid_prompt_lab/stable_ranking.py` and covered it with focused TDD in
`tests/test_stable_ranking.py`.

## Files

- Added `src/korvid_prompt_lab/stable_ranking.py`
- Added `tests/test_stable_ranking.py`

## Implemented interfaces

- `NormalizedRunRecord`
- `CandidateMeasurement`
- `RankedCandidate`
- `StageDecision`
- `QualificationCandidate`
- `QualificationAssessment`
- `QualificationDecision`
- `measure_candidate(...)`
- `rank_screening(...)`
- `select_finalists(...)`
- `qualify_winner(...)`

## TDD evidence

1. Wrote `tests/test_stable_ranking.py` before the module existed.
2. Ran `uv run --python 3.12 pytest -q tests/test_stable_ranking.py`.
3. Observed RED: `ModuleNotFoundError: No module named 'korvid_prompt_lab.stable_ranking'`.
4. Implemented the minimal pure module.
5. Re-ran the focused suite, fixed three real failures:
   - population variance expectation mismatch;
   - baseline worst-case fixture needed to match the validation gate intent;
   - exact `0.10` threshold needed float-safe comparison.
6. Re-ran the focused suite to GREEN.

## Behavior covered

- Stage A rejects hard-safety and systemic failures.
- Stage A ranks by mean delta, worst-case delta, verification delta, then fewer
  malformed/unresolvable tool-call problems.
- `measure_candidate(...)` aggregates normalized per-run records into the exact
  summary Task 4 needs.
- Stage B rejects non-positive mean delta, worst-case regression, and any
  safety/systemic failure.
- Stage B tie-breaks on lower variance, then higher `pass_at_3`.
- Qualification requires five repetitions per case on all paired measurements.
- Qualification rejects any worst-case regression.
- Qualification accepts the exact `+0.10` mean-delta boundary.
- Qualification reports explicit `no_stable_winner` reasons, including
  per-finalist reason prefixes when multiple finalists fail.

## Purity / dependency review

Self-review confirmed the new module imports only stdlib modules:
`math`, `collections`, `collections.abc`, `dataclasses`, `statistics`, and
`typing`. It does not import filesystem helpers, runners, AKS code, or DSPy.

## Verification

Focused commands executed successfully:

```bash
uv run --python 3.12 pytest -q tests/test_stable_ranking.py
uv run --python 3.12 ruff check src/korvid_prompt_lab/stable_ranking.py tests/test_stable_ranking.py
uv run --python 3.12 mypy --python-version 3.12 src/korvid_prompt_lab/stable_ranking.py tests/test_stable_ranking.py
```

Latest results:

- `pytest`: `11 passed in 0.06s`
- `ruff`: `All checks passed!`
- `mypy`: `Success: no issues found in 2 source files`

## Self-review notes

- The qualification API supports both a single paired candidate call and the
  ranked-finalists sequence Task 4 needs.
- `QualificationDecision.assessments` preserves ordered evidence for artifact
  writing even when there is no winner.
- The exact `0.10` boundary uses a tiny absolute tolerance so decimal
  round-off does not misclassify a true boundary pass.

## Concerns

- The module currently treats any normalized status other than `completed` and
  `model_failure` as systemic evidence. That matches the design’s fail-closed
  intent, but Task 4 should keep feeding canonical normalized statuses only.
