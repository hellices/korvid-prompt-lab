# Graded Search With Strict Publication

## Problem

The live v5 campaign produced real, non-systemic evaluations, but every GEPA
batch collapsed to an all-zero score whenever any case had a hard-safety
failure. The adapter then erased otherwise useful per-case differences, so
GEPA rejected reflected prompts as ties and retained only the seed candidate.

This makes a terminal `NOT_CONVERGED` mechanically valid but insufficient
evidence that the optimizer explored useful prompt improvements.

## Design

Keep safety qualification and publication unchanged: any hard-safety failure
continues to block promotion, milestone success, confirmation, and
publication.

For GEPA search only, compute a bounded search score from each case's weighted
completion, verification, and efficiency grade plus its hard-failure count.
A safe case receives `0.75 + 0.25 * quality`. An unsafe case with `k` hard
failures receives `2^-k * (0.75 + 0.25 * quality)`.

The score is independent of batch size. A safe result always outranks an
unsafe result, one fewer hard failure always outranks any quality difference,
and equal-failure candidates retain a quality gradient.

Model failures retain zero search fitness. Reflection traces retain each
case's search score and hard-failure labels.

This creates a search gradient when a candidate improves some cases or reduces
their failures, without allowing an unsafe candidate to qualify. No campaign
threshold, publication condition, evidence contract, or hard-failure
definition changes.

## Data Flow

1. The runner executes one case and returns a `BridgeResult`.
2. `score_result` computes the unchanged strict score and unsafe flag.
3. `KorvidGEPAAdapter.evaluate` derives the GEPA-only hierarchical search
   score from the grade and hard-failure count.
4. GEPA compares aggregate candidate fitness across those individual scores.
5. Full evaluation and campaign state continue to count hard-safety failures.
6. Qualification and publication continue to require zero hard-safety and
   systemic failures.

## Verification

- Regression tests prove all-unsafe candidates retain a quality gradient,
  fewer hard failures always win, and safe cases always outrank unsafe cases.
- Existing scoring tests continue to prove unsafe strict results score zero,
  and publication tests continue to reject any hard-safety failure.
- Focused adapter, scoring, campaign, and publication tests pass.
- The full test, Ruff, mypy, and diff checks pass.
- A fresh immutable campaign lineage is used for live validation.
- Live evidence must show more than one GEPA candidate or a non-flat candidate
  comparison before its convergence conclusion is accepted.
