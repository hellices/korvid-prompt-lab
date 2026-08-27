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

For GEPA search only, preserve the score returned for each individual case.
An unsafe case remains scored according to the existing `score_result`
contract, while its presence no longer overwrites scores for unrelated cases
in the same minibatch. Reflection traces retain each case's actual score and
hard-failure labels.

This creates a search gradient when a candidate improves some cases or reduces
their failures, without allowing an unsafe candidate to qualify. No campaign
threshold, publication condition, evidence contract, or hard-failure
definition changes.

## Data Flow

1. The runner executes one case and returns a `BridgeResult`.
2. `score_result` computes that case's score and unsafe flag.
3. `KorvidGEPAAdapter.evaluate` appends the individual score and trace.
4. GEPA compares aggregate candidate fitness across those individual scores.
5. Full evaluation and campaign state continue to count hard-safety failures.
6. Qualification and publication continue to require zero hard-safety and
   systemic failures.

## Verification

- A regression test proves one unsafe result does not erase a safe sibling's
  score.
- Existing tests continue to prove unsafe results themselves score zero and
  strict publication rejects any hard-safety failure.
- Focused adapter, scoring, campaign, and publication tests pass.
- The full test, Ruff, mypy, and diff checks pass.
- A fresh immutable campaign lineage is used for live validation.
- Live evidence must show more than one GEPA candidate or a non-flat candidate
  comparison before its convergence conclusion is accepted.
