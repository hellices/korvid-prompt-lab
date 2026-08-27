# Persisted Search Incumbent Design

## Problem

The initial campaign state assigns the unevaluated seed a synthetic score with
zero hard-safety failures. Search promotion then compares real candidates
against that synthetic score. A candidate that improves observed failures from
25 to 20 cannot replace the seed because any nonzero hard-failure count ranks
below the synthetic zero.

The next campaign action consequently restarts from the original seed and
cannot accumulate partial prompt improvements.

## Design

Treat the existing campaign `champion` as the search incumbent. Safe comparison
evidence already classifies each SEARCH as improved, unchanged, or regressed.
The first improved, non-systemic SEARCH outcome establishes a different
partially unsafe candidate as the incumbent and resets stagnation. An unchanged
first SEARCH records the seed's real score and increments stagnation.

Subsequent SEARCH outcomes use the existing strict ordering:

1. systemic outcomes never promote;
2. core regressions never promote;
3. fewer hard-safety failures rank first;
4. aggregate score and pass metrics break remaining ties.

Milestone, confirmation, qualification, and publication gates remain
unchanged. An unsafe search incumbent may be persisted in safe campaign
evidence and used as the next optimization seed, but it cannot reach
`QUALIFIED` or be published.

The first-search condition is derived from immutable cursor and accounting
state: tier 0, stage 0, seed 0, zero metric calls, zero stagnation, and the
incumbent equal to the seed. The validated comparison outcome is carried into
`AttemptOutcome`; it is not trusted from workflow input. No persisted state
schema migration is required; a fresh campaign lineage is still required for
live execution.

## Artifact Behavior

`champion-candidate.yaml` remains the persisted search incumbent artifact.
Campaign summaries must continue to state that publication is blocked while
qualification gates fail. The artifact name does not imply publication.

## Verification

- The first improved unsafe candidate replaces the synthetic initial score.
- The first evaluation of an unchanged seed records its real score.
- A later candidate with fewer hard failures becomes the incumbent.
- A later candidate with more hard failures does not replace it.
- Systemic and core-regression outcomes never become incumbents.
- Unsafe incumbents fail milestone/confirmation publication gates.
- Workflow packaging carries the incumbent candidate into the next action.
