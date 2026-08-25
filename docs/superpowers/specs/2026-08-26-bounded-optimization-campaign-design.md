# Bounded Optimization Campaign Design

**Date:** 2026-08-26  
**Status:** Approved for implementation planning

## Problem

The current Grounding workflow proves that Prompt Lab can evaluate a seed and a
GEPA-selected candidate under the same contract, render an honest Before/After
comparison, block unsafe publication, and clean up AKS resources. The live canary
correctly reported `UNCHANGED` and blocked publication.

That is pipeline validation, not product success. Prompt Lab's product goal is to
find a prompt that qualifies for publication on a small model, or to produce
bounded, reproducible evidence that the selected model did not converge within
the declared search budget.

Repeatedly dispatching the same 12-call workflow is not sufficient:

- one search seed can repeatedly explore the same neighborhood;
- increasing a single GEPA budget does not isolate seed sensitivity;
- the current two cases are reused across train, validation, and milestone
  evaluation, so a passing result would not demonstrate holdout generalization;
- infrastructure failures can consume budget without providing optimization
  evidence;
- an incapable model could consume unbounded AKS time without a defensible stop
  decision.

## Goals

1. Continue experiments automatically until a candidate qualifies or a declared
   campaign limit is reached.
2. Keep hard-safety and publication policy fail-closed.
3. Separate search, selection, and final qualification evidence.
4. Preserve every round's safe, reproducible decision surface in GitHub Actions.
5. Distinguish prompt-search exhaustion from infrastructure failure and model
   capability limits.
6. Prevent unbounded compute use and silent overfitting.

## Non-goals

- Guarantee that every small model can pass the qualification gate.
- Publish a candidate that merely improves aggregate score.
- Expose raw model answers, reflection transcripts, credentials, kubeconfig, or
  optimizer state in GitHub artifacts.
- Treat retries for AKS, runner, bridge, or Korvid failures as optimization
  attempts.
- Build a general-purpose experiment scheduler outside GitHub Actions.

## Success Semantics

### Candidate qualification

A candidate is `QUALIFIED` only when a full milestone evaluation reports all of:

- `systemic_failures == 0`;
- `hard_safety_failures == 0`;
- `pass_at_3 == 1.0`;
- `pass_at_5 == 1.0`;
- the existing publication contract and provenance checks pass.

The candidate must then pass one independent confirmatory milestone evaluation
using fresh repetition seeds and the same immutable model, Prompt Lab, Korvid,
campaign, serving, and prompt fingerprints. Confirmation is not another
optimization round and does not expose milestone evidence to GEPA.

If the confirmation fails, the candidate remains unpublished and the campaign
continues while budget remains. A candidate is never called qualified based only
on validation metrics or a single milestone pass.

### Bounded non-convergence

A campaign that exhausts its search budget without a qualified candidate ends as
`NOT_CONVERGED`, not as a successful workflow. Its result must include:

- the best safe candidate found;
- the exact budget consumed;
- validation and milestone blockers;
- per-category failure movement;
- the reason the search stopped;
- whether the evidence indicates prompt-search stagnation or a possible model
  capability ceiling.

`NOT_CONVERGED` is useful evidence but does not permit publication.

## Evaluation Data Separation

The campaign case pack must contain three disjoint sets:

1. **Train cases** provide GEPA metric feedback.
2. **Validation cases** rank candidates and control promotion between search
   stages.
3. **Milestone holdout cases** are used only for final qualification and
   confirmation.

No case ID may appear in more than one set. Milestone prompts, grades, failure
details, and aggregate results must not be supplied to the reflection model or
used to propose the next candidate.

The existing two-case AKS campaign remains useful as a pipeline canary but is not
a qualification campaign. Before automated qualification begins, the case pack
must be expanded with representative read-before-write, target selection,
approval denial, postcondition verification, and non-mutating request cases.

Each model family and immutable model digest has its own campaign leaderboard.
Evidence from one model cannot qualify a prompt for another model.

## Campaign Architecture

### Campaign manifest

A versioned campaign-control manifest declares:

- the base evaluation campaign and initial candidate;
- ordered train, validation, and milestone case IDs;
- search seeds;
- staged GEPA metric-call budgets;
- total metric-call and wall-clock limits;
- maximum infrastructure retries per attempt;
- stagnation thresholds;
- qualification and confirmation requirements;
- the ordered small-model tiers that may be evaluated.

All limits are required and positive. The controller rejects overlapping case
sets, mutable model references, stale or untrusted Korvid revisions, and
incomplete limits before allocating AKS capacity.

### Campaign controller

A Prompt Lab command owns campaign state and makes deterministic next-step
decisions. GitHub Actions remains the outer orchestrator and invokes one bounded
controller operation at a time.

The controller:

1. validates campaign identity and immutable revisions;
2. records a fresh baseline for the initial candidate;
3. schedules search attempts across the current stage's declared seeds;
4. ingests validated optimization and evaluation summaries;
5. ranks candidates using validation evidence;
6. promotes only non-regressing candidates;
7. runs milestone evaluation only for eligible finalists;
8. requests confirmation only after a full milestone pass;
9. stops as `QUALIFIED`, `NOT_CONVERGED`, or `SYSTEM_ERROR`.

Campaign state is a typed JSON artifact keyed by a deterministic campaign ID.
State contains safe summaries, fingerprints, counters, and decisions, not GEPA
state or raw conversations. A resumed workflow verifies every referenced
artifact and revision before continuing.

### Search stages

The default search policy has three bounded stages:

1. **Explore:** several independent seeds with a small metric-call budget.
2. **Refine:** only the leading safe candidates receive a larger budget.
3. **Qualify:** finalists receive the untouched full milestone pack.

Exact budgets belong in the campaign manifest rather than application code. The
first production manifest should begin conservatively and may be revised only as
a new manifest revision, never during a running campaign.

GEPA attempts are independent and never share `gepa_state.bin`. A promoted
candidate becomes an explicit seed candidate for a new invocation with a new run
identity.

## Candidate Ranking and Promotion

Candidates are ranked on validation evidence in this order:

1. zero systemic failures;
2. no hard-safety regression relative to the current champion;
3. no regression in any core comparison metric;
4. greater hard-safety failure reduction;
5. greater aggregate score;
6. greater pass@3, then pass@5;
7. stable fingerprint ordering as a deterministic tie-breaker.

A candidate is promoted only when:

- its validation contract matches the champion's contract;
- it has zero systemic failures;
- no core metric regresses;
- at least one core metric improves; and
- its fingerprint differs from the champion.

Changed-but-flat and identical candidates are retained as evidence but are not
promoted. A later seed may continue from the current champion while budget
remains.

## Stagnation and Model Capability Escalation

The controller declares search stagnation when the manifest's consecutive
attempt limit is reached with no promotable candidate. It also records dominant
failure categories, including:

- `write_before_fresh_read`;
- `wrong_target_write`;
- `success_without_postcondition_read`.

Stagnation does not prove that a model is incapable. It is evidence that the
declared prompt-search strategy and budget did not improve it.

When all stages for one immutable model digest are exhausted, that model ends as
`NOT_CONVERGED`. If the manifest contains another approved small-model tier, a
new independent campaign may begin for that model. The prior model's candidate,
scores, and milestone evidence are not reused as qualification evidence.

This prevents unlimited spending on the 0.6B model while still preserving a
clear record of its prompt-optimization ceiling.

## Infrastructure and Error Handling

Before any experiment, Prompt Lab must pin a reviewed post-squash Korvid SHA that
satisfies both repository trust provenance and bridge importability.

Attempt outcomes are classified as:

- `EVIDENCE`: validated exit `0` or hard-safety exit `1`;
- `SYSTEM_ERROR`: exit `70`, malformed or contradictory evidence, AKS failure,
  runner failure, bridge failure, or the Korvid `#agent-chat` race;
- `CONFIG_ERROR`: invalid manifest, overlapping sets, mutable identity, or
  unsupported inputs.

System and configuration failures never update candidate scores, consume GEPA
metric-call budget, or trigger model escalation. They still count toward the
campaign's wall-clock safety limit, which bounds total operator and AKS exposure
rather than model evidence. A system failure may use only the manifest's bounded
retry allowance. Cleanup and restoration remain mandatory after every attempt.

Unexpected optimizer failure never falls back to the seed as if optimization
succeeded.

## GitHub Actions Decision Surface

The campaign Job Summary begins with `# Optimization Campaign Outcome` and shows:

- overall state: `RUNNING`, `QUALIFIED`, `NOT_CONVERGED`, or `SYSTEM_ERROR`;
- model family and immutable digest;
- current champion fingerprint;
- stage and attempt number;
- cumulative metric calls and wall-clock use versus limits;
- a leaderboard of candidate validation metrics;
- category-level movement from the initial baseline;
- milestone and confirmation state;
- publication eligibility and blockers;
- the controller's next decision or final stop reason.

Each underlying Grounding round retains its existing `Grounding Round Outcome`.
The campaign summary links to those safe round summaries without copying raw
evidence.

The safe campaign artifact contains:

- `campaign-state.json`;
- `campaign-summary.json`;
- `campaign-summary.md`;
- safe candidate YAML files;
- safe references to constituent round artifacts.

It excludes the same sensitive material as the existing safe-evidence package.
All artifact references must resolve within downloaded safe packages.

## Workflow Behavior

GitHub Actions uses a concurrency key per campaign ID so two controllers cannot
advance the same campaign simultaneously. A workflow invocation performs at
most one expensive search or evaluation attempt, persists updated state, and
dispatches the next invocation only when the state says `RUNNING`.

The continuation token is the immutable campaign ID plus the hash of the prior
state. The controller rejects stale state updates. This makes retries idempotent
and prevents two completed jobs from selecting different champions.

Protected-environment approval remains required for AKS and reflection
credentials. `QUALIFIED` does not automatically publish; it makes the existing
publication action eligible for an explicit approval.

## Testing and Verification

Implementation must use test-driven development and cover:

- disjoint case-set and immutable-identity validation;
- deterministic candidate ranking and tie-breaking;
- non-regression promotion rules;
- staged seed and budget scheduling;
- total-budget and wall-clock stopping;
- stagnation and model-tier transition;
- milestone isolation from optimizer inputs;
- confirmatory evaluation success and failure;
- system failures not consuming experiment budget;
- bounded retries and mandatory cleanup;
- resumable, idempotent state transitions;
- stale-state rejection;
- safe artifact allowlisting and reference resolution;
- exact GitHub Actions summary and artifact contracts.

Process-level tests must exercise `QUALIFIED`, `NOT_CONVERGED`, and
`SYSTEM_ERROR` paths. A live rollout starts with a low-budget canary, verifies
cleanup, then runs the first bounded production campaign. A publication claim
requires fresh evidence from both qualification and confirmation evaluations.

## Rollout Order

1. Replace the stale Korvid pin and verify trust plus bridge importability.
2. Expand and review the disjoint AKS case pack.
3. Implement the typed campaign manifest, state machine, and safe summaries.
4. Add one-attempt GitHub Actions orchestration and idempotent continuation.
5. Run a low-budget controller canary.
6. Start the 0.6B bounded campaign.
7. If it ends `NOT_CONVERGED`, retain the evidence and start the next approved
   small-model tier rather than silently relaxing the publication gate.

## Decision

Prompt Lab will not call the `UNCHANGED` canary an optimization success. It will
continue through a bounded, multi-seed, staged campaign until a candidate passes
the full publication gate and an independent confirmation, or until the
declared limits establish `NOT_CONVERGED`.
