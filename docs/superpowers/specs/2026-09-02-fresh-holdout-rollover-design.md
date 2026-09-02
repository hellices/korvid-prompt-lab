# Fresh-Holdout Stable Search Rollover Design

**Date:** 2026-09-02

## Goal

Continue the Korvid `small` prompt search after a valid
`no_stable_winner` result without tuning against an already observed
milestone.

The immediate objective is one v3 campaign against the AKS-hosted
`qwen3:0.6b` model. The campaign either:

- produces a candidate that passes the existing stable-winner contract; or
- records another valid `no_stable_winner` and stops with
  `fresh_holdout_exhausted`.

The rollover must not claim that repeated testing guarantees improvement.
It guarantees only that a promoted prompt passed a fresh qualification set
under the stated budget and safety gates.

## Why the v2 Result Is Retained

The v2 result is a checkpoint, not the desired endpoint. It prevents four
invalid behaviors:

- promoting the Stage B `+0.1222` result after it regressed on milestone;
- retesting the same candidate space without learning from its failures;
- treating a serving collapse as prompt-quality evidence;
- reusing the observed v2 milestone as if it were still a holdout.

The v2 milestone is now development evidence. It may inform v3 candidates,
but it may never qualify them.

## Scope

This increment adds:

- immutable prior-campaign lineage validation;
- scenario-consumption tracking by Korvid version and scenario digests;
- deterministic fresh-holdout rollover;
- a separate deterministic rollover candidate matrix;
- a bounded v3 CLI path;
- unit, integration, and live AKS verification.

It does not add:

- unbounded search;
- reuse of a consumed milestone for promotion;
- synthetic scenarios;
- mutation of Korvid's installed prompt or scenario package;
- persistence of raw answers, fixture state, endpoints, or credentials.

## Inputs and Trust Boundaries

The rollover accepts a prior stable-search artifact root and a fresh output
root. It reads only these prior files:

- `stable-search-summary.json`;
- `scenario-manifest.json`;
- `candidate-manifest.json`;
- `stage-c/qualification-summary.json`.

Every prior file must be a regular file confined beneath the supplied prior
root. The rollover rejects symlinks, path traversal, missing files, malformed
JSON, unexpected schema versions, non-finite numbers, and a prior decision
other than `no_stable_winner`.

The installed Korvid wheel remains authoritative for scenario content.
Prior scenario IDs and question/fixture digests must match the installed
catalog and Korvid version exactly. A mismatch aborts before any model call.

## Scenario Rollover

The v2 manifest consumed 18 of the 25 installed Korvid scenarios. The
remaining seven scenarios are the only eligible v3 milestone pool:

- `missing-configmap-mount`;
- `node-pressure-eviction`;
- `pending-insufficient-cpu`;
- `pvc-pending-no-storageclass`;
- `pvc-wait-for-first-consumer`;
- `service-endpoints-not-ready`;
- `service-selector-mismatch`.

The rollover builder works from digests, not a hard-coded ID list:

1. Load and validate all prior assignments.
2. Mark every prior train, validation, and milestone assignment as consumed.
3. Match the installed catalog against the consumed `(question_sha256,
   fixture_sha256)` pairs.
4. Select six untouched records deterministically for milestone, balanced
   across available scenario classes.
5. Select six prior consumed records for train and six different consumed
   records for validation.
6. Preserve one untouched record as an audit reserve. It is not enough to
   support another valid campaign by itself.

Candidate generation receives prior aggregate measurements but does not
receive the v3 milestone questions, fixture data, per-case scores, or IDs.
The milestone manifest may be materialized for execution, but it is not an
input to the candidate builder or proposer.

## Rollover Candidate Matrix

The original candidate matrix and IDs remain unchanged so that v2 evidence
stays reproducible. A separate v3 matrix starts from the strongest v2
development signal and addresses its observed failure mode.

The seed append is:

```text
name the observed evidence and its source before the final conclusion.
```

The new axes are:

- **decisive-read-first:** gather the smallest relevant read-only evidence
  needed to distinguish likely causes before concluding;
- **continue-before-uncertainty:** do not stop merely because initial evidence
  is insufficient; inspect the next highest-value source first;
- **bounded-uncertainty:** after relevant read-only evidence is exhausted,
  state exactly what remains unknown and stop instead of guessing;
- **evidence-linked-conclusion:** tie each conclusion to observed evidence and
  avoid unsupported remediation.

The deterministic matrix contains the four single-axis variants, three
targeted pairs, and one all-axis candidate. Every candidate:

- preserves the installed baseline `system` component byte-for-byte;
- adds only a canonical `append`;
- remains at most 480 characters;
- records the prior finalist fingerprint and v2 receipt hash in metadata;
- has an ID and fingerprint derived deterministically from its content.

No v3 candidate text may depend on the untouched milestone identities or
contents.

## Execution and Budget

The existing Stage A/B/C orchestrator and stable-winner contract are reused:

- Stage A: train screening, one repetition;
- Stage B: validation, three repetitions;
- Stage C: validation plus fresh milestone, five repetitions;
- no more than 306 target-model calls;
- validation mean delta at least `+0.10`;
- milestone mean delta at least `+0.10`;
- no per-case worst-score regression;
- exactly five repetitions per qualification case;
- zero hard-safety failures;
- zero systemic failures.

An optional bounded proposer may occupy the existing reserved candidate slot.
Its input is restricted to consumed aggregate v2/v3 development measurements.
It never receives fresh milestone data.

The run stops immediately on:

- a stable winner;
- serving collapse or another systemic failure;
- the 306-call budget;
- manifest or lineage mismatch;
- fresh-holdout exhaustion.

`no_stable_winner` is a valid campaign result but not project success. After a
valid v3 no-winner result, the next project step is to expand the scenario
bank before another qualification attempt.

## Artifacts

The v3 output root remains immutable and contains the existing normalized
artifacts plus `rollover-lineage.json`. The lineage records:

- prior campaign decision and receipt SHA-256;
- prior scenario-manifest SHA-256;
- consumed scenario digests;
- fresh milestone digests;
- development/fresh split counts;
- candidate-matrix version;
- target-model identity;
- call budget and actual call count;
- terminal reason.

It must not contain raw model answers, raw upstream JSON, scenario questions,
fixture state, reflection transcripts, kubeconfig data, endpoint values, or
credentials.

If a winner passes every gate, the CLI writes the existing exact append
Candidate YAML. Otherwise it writes no winner YAML.

## CLI

Add a separate command so the original campaign remains reproducible:

```bash
korvid-prompt-lab stable-search-rollover \
  --prior-artifact-root <v2-root> \
  --artifact-root <fresh-v3-root> \
  --json
```

The command fails closed if either root already violates the immutable
artifact contract. It prints only normalized JSON in `--json` mode and only
bounded error labels for systemic failures.

## AKS Lifecycle

The live task:

1. records the original `modeleval` node-pool count;
2. scales up only if required;
3. verifies the expected Ollama model digest and serving readiness;
4. runs v3 through a supervised loopback port-forward;
5. terminates only the exact supervisor/port-forward processes it started;
6. removes temporary kubeconfig data;
7. restores the original node-pool count;
8. verifies provisioning state `Succeeded`.

Cleanup runs for winner, no-winner, timeout, interruption, and systemic error.

## Testing

Tests must prove:

- prior artifact confinement and schema validation;
- rejection of symlinks, changed Korvid versions, and digest mismatch;
- consumed scenarios cannot re-enter milestone;
- fresh milestone records never enter candidate/proposer inputs;
- deterministic balanced rollover splits;
- deterministic append-only rollover candidates;
- the 306-call ceiling;
- five-repetition qualification on the fresh milestone;
- no winner YAML on rejection;
- winner YAML only after every unchanged gate passes;
- bounded JSON and error output;
- cleanup behavior for every terminal state.

A fake-runner integration test must demonstrate both a qualifying rollover
winner and `fresh_holdout_exhausted`. The live campaign must record the actual
v3 result without claiming improvement unless every promotion gate passes.
