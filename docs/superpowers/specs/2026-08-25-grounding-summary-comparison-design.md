# Grounding Before/After Summary Design

## Purpose

Grounding Round evidence is complete but not decision-friendly. The current
GitHub Job Summary presents the final aggregate, pass rates, safety failures,
and per-run detail in separate sections. An operator must find an earlier run
and manually interpret whether each number increasing or decreasing is good.

The summary must answer three questions at the top of the page:

1. Did the prompt actually change?
2. Which comparable metrics improved, stayed unchanged, or regressed?
3. Is the resulting candidate publishable?

Prompt text diff rendering is out of scope. Raw model answers, reflection
conversations, GEPA state, credentials, and Kubernetes access material remain
excluded.

## Decision

An `optimize-evaluate` round will produce a paired comparison inside one
workflow. The seed candidate and the best candidate will use the same target
model, campaign, milestone cases, repetition count, evaluator revision, and
serving contract. The Job Summary will start with a compact semantic comparison
table before any detailed evidence.

This is preferred over comparing with a prior GitHub Actions run because a
prior run can differ in revision, cases, repetitions, or serving conditions. It
is preferred over comparing GEPA validation scores because validation scores
alone cannot explain final pass rates and safety failures.

## Round Execution

### Optimize-evaluate rounds

The round orchestrator will use this order:

1. Complete the existing trust checks, node-pool scale-up, and AKS preflight.
2. Evaluate the seed candidate against the final evaluation contract and store
   the evidence under a dedicated before-evaluation root.
3. Accept evaluation exit code `1` only when a complete evaluation summary
   exists, `systemic_failures` is zero, and the failure is therefore the
   expected hard-safety gate. Missing or malformed evidence and any systemic
   failure stop the round.
4. Run optimization. Optimization failure remains fatal and never falls back
   to the seed.
5. Compare the seed and best-candidate fingerprints.
6. If the fingerprints differ, evaluate the best candidate with the exact same
   evaluation contract and store it under the after-evaluation root.
7. If the fingerprints are identical, reuse the before evidence as the final
   evaluation evidence. Do not run the same prompt a second time and do not
   present sampling variance as prompt improvement.
8. Build safe comparison evidence, render the summary, and preserve the
   existing promotion gate and cleanup behavior.

This adds one full seed evaluation only when optimization produces a changed
candidate. When the candidate is unchanged, the round performs one final
campaign evaluation, as it does today.

### Evaluate-only rounds

Evaluate-only rounds have no optimizer-owned before/after pair. Their summary
will show a single-result headline and the existing final metrics without a
delta column. No prior run is inferred automatically.

## Comparison Contract

The safe-evidence package will add `comparison-summary.json` with
`schema_version: 1`. It will contain only allowlisted derived values:

- comparison status: `changed`, `unchanged`, or `not_applicable`
- seed and best candidate fingerprints
- the exact shared evaluation contract identity
- before and after aggregate score
- before and after pass@3 and pass@5
- before and after total hard-safety failures
- before and after systemic failures
- before and after hard-failure counts by category
- promotion eligibility and blockers

For changed candidates, before and after evidence must have the same campaign,
model set, evaluated case IDs, repetitions, execution mode, Prompt Lab
revision, and Korvid revision. A mismatch fails closed rather than rendering a
misleading delta.

For unchanged candidates, the before and after metric values in the comparison
contract are the same evidence and every delta is zero. The comparison status,
not score noise, is the authoritative outcome.

The package will preserve the existing final `evaluation-summary.json`,
`best-candidate.yaml`, `optimization-summary.json`, and blank-answer response
projections. It will additionally include the allowlisted before evaluation
summary and before response projections when the candidate changed, so the
comparison remains auditable. Artifact filenames and JSON keys continue to pass
the existing credential and forbidden-name checks.

## Job Summary Layout

The first screen of `round-summary.md` will use this order:

### 1. Outcome headline

One unambiguous line:

- `✅ IMPROVED — candidate changed; no core metric regressed`
- `⚠️ REGRESSED — candidate changed; one or more core metrics regressed`
- `➖ UNCHANGED — optimizer retained the seed prompt`
- `ℹ️ SINGLE EVALUATION — no before/after pair`

The headline describes comparison outcome, not workflow conclusion. Promotion
remains a separate decision.

### 2. Before vs after table

Changed and unchanged optimize-evaluate rounds render:

| Metric | Before | After | Delta | Result |
| --- | ---: | ---: | ---: | --- |
| Aggregate score | 0.000 | 0.020 | +0.020 | ✅ improved |
| pass@3 | 0.000 | 0.000 | 0.000 | ➖ unchanged |
| pass@5 | 0.000 | 0.000 | 0.000 | ➖ unchanged |
| Hard safety failures | 15 | 13 | -2 | ✅ improved |
| Systemic failures | 0 | 0 | 0 | ➖ unchanged |
| `write_before_fresh_read` | 10 | 8 | -2 | ✅ improved |
| `wrong_target_write` | 5 | 4 | -1 | ✅ improved |

Metric direction is semantic:

- aggregate score, pass@3, and pass@5 improve when they increase
- systemic, total hard-safety, and per-category failure counts improve when
  they decrease
- equal values are unchanged

The renderer uses explicit words and symbols together; it never relies on
color or an arrow whose meaning changes by row. Per-category rows use the union
of categories present before and after, with absent categories treated as zero.

### 3. Decision digest

Three short bullets follow the table:

- Prompt: changed or unchanged, with both fingerprints
- Net: count of improved, unchanged, and regressed metrics
- Publication: eligible or blocked, with the blocker names

If the prompt is unchanged, the digest explicitly says that the optimizer did
not produce an improved prompt even if a historical run had a different score.

### 4. Detailed evidence

The existing overview, per-model scores, status counts, hard-failure counts,
per-run table, artifacts, and reproduction command remain available under:

```html
<details>
<summary>Detailed round evidence</summary>
...
</details>
```

This preserves auditability while keeping the decision surface short. The same
top section appears in the sticky PR comment because it consumes the generated
round summary.

## Outcome Semantics

Core metrics are aggregate score, pass@3, pass@5, total hard-safety failures,
and systemic failures. The headline is:

- `UNCHANGED` whenever seed and best fingerprints match
- `REGRESSED` when a changed candidate regresses on any core metric
- `IMPROVED` when a changed candidate improves at least one core metric and
  regresses on none
- `UNCHANGED` when a changed candidate has no core metric movement

Per-category safety changes are shown in the table and net counts but do not
override a core-metric regression. Promotion eligibility continues to use the
authoritative existing policy; an `IMPROVED` candidate can still be blocked
from publication.

No epsilon is applied in version 1. Values are compared after schema validation
using their stored numeric values and formatted consistently for display.

## Error Handling

- A before evaluation with hard-safety failures may continue only with complete,
  live, internally consistent evidence and zero systemic failures.
- Any systemic failure, missing summary, mismatched evaluation contract,
  non-finite metric, unknown comparison field, or fingerprint mismatch fails
  the round before artifact publication.
- An optimization failure remains fatal.
- An after evaluation with hard-safety failures produces comparison evidence
  and then preserves the existing non-zero safety-gate conclusion.
- Cleanup remains registered before scale-up and runs for success, failure, and
  cancellation.
- The renderer never substitutes zero for a missing metric. A legitimately
  absent optional pass rate displays `N/A` and has no delta or trend.

## Compatibility

- Existing evaluate-only CLI behavior remains supported.
- Existing hosted reflection and credentialless Ollama provider behavior is
  unchanged.
- Existing raw-answer redaction and safe-evidence allowlist remain mandatory.
- Publication consumes the final after summary, or the single reused summary
  when the candidate is unchanged.
- Detailed report fields and reproduction commands remain present.

## Testing

### Unit tests

- higher-is-better and lower-is-better direction calculation
- unchanged and changed fingerprints
- improved, regressed, unchanged, and single-evaluation headlines
- union of hard-failure categories
- optional pass rates rendered as `N/A`
- comparison contract mismatch and non-finite value rejection
- compact section appears before detailed evidence
- detailed evidence remains inside a valid `<details>` block

### Safe-evidence tests

- comparison summary contains only allowlisted fields
- before and after fingerprints match their response projections
- before raw answers are blanked exactly like after raw answers
- raw requests, audit data, credentials, and GEPA state remain absent
- unchanged candidates do not duplicate response evidence

### Orchestrator tests

- expected before safety-gate exit continues to optimization
- before systemic failure aborts before optimization
- changed candidate evaluates both sides with identical arguments
- unchanged candidate performs no duplicate final evaluation
- optimization failure remains fatal
- after safety gate still uploads the summary and restores compute

### Workflow tests

- Job Summary appends the new comparison-first Markdown
- sticky PR comment reads the same safe summary
- safe-evidence upload and cleanup guards remain unchanged

## Success Criteria

An operator opening a completed optimize-evaluate run can determine, without
opening an artifact or another run:

- whether the prompt changed
- each core metric's before value, after value, signed delta, and semantic
  direction
- the safety categories that improved or regressed
- whether publication is allowed and why

The comparison is generated only from same-round, contract-compatible evidence
and does not weaken the existing security or cleanup boundaries.
