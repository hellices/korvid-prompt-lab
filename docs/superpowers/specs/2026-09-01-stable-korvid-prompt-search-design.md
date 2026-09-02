# Stable Korvid Prompt Search

## Goal

Produce a prompt candidate that repeatedly improves the installed Korvid
`small` prompt on the AKS-hosted `qwen3:0.6b` model, or produce conclusive
evidence that the bounded candidate space has no promotable winner.

This work starts from the merged `korvid_readonly` backend. The installed
Korvid wheel remains the source of truth for the baseline prompt, scenario
definitions, tool arm, grading, and execution behavior.

## Success Criterion

A candidate is promotable only when all of the following hold:

- its mean score exceeds the paired baseline by at least `0.10` on both the
  validation and milestone splits;
- each finalist and the baseline have five repetitions per case on both
  splits;
- its worst per-case mean does not regress below the paired baseline;
- it produces zero hard-safety failures;
- it produces zero systemic failures;
- every comparison uses the same installed Korvid version, scenario
  fingerprints, serving model identity, and evaluation contract.

An isolated high score is not a winner. A candidate that fails any gate is
recorded and rejected.

## Approach

Use a structured-first, proposer-second hybrid search.

Free-form GEPA search is not the first stage because previous `qwen3:0.6b`
reflection runs generated blank or scenario-specific proposals, while the
`qwen3:4b` proposer timed out. A deterministic candidate matrix provides
interpretable coverage and guarantees that the target model evaluates distinct
instructions. A stronger proposer is allowed only after the structured search
identifies a repeatable failure axis.

### Rejected Alternatives

1. **Free-form GEPA only.** This repeats the previously observed proposer
   failure mode and provides no guarantee that a useful candidate is produced.
2. **Manual candidate only.** This is reliable and interpretable but cannot
   refine a surviving candidate from measured failure feedback.
3. **Full factorial search.** Four binary axes create sixteen combinations,
   many of which are redundant and unnecessarily expensive on live GPU
   capacity.

## Candidate Space

The system prompt is never replaced. Every candidate preserves the exact
installed `small` system prompt and changes only the optional append component.

Four bounded instruction axes are available:

1. **Evidence first:** inspect runtime evidence before stating a diagnosis.
2. **One tool at a time:** choose the single highest-value read tool, inspect
   its result, then decide the next step.
3. **Cite before conclusion:** name the observed evidence and its source before
   the final conclusion.
4. **Stop with uncertainty:** when evidence is insufficient, state what is
   missing and stop instead of guessing.

The initial matrix contains eight candidates:

- one candidate for each single axis;
- three pairwise candidates selected for complementary behavior:
  evidence-first plus one-tool-at-a-time, evidence-first plus citation, and
  citation plus uncertainty;
- one concise candidate containing all four axes.

Each append is canonical, nonblank, and at most 480 characters. Candidate IDs
and fingerprints are deterministic from the axis selection and exact text.

## Scenario Stratification

Read scenario metadata from the installed Korvid wheel at runtime. Do not copy
scenario bodies into Prompt Lab.

Build a deterministic manifest that assigns read-only scenarios to broad
failure classes such as workload health, image/configuration, scheduling and
resources, networking, storage, and healthy/no-fault controls. Within each
class, assign scenario IDs to train, validation, and milestone splits by a
stable hash. The split builder must:

- keep scenario IDs disjoint across splits;
- include at least two failure classes in every split;
- include a healthy/no-fault control in validation or milestone when the
  installed catalog provides one;
- persist installed Korvid version and scenario fingerprints;
- fail closed when the installed catalog cannot satisfy the configured split
  size.

The first live campaign uses six train, six validation, and six milestone
scenarios. If the installed catalog has fewer than eighteen suitable scenarios,
the builder reduces all splits to the same largest feasible size while keeping
at least four scenarios per split.

## Search Stages

### Stage A: Structured Screening

Evaluate the baseline and all eight candidates on the train split with one
repetition per case.

Reject a candidate immediately when it has a hard-safety or systemic failure.
Rank the remaining candidates by:

1. mean score delta over the paired baseline;
2. worst per-case delta;
3. evidence verification delta;
4. fewer malformed or unresolvable tool calls.

Keep the top three candidates. Screening results are discovery evidence only
and cannot promote a prompt.

### Stage B: Repeated Validation

Evaluate the baseline and the three survivors on validation with three
repetitions per case.

Keep at most two finalists that:

- have positive mean delta;
- have no negative worst per-case delta;
- have zero safety and systemic failures.

Rank finalists by mean delta, then lower score variance, then higher pass-at-3.

### Stage C: Qualification

Evaluate the baseline and each finalist on validation and milestone with five
repetitions per case. Apply the success criterion without exceptions.

The baseline is re-evaluated in the same live window as the finalists. Historical
baseline scores are displayed for context but never used as the qualification
denominator.

## Optional Bounded Proposer

Run the proposer only when Stage B produces a finalist with positive mean delta
but a consistent bounded failure signal, such as low evidence verification or
too many unresolvable tool calls.

The proposer receives:

- the finalist append;
- aggregate bounded counts and booleans;
- the single selected failure axis;
- a requirement to return one append no longer than 480 characters.

It does not receive raw model answers, cluster data, logs, credentials, or
scenario bodies. Use `qwen3:4b` with a dedicated proposer timeout. Blank,
overlong, noncanonical, or timed-out proposals are rejected, and the structured
finalist remains unchanged. The proposed variant must pass Stage B and Stage C
like every other finalist.

## Data Flow

1. Materialize the installed Korvid baseline candidate.
2. Discover and fingerprint the installed scenario catalog.
3. Write an immutable split manifest and deterministic candidate manifest.
4. Start AKS model capacity and establish a private local forwarding endpoint.
5. Run Stage A, Stage B, and Stage C in separate immutable artifact roots.
6. Normalize and redact every run through the existing `korvid_readonly`
   evidence contract.
7. Produce one campaign summary containing baseline/candidate score movement,
   per-case movement, variance, pass metrics, safety/systemic counts, and the
   final decision.
8. Restore AKS capacity to its original state and remove temporary kubeconfig,
   port-forward, and raw process artifacts.

## Failure Handling

- A systemic runner failure aborts the current stage; it is never scored as
  zero.
- A model failure is a scored zero only when it satisfies the installed
  Korvid exit and JSON contract.
- A candidate safety failure rejects that candidate and prevents promotion.
- A proposer failure does not invalidate structured evaluations.
- A serving model, Korvid version, prompt, tool arm, scenario, or provenance
  mismatch aborts the stage.
- An interrupted campaign does not resume mutable optimizer state. A retry
  uses a new immutable run identity and artifact root.

## Cost and Runtime Bounds

The default maximum live evaluations are:

- Stage A: `9 candidates × 6 cases × 1 repetition = 54`;
- Stage B: `4 candidates × 6 cases × 3 repetitions = 72`;
- Stage C: `3 candidates × 12 cases × 5 repetitions = 180`.

The absolute maximum is 306 target-model evaluations. Stage A and Stage B may
reduce this total by rejecting candidates early. The campaign records actual
and conservative bounded metric-call usage. AKS capacity is scaled down after
completion or failure.

## Artifacts

Persist:

- baseline candidate fingerprint and installed Korvid version;
- split and candidate manifests;
- normalized redacted run evidence;
- per-stage rankings and rejection reasons;
- paired baseline/candidate aggregate and per-case movement;
- repetition variance and pass metrics;
- exact winning append, if any;
- final decision: `promote`, `no_stable_winner`, or `system_error`.

Do not persist raw answers, raw upstream Korvid JSON, fixture state, kubeconfig,
credentials, endpoint secrets, or reflection transcripts.

## Verification

Unit and integration tests must cover:

- deterministic candidate and split generation;
- split disjointness and class coverage;
- stage ranking and early rejection;
- paired baseline comparison;
- variance, worst-case, and five-repetition qualification gates;
- proposer input redaction and output bounds;
- no-winner behavior;
- immutable run identities and artifact provenance;
- cleanup on success, rejection, timeout, and interruption.

A deterministic fake-runner integration test must produce one known winner.
The real AKS campaign is successful only when a candidate passes the stated
qualification gate; otherwise `no_stable_winner` is the correct result.
