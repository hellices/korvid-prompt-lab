# Read-Only Korvid Prompt Optimization

## Goal

Evaluate the currently shipped Korvid prompt and improve it with Prompt Lab's
existing GEPA, comparison, and campaign machinery.

## Scope

The optimization target is Korvid's shipped read-only scenario pack. The
installed `korvid[agent]>=0.3` package remains the source of scenarios,
profiles, prompt composition, runner behavior, and grading.

Conversational journeys are baseline regression evidence only because Korvid
0.3 does not expose prompt override flags for that CLI. Write/approval journeys
remain in Korvid tests and outside Prompt Lab optimization.

## Serving Backend

Add a `korvid_readonly` campaign serving backend with:

- provider: `ollama` or `openai-compat`;
- base URL from an environment reference;
- profile: `small` or `full`;
- timeout.

For each `EvalCase`, the runner loads the bundled scenario with the same ID,
verifies the authored question equals the campaign prompt, copies exactly that
fixture into a private temporary pack, and invokes:

```text
python -m korvid.evals
  --scenarios <one-case-pack>
  --reps 1
  --profile <profile>
  --system-prompt-file <candidate-system>
  [--prompt-append-file <candidate-append>]
  --json <private-output>
```

The process receives model endpoint/provider configuration through
`KORVID_EVAL_*` environment variables. Raw requests, credentials, and upstream
JSON are not published.

## Baseline

Prompt Lab imports `korvid.agent.profiles.build_profile` from the installed
wheel and materializes the selected profile's current `system_prompt` as the
seed candidate. No hard-coded prompt copy is stored in this repository.

A CLI command writes this immutable candidate YAML and records:

- installed Korvid distribution version;
- profile name;
- prompt fingerprint.

The current shipped prompt and optimized candidates therefore use the exact
same override path during comparison.

## Score Mapping

One Korvid scenario run maps to Prompt Lab metrics:

- completion: `1.0` when `diagnosis_success`, otherwise `0.0`;
- verification: `1.0` when `evidence_fetched`, otherwise `0.0`;
- efficiency: on-target calls divided by total calls, or `1.0` when no calls;
- hard failures: successful write safety violations and attempted write tools;
- model failure: a graded run carrying a provider/session error;
- systemic failure: malformed JSON, identity mismatch, process error, timeout,
  or missing artifact.

Reflection traces expose only bounded answer, missing mention/evidence counts,
tool counts, citation coverage/precision, and hard-failure labels.

## CLI and Reuse

Add:

- `korvid-prompt-lab korvid-baseline`;
- the `korvid_readonly` serving backend accepted by existing `evaluate` and
  `optimize`.

No separate optimizer is introduced. Existing GEPA search, actual metric-call
accounting, before/after comparison, and immutable artifacts remain the
control plane.

## Verification

- Baseline prompt equals the installed profile at runtime.
- One-case fixture selection is exact and isolated.
- Candidate prompt files are private and deleted.
- JSON identity and metric parsing fail closed.
- Concurrent runs do not share prompt or pack state.
- A fake CLI contract suite covers every outcome.
- A real installed-wheel scripted smoke test covers profile and fixture APIs.
- A live AKS canary demonstrates a non-flat candidate comparison.

