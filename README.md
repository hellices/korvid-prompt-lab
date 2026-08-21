# Korvid Prompt Lab

Korvid Prompt Lab is a small control plane for validating prompt candidates,
running deterministic bridge evaluations, exercising read-only AKS preflight,
and publishing prompt bundles with a common-first, safety-gated override policy.

Run commands from the repository root so relative fixture and artifact paths
resolve as documented.

## Install with `uv`

```bash
uv sync --python 3.12 --extra dev
```

CLI entrypoint:

```bash
uv run --python 3.12 korvid-prompt-lab --help
```

## CLI commands

### Validate

Loads a candidate and campaign, applies strict schema checks, and confirms model
coverage.

```bash
uv run --python 3.12 korvid-prompt-lab validate \
  --candidate examples/candidates/shipped-small.yaml \
  --campaign examples/campaigns/local-smoke.yaml
```

### Evaluate

Runs the selected cases through the configured bridge, writes request/response
artifacts, emits an `evaluation-summary.json` with candidate/campaign identity,
case/model coverage, and the exact train/validation/milestone sets used as
publication provenance, and fails if any hard safety failure occurs.

`--train-case-id` and `--validation-case-id` are **required**. Both must name at
least one evaluated case and the two sets must be disjoint, so a bundle can never
claim validation evidence that is really its own training evidence.

```bash
uv run --python 3.12 korvid-prompt-lab evaluate \
  --candidate examples/candidates/shipped-small.yaml \
  --campaign examples/campaigns/local-smoke.yaml \
  --artifact-root artifacts/evaluate/local-smoke \
  --train-case-id smoke-happy \
  --validation-case-id smoke-guardrail \
  --json
```

Useful flags:

- `--case-id <id>` to limit evaluation to selected cases
- `--train-case-id <id>` (required) recorded train split
- `--validation-case-id <id>` (required) recorded validation split, disjoint from train
- `--milestone-case-id <id>` to record an explicit milestone pack; `milestone_passed`
  stays `false` unless the recorded pack is exactly the required pack and it ran
- `--bundle-kind common|model-specific` to record promotion intent
- `--json` to print the summary JSON to stdout

### Optimize

Requires a reflection-model configuration and delegates bounded search to
`optimize_campaign(...)`. `--train-case-id` and `--validation-case-id` are
**required** and must be disjoint, so search never validates on the cases it
learned from.

```bash
uv run --python 3.12 korvid-prompt-lab optimize \
  --candidate examples/candidates/shipped-small.yaml \
  --campaign examples/campaigns/local-smoke.yaml \
  --artifact-root artifacts/optimize/local-smoke \
  --max-metric-calls 2 \
  --reflection-model openai/gpt-4.1-mini \
  --seed 0 \
  --train-case-id smoke-happy \
  --validation-case-id smoke-guardrail
```

`optimization-summary.json` records `train_case_ids`, `validation_case_ids`, the
seed fingerprint, and `best_candidate_differs_from_seed` so a silent no-op search
is visible in the artifact.

#### Run identity, seeds, and contamination safety

GEPA resumes any `gepa_state.bin` it finds in its `run_dir`. A shared, stable
directory therefore turns a second `optimize` into a silent no-op that reports the
*previous* search's candidate and provenance. Korvid Prompt Lab makes every
invocation self-contained instead:

- every run derives a `run_id` from its full immutable identity — campaign id,
  candidate id, seed candidate fingerprint, train case ids, validation case ids,
  `--max-metric-calls`, `--seed`, and the proposal source
  (`none` / `reflection_lm` / `candidate_proposer`);
- all artifacts live under `<artifact-root>/invocations/<run_id>/`, so a changed
  seed always starts a fresh search and never inherits stale state;
- previous invocations stay immutable: nothing is written outside the current
  invocation directory;
- there is **no resume feature**. Re-running an identical identity fails closed
  with `optimization invocation directory already exists: ...`, and an existing
  `gepa_state.bin` is refused rather than resumed.

```text
artifacts/optimize/local-smoke/invocations/<run_id>/
  run-identity.json          # the exact identity the run_id was derived from
  gepa/                      # GEPA run_dir (state, logs) for this invocation only
  runs/                      # bridge request/response artifacts for this invocation
  best-candidate.yaml
  optimization-summary.json
```

`--seed` (default `0`) must be a non-negative integer; it is passed to GEPA and is
part of the run identity, so changing it is the normal way to run a second,
independent search into the same artifact root.

#### GEPA proposal contract

`KorvidGEPAAdapter` declares GEPA's optional `propose_new_texts` attribute and
leaves it `None`. GEPA reads that attribute on every reflective mutation; without
it the mutation step raises inside GEPA's own `try/except`, is logged, and the
search silently degrades to "no candidate proposed". Proposals themselves stay
outside the adapter:

- `--reflection-model` builds a DSPy reflection LM and passes
  `DSPyInstructionProposer` as GEPA's `custom_candidate_proposer` (reflection only);
- library callers may inject a deterministic proposer with
  `optimize_campaign(..., candidate_proposer=...)`; `reflection_lm` and
  `candidate_proposer` are mutually exclusive.

### AKS-backed serving

`aks-check` performs a read-only preflight for the `aks_port_forward` backend.
It validates the cluster, namespace, Service, Ready endpoints, loopback-only
port-forward, and `/v1/models` advertisement without changing the cluster.

`evaluate` and `optimize` use the same backend: for an `aks_port_forward`
campaign they open exactly one loopback port-forward, keep it open for the whole
run, pass the resulting `http://127.0.0.1:<port>` base URL to every bridge
request as `runtime.model_endpoint`, and terminate only that forward (and its
temporary kubeconfig) when the run ends, including on failure.

An `aks_port_forward` campaign must therefore declare the reviewed local Korvid
bridge command explicitly:

```yaml
serving:
  backend: aks_port_forward
  resource_group: rg-pension-guard
  cluster_name: aks-shared-runners
  namespace: env:KORVID_AKS_NAMESPACE
  service: env:KORVID_AKS_SERVICE
  model: env:KORVID_AKS_MODEL
  command:
    - korvid-bridge
    - --request
    - "{request}"
    - --response
    - "{response}"
```

Bridge commands are argument lists: no shell, no `env:` interpolation, and both
`{request}` and `{response}` placeholders are required. The endpoint is delivered
inside the request JSON, never on a shared or public address; the runner rejects
any endpoint that is not a loopback `http://` URL with an explicit port.

```bash
export KORVID_AKS_NAMESPACE=korvid
export KORVID_AKS_SERVICE=korvid-api
export KORVID_AKS_MODEL=qwen3-4b

uv run --python 3.12 korvid-prompt-lab aks-check \
  --campaign examples/campaigns/aks-shared-runners.yaml \
  --artifact-root artifacts/aks-check/shared-runners
```

The included AKS example targets the reviewed shared-runner environment:

- resource group: `rg-pension-guard`
- cluster: `aks-shared-runners`

Before running it, make sure `az login` is current and your active subscription
can resolve that exact resource group and AKS cluster.

### Publish

`publish` applies the reviewed promotion policy and writes an immutable registry
bundle, registry index, and Markdown scoreboard.

Minimal model metadata schema:

```json
{
  "model_family": "mock-small",
  "model_name": "mock-small@2026-08-21",
  "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "quantization": "fp16",
  "context_length": 8192,
  "serving_engine": "korvid-process"
}
```

Example publish flow:

```bash
uv run --python 3.12 korvid-prompt-lab publish \
  --candidate examples/candidates/shipped-small.yaml \
  --campaign examples/campaigns/local-smoke.yaml \
  --model-metadata model-metadata.json \
  --evaluation-summary artifacts/evaluate/local-smoke/evaluation-summary.json \
  --registry-root registry \
  --minimum-model-improvement 0.02
```

## Campaign runtime policy

A campaign owns every runtime knob; candidates and the optimizer can never change
them. Alongside `repetitions`, `models`, `cases`, and `serving`, a campaign may
declare how long one bridge invocation is allowed to take:

```yaml
schema_version: 1
campaign_id: local-smoke
repetitions: 5
bridge_timeout_seconds: 60
```

- `bridge_timeout_seconds` is optional and defaults to `300` (five minutes), which
  is sized for a real Korvid bridge doing a full model-backed operation.
- It must be a strictly positive, finite number; `0`, negatives, `null`, strings,
  booleans, `.inf`, and `.nan` are rejected at load time.
- `evaluate` and `optimize` pass it into every `KorvidProcessRunner`, so both the
  direct evaluation loop and the GEPA search enforce exactly the same per-bridge
  budget. Exceeding it is a systemic failure (`bridge timed out after N seconds`),
  never a low score.

## Bridge request/response schema

The process bridge receives a request JSON and must write a response JSON.
Requests always include:

```json
{
  "protocol_version": 1,
  "candidate_fingerprint": "<sha256>",
  "candidate": {
    "schema_version": 1,
    "candidate_id": "shipped-small",
    "components": {
      "system": "...",
      "append": "...",
      "tool.scale_resource": "..."
    },
    "metadata": {
      "source": "shipped"
    }
  },
  "case": {
    "case_id": "smoke-happy",
    "template_id": "smoke-template",
    "prompt": "...",
    "model": "mock-small",
    "repetition": 1,
    "seed": 0
  },
  "runtime": {
    "campaign_id": "local-smoke",
    "repetitions": 5,
    "artifact_dir": "artifacts/evaluate/local-smoke/runs/...",
    "model_endpoint": null
  }
}
```

`runtime.model_endpoint` is `null` for `process` serving and carries the exact
loopback base URL (for example `http://127.0.0.1:41001`) for `aks_port_forward`
serving. A bridge that talks to the AKS-hosted model must read it from there.

Responses must include protocol version `1`, the exact candidate fingerprint, a
matching `request_identity`, `status`, `answer`, `journal`, `usage`, `error`,
and either:

- `status: "completed"` with a grade object containing `completion`,
  `verification`, `efficiency`, and `hard_failures`, or
- `status: "model_failure"` with `grade: null`

Any other status is treated as systemic and aborts evaluation/publication.

## Fake smoke path and real Korvid bridge integration

The local smoke campaign uses the bundled fake bridge:

```text
tests/fixtures/fake_korvid_bridge.py
```

That lets you verify contracts end-to-end without AKS access. For a real Korvid
process bridge, replace the `serving.command` list with the reviewed executable
and preserve the `{request}` / `{response}` placeholders so the runner can pass
artifact paths explicitly.

For AKS-backed serving, keep the campaign on `backend: aks_port_forward`, declare
its own `serving.command`, and use `korvid-prompt-lab aks-check` before any live
evaluation. `evaluate` and `optimize` then run the whole campaign inside a single
loopback port-forward.

## Safety and promotion semantics

- Hard safety failures zero the affected run score.
- Any hard safety failure causes `evaluate` to return exit code `1`.
- Systemic bridge failures abort the run instead of fabricating a score.
- Publication never mutates an existing prompt bundle payload.
- Common bundles publish first.
- Model-specific bundles publish only when:
  - a matching common baseline already exists,
  - the milestone evaluation passed,
  - the effective score beats the common baseline by **strictly more** than the
    configured minimum improvement (`--minimum-model-improvement`, default
    `0.02`), so a tie or noise-sized gain never forks the prompt, and
  - hard safety failures remain at zero.

This preserves the common-first rollout and keeps model overrides explicitly
safety-gated.

### pass^3 and pass^5

`pass_at_3` and `pass_at_5` are true `pass^k` metrics: the share of
case/model groups whose **first k repetitions all passed**. A single lucky
repetition never counts as a pass.

A repetition passes only when the bridge reports authoritative success:

- `status` is `completed` — an executed `model_failure` never passes;
- the run carries no hard safety failure;
- `grade.completion` is exactly `1.0`, meaning the requested operation actually
  finished.

The weighted score is deliberately **not** the pass criterion. A half-finished
operation still earns a positive score through `verification` and `efficiency`
(for example `completion: 0.0, verification: 1.0, efficiency: 1.0` scores `0.40`),
and counting that as a pass would overstate reliability. Such a run raises the
aggregate score and still fails `pass^k`.

When a campaign records fewer than `k` repetitions for any group, the summary
reports `null` (`insufficient-evidence` in the text output) instead of inventing
a score, and `publish` refuses the bundle until the required repetitions exist.
Publishable campaigns therefore need `repetitions: 5` or more; the bundled
examples use five.

Additional CLI publish gates:

- `common` publication requires the full campaign case pack and the full model
  matrix recorded in the evaluation summary.
- `model-specific` publication requires the full milestone case pack recorded in
  the evaluation summary.
- every bundle requires non-empty, disjoint `case_sets.train` and
  `case_sets.validation` drawn from the campaign cases.

## Model matrix

| Model family | Usage | Backend | Notes |
| --- | --- | --- | --- |
| `mock-small` | local smoke validation | process | Uses the bundled fake bridge for contract checks. |
| `qwen3-4b` | AKS discovery / low-cost iteration | aks_port_forward | Suggested first live target. |
| `qwen3-8b` | broader validation | aks_port_forward | Use for stronger follow-up evaluation. |
| `qwen3-14b` | milestone / publication gate | aks_port_forward | Use for the final reviewed milestone pack. |

## Artifacts

Typical outputs:

- `artifacts/evaluate/.../runs/.../request.json`
- `artifacts/evaluate/.../runs/.../response.json`
- `artifacts/evaluate/.../evaluation-summary.json`
- `artifacts/optimize/.../invocations/<run_id>/run-identity.json`
- `artifacts/optimize/.../invocations/<run_id>/best-candidate.yaml`
- `artifacts/optimize/.../invocations/<run_id>/optimization-summary.json`
- `registry/index.json`
- `registry/scoreboard.md`
- `registry/bundles/<model-family>/<version>/prompt-bundle.yaml`
- `registry/bundles/<model-family>/<version>/evaluation-summary.json`

## Non-goals

Korvid Prompt Lab does **not**:

- deploy or modify AKS workloads;
- expose the shared-runner endpoint publicly;
- auto-promote a bundle directly into Korvid production runtime;
- invent model digests or serving metadata;
- bypass safety failures to improve aggregate scores.
