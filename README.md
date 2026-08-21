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
artifacts, emits an `evaluation-summary.json` with candidate/campaign identity
and case/model coverage metadata for `publish`, and fails if any hard safety
failure occurs.

```bash
uv run --python 3.12 korvid-prompt-lab evaluate \
  --candidate examples/candidates/shipped-small.yaml \
  --campaign examples/campaigns/local-smoke.yaml \
  --artifact-root artifacts/evaluate/local-smoke \
  --json
```

Useful flags:

- `--case-id <id>` to limit evaluation to selected cases
- `--bundle-kind common|model-specific` to record promotion intent
- `--json` to print the summary JSON to stdout

### Optimize

Requires a reflection-model configuration and delegates bounded search to
`optimize_campaign(...)`.

```bash
uv run --python 3.12 korvid-prompt-lab optimize \
  --candidate examples/candidates/shipped-small.yaml \
  --campaign examples/campaigns/local-smoke.yaml \
  --artifact-root artifacts/optimize/local-smoke \
  --max-metric-calls 2 \
  --reflection-model openai/gpt-4.1-mini
```

Optional flags:

- `--train-case-id <id>` to filter the train split
- `--validation-case-id <id>` to filter the validation split

### AKS preflight

`aks-check` performs a read-only preflight for the `aks_port_forward` backend.
It validates the cluster, namespace, Service, Ready endpoints, loopback-only
port-forward, and `/v1/models` advertisement without changing the cluster.

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
  --registry-root registry
```

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
    "repetitions": 1,
    "artifact_dir": "artifacts/evaluate/local-smoke/runs/..."
  }
}
```

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

For AKS-backed serving, keep the campaign on `backend: aks_port_forward` and use
`korvid-prompt-lab aks-check` before any live evaluation.

## Safety and promotion semantics

- Hard safety failures zero the affected run score.
- Any hard safety failure causes `evaluate` to return exit code `1`.
- Systemic bridge failures abort the run instead of fabricating a score.
- Publication never mutates an existing prompt bundle payload.
- Common bundles publish first.
- Model-specific bundles publish only when:
  - a matching common baseline already exists,
  - the milestone evaluation passed,
  - the effective score beats the common baseline by the configured minimum
    improvement, and
  - hard safety failures remain at zero.

This preserves the common-first rollout and keeps model overrides explicitly
safety-gated.

Additional CLI publish gates:

- `common` publication requires the full campaign case pack and the full model
  matrix recorded in the evaluation summary.
- `model-specific` publication requires the full milestone case pack recorded in
  the evaluation summary.

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
- `artifacts/optimize/.../best-candidate.yaml`
- `artifacts/optimize/.../optimization-summary.json`
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
