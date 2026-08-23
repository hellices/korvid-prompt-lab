# AKS Ollama Reflection Design

**Date:** 2026-08-23

## Goal

Run the first self-contained `optimize-evaluate` Grounding Round against
`qwen3:0.6b`, using a larger model already stored in the same AKS Ollama
deployment as the GEPA reflection model. The round must produce comparable,
redacted GitHub Actions evidence without introducing an external model
credential.

The first teacher is `qwen3:14b`. It is materially stronger than the target,
is already present on the persistent Ollama volume, and is a safer latency
choice than the available 30B-70B models on the current Spot CPU pool.

## Context

The existing Grounding workflow already supports bounded `optimize-evaluate`
rounds, disjoint train and validation cases, exact revision provenance,
protected Environment approval, safe-evidence projection, and unconditional
node-pool cleanup. Its reflection path currently assumes every provider needs
an API credential. That assumption blocks the in-cluster Ollama provider even
though the ARC runner and Ollama service share the same AKS network.

The live baseline for `qwen3:0.6b` scored `0.0`, with ten
`write_before_fresh_read` failures and five `wrong_target_write` failures.
Optimization should specifically search for an instruction that forces a
fresh read and explicit target verification before any write.

## Considered Approaches

### 1. In-cluster Ollama teacher

Use `ollama_chat/qwen3:14b` through the Kubernetes service DNS name. This keeps
prompts and proposals inside AKS, adds no provider secret, reuses the existing
Spot pool, and directly exercises the repository's intended model lab.

This is the selected approach. Its trade-off is slower CPU inference, bounded
by the existing 150-minute round step and a small GEPA call budget.

### 2. External hosted teacher

OpenAI, Anthropic, or Gemini would likely produce stronger proposals with
lower latency. It would require a new protected secret, external prompt
transmission, and provider cost. Existing hosted-provider support remains
available but is not required for the first round.

### 3. Curated candidate comparison

Manually write several safety-oriented candidates and evaluate them without a
reflection model. This is deterministic and simple, but does not validate the
GEPA optimization path and does not meet the immediate goal.

## Architecture

The GitHub Actions runner remains in `arc-runners-prompt-lab`, while Ollama
remains in the `ollama` namespace on the `modeleval` pool. For
`optimize-evaluate` only, the runner supplies LiteLLM with:

```text
model: ollama_chat/qwen3:14b
OLLAMA_API_BASE: http://ollama.ollama.svc.cluster.local:11434
```

The cluster-local address is constructed from the already validated service
and namespace configuration. The workflow never uses the service's external
load-balancer address.

Provider policy is explicit:

- `ollama` and `ollama_chat` require a model value and no API credential;
- `openai`, `anthropic`, `cohere`, `gemini`, and `google` retain their
  provider-standard protected credential mapping;
- unknown providers continue to use the existing OpenAI-compatible credential
  convention;
- evaluate-only rounds materialize neither a reflection model nor a
  reflection credential.

The access bootstrap accepts a model without a credential only for the two
Ollama provider prefixes. Hosted providers still require model and credential
together. A stale hosted-provider secret is not injected into an Ollama round.

## Data Flow

1. The workflow validates dispatch inputs and exact Prompt Lab and Korvid
   revision provenance before obtaining credentials.
2. Azure OIDC and the read-only Korvid GitHub App token are materialized under
   the protected `aks-grounding` Environment.
3. The round snapshots the `modeleval` node count, installs the cleanup trap,
   and scales the pool from zero to one when needed.
4. Existing AKS preflight waits for the Ollama deployment and target
   `qwen3:0.6b` endpoint.
5. GEPA evaluates the seed and proposals on
   `aks-scale-deployment-up`, while `ollama_chat/qwen3:14b` proposes revised
   instructions through cluster DNS.
6. The unique `best-candidate.yaml` is evaluated on the disjoint
   `aks-restart-denied` validation case and the complete milestone pack, using
   the existing five repetitions per case.
7. The GitHub summary compares the optimized result with the recorded baseline
   and reports score, pass-at-k, hard safety failures, candidate fingerprint,
   and promotion eligibility.
8. Safe evidence includes the best candidate and closed projections only.
   GEPA state, prompts sent to the teacher, raw model answers, logs, and
   credentials remain excluded.
9. Cleanup restores the original node count on success, failure, timeout, or
   cancellation.

## Failure Handling

- Missing Ollama model configuration fails before the node pool is read.
- Ollama never silently falls back to a hosted-provider credential.
- Hosted providers still fail before scale-up when their credential is absent.
- An unreachable cluster-local endpoint or missing teacher model fails the
  optimization; the seed candidate is never substituted as a successful
  result.
- Zero or multiple `best-candidate.yaml` outputs remain fatal.
- A generated candidate that fails the authoritative safety gate remains
  non-promotable even if its aggregate score improves.
- Cleanup behavior and safe-evidence upload remain independent of the round
  conclusion.

## Testing

Unit and infrastructure tests will prove:

- Ollama provider prefixes do not require or receive an API credential;
- hosted providers still require the correct provider-standard credential;
- evaluate-only rounds do not materialize reflection configuration;
- the API base uses the validated namespace and service and stays
  cluster-local;
- unsafe or malformed reflection model values fail before Azure compute work;
- no credential appears in argv, summaries, or safe evidence;
- existing cleanup, artifact allowlist, workflow provenance, lint, type, and
  shell/YAML checks continue to pass.

After merge, remote validation has two stages:

1. A canary `optimize-evaluate` round with four metric calls verifies Ollama
   reflection connectivity, candidate generation, safe evidence, and cleanup.
2. If the canary is infrastructure-clean, a full round with twelve metric
   calls and seed zero evaluates the optimized candidate against the live
   baseline.

The canary may fail the safety gate and still count as infrastructure success.
The full candidate is publishable only if the existing promotion policy marks
it eligible.

## Success Criteria

- The canary and full runs reach the GEPA reflection path without an external
  API credential.
- Both runs upload only allowlisted safe evidence and restore
  `modeleval` to its original count.
- The full run records a best-candidate fingerprint distinct from the seed, or
  explicitly reports that the search was a no-op.
- The full result is compared with the `qwen3:0.6b` baseline; no claim of
  improvement or publication is made unless the recorded metrics and safety
  gate support it.
