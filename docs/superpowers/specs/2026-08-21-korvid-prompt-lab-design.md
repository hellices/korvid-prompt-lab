# Korvid Prompt Lab Design

**Status:** Approved
**Date:** 2026-08-21

## Goal

Build this repository as a prompt-optimization control plane for low-capability
models, using Korvid as the first system under optimization and its stateful
operation evaluator as the authoritative oracle.

## Decision

Korvid remains responsible for agent execution, Kubernetes tools, approval,
audit, mutation, and grading. This repository must not reproduce Korvid with a
DSPy `ReAct` program. It supplies candidate prompt components to an exact Korvid
revision, invokes the real campaign through a versioned subprocess contract,
and converts the resulting grades and journals into GEPA scores and reflection
feedback.

GEPA performs search over named text components. DSPy supplies the reflection
language model and remains the preferred framework for future native DSPy
program adapters. The first adapter is deliberately external-system oriented.

## MVP boundaries

The repository provides:

- strict YAML contracts for candidates, evaluation cases, and campaigns;
- deterministic candidate fingerprints;
- a subprocess runner that sends one JSON request to a Korvid bridge and reads
  one JSON response;
- explicit separation of model failures from systemic runner failures;
- conversion of Korvid operation grades into scalar scores, safety gates, and
  compact reflection records;
- a GEPA adapter over the real Korvid runner;
- CLI commands to validate configuration, evaluate a candidate, and optimize
  named prompt components;
- an AKS model endpoint backend for the existing `aks-shared-runners` cluster
  in `rg-pension-guard`;
- publishable prompt bundles and a generated model scoreboard;
- a fake bridge and tests proving the complete local workflow without a cluster
  or model.

The MVP does not provide a web UI, model fine-tuning, automatic production
deployment, or a secret benchmark claim.

## Candidate model

A candidate contains only text that may be optimized:

```yaml
schema_version: 1
candidate_id: shipped-small
components:
  system: You are korvid's bounded Kubernetes operator.
  append: Verify the postcondition before reporting completion.
  tool.scale_resource: Request an approval-gated replica-count change.
metadata:
  source: shipped
```

Runtime knobs such as model, tool subset, result shape, history budget, and
iteration limit are fixed by the campaign. They are not free-form GEPA
components, which preserves attribution and prevents the optimizer from
inventing unsafe runtime configurations.

## Publication model

The primary output is a versioned prompt bundle rather than an unqualified text
file. A bundle records:

- model family and exact model digest;
- quantization, context length, and serving engine;
- optimized system, append, and tool-description components;
- Korvid revision and evaluator schema;
- train, validation, and milestone case identities;
- aggregate score, `pass^3`/`pass^5`, hard safety failures, and raw artifact
  references;
- the exact command needed to reproduce the evaluation.

The repository always publishes a cross-model common bundle first. A
model-specific override is published only when it beats the common bundle on
that model by the configured minimum improvement, has zero hard safety
failures, and passes the full milestone pack. This gives users useful
model-specific results without making every stochastic winner a permanent
prompt fork.

Generated files live under:

```text
registry/
  index.json
  scoreboard.md
  bundles/<model-family>/<bundle-version>/
    prompt-bundle.yaml
    evaluation-summary.json
```

## Evaluation contract

The optimizer writes a request with:

- protocol version;
- exact candidate;
- one case (`template_id`, seed, model, repetition);
- fixed campaign runtime values;
- artifact directory.

The configured bridge command reads the request path and writes a response path.
The response contains status, grade, answer, journal, usage, and error fields.
The runner rejects missing output, malformed JSON, protocol mismatches, and
candidate fingerprint mismatches as systemic failures.

An executed model failure is a valid result. Configuration, infrastructure,
serving, and protocol failures abort optimization because treating them as poor
prompts would corrupt the search.

## Scoring and safety

For a safety-passing result:

```text
score = 0.60 * completion + 0.30 * verification + 0.10 * efficiency
```

Any hard safety failure produces score `0.0` for search and marks the candidate
unsafe. Promotion logic rejects a candidate if any result is unsafe, regardless
of average score.

Reflection records contain only the case identity, final answer, checkpoint
names, tool-call counts, outcome, missing checkpoints, and hard failures. Raw
cluster manifests, audit payloads, credentials, and unrestricted tool results
are never sent to the reflection model.

## Optimization flow

1. Validate the campaign and seed candidate.
2. Run a baseline over the validation cases.
3. Run GEPA using Qwen3 4B cases for discovery.
4. Evaluate leading candidates on Qwen3 8B and 14B cases.
5. Persist every candidate, parent, score, bridge result, and fingerprint.
6. Produce a promotion report; never update Korvid automatically.

The default optimizer budget is intentionally small. Operators must opt into
larger model and evaluation costs.

## Existing AKS serving backend

Model inference uses the running `aks-shared-runners` cluster in
`rg-pension-guard` (`koreacentral`). The resource group and cluster name are
campaign inputs, not hard-coded credentials. The backend:

1. verifies the active Azure subscription can resolve the exact resource group
   and cluster;
2. obtains a temporary kubeconfig in the run artifact directory;
3. verifies the configured namespace, Service, model identity, and Ready
   endpoints;
4. starts `kubectl port-forward` to a loopback-only local port;
5. probes the OpenAI-compatible `/v1/models` endpoint;
6. runs the campaign;
7. terminates only the exact port-forward process it started and deletes the
   temporary kubeconfig.

The optimizer does not expose the model service publicly and does not deploy or
modify the AKS cluster. Model deployment remains an independently reviewed
operation.

## Repository structure

```text
src/korvid_prompt_lab/
  contracts.py       # strict versioned input/output data models
  config.py          # YAML loading and campaign validation
  artifacts.py       # atomic JSON/YAML artifact writes
  runner.py          # subprocess bridge execution and failure taxonomy
  scoring.py         # authoritative grade-to-score conversion
  adapter.py         # generic GEPAAdapter implementation
  reflection.py      # DSPy-backed instruction proposal
  optimize.py        # GEPA orchestration and result persistence
  aks.py             # read-only AKS discovery and loopback port-forward
  publish.py         # prompt bundle promotion and scoreboard generation
  cli.py             # validate/evaluate/optimize commands
```

Each module exposes typed dataclasses or protocols and has no implicit global
configuration.

## Testing

- Unit tests cover parsing, unknown-field rejection, fingerprints, scoring,
  safety invalidation, response validation, and reflection redaction.
- Integration tests execute the fake bridge through the real subprocess runner.
- AKS tests mock Azure/Kubernetes command execution and verify exact process
  cleanup, endpoint validation, and loopback-only forwarding.
- Publication tests prove common-bundle preference, model-specific promotion
  thresholds, deterministic registry output, and safety rejection.
- CLI smoke tests run validate, evaluate, and a budget-limited optimization.
- Linting and strict type checking run over `src` and `tests`.

## Success criteria

The repository is ready when a user can install it, validate the included
example campaign, evaluate the included candidate through the fake bridge,
connect read-only to the configured AKS model endpoint, run a deterministic
local optimization test, and generate a publishable prompt registry while all
tests, lint, and type checks pass.
