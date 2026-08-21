# Korvid Prompt Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the empty repository into a tested DSPy/GEPA prompt-optimization control plane that evaluates candidates through a versioned Korvid subprocess contract and publishes verified common and model-specific prompt bundles.

**Architecture:** Typed YAML configuration feeds immutable candidate and case contracts. A subprocess runner invokes the real Korvid bridge (or the bundled fake), a GEPA adapter turns authoritative grades into scores and compact reflection datasets, and a CLI validates, evaluates, and optimizes candidates while persisting reproducible artifacts.

**Tech Stack:** Python 3.11+, `dspy`, `gepa`, PyYAML, pytest, mypy, ruff, standard-library argparse/subprocess/dataclasses.

## Global Constraints

- Korvid remains the system under optimization; do not reproduce its agent loop.
- Candidate text is separate from fixed runtime knobs.
- Safety failures cannot be averaged away.
- Systemic bridge failures abort optimization; executed model failures are scored.
- Reflection data must not include raw manifests, audit payloads, credentials, or unrestricted tool output.
- The existing AKS model service is reached only through loopback port-forward; this repository does not deploy or mutate it.
- Prefer a common prompt bundle; publish a model-specific override only after a configured improvement and a zero-safety-failure milestone run.

---

### Task 1: Repository and strict contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/korvid_prompt_lab/__init__.py`
- Create: `src/korvid_prompt_lab/contracts.py`
- Create: `src/korvid_prompt_lab/config.py`
- Create: `tests/test_contracts.py`
- Create: `examples/candidates/shipped-small.yaml`
- Create: `examples/campaigns/local-smoke.yaml`
- Create: `examples/campaigns/aks-shared-runners.yaml`

**Interfaces:**
- Produces: `Candidate.from_mapping`, `Candidate.fingerprint`, `EvalCase`, `Campaign`, `load_candidate`, `load_campaign`.

- [ ] **Step 1: Write failing contract tests**

Test valid YAML loading, deterministic SHA-256 fingerprints, duplicate case IDs,
unknown fields, invalid component keys, invalid repetitions, and model coverage.

- [ ] **Step 2: Run the focused test**

Run: `pytest tests/test_contracts.py -q`
Expected: FAIL because the package does not exist.

- [ ] **Step 3: Implement immutable dataclasses and strict parsers**

Use `dataclass(frozen=True, slots=True)`. Accept only schema version `1`.
Components must be non-empty string keys and values; allowed keys are `system`,
`append`, and `tool.<tool-name>`. Canonical JSON with sorted keys produces the
fingerprint. Reject unknown keys at every YAML object boundary.
Campaign serving configuration supports `process` and `aks_port_forward`
backends; the AKS example names resource group `rg-pension-guard`, cluster
`aks-shared-runners`, and obtains namespace, Service, and model values from
environment variables rather than committing credentials.

- [ ] **Step 4: Add package metadata and examples**

Declare runtime dependencies `dspy`, `gepa`, and `PyYAML`; development
dependencies `pytest`, `mypy`, and `ruff`. Register
`korvid-prompt-lab = korvid_prompt_lab.cli:main`.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/test_contracts.py -q`
Expected: PASS.

Commit: `feat: add prompt lab contracts`

### Task 2: Artifact store, scoring, and bridge runner

**Files:**
- Create: `src/korvid_prompt_lab/artifacts.py`
- Create: `src/korvid_prompt_lab/scoring.py`
- Create: `src/korvid_prompt_lab/runner.py`
- Create: `tests/fixtures/fake_korvid_bridge.py`
- Create: `tests/test_scoring.py`
- Create: `tests/test_runner.py`

**Interfaces:**
- Consumes: `Candidate`, `EvalCase`, `Campaign`.
- Produces: `OperationGrade`, `BridgeResult`, `ScoredResult`,
  `score_result(result)`, `KorvidProcessRunner.run(candidate, case, run_dir)`.

- [ ] **Step 1: Write failing scoring and runner tests**

Cover the `0.6/0.3/0.1` score, hard-failure zeroing, systemic status rejection,
model-failure acceptance, timeout, non-zero exit, missing output, malformed JSON,
protocol mismatch, fingerprint mismatch, and atomic artifact creation.

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/test_scoring.py tests/test_runner.py -q`
Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement score types and runner**

Use `subprocess.run(..., shell=False, timeout=...)`. Expand only the literal
tokens `{request}` and `{response}` in a command list. Write requests and
responses as UTF-8 JSON. Raise typed `BridgeSystemError` subclasses for systemic
failures. Return normal `BridgeResult` objects for `completed` and
`model_failure` statuses.

- [ ] **Step 4: Implement the fake bridge**

The fixture reads the request, derives deterministic outcomes from case tags,
and writes a version-1 response carrying the exact candidate fingerprint. It
supports flags for malformed output, timeout, protocol mismatch, and systemic
failure so integration tests exercise the real subprocess boundary.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/test_scoring.py tests/test_runner.py -q`
Expected: PASS.

Commit: `feat: add Korvid bridge runner`

### Task 3: GEPA adapter and DSPy reflection

**Files:**
- Create: `src/korvid_prompt_lab/adapter.py`
- Create: `src/korvid_prompt_lab/reflection.py`
- Create: `src/korvid_prompt_lab/optimize.py`
- Create: `tests/test_adapter.py`
- Create: `tests/test_reflection.py`
- Create: `tests/test_optimize.py`

**Interfaces:**
- Consumes: `KorvidProcessRunner`, campaign cases, candidate component maps.
- Produces: `KorvidGEPAAdapter.evaluate`, `make_reflective_dataset`,
  `DSPyInstructionProposer.__call__`, `optimize_campaign`.

- [ ] **Step 1: Write failing adapter tests**

Assert one output/score/trajectory per case, zero score for unsafe candidates,
systemic-error propagation, compact reflection records, sensitive-field
exclusion, and candidate component immutability.

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/test_adapter.py tests/test_reflection.py tests/test_optimize.py -q`
Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement `KorvidGEPAAdapter`**

Implement GEPA's `evaluate(batch, candidate, capture_traces=False)` and
`make_reflective_dataset(candidate, eval_batch, components_to_update)`.
Trajectories contain only typed safe summaries. Create one run directory per
candidate fingerprint and case execution.

- [ ] **Step 4: Implement DSPy-backed proposals**

Define a `dspy.Signature` taking current component text and JSON reflection
records and returning revised component text. Instantiate `dspy.Predict` lazily
so validation and evaluation do not require a reflection model. Reject blank
proposals and components not requested by GEPA.

- [ ] **Step 5: Implement GEPA orchestration**

Call `gepa.optimize` with the adapter, seed component map, train/validation
cases, bounded `max_metric_calls`, optional DSPy proposer, and artifact log
directory. Persist the best candidate as strict candidate YAML plus an
optimization summary JSON.

- [ ] **Step 6: Verify and commit**

Run: `pytest tests/test_adapter.py tests/test_reflection.py tests/test_optimize.py -q`
Expected: PASS.

Commit: `feat: add GEPA optimization adapter`

### Task 4: Existing AKS serving backend

**Files:**
- Create: `src/korvid_prompt_lab/aks.py`
- Create: `tests/test_aks.py`

**Interfaces:**
- Consumes: AKS serving configuration.
- Produces: `AKSPortForward`, a context manager exposing `base_url: str` and
  cleaning up only its own process and kubeconfig.

- [ ] **Step 1: Write failing AKS tests**

Mock command execution and cover missing resource group, wrong cluster,
non-Ready endpoints, port-forward early exit, model-probe mismatch, loopback
binding, unique temporary kubeconfig, and exact process cleanup.

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/test_aks.py -q`
Expected: FAIL because the AKS backend does not exist.

- [ ] **Step 3: Implement read-only discovery and forwarding**

Use argument lists with `az aks show`, `az aks get-credentials --file`, and
`kubectl --kubeconfig ... get service/endpoints`. Start
`kubectl port-forward --address 127.0.0.1`; probe `/v1/models`; never print
subscription identifiers or kubeconfig content; terminate and wait for only
the spawned process in `__exit__`.

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/test_aks.py -q`
Expected: PASS.

Commit: `feat: add AKS model endpoint backend`

### Task 5: Prompt bundle publishing

**Files:**
- Create: `src/korvid_prompt_lab/publish.py`
- Create: `tests/test_publish.py`

**Interfaces:**
- Consumes: candidate, campaign, model metadata, and evaluation summary.
- Produces: `PromptBundle`, `PromotionDecision`, `publish_bundle`, and
  `render_scoreboard`.

- [ ] **Step 1: Write failing publication tests**

Cover deterministic bundle versions, zero-safety gate, common-bundle
preference, minimum model-specific improvement, exact model digest
requirements, registry index ordering, and Markdown scoreboard generation.

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/test_publish.py -q`
Expected: FAIL because publication support does not exist.

- [ ] **Step 3: Implement promotion and registry writes**

Write immutable bundle directories under `registry/bundles`, update
`registry/index.json` atomically, and regenerate `registry/scoreboard.md`.
Reject model-specific promotion unless its paired common baseline exists and
the improvement threshold is met.

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/test_publish.py -q`
Expected: PASS.

Commit: `feat: publish verified prompt bundles`

### Task 6: CLI, documentation, and full verification

**Files:**
- Create: `src/korvid_prompt_lab/cli.py`
- Create: `tests/test_cli.py`
- Create: `README.md`
- Create: `.gitignore`

**Interfaces:**
- Consumes: configuration loaders, runner, and optimizer.
- Produces: `korvid-prompt-lab validate`, `evaluate`, `optimize`, `aks-check`,
  and `publish`.

- [ ] **Step 1: Write failing CLI tests**

Test help, validation success, malformed configuration exit `2`, fake-bridge
evaluation, systemic bridge exit `1`, JSON summary output, AKS endpoint
preflight, and prompt bundle publication.

- [ ] **Step 2: Run focused CLI tests**

Run: `pytest tests/test_cli.py -q`
Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement argparse commands**

`validate` loads and cross-validates candidate/campaign files. `evaluate`
executes selected cases and fails when any result is unsafe.
`optimize` requires reflection-model configuration and invokes
`optimize_campaign`. `publish` applies the promotion policy and writes the
registry. `aks-check` performs read-only serving preflight. Print concise
summaries to stdout and diagnostics to stderr.

- [ ] **Step 4: Write operator documentation**

Document installation with `uv`, the bridge request/response schema, fake
smoke commands, real Korvid bridge integration, `rg-pension-guard` /
`aks-shared-runners` port-forward setup, safety semantics, common versus
model-specific promotion, model matrix, artifacts, and the non-goals.

- [ ] **Step 5: Run complete verification**

Run:

```bash
pytest -q
ruff check .
mypy src tests
korvid-prompt-lab validate \
  --candidate examples/candidates/shipped-small.yaml \
  --campaign examples/campaigns/local-smoke.yaml
```

Expected: all commands pass.

- [ ] **Step 6: Commit**

Commit: `feat: complete Korvid prompt lab MVP`
