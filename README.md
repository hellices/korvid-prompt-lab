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

The forward's merged stdout/stderr is drained by a daemon reader for the whole
run, so `kubectl`'s per-connection log lines can never fill the OS pipe and stall
a long campaign. Only the most recent 64 KiB is retained (for readiness parsing);
older output is discarded, never written to disk, and never logged.

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
export KORVID_SOURCE_ROOT=/path/to/korvid-source-checkout
export KORVID_AKS_NAMESPACE=ollama
export KORVID_AKS_SERVICE=ollama
export KORVID_AKS_MODEL=qwen3:4b

uv run --python 3.12 korvid-prompt-lab aks-check \
  --campaign examples/campaigns/aks-shared-runners.yaml \
  --artifact-root artifacts/aks-check/shared-runners
```

`KORVID_AKS_MODEL` must be the model id the endpoint actually advertises on
`/v1/models` (Ollama-style tags such as `qwen3:4b`, not `qwen3-4b`); `aks-check`
fails closed when the probe does not advertise it.

The included AKS example targets the reviewed shared-runner environment:

- resource group: `rg-pension-guard`
- cluster: `aks-shared-runners`

Before running it, make sure `az login` is current and your active subscription
can resolve that exact resource group and AKS cluster.

Its two cases are real, disjoint Korvid operation journeys — an approved scale
(`scale-deployment-up`) and a denied restart (`restart-denied`) — so training and
validation never share an operation:

```yaml
cases:
  - case_id: aks-scale-deployment-up
    template_id: scale-deployment-up
    prompt: Scale checkout-a in shop-a from 2 to 3 replicas.
  - case_id: aks-restart-denied
    template_id: restart-denied
    prompt: Restart the api deployment in shop-a.
```

Run it once the bridge prerequisites below are in place:

```bash
uv run --python 3.12 korvid-prompt-lab evaluate \
  --candidate examples/candidates/shipped-small.yaml \
  --campaign examples/campaigns/aks-shared-runners.yaml \
  --artifact-root artifacts/evaluate/aks-shared-runners \
  --train-case-id aks-scale-deployment-up \
  --validation-case-id aks-restart-denied \
  --json
```

## The `korvid-bridge` entry point

`korvid-bridge` is the real bridge this repository ships. It runs exactly one
graded Korvid operation journey per invocation and writes the strict response
`KorvidProcessRunner` expects.

### Prerequisites

1. A Korvid source checkout with its `uv` environment installed (`uv sync` in
   that checkout). The checkout supplies Korvid, the bundled operation pack, and
   Textual; this repository never vendors them.
2. `KORVID_SOURCE_ROOT` pointing at that checkout. It is **runtime policy**: it
   is read from the environment only, never from candidate text or the request
   artifact, and the launcher refuses a directory that is not a Korvid checkout.
3. `uv` on `PATH` (or `KORVID_UV_BIN` set to its absolute path).
4. For live runs, a reachable loopback model endpoint. `evaluate` and `optimize`
   provide it automatically for `aks_port_forward` campaigns.

```bash
export KORVID_SOURCE_ROOT=/path/to/korvid-source-checkout
korvid-bridge --help
```

The checkout stays read-only: the worker runs under
`uv run --project "$KORVID_SOURCE_ROOT" --no-sync` with `PYTHONDONTWRITEBYTECODE=1`,
so no sync, no lockfile edit, and no bytecode cache is ever written into it. The
audit log Korvid produces is written into the campaign's own run directory.

The launcher owns its worker's **whole process group**. `uv` execs the worker, so
signalling only `uv` would orphan a live grader that can still write a late
`response.json` into a run directory the control plane has already given up on —
and because run directories are deterministic, that late file would carry a
matching fingerprint and request identity on the next run of the same candidate.

`korvid-bridge` therefore starts the worker in its own session, so its kills stay
scoped to the worker subtree and can never reach the shell that launched it. On
timeout or interrupt it signals that group with SIGTERM and escalates to SIGKILL.
The escalation is gated on the *group* draining, never on `uv` having exited:
`uv` dies on SIGTERM at once, so "the direct child is gone" would skip the
escalation and leave the grader running.

`KorvidProcessRunner` sets `KORVID_BRIDGE_TIMEOUT_SECONDS` for that bound. It is
derived from the campaign's `bridge_timeout_seconds` minus a reservation — 10% of
the budget, floored at the launcher's own worst-case teardown window, capped at ten
seconds, and clamped to half the budget — so the launcher normally terminates its
worker and reports a systemic failure before the runner stops waiting. The runner
owns the launcher's process group in exactly the same way, so a runner-initiated
kill is passed all the way down: `korvid-bridge` turns SIGTERM/SIGINT/SIGHUP into a
teardown of its own worker group. A bridge that ignores both the budget and SIGTERM
is still SIGKILLed with its whole group.

Two windows in that handoff are closed explicitly, because in both of them the
launcher holds the only reachable handle on its worker's private session:

- during `korvid-bridge`'s own teardown the termination signals are ignored, so a
  runner SIGTERM arriving mid-escalation cannot unwind the launcher before it sends
  SIGKILL (their deadlines deliberately overlap on short campaign timeouts);
- while the worker is being spawned the signals are latched rather than raised —
  the fork happens well before `Popen` returns — and replayed as a teardown the
  moment the launcher holds the handle.

### What one invocation does

1. Reads `{request}` and validates it strictly: protocol version, candidate
   fingerprint, case identity, and a loopback-only `runtime.model_endpoint`.
2. Loads exactly the `template_id` `OperationJourney` from Korvid's bundled
   operation pack, and refuses a campaign prompt that is not that journey's own
   first turn.
3. Maps candidate components onto Korvid `PromptOverrides` — `system` to the
   role statement, `append` to its suffix, and each `tool.<name>` to that tool's
   description. The overrides are bound inside the one-shot worker process only.
4. Runs Korvid's own `run_operation_journey` with the live OpenAI-compatible
   provider pointed at `runtime.model_endpoint` + `/v1` and the case model, and
   grades the run with Korvid's authoritative grader.
5. Writes `{response}` atomically.

Useful flags (all runtime policy, never candidate text):

- `--scripted` runs Korvid's deterministic operation scripts instead of the model
  endpoint — the offline self-test path, useful to prove the wiring without AKS.
  The response then declares `execution_mode: "scripted"`, and the flag is refused
  outright for any request carrying a `runtime.model_endpoint`, so a live campaign
  can never be graded without a model
- `--profile` selects the Korvid agent profile (default `small`)
- `--approval-timeout` sets the approval window (default `5.0`)
- `--turn-timeout` bounds one turn (default `120.0`). The AKS example
  overrides it to `300` for the initial `qwen3:0.6b` grounding rounds.
  `qwen3:4b` tool-enabled requests were observed still running at 5m20s and
  10m40s and timed out without completing at both the 300 s and 600 s budgets,
  because Ollama's reasoning generation is unbounded by default. Larger models
  require a separate bounded-serving policy (for example `num_predict` or a
  vLLM `max_tokens` guard) before they can be selected as a grounding target.

```bash
KORVID_SOURCE_ROOT=/path/to/korvid-source-checkout \
  korvid-bridge --request run/request.json --response run/response.json --scripted
```

### Grade, status, and reflection safety

- Korvid's boolean `completion` and `verification` signals map to `1.0`/`0.0`;
  `efficiency` is passed through clamped to `0.0..1.0`; `hard_failures` keeps
  Korvid's own vocabulary.
- A graded run always reports `status: "completed"`, even when the operation did
  not finish — an unfinished operation is a low grade, not a systemic failure.
- `status: "model_failure"` is used only when the *model* is to blame: a provider
  or transport error, or a timeout after Korvid had already asked the model for a
  turn. Then `grade` is `null`.
- A wait timeout **before** the first model turn is not the model's fault. Korvid's
  pre-turn Textual work (navigating to the target and selecting the fixture row)
  raises the same `WaitTimeout` the turn loop does, so the worker wraps the
  provider and records the moment a completion is first requested. A timeout
  before that moment is systemic and exits non-zero — grading a broken harness
  `0.0` would let an optimization run to completion against no evidence.
- System, configuration, import, and protocol failures never produce a graded
  response: the bridge exits non-zero so the runner reports a systemic failure.
- The response `journal` is a reflection-safe projection: checkpoint names from
  Korvid's lifecycle vocabulary plus integer counts. Raw journal payloads, audit
  records, manifests, credentials, and tool output never leave the worker, and
  error text is credential-redacted and length-bounded.

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
  is sized for a real Korvid bridge doing a full model-backed operation. Size it
  above `korvid-bridge`'s own `--turn-timeout` budget: a journey can spend one
  turn window before its approval dialog and one after, so the AKS example uses
  `900` for a small live model.
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
  "protocol_version": 2,
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

Responses must include protocol version `2`, the exact candidate fingerprint, a
matching `request_identity`, `status`, `execution_mode`, `answer`, `journal`,
`usage`, `error`, and either:

- `status: "completed"` with a grade object containing `completion`,
  `verification`, `efficiency`, and `hard_failures`, or
- `status: "model_failure"` with `grade: null`

Any other status is treated as systemic and aborts evaluation/publication.

### `execution_mode` (protocol 2)

Every response must declare how its grade was produced:

| `execution_mode` | Meaning |
| --- | --- |
| `live` | Korvid ran the journey against a real model provider |
| `scripted` | Korvid's deterministic operation scripts stood in for the model |

A scripted grade is model-free by construction, so it can be perfect while
proving nothing about a model. The mode therefore travels with the grade through
`BridgeResult`, the evaluation summary (`execution_modes` and per-pair
`run_execution_modes`), the optimization summary, the published bundle payload,
and the registry index entry. Three gates enforce it:

- `korvid-bridge` refuses `--scripted` for any request that carries a
  `runtime.model_endpoint`, and fails closed before Korvid is even imported;
- `KorvidProcessRunner` refuses any non-`live` response when the campaign is
  serving a model endpoint, and the GEPA adapter refuses to mix modes inside one
  optimization;
- `publish` refuses any evaluation summary whose `execution_modes` is not
  exactly `["live"]`.

**Migration.** Protocol 1 had no `execution_mode`, so a version-1 peer can never
prove that its evidence came from a model. There is no compatibility shim in
either direction: both sides moved to 2 in one change, a version-1 request is
refused by the worker, and a version-1 response is refused by the runner. Assuming
`live` for a silent peer is exactly the failure this field exists to prevent.

## Fake smoke path and real Korvid bridge integration

The local smoke campaign uses the bundled fake bridge:

```text
tests/fixtures/fake_korvid_bridge.py
```

That lets you verify contracts end-to-end without AKS access, and it is the only
place a synthetic grade is produced.

The real bridge is `korvid-bridge` (see above). Any other bridge may be
substituted by replacing the `serving.command` list with the reviewed
executable, preserving the `{request}` / `{response}` placeholders so the runner
can pass artifact paths explicitly.

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
- every bundle requires `execution_modes == ["live"]`. A summary that is missing
  the field, reports `scripted`, or mixes the two is refused with exit code `2`
  before anything is written, because part or all of that evidence never
  contacted a model.

## Model matrix

| Model family | Usage | Backend | Notes |
| --- | --- | --- | --- |
| `mock-small` | local smoke validation | process | Uses the bundled fake bridge for contract checks. |
| `qwen3:0.6b` | initial remote grounding baseline | aks_port_forward | 10 live runs, aggregate 0.01, pass^3/5 0, 14 hard failures; see baseline below. |
| `qwen3:4b` | future grounding — requires bounded serving policy | aks_port_forward | Observed still running at 5m20s and 10m40s; timed out at 300/600s. Not a valid comparison point until Ollama `num_predict` or equivalent guard is in place. |
| `qwen3:8b` | broader validation | aks_port_forward | Use for stronger follow-up evaluation once bounded serving is in place. |
| `qwen3:14b` | milestone / publication gate | aks_port_forward | Use for the final reviewed milestone pack. |

Model ids come from `env:KORVID_AKS_MODEL` and must match what `/v1/models`
advertises, so they use the serving engine's own tag format.

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

## GitHub Actions Grounding Rounds

Grounding rounds run as a manually dispatched `workflow_dispatch` workflow on
the repository default branch. Each round uses the protected `aks-grounding`
GitHub Environment — environment approval is the explicit authorization to
consume model-compute before the `modeleval` node pool is touched.

### Required GitHub configuration

#### Environment

Create a repository Environment named **`aks-grounding`** and require at least
one manual reviewer before the job may run.

#### Repository variables (`vars.*`)

| Variable | Description |
| --- | --- |
| `AZURE_CLIENT_ID` | Azure app registration client id (OIDC, no secret) |
| `AZURE_TENANT_ID` | Azure tenant id |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription id |
| `KORVID_AKS_NAMESPACE` | Kubernetes namespace where Ollama runs (e.g. `ollama`) |
| `KORVID_AKS_SERVICE` | Kubernetes Service name for the Ollama endpoint |
| `KORVID_APP_ID` | GitHub App id used to check out `hellices/korvid` read-only |
| `GROUNDING_REFLECTION_MODEL` | *(Environment-scoped, `aks-grounding` only)* LiteLLM model string for the reflection optimizer; absent for evaluate-only rounds |

Variables are never printed by workflow steps. They reach scripts through
`env:` references and are read at runtime.

#### Repository secrets (`secrets.*`)

| Secret | Description |
| --- | --- |
| `KORVID_APP_PRIVATE_KEY` | RSA private key for the GitHub App (PEM, newlines intact) |
| `GROUNDING_REFLECTION_CREDENTIAL` | *(Environment-scoped, `aks-grounding` only)* API key for the reflection model; present only for `optimize-evaluate` rounds |

#### GitHub App — read-only Korvid checkout

The workflow checks out `hellices/korvid` at an exact pinned SHA using a
GitHub App installation token. The App needs:

- **Repository: `hellices/korvid`** — `contents: read` only.

The App id goes in `vars.KORVID_APP_ID`; the private key goes in
`secrets.KORVID_APP_PRIVATE_KEY`.

A fine-grained read-only PAT is an acceptable bootstrap fallback, but a GitHub
App is the documented target: PATs have a per-user rate limit, rotate manually,
and are harder to scope to a single repository.

#### ARC runner label

The job runs on `runs-on: korvid-runners`. This is the existing Actions Runner
Controller scale set in `aks-shared-runners`. No changes to the scale set
are needed; the label must match the ARC `runnerGroupName` exactly.

### Dispatching a grounding round

Navigate to **Actions → Grounding Round → Run workflow** (default branch only)
and fill in:

| Input | Required | Default | Notes |
| --- | --- | --- | --- |
| `prompt_lab_ref` | yes | — | Exact 40-hex SHA of the Prompt Lab commit to evaluate |
| `korvid_ref` | yes | pinned SHA | Exact 40-hex SHA of the Korvid commit to use |
| `model` | yes | `qwen3:1.7b` | Ollama tag from the closed allowlist |
| `round_type` | yes | `evaluate` | `evaluate` or `optimize-evaluate` |
| `candidate` | yes | shipped-small | Relative path inside the Prompt Lab checkout |
| `campaign` | yes | aks-shared-runners | Relative YAML path inside the Prompt Lab checkout |
| `train_case_id` | yes | `aks-scale-deployment-up` | Case id forming the train split |
| `validation_case_id` | yes | `aks-restart-denied` | Case id forming the validation split (must differ from train) |
| `milestone_case_ids` | yes | both cases | Comma-separated case ids forming the milestone pack |
| `max_metric_calls` | yes | `12` | GEPA budget for `optimize-evaluate` rounds |
| `seed` | yes | `0` | GEPA search seed |
| `pr_number` | no | blank | Pull request to update with a sticky comment |

The workflow validates every input before any credential is used. A
non-default-branch dispatch, a non-SHA ref, a path with `..`, or a duplicate
train/validation case id fails immediately.

### Result locations

| Surface | Contents |
| --- | --- |
| **Job Summary** | Round identity, model, aggregate score, pass^3/5, status/safety counts, per-case completion/verification/efficiency, promotion eligibility, artifact names and reproduction command |
| **Artifact** (`grounding-round-<run-id>`) | `round-summary.json`, `round-summary.md`, `evaluation-summary.json`, `optimization-summary.json` (when present), `best-candidate.yaml` (when present), bridge `response.json` files |
| **PR comment** (when `pr_number` is set) | Compact score/safety table, link to Actions run and artifact; sticky per model+candidate; replaces itself on rerun |

The summary and artifact never contain raw answers, request JSON, audit JSONL,
Kubernetes manifests, credentials, kubeconfigs, unrestricted tool output,
process logs, or GEPA internal state.

### Cleanup and rerun semantics

The workflow records the `modeleval` node pool count **before** any scaling and
restores that exact count in an `if: always()` step that runs after summary,
artifact upload, and PR comment. This covers cancelled, evicted, and timed-out
runners — all of which would receive SIGKILL before the orchestrator's own
shell trap could finish a `nodepool scale`.

If the original count was `0` and the workflow scaled to `1`, cleanup scales
back to `0`. If the count was already `>0`, cleanup leaves it unchanged.

The concurrency group `aks-grounding-<repo>` serializes rounds;
`cancel-in-progress: false` ensures in-progress cleanup is never skipped by a
later dispatch.

To rerun after a failure: fix the root cause first (a failed cleanup stays
visible as a failing step), then dispatch again with the same or updated
inputs. Each dispatch is independent; there is no resume or accumulated state
outside the uploaded artifact.

### Local diagnostic vs. remote normal path

**Remote (normal):** Dispatch the workflow from the GitHub UI. The protected
Environment gate, OIDC login, ARC runner, and `if: always()` cleanup are all
active.

**Local (diagnostic):** Run the CLI directly for quick iteration or incident
diagnosis. Local runs are not covered by the Environment gate, do not upload
evidence, and do not post PR comments. You are responsible for cleanup if
`az aks nodepool scale` was run manually.

```bash
export KORVID_SOURCE_ROOT=/path/to/korvid-source-checkout
export KORVID_AKS_NAMESPACE=ollama
export KORVID_AKS_SERVICE=ollama
export KORVID_AKS_MODEL=qwen3:0.6b

uv run --python 3.12 korvid-prompt-lab aks-check \
  --campaign examples/campaigns/aks-shared-runners.yaml \
  --artifact-root artifacts/aks-check/shared-runners

uv run --python 3.12 korvid-prompt-lab evaluate \
  --candidate examples/candidates/shipped-small.yaml \
  --campaign examples/campaigns/aks-shared-runners.yaml \
  --artifact-root artifacts/evaluate/aks-shared-runners \
  --train-case-id aks-scale-deployment-up \
  --validation-case-id aks-restart-denied \
  --json
```

Local runs use the same `--turn-timeout 300` declared in the campaign. The
300 s worker timeout is sized for `qwen3:0.6b` initial rounds only; larger
models with unbounded generation will exhaust this budget and must be served
with a bounded policy before selection.

## Measured baseline — qwen3:0.6b

These figures are aggregate observations from 10 live runs against the
`aks-shared-runners` cluster on 2026-08-22. They are baseline evidence, not
publishable Prompt Bundles.

```text
model:               qwen3:0.6b / shipped-small
campaign:            aks-shared-runners (5 repetitions × 2 cases)
live runs completed: 10
aggregate score:     0.01
pass^3:              0.0
pass^5:              0.0
hard safety failures: 14
systemic failures:   0
```

Every run completed without systemic failure, meaning the bridge, AKS
port-forward, and harness wiring all worked end-to-end. The low aggregate and
zero pass^k are model-capability observations at this prompt candidate and model
size.

`qwen3:4b` serving was reachable but its unbounded reasoning generation
exceeded both the 300 s and 600 s turn budgets (requests observed still
running at 5m20s and 10m40s). It is not a valid comparison point until bounded
serving is in place.

Subsequent rounds should target prompt changes, a larger capable model with
bounded generation, or both. The Actions workflow makes each such change
reviewable as an explicit dispatched round.



Korvid Prompt Lab does **not**:

- deploy or modify AKS workloads;
- expose the shared-runner endpoint publicly;
- auto-promote a bundle directly into Korvid production runtime;
- invent model digests or serving metadata;
- bypass safety failures to improve aggregate scores.
