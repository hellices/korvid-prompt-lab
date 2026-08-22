# GitHub Actions Grounding Rounds

## Goal

Run reproducible Korvid prompt-evaluation and optimization rounds remotely,
using the existing `korvid-runners` Actions Runner Controller scale set in
`aks-shared-runners`. Each round must leave a concise GitHub Actions summary,
safe downloadable evidence, and an optional Pull Request comment.

The workflow replaces operator-local orchestration. Local execution remains a
development and incident-diagnosis path, not the normal grounding path.

## Considered approaches

### 1. Existing AKS ARC runner with manual dispatch

Use `runs-on: korvid-runners`, Azure workload federation, and a protected
`aks-grounding` GitHub Environment. A manually dispatched round selects the
candidate, model, Korvid revision, and operation budget.

This is the selected approach. The runner is already present in the target
cluster, avoids private-cluster networking changes, and gives explicit cost and
credential approval before model nodes start.

### 2. GitHub-hosted runner with Azure OIDC

This has lower runner administration overhead, but requires the AKS API server
and model route to be reachable from GitHub-hosted infrastructure. It would
weaken the current private, loopback-only serving posture or require additional
networking.

### 3. Automatic Pull Request trigger

Automatically grounding every PR is convenient but unsafe and expensive. PR
code would execute near cloud credentials, fork behavior is difficult to secure,
and repeated pushes can keep the model pool running. `pull_request_target` is
explicitly prohibited.

## Trigger and trust boundary

The primary trigger is `workflow_dispatch` on the repository default branch.
Inputs are:

- Prompt Lab git ref, defaulting to the selected workflow revision.
- Korvid git revision, required and recorded verbatim.
- Candidate path.
- Model from a closed allowlist.
- Train, validation, and milestone case identifiers.
- Round type: `evaluate` or `optimize-evaluate`.
- Metric-call budget for optimization.
- Optional Pull Request number for a result comment.

The job uses the protected `aks-grounding` Environment. Environment approval is
the authorization to consume model compute. The workflow never executes
untrusted fork code with Azure or cross-repository credentials.

The existing ARC runner scale-set name is `korvid-runners`.

## Credentials

Azure login uses GitHub OIDC through `azure/login`; no Azure client secret is
stored. Azure identifiers are repository or Environment configuration and are
never printed by workflow steps.

The Korvid source checkout uses a read-only GitHub App installation token. A
fine-grained read-only token is an acceptable bootstrap fallback, but the
documented target is a GitHub App token.

Reflection-model credentials, when optimization is enabled, are scoped to the
protected Environment and are not available to evaluation-only rounds.

## Round lifecycle

1. Check out Prompt Lab and the pinned Korvid revision into separate paths.
2. Install the pinned Python and `uv`; install Prompt Lab without generating or
   uploading an environment-specific lockfile.
3. Authenticate to Azure with OIDC.
4. Record the original `modeleval` node count.
5. Scale the existing `modeleval` pool to one node only when its original count
   is zero.
6. Wait for the node pool, Ollama pod, Service endpoint, and advertised model.
7. Run `korvid-prompt-lab aks-check`.
8. Run the selected evaluation or bounded optimization/evaluation sequence.
9. Render a redacted round summary.
10. Upload only safe evidence.
11. Optionally update one sticky PR comment for the model/candidate combination.
12. In both a shell trap and an `if: always()` cleanup step, restore the exact
    original node count.

The workflow uses a repository-wide concurrency group for AKS grounding.
Concurrent rounds do not cancel one another, because cancellation during
cleanup could leave model compute running.

## Result model

The GitHub Job Summary contains:

- round identity, workflow run link, Prompt Lab revision, and Korvid revision;
- candidate id and fingerprint;
- model and live execution mode;
- aggregate score and per-model score;
- `pass^3` and `pass^5`;
- safe/model/systemic run counts;
- hard-safety failure counts grouped by failure name;
- per-case completion, verification, efficiency, status, and elapsed duration;
- promotion eligibility and the exact reason when blocked;
- artifact names and reproduction command.

The summary never includes raw answers, Kubernetes manifests, audit records,
credentials, kubeconfigs, request payloads, unrestricted tool output, or raw
process logs.

Safe uploaded evidence is limited to:

- `round-summary.json`;
- `round-summary.md`;
- `evaluation-summary.json`;
- `optimization-summary.json`, when present;
- `best-candidate.yaml`, when present;
- bridge `response.json` files, which already use the closed safe projection.

Files named `audit.jsonl`, request artifacts, kubeconfigs, subprocess output,
and GEPA internal state are excluded.

The optional PR comment contains the compact score/safety table and links to the
Actions run and uploaded artifact. It does not duplicate raw evidence.

## Failure semantics

- Azure, AKS, kubeconfig, bridge, protocol, source-checkout, or serving failures
  fail the workflow.
- Model failures remain scored results and appear in the summary.
- Any hard safety failure makes the round unsuccessful for promotion and makes
  the evaluation command non-zero, but summary and artifact generation still
  run.
- A failed optimization cannot fall back to the seed and claim success.
- Missing or malformed result files fail summary generation.
- Cleanup failure is a separate failing step and is never hidden by the
  evaluation result.

## Model-pool safety

The workflow may only scale the existing `modeleval` pool between its recorded
original count and one. It cannot create, delete, resize, relabel, or retag node
pools, and it cannot deploy or mutate model workloads.

If the original count is greater than zero, cleanup leaves it unchanged. If the
workflow scaled zero to one, cleanup restores zero.

## Testing

- Unit-test summary rendering with safe, unsafe, model-failure, systemic, and
  malformed inputs.
- Unit-test safe-evidence packaging against path traversal and forbidden files.
- Test node-count restoration planning as a pure function.
- Validate workflow YAML and all referenced scripts in pytest.
- Execute workflow shell logic against fake `az` and `kubectl` binaries.
- Keep live AKS execution manual and protected; CI must not consume model
  compute during ordinary tests.

## Initial operating policy

The first remote rounds use `qwen3:0.6b`. A live local diagnostic round on
2026-08-22 completed all ten Korvid runs but produced aggregate `0.01`,
`pass^3=0`, `pass^5=0`, and fourteen hard safety failures. `qwen3:4b` serving
was healthy but its unbounded reasoning exceeded the current turn budget.

These are baseline observations, not publishable Prompt Bundles. The Actions
workflow exists to make subsequent prompt changes and model comparisons
reviewable as explicit rounds.
