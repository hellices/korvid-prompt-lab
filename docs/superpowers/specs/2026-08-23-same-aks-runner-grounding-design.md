# Same-AKS Runner and Grounding Design

## Goal

Run Prompt Lab grounding rounds entirely inside the existing
`rg-pension-guard` AKS environment:

- GitHub Actions jobs run on ARC runners in `aks-shared-runners`.
- Model inference runs on the existing `modeleval` Spot node pool.
- The workflow temporarily scales `modeleval` from zero to one and restores its
  original count.
- Korvid remains the authoritative evaluator and only safe evidence leaves the
  cluster.

This design does not move Korvid CI or reuse its repository-scoped runner
registration. It adds a Prompt Lab-specific runner scale set to the same
cluster.

## Current State

The cluster already has the required compute separation:

| Purpose | Current configuration |
| --- | --- |
| General AKS system pool | `nodepool1`, `Standard_D2s_v5`, one node |
| ARC runner compute | dynamically provisioned nodes labeled `workload=gha-runner` |
| Model evaluation | `modeleval`, `Standard_D32s_v5`, Spot, count zero |
| Model scheduling | `purpose=korvid-model-eval`, `workload=ollama:NoSchedule` |
| Model service | `ollama/ollama`, port `11434` |
| Cluster identity | system-assigned identity, OIDC and Workload Identity enabled |
| Kubernetes authorization | Microsoft Entra authentication with Azure RBAC enabled |

`korvid-runners` is registered to `https://github.com/hellices/korvid`.
Repository-scoped ARC runner scale sets cannot accept jobs from
`hellices/korvid-prompt-lab`, even when the workflow uses the same runner label.
The Prompt Lab repository currently has no self-hosted runner and no
`aks-grounding` Environment.

## Chosen Architecture

Add a separate repository-scoped ARC scale set:

| Setting | Value |
| --- | --- |
| Scale set name / workflow label | `prompt-lab-runners` |
| GitHub config URL | `https://github.com/hellices/korvid-prompt-lab` |
| Kubernetes namespace | `arc-runners-prompt-lab` |
| Controller namespace | existing `arc-systems` |
| Runner placement | `nodeSelector: workload=gha-runner` |
| Runner minimum | `0` |
| Runner maximum | `1` initially |
| Runner image | `acrpensionguard.azurecr.io/runner-base:prompt-lab-v1` |
| Kubernetes service account | no Kubernetes API permission |

The runner and model workloads share the cluster network but not the compute
pool:

```text
GitHub workflow_dispatch
        |
        v
prompt-lab-runners (ARC, runner node)
        |
        | Azure OIDC + az aks get-credentials
        | kubectl port-forward over the Kubernetes API
        v
ollama/ollama Service
        |
        v
Ollama pod on modeleval Spot node
```

The workflow continues to use `kubectl port-forward` and a loopback model
endpoint. Direct ClusterIP access is not introduced because the bridge already
enforces a loopback-only trust boundary and the runner pod intentionally has no
Kubernetes RBAC credential.

## Components

### Prompt Lab ARC Scale Set

The scale set is installed independently of `korvid-runners`. It uses a
repository-scoped GitHub App installation for runner registration. Changing or
deleting this scale set must not affect Korvid CI.

Runner pods:

- schedule only on runner nodes;
- do not tolerate the `modeleval` taints;
- use ephemeral work directories;
- receive no Kubernetes service-account token with useful permissions;
- run one grounding job at a time;
- are destroyed after the job.

The current `runner-base:v1` image has Azure CLI, Git, Bash, jq, and Python
3.12, but does not have `kubectl`, `kubelogin`, or `uv`. Build
`runner-base:prompt-lab-v1` from that reviewed base and add exactly kubectl
v1.35.6, kubelogin v0.2.19, and uv 0.10.9. The image build must verify all
required commands and must run jobs as the existing non-root `runner` user.

The initial maximum is one because the workflow has repository-wide
non-cancelling concurrency and the model pool supports a single evaluation
node. Increasing runner concurrency without adding isolated model capacity is
out of scope.

### Model Pool and Ollama

The existing `modeleval` pool remains the only target for Ollama:

- count is normally zero;
- allowed round lifecycle is zero to one to zero;
- an original count of one is left unchanged;
- any count outside zero or one fails closed;
- Spot eviction is treated as a visible round failure;
- no new node pool or model service is created.

The Ollama deployment keeps its current node selector and tolerations. The
workflow validates the cluster identity, namespace, Service, endpoints, and
advertised model before evaluation.

### GitHub Environment

Create the repository Environment `aks-grounding` with required reviewers.
Store configuration at Environment scope so other workflows cannot
accidentally inherit grounding privileges.

Required variables:

- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`
- `KORVID_AKS_NAMESPACE=ollama`
- `KORVID_AKS_SERVICE=ollama`
- `KORVID_APP_ID`

Required secrets:

- `KORVID_APP_PRIVATE_KEY`
- `GROUNDING_REFLECTION_CREDENTIAL` only when optimize-evaluate rounds are used

The Azure federated credential is restricted to the
`aks-grounding` Environment subject for
`hellices/korvid-prompt-lab`. It does not use a client secret.

### Azure Authorization

The workflow identity receives only the permissions required to:

1. read the target AKS cluster and obtain user credentials;
2. read the `modeleval` node pool;
3. scale only `modeleval` between zero and one.

It must not have permission to create/delete clusters, node pools, or unrelated
resource-group resources. Because this cluster has Azure RBAC for Kubernetes
authorization enabled, an Entra identity is not authorized through a
Kubernetes `RoleBinding`. Use a custom Azure role assignment at the exact AKS
resource ID with `/namespaces/ollama` appended, resolved dynamically with
`az aks show --query id`, and include only these DataActions:

- `Microsoft.ContainerService/managedClusters/services/read`
- `Microsoft.ContainerService/managedClusters/endpoints/read`
- `Microsoft.ContainerService/managedClusters/pods/read`
- `Microsoft.ContainerService/managedClusters/pods/write`
- `Microsoft.ContainerService/managedClusters/apps/deployments/read`

`pods/write` is required by the Kubernetes port-forward subresource; the role
must not include Secrets, service accounts, exec, roles, role bindings, or
resources outside `ollama`. Use a separate custom management-plane role,
assigned at the `modeleval` agent-pool resource, for agent-pool read/write.

## Workflow Changes

The Grounding Round workflow changes from:

```yaml
runs-on: korvid-runners
```

to:

```yaml
runs-on: prompt-lab-runners
```

All existing controls remain:

- `workflow_dispatch` only;
- protected `aks-grounding` Environment;
- exact Prompt Lab and Korvid revision provenance;
- pinned actions;
- read-only checkout credentials;
- one non-cancelling round at a time;
- 180-minute job timeout and 150-minute orchestrator timeout;
- `if: always()` node restoration;
- allowlisted Job Summary, artifact, and optional PR comment.

The workflow must validate runner prerequisites before scaling model compute:
`az`, `kubectl`, `kubelogin`, `uv`, Git, Bash, and Python 3.12 support.

## Data and Credential Flow

1. GitHub queues the manually approved job to `prompt-lab-runners`.
2. ARC creates one ephemeral runner pod on runner compute.
3. The workflow proves both source revisions before obtaining credentials.
4. GitHub mints the read-only Korvid App token.
5. GitHub OIDC authenticates the Environment-bound Azure identity.
6. The workflow records the original `modeleval` count.
7. The orchestrator scales zero to one when necessary.
8. AKS validation creates a temporary kubeconfig and loopback port-forward.
9. Korvid evaluates the selected model; Prompt Lab writes raw artifacts only
   inside the ephemeral runner workspace.
10. Prompt Lab projects the strict safe-evidence allowlist.
11. GitHub uploads safe evidence and renders the Job Summary.
12. Shell and workflow cleanup restore the original node count.
13. ARC destroys the runner pod and its workspace.

No kubeconfig, raw answer, audit log, request payload, model manifest,
reflection credential, or optimizer state is uploaded or posted to a PR.

## Failure Handling

- Runner registration failure: no job starts and no model node is scaled.
- Environment rejection: no credentials are minted.
- Revision or prerequisite failure: fail before Azure login or model scaling.
- Permanent AKS/auth/config failure: fail immediately.
- Transient endpoint/model readiness: retry only the typed temporary failure.
- Model failure: record a model failure without pretending it is systemic.
- Hard safety failure: produce safe scored evidence, fail the job, never
  publish.
- Systemic failure: fail without manufacturing a summary.
- Job cancellation or timeout: shell trap attempts restore; the independent
  `if: always()` step re-reads and restores the count.
- Runner loss before cleanup: an external scheduled guard checks for a stale
  `modeleval=1` condition and alerts; automatic blind scale-down is not part of
  this workflow because another approved round or operator may own the node.

## Verification

### Static and Unit Verification

- workflow contract tests require `runs-on: prompt-lab-runners`;
- ARC configuration tests verify repository URL, namespace, min/max runners,
  runner-node selector, and absence of model-pool tolerations;
- credential and permission documentation remains synchronized with workflow
  references;
- existing lifecycle, safe-evidence, injection, and provenance tests remain
  green.

### Deployment Verification

1. Confirm `prompt-lab-runners` listener is Ready.
2. Dispatch a no-model prerequisite/trust smoke job and observe one ephemeral
   runner on runner compute.
3. Confirm the runner cannot schedule on `modeleval`.
4. Run an evaluate-only `qwen3:0.6b` grounding round.
5. Observe `modeleval` zero to one, Ollama Ready, evaluation completion, and
   one to zero restoration.
6. Verify Job Summary and `safe-evidence` contain no excluded files.
7. Force a controlled evaluation failure and cancellation; verify restoration
   in both cases.
8. Confirm the runner pod and temporary workspace disappear after the job.

## Success Criteria

- Prompt Lab jobs are accepted by a repo-scoped ARC scale set in
  `aks-shared-runners`.
- Korvid CI continues using its existing scale set without interruption.
- Runner pods and Ollama never share a node pool.
- Model compute is zero when no round owns it.
- A completed remote round produces only safe evidence.
- Failure, safety rejection, timeout, and cancellation leave `modeleval` at
  its recorded original count.
- No long-lived Azure secret, kubeconfig, or runner workspace remains.

## Out of Scope

- Organization-wide shared runners.
- Multiple simultaneous grounding rounds.
- New model-serving infrastructure or GPU node pools.
- Direct ClusterIP model access from the bridge.
- Automatic publication of unsafe or synthetic evidence.
