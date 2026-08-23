# Same-AKS Runner and Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register a dedicated Prompt Lab ARC runner scale set in `aks-shared-runners`, configure least-privilege GitHub/Azure access, and execute Grounding Round jobs against Ollama on the existing `modeleval` Spot pool.

**Architecture:** Add `prompt-lab-runners` as a repo-scoped ARC scale set without changing `korvid-runners`. Runner pods stay on `workload=gha-runner` compute while Ollama stays on `modeleval`; the existing workflow uses Azure OIDC, a loopback `kubectl port-forward`, strict evidence projection, and exact node-count restoration.

**Tech Stack:** GitHub Actions, Actions Runner Controller 0.14.2, Helm, AKS, Azure CLI, GitHub CLI, Bash, Kubernetes RBAC, Python 3.12, pytest, Ruff, mypy.

## Global Constraints

- Target only resource group `rg-pension-guard`, cluster `aks-shared-runners`, and node pool `modeleval`.
- Keep existing `korvid-runners` and `hellices/korvid` CI unchanged.
- Use repo-scoped ARC scale set name and workflow label `prompt-lab-runners`.
- Use namespace `arc-runners-prompt-lab`, `minRunners: 0`, and `maxRunners: 1`.
- Runner pods select `workload=gha-runner` and never tolerate `modeleval` taints.
- Ollama remains in namespace/service `ollama/ollama` on `modeleval`.
- Preserve `workflow_dispatch`, protected Environment `aks-grounding`, exact revision provenance, pinned actions, and non-cancelling concurrency.
- Preserve loopback-only model access through `kubectl port-forward`.
- Never place GitHub App private keys, reflection credentials, kubeconfigs, or Azure tokens in source, CLI arguments, logs, artifacts, or PR comments.
- Apply TDD for every repository behavior change: RED, GREEN, refactor, focused test, then commit.
- Do not dispatch the production Grounding Round until its workflow is on the default branch.

---

## File Structure

- Create `infra/arc/prompt-lab-runners-values.yaml`: declarative ARC scale-set values.
- Create `infra/arc/prompt-lab-runner-service-account.yaml`: namespace and tokenless runner service account.
- Create `infra/azure/grounding-kubernetes-role.json.tpl`: namespace-scoped Azure RBAC DataActions for Kubernetes API access.
- Create `scripts/install-prompt-lab-runner.sh`: secret-safe ARC installation and readiness verification.
- Create `scripts/configure-grounding-access.sh`: GitHub Environment, Azure OIDC, Azure role, and Kubernetes RBAC bootstrap.
- Create `scripts/verify-grounding-deployment.sh`: read-only post-deployment contract checks.
- Create `tests/test_grounding_infrastructure.py`: offline contract and injection tests for all infrastructure files.
- Modify `.github/workflows/grounding-round.yml`: select `prompt-lab-runners`.
- Modify `tests/test_grounding_workflow.py`: enforce the new runner label and preserved trust controls.
- Modify `README.md`: document installation, access bootstrap, pre-merge verification, and post-merge live dispatch.

---

### Task 1: Declare the Prompt Lab ARC Scale Set

**Files:**
- Create: `infra/arc/prompt-lab-runners-values.yaml`
- Create: `infra/arc/prompt-lab-runner-service-account.yaml`
- Create: `tests/test_grounding_infrastructure.py`

**Interfaces:**
- Consumes: existing AKS label `workload=gha-runner`, image `acrpensionguard.azurecr.io/runner-base:v1`, ARC chart `gha-runner-scale-set` 0.14.2.
- Produces: Helm values for release `prompt-lab-runners` and service account `prompt-lab-runners-no-permission`.

- [ ] **Step 1: Write failing ARC configuration tests**

Add tests that parse the two YAML files and require the exact contract:

```python
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VALUES = ROOT / "infra/arc/prompt-lab-runners-values.yaml"
SERVICE_ACCOUNT = ROOT / "infra/arc/prompt-lab-runner-service-account.yaml"


def test_prompt_lab_runner_values_are_repo_scoped_and_serial() -> None:
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    assert values["githubConfigUrl"] == "https://github.com/hellices/korvid-prompt-lab"
    assert values["githubConfigSecret"] == "prompt-lab-runners-github-app"
    assert values["runnerScaleSetName"] == "prompt-lab-runners"
    assert values["minRunners"] == 0
    assert values["maxRunners"] == 1


def test_prompt_lab_runners_cannot_schedule_on_model_compute() -> None:
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    pod = values["template"]["spec"]
    assert pod["serviceAccountName"] == "prompt-lab-runners-no-permission"
    assert pod["automountServiceAccountToken"] is False
    assert pod["nodeSelector"] == {"workload": "gha-runner"}
    assert pod["tolerations"] == [
        {
            "key": "gha-runner",
            "operator": "Equal",
            "value": "true",
            "effect": "NoSchedule",
        }
    ]
    assert all(item["key"] != "workload" for item in pod["tolerations"])


def test_runner_service_account_is_tokenless_and_role_free() -> None:
    docs = list(yaml.safe_load_all(SERVICE_ACCOUNT.read_text(encoding="utf-8")))
    assert [doc["kind"] for doc in docs] == ["Namespace", "ServiceAccount"]
    assert docs[0]["metadata"]["name"] == "arc-runners-prompt-lab"
    assert docs[1]["metadata"]["namespace"] == "arc-runners-prompt-lab"
    assert docs[1]["automountServiceAccountToken"] is False
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_grounding_infrastructure.py
```

Expected: FAIL because the infrastructure YAML files do not exist.

- [ ] **Step 3: Add the ARC values**

Create `infra/arc/prompt-lab-runners-values.yaml`:

```yaml
githubConfigUrl: https://github.com/hellices/korvid-prompt-lab
githubConfigSecret: prompt-lab-runners-github-app
runnerScaleSetName: prompt-lab-runners
minRunners: 0
maxRunners: 1

template:
  metadata:
    annotations:
      karpenter.sh/do-not-disrupt: "true"
  spec:
    serviceAccountName: prompt-lab-runners-no-permission
    automountServiceAccountToken: false
    nodeSelector:
      workload: gha-runner
    tolerations:
      - key: gha-runner
        operator: Equal
        value: "true"
        effect: NoSchedule
    containers:
      - name: runner
        image: acrpensionguard.azurecr.io/runner-base:v1
        command:
          - /home/runner/run.sh
        resources:
          requests:
            cpu: "1"
            memory: 3Gi
          limits:
            memory: 5Gi
```

Create `infra/arc/prompt-lab-runner-service-account.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: arc-runners-prompt-lab
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prompt-lab-runners-no-permission
  namespace: arc-runners-prompt-lab
automountServiceAccountToken: false
```

- [ ] **Step 4: Run focused tests and Helm rendering**

Run:

```bash
uv run pytest -q tests/test_grounding_infrastructure.py
helm template prompt-lab-runners \
  --namespace arc-runners-prompt-lab \
  --version 0.14.2 \
  -f infra/arc/prompt-lab-runners-values.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  >/tmp/prompt-lab-runners-rendered.yaml
```

Expected: tests PASS; Helm renders an `AutoscalingRunnerSet` whose GitHub URL,
runner scale set name, service account, selector, and maximum match the test.
Delete `/tmp/prompt-lab-runners-rendered.yaml`.

- [ ] **Step 5: Commit**

```bash
git add infra/arc/prompt-lab-runners-values.yaml \
  infra/arc/prompt-lab-runner-service-account.yaml \
  tests/test_grounding_infrastructure.py
git commit -m "feat(arc): declare Prompt Lab runner scale set"
```

---

### Task 2: Bind Grounding Round to the Dedicated Runner

**Files:**
- Modify: `.github/workflows/grounding-round.yml:104-112`
- Modify: `tests/test_grounding_workflow.py`
- Modify: `README.md:590-660`

**Interfaces:**
- Consumes: ARC label `prompt-lab-runners` from Task 1.
- Produces: a workflow that queues only to the Prompt Lab scale set.

- [ ] **Step 1: Write the failing workflow test**

Replace the old label assertion with:

```python
def test_grounding_job_uses_the_prompt_lab_repo_scoped_runner() -> None:
    workflow = load_workflow()
    job = workflow["jobs"]["grounding"]
    assert job["runs-on"] == "prompt-lab-runners"
    assert "korvid-runners" not in str(job["runs-on"])
```

Also keep the existing assertions for:

```python
assert job["environment"] == "aks-grounding"
assert workflow["concurrency"]["cancel-in-progress"] is False
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run pytest -q tests/test_grounding_workflow.py -k repo_scoped_runner
```

Expected: FAIL, actual label is `korvid-runners`.

- [ ] **Step 3: Change the workflow label**

Update:

```yaml
jobs:
  grounding:
    name: Grounding round
    runs-on: prompt-lab-runners
    environment: aks-grounding
```

Do not change permissions, concurrency, checkout pins, timeouts, or cleanup.

- [ ] **Step 4: Update operator documentation**

Document:

- `korvid-runners` is repository-scoped to `hellices/korvid` and cannot serve
  Prompt Lab.
- `prompt-lab-runners` is installed in the same cluster and uses runner
  compute only.
- Grounding model compute remains `modeleval`.
- The workflow will remain queued until the new listener and scale set are
  Ready.

- [ ] **Step 5: Run workflow and documentation contract tests**

Run:

```bash
uv run pytest -q tests/test_grounding_workflow.py tests/test_contracts.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/grounding-round.yml \
  tests/test_grounding_workflow.py README.md
git commit -m "ci: route grounding to Prompt Lab runners"
```

---

### Task 3: Bootstrap GitHub Environment and Least-Privilege Azure RBAC

**Files:**
- Create: `infra/azure/grounding-kubernetes-role.json.tpl`
- Create: `scripts/configure-grounding-access.sh`
- Modify: `tests/test_grounding_infrastructure.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: an authenticated `gh`, `az`, `kubectl`, `kubelogin`, and the
  Environment variables `KORVID_APP_ID` and `KORVID_APP_PRIVATE_KEY_FILE`.
- Produces: Environment `aks-grounding`, Environment variables/secrets, Entra
  app `korvid-prompt-lab-grounding`, Environment-bound federated credential,
  node-pool scaler role assignment, and namespace-scoped Azure RBAC DataActions.

- [ ] **Step 1: Write failing access-boundary tests**

Add tests that require:

```python
ACCESS_SCRIPT = ROOT / "scripts/configure-grounding-access.sh"
RBAC_TEMPLATE = ROOT / "infra/azure/grounding-kubernetes-role.json.tpl"


def test_grounding_access_is_environment_bound_and_nodepool_scoped() -> None:
    body = ACCESS_SCRIPT.read_text(encoding="utf-8")
    assert "repo:hellices/korvid-prompt-lab:environment:aks-grounding" in body
    assert "agentPools/modeleval" in body
    assert "namespaces/ollama" in body
    assert "AZURE_CLIENT_SECRET" not in body
    assert "gh variable set AZURE_CLIENT_ID --env aks-grounding" in body
    assert "gh secret set KORVID_APP_PRIVATE_KEY --env aks-grounding" in body


def test_grounding_kubernetes_role_has_only_required_data_actions() -> None:
    role = json.loads(RBAC_TEMPLATE.read_text(encoding="utf-8"))
    assert role["Actions"] == []
    assert role["DataActions"] == [
        "Microsoft.ContainerService/managedClusters/apps/deployments/read",
        "Microsoft.ContainerService/managedClusters/endpoints/read",
        "Microsoft.ContainerService/managedClusters/pods/read",
        "Microsoft.ContainerService/managedClusters/pods/write",
        "Microsoft.ContainerService/managedClusters/services/read",
    ]
    assert role["NotDataActions"] == []
    assert "secrets" not in json.dumps(role)
    assert "exec/action" not in json.dumps(role)


def test_access_script_never_places_secrets_on_cli_arguments() -> None:
    body = ACCESS_SCRIPT.read_text(encoding="utf-8")
    assert "--from-literal" not in body
    assert "cat \"$KORVID_APP_PRIVATE_KEY_FILE\" | gh secret set" in body
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_grounding_infrastructure.py
```

Expected: FAIL because the Azure role template and access script do not exist.

- [ ] **Step 3: Add the namespace-scoped Azure RBAC role template**

Create `infra/azure/grounding-kubernetes-role.json.tpl`:

```json
{
  "Name": "Korvid Prompt Lab Grounding Kubernetes Access",
  "Description": "Read Ollama discovery resources and open a pod port-forward in the ollama namespace.",
  "Actions": [],
  "NotActions": [],
  "DataActions": [
    "Microsoft.ContainerService/managedClusters/apps/deployments/read",
    "Microsoft.ContainerService/managedClusters/endpoints/read",
    "Microsoft.ContainerService/managedClusters/pods/read",
    "Microsoft.ContainerService/managedClusters/pods/write",
    "Microsoft.ContainerService/managedClusters/services/read"
  ],
  "NotDataActions": [],
  "AssignableScopes": ["__SUBSCRIPTION_SCOPE__"]
}
```

- [ ] **Step 4: Implement the idempotent access script**

The script must:

1. require `gh`, `az`, `jq`, `kubectl`, and `kubelogin`;
2. discover subscription and tenant IDs without printing them;
3. create or reuse Entra app `korvid-prompt-lab-grounding`;
4. create or replace a federated credential with:

```json
{
  "name": "github-aks-grounding",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:hellices/korvid-prompt-lab:environment:aks-grounding",
  "audiences": ["api://AzureADTokenExchange"]
}
```

5. assign `Azure Kubernetes Service Cluster User Role` at the cluster resource;
6. create a custom role containing only:

```json
[
  "Microsoft.ContainerService/managedClusters/agentPools/read",
  "Microsoft.ContainerService/managedClusters/agentPools/write"
]
```

   and assign it at the exact ID returned by:

```bash
az aks nodepool show \
  --resource-group rg-pension-guard \
  --cluster-name aks-shared-runners \
  --name modeleval \
  --query id \
  --output tsv
```

7. render the Kubernetes DataActions role with the subscription scope, create
   or update it, and assign it to the service principal at the namespace scope
   computed without hardcoding resource IDs:

```bash
AKS_ID="$(az aks show \
  --resource-group rg-pension-guard \
  --name aks-shared-runners \
  --query id \
  --output tsv)"
KUBERNETES_SCOPE="${AKS_ID}/namespaces/ollama"
```

   The cluster uses Azure RBAC for Kubernetes authorization; do not create a
   Kubernetes `RoleBinding`. `pods/write` is retained because Azure maps
   port-forward authorization through the pod write DataAction.
8. create/update `aks-grounding`, using the authenticated GitHub user as a
   required reviewer;
9. set the six Environment variables from the design;
10. read the Korvid private key only from
    `KORVID_APP_PRIVATE_KEY_FILE` through stdin:

```bash
cat "$KORVID_APP_PRIVATE_KEY_FILE" |
  gh secret set KORVID_APP_PRIVATE_KEY \
    --env aks-grounding \
    --repo hellices/korvid-prompt-lab
```

Use `mktemp -d`, `chmod 700`, and an EXIT trap for federated credential and role
definition JSON. Do not enable `set -x`.

- [ ] **Step 5: Run tests and shell validation**

Run:

```bash
uv run pytest -q tests/test_grounding_infrastructure.py
bash -n scripts/configure-grounding-access.sh
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add infra/azure/grounding-kubernetes-role.json.tpl \
  scripts/configure-grounding-access.sh \
  tests/test_grounding_infrastructure.py README.md
git commit -m "feat(aks): bootstrap grounding access"
```

---

### Task 4: Install and Verify the Prompt Lab Runner Scale Set

**Files:**
- Create: `scripts/install-prompt-lab-runner.sh`
- Create: `scripts/verify-grounding-deployment.sh`
- Modify: `tests/test_grounding_infrastructure.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `ARC_GITHUB_APP_ID`,
  `ARC_GITHUB_APP_INSTALLATION_ID`, and
  `ARC_GITHUB_APP_PRIVATE_KEY_FILE`; ARC values and service account from Task 1.
- Produces: deployed Helm release `prompt-lab-runners` and read-only deployment
  verification.

- [ ] **Step 1: Write failing installer and verifier tests**

Require the scripts to contain:

```python
def test_runner_installer_pins_arc_and_handles_secrets_through_files() -> None:
    body = (ROOT / "scripts/install-prompt-lab-runner.sh").read_text(encoding="utf-8")
    assert "gha-runner-scale-set" in body
    assert "--version 0.14.2" in body
    assert "--from-file=github_app_private_key=" in body
    assert "--from-literal" not in body
    assert "trap cleanup EXIT" in body


def test_deployment_verifier_is_read_only() -> None:
    body = (ROOT / "scripts/verify-grounding-deployment.sh").read_text(encoding="utf-8")
    for forbidden in ("kubectl apply", "kubectl delete", "helm upgrade", "az aks nodepool scale"):
        assert forbidden not in body
    assert "prompt-lab-runners" in body
    assert "modeleval" in body
    assert "safe-evidence" in body
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_grounding_infrastructure.py
```

Expected: FAIL because the scripts do not exist.

- [ ] **Step 3: Implement the secret-safe installer**

The installer must:

- verify the active subscription and exact AKS cluster;
- create a mode-0700 temporary directory;
- copy app ID, installation ID, and private key into mode-0600 files;
- apply the Namespace and ServiceAccount manifest;
- create/update the secret with:

```bash
kubectl -n arc-runners-prompt-lab create secret generic \
  prompt-lab-runners-github-app \
  --from-file=github_app_id="$tmp/github_app_id" \
  --from-file=github_app_installation_id="$tmp/github_app_installation_id" \
  --from-file=github_app_private_key="$tmp/github_app_private_key" \
  --dry-run=client -o yaml |
  kubectl apply -f -
```

- install the pinned chart:

```bash
helm upgrade --install prompt-lab-runners \
  --namespace arc-runners-prompt-lab \
  --version 0.14.2 \
  --values infra/arc/prompt-lab-runners-values.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --wait --timeout 10m
```

- verify `AutoscalingRunnerSet.spec.githubConfigUrl`,
  `minRunners`, `maxRunners`, and listener readiness;
- remove all temporary files on EXIT.

- [ ] **Step 4: Implement the read-only verifier**

The verifier must check:

- ARC release status is deployed;
- the scale set URL is `hellices/korvid-prompt-lab`;
- min/max are zero/one;
- runner template selects `workload=gha-runner`;
- no runner toleration matches `workload=ollama`;
- the Prompt Lab workflow uses `prompt-lab-runners`;
- GitHub Environment exists and required variable/secret names exist;
- `modeleval` count is zero or one and provisioning state is `Succeeded`;
- Ollama selector/tolerations still target `modeleval`;
- the workflow artifact upload path ends at `safe-evidence/`.

Print identities and secret names only, never secret values.

- [ ] **Step 5: Run focused validation**

Run:

```bash
uv run pytest -q tests/test_grounding_infrastructure.py
bash -n scripts/install-prompt-lab-runner.sh \
  scripts/verify-grounding-deployment.sh
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/install-prompt-lab-runner.sh \
  scripts/verify-grounding-deployment.sh \
  tests/test_grounding_infrastructure.py README.md
git commit -m "feat(arc): install Prompt Lab runners"
```

---

### Task 5: Apply Infrastructure and Prove a Remote Grounding Round

**Files:**
- Modify: `README.md`
- Modify: `docs/review-fix-report.md`

**Interfaces:**
- Consumes: approved GitHub App installation, the scripts from Tasks 3-4, and
  the merged Grounding Round workflow on `main`.
- Produces: a Ready listener, an ephemeral Prompt Lab runner, one real
  evaluate-only round, safe evidence, and verified node restoration.

- [ ] **Step 1: Verify prerequisites before mutation**

Run:

```bash
gh api repos/hellices/korvid-prompt-lab --jq '{default_branch,permissions}'
az aks nodepool show \
  --resource-group rg-pension-guard \
  --cluster-name aks-shared-runners \
  --name modeleval \
  --query '{count:count,provisioningState:provisioningState}' \
  --output json
```

Expected: admin access to the repository; `modeleval` count `0` and state
`Succeeded`.

- [ ] **Step 2: Apply GitHub/Azure/Kubernetes access**

With the GitHub App installed on both `hellices/korvid-prompt-lab` and
`hellices/korvid`, load its ID and private-key path into the shell through the
operator's secret manager, then verify they are present:

```bash
: "${KORVID_APP_ID:?load KORVID_APP_ID from the secret manager}"
: "${KORVID_APP_PRIVATE_KEY_FILE:?load the absolute PEM path from the secret manager}"
test -r "$KORVID_APP_PRIVATE_KEY_FILE"
./scripts/configure-grounding-access.sh
```

Expected: Environment and role setup succeeds without printing the private key,
tenant, subscription, or tokens.

- [ ] **Step 3: Install the runner scale set**

Load the ARC GitHub App installation values through the operator's secret
manager, then verify them:

```bash
: "${ARC_GITHUB_APP_ID:?load ARC_GITHUB_APP_ID from the secret manager}"
: "${ARC_GITHUB_APP_INSTALLATION_ID:?load ARC_GITHUB_APP_INSTALLATION_ID from the secret manager}"
: "${ARC_GITHUB_APP_PRIVATE_KEY_FILE:?load the absolute PEM path from the secret manager}"
test -r "$ARC_GITHUB_APP_PRIVATE_KEY_FILE"
./scripts/install-prompt-lab-runner.sh
```

Expected: Helm release deployed and listener Ready with zero idle runners.

- [ ] **Step 4: Run the read-only deployment verifier**

Run:

```bash
./scripts/verify-grounding-deployment.sh
```

Expected: every check passes; `modeleval` remains zero.

- [ ] **Step 5: Run repository validation**

Run:

```bash
KORVID_SOURCE_ROOT=/Users/hwang-inhwan/workspace/kube uv run pytest -q
uv run ruff check .
uv run mypy --python-version 3.12 src tests
bash -n scripts/*.sh
uv run python - <<'PY'
from pathlib import Path
import yaml

for path in Path(".github/workflows").glob("*.yml"):
    assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)
    print(f"{path}: YAML OK")
PY
```

Expected: all tests, lint, types, Bash, and YAML validation pass.

- [ ] **Step 6: Push and update Draft PR**

Push the reviewed commits to `feat/prompt-lab-mvp` and update Draft PR #1 with:

- runner scale set name and cluster;
- access boundaries;
- test results;
- deployment verifier result;
- the fact that the live Grounding Round waits for merge because the workflow
  intentionally executes only from the default branch.

- [ ] **Step 7: After merge, dispatch the live evaluate-only round**

Resolve the merged Prompt Lab SHA:

```bash
PROMPT_LAB_REF="$(gh api repos/hellices/korvid-prompt-lab/commits/main --jq .sha)"
gh workflow run grounding-round.yml \
  --repo hellices/korvid-prompt-lab \
  --ref main \
  -f prompt_lab_ref="$PROMPT_LAB_REF" \
  -f korvid_ref=fc7eece2adb66a5b2a18d378bdfd7503ddbdd2ca \
  -f model=qwen3:0.6b \
  -f round_type=evaluate \
  -f candidate=examples/candidates/shipped-small.yaml \
  -f campaign=examples/campaigns/aks-shared-runners.yaml \
  -f train_case_id=aks-scale-deployment-up \
  -f validation_case_id=aks-restart-denied \
  -f milestone_case_ids=aks-scale-deployment-up,aks-restart-denied \
  -f max_metric_calls=12 \
  -f seed=0
```

Expected: one ephemeral runner is created on runner compute; `modeleval` goes
zero to one; Ollama becomes Ready; the round produces a Job Summary and
`safe-evidence`; `modeleval` returns to zero.

- [ ] **Step 8: Verify cleanup and safe evidence**

Run:

```bash
az aks nodepool show \
  --resource-group rg-pension-guard \
  --cluster-name aks-shared-runners \
  --name modeleval \
  --query '{count:count,provisioningState:provisioningState}' \
  --output json
gh run list --repo hellices/korvid-prompt-lab \
  --workflow grounding-round.yml --limit 1
```

Download only the `safe-evidence` artifact and assert it contains no
`request.json`, `audit.jsonl`, kubeconfig, raw log, manifest, credential, or
GEPA state.

- [ ] **Step 9: Document evidence and commit**

Append the workflow run ID, final node count, artifact manifest, and observed
model result to `docs/review-fix-report.md`. Update README only if the live
result contradicts an operator instruction.

```bash
git add README.md docs/review-fix-report.md
git commit -m "docs: record same-AKS grounding validation"
```

---

## Final Review Gate

- [ ] Request an independent whole-change code review from the design commit
  through the final implementation commit.
- [ ] Fix every Critical or Important finding with a failing regression test.
- [ ] Re-run the complete verification commands from Task 5.
- [ ] Confirm `modeleval` count is `0` and provisioning state is `Succeeded`.
- [ ] Confirm the Prompt Lab ARC scale set has zero idle runners.
- [ ] Confirm `korvid-runners` remains registered to `hellices/korvid`.
- [ ] Push the reviewed branch and update Draft PR #1.
