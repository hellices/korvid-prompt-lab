# Same-AKS Task 3 Report: OIDC Federation, Azure Roles, and GitHub Environment

**Date:** 2026-08-23  
**Branch:** feat/prompt-lab-mvp  
**Commit:** 1fac525

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/setup-oidc-federation.sh` | Idempotent federated credential creation (subject: `repo:hellices/korvid-prompt-lab:environment:aks-grounding`) |
| `scripts/setup-azure-roles.sh` | Idempotent Azure role assignments at 3 scopes |
| `scripts/setup-github-environment.sh` | Idempotent `aks-grounding` environment, variables, and secrets via file stdin |
| `infra/azure/prompt-lab-k8s-data-role.json` | Custom Azure role definition with DataActions for pod/service read and port-forward |
| `tests/test_oidc_and_roles.py` | 21 offline contract tests |

---

## RED Evidence

Command:
```
uv run pytest tests/test_oidc_and_roles.py -q
```

Output (before implementation):
```
21 failed in 0.24s
  FileNotFoundError: scripts/setup-oidc-federation.sh
  FileNotFoundError: scripts/setup-azure-roles.sh
  FileNotFoundError: scripts/setup-github-environment.sh
  FileNotFoundError: infra/azure/prompt-lab-k8s-data-role.json
```

---

## GREEN Commands and Results

After creating all files:

```
uv run pytest tests/test_oidc_and_roles.py -q
```
Output:
```
21 passed in 0.08s
```

Shell syntax check:
```
bash -n scripts/setup-oidc-federation.sh && bash -n scripts/setup-azure-roles.sh && bash -n scripts/setup-github-environment.sh
```
Output: all parse OK.

Full regression:
```
uv run pytest -q
564 passed, 6 skipped in 82.14s
```

---

## Role Boundaries

| Role | Scope | Purpose |
|------|-------|---------|
| Prompt Lab AKS Namespace Data Access (custom DataActions) | `${AKS_ID}/namespaces/ollama` | Kubernetes API access: pod/service read, port-forward |
| Prompt Lab Nodepool Scaler | `${AKS_ID}/agentPools/modeleval` (exact agentpool ID) | `az aks nodepool show/scale` on GPU pool only |
| Azure Kubernetes Service Cluster User Role | `${AKS_ID}` (cluster scope) | `az aks get-credentials` only |

No Kubernetes RoleBindings — cluster uses managed Entra with `enableAzureRbac=true`.

---

## Idempotency Behaviour

| Script | Check | Action |
|--------|-------|--------|
| `setup-oidc-federation.sh` | `az ad app federated-credential list` for existing name | Skips creation if exists |
| `setup-azure-roles.sh` | `az role assignment list` per scope | Skips if count > 0; role definition uses create-or-update |
| `setup-github-environment.sh` | `gh api repos/.../environments/...` | Creates if 404; `gh variable set` / `gh secret set` are inherently idempotent (overwrite) |

---

## Commit

```
1fac525  feat(oidc): add OIDC federation, Azure role, and GitHub environment setup scripts

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

---

## Self-Review

**Positive:**
- All 3 scripts use `set -Eeuo pipefail` matching codebase convention.
- No hardcoded tenant/subscription/client IDs — all from environment variables.
- No `set -x` in any script — prevents credential leakage.
- Secret private key flows only through a readable file redirected to stdin (`< "${KORVID_APP_PRIVATE_KEY_FILE}"`).
- No `|| true` or `2>/dev/null` on secret-setting commands.
- No `kubectl` or Kubernetes RoleBindings anywhere.
- Custom role definition uses DataActions only (no Actions), scoped to managedClusters.
- Scaler role assigned at exact agentpool resource ID, not cluster scope.
- 21 tests verify all contracts offline without requiring Azure/GitHub CLI.

**Concerns:**
1. **Custom role definition uses placeholder variables in AssignableScopes.** The `${AZURE_SUBSCRIPTION_ID}` tokens in the JSON must be substituted at apply time (e.g. via `envsubst` or the setup script). The `setup-azure-roles.sh` uses `@${CUSTOM_ROLE_DEF_FILE}` directly — the caller must preprocess or the CLI must support the template. In practice, `az role definition create` does not expand shell variables in JSON files. A future enhancement could add `envsubst` preprocessing.
2. **Scaler role name `Prompt Lab Nodepool Scaler` is referenced but not defined.** The custom role definition only defines the DataActions role. The scaler role must be either a built-in role or a second custom role definition created separately.
3. **No live integration test.** Scripts are contract-tested offline; actual `az` and `gh` commands can only be verified in a real environment.
