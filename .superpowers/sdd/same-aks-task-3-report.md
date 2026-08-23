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

---

# Same-AKS Task 3 Repair (pre-review)

**Date:** 2026-08-23
**Branch:** feat/prompt-lab-mvp
**Supersedes:** everything above — the original implementation was not
deployable and did not match the plan interface.

## What was wrong

| # | Defect in commit `1fac525` | Consequence |
|---|---|---|
| 1 | Three split scripts (`setup-oidc-federation.sh`, `setup-azure-roles.sh`, `setup-github-environment.sh`) instead of the planned `scripts/configure-grounding-access.sh` | Competing half-bootstraps, no single idempotent entry point, and the plan interface unmet |
| 2 | `infra/azure/prompt-lab-k8s-data-role.json` used `pods/portforward/action` and `services/portforward/action` | Those DataActions do not exist on AKS; `az role definition create` would reject the definition. `apps/deployments/read` and `endpoints/read` were missing, so the workflow's own discovery calls would be denied |
| 3 | `AssignableScopes` held literal `${AZURE_SUBSCRIPTION_ID}` shell text in a JSON file passed straight to `az` | `az` does not expand shell variables; the role definition would be created with a garbage scope or rejected |
| 4 | The `Prompt Lab Nodepool Scaler` role was assigned but never defined | `az role assignment create` fails with "role not found" |
| 5 | Node pool scope was hand-built by string concatenation, never `az aks nodepool show --query id` | A drifted resource id would be assigned silently |
| 6 | Subscription/tenant/app object id were required as *inputs*; no app, service principal, or environment reviewer was ever created | Bootstrap could not run from a clean slate |
| 7 | The federated credential was skipped whenever a credential with the same *name* existed | A drifted subject or audience was never corrected |
| 8 | 21 tests only grepped for substrings | Every defect above passed the suite |

## Repair

| File | Change |
|------|--------|
| `scripts/configure-grounding-access.sh` | **new** — the single idempotent orchestrator |
| `infra/azure/grounding-kubernetes-role.json.tpl` | **new** — exact five DataActions, `__SUBSCRIPTION_SCOPE__` placeholder |
| `tests/test_grounding_infrastructure.py` | **+37 tests** — six file-contract tests plus 31 that execute the script against strict stateful fake `az`/`gh`/`kubectl` binaries |
| `README.md` | bootstrap usage, the authorization boundary table, and the DataActions rationale |
| `scripts/setup-oidc-federation.sh`, `scripts/setup-azure-roles.sh`, `scripts/setup-github-environment.sh`, `infra/azure/prompt-lab-k8s-data-role.json`, `tests/test_oidc_and_roles.py` | **removed** — no competing path is left behind |

The fakes keep cloud state on disk (application, service principal, federated
credentials, role definitions, assignments), refuse any command outside the
contract, record every argv, payload mode, and stdin byte, and can be told to
fail a specific call once or always. That is what makes the tests behavioural:
they assert scopes, actions, reviewer payload, create-vs-reuse, drift repair,
retry, failure propagation, and the absence of leakage — not the presence of
strings. Nothing they run touches a real subscription or repository.

## RED

Command:

```bash
uv run --python 3.12 pytest -q tests/test_grounding_infrastructure.py
```

Output (new tests added, implementation absent):

```
26 failed, 19 passed in 0.97s
FAILED test_grounding_access_is_environment_bound_and_nodepool_scoped
FAILED test_grounding_kubernetes_role_has_only_required_data_actions
FAILED test_grounding_kubernetes_role_excludes_unsupported_portforward_actions
FAILED test_access_script_never_places_secrets_on_cli_arguments
FAILED test_access_script_is_strict_bash_without_tracing_or_kubernetes_rbac
FAILED test_split_access_scripts_are_retired
FAILED test_cold_bootstrap_creates_identity_roles_and_environment
FAILED test_bootstrap_assigns_exactly_three_scoped_roles
FAILED test_rendered_role_definitions_carry_exact_actions_and_scope
FAILED test_no_unexpanded_template_placeholder_reaches_azure
... (16 more)
  bash: scripts/configure-grounding-access.sh: No such file or directory
```

A second RED appeared during implementation and was fixed rather than
worked around: `test_transient_principal_creation_is_retried` and
`test_retries_are_bounded_and_then_fail` proved that the `lookup` helper
swallowed a failing `az` call inside `retry` (because `set -e` is suspended in
an `if` condition), so a transient Entra replication error would have produced
an empty principal id instead of a retry.

```
2 failed, 45 passed in 37.12s
  configure-grounding-access: the service principal was created without an object id
  assert 1 == 3   # only one `az ad sp create` attempt was ever made
```

## GREEN

```bash
uv run --python 3.12 pytest -q tests/test_grounding_infrastructure.py
```

```
47 passed in 34.43s
```

```bash
bash -n scripts/configure-grounding-access.sh
```

```
(no output — parses cleanly)
```

## Full verification

```bash
uv run --python 3.12 pytest -q
```

```
580 passed, 6 skipped in 125.13s (0:02:05)
```

```bash
uv run --python 3.12 ruff check .
```

```
All checks passed!
```

```bash
uv run --python 3.12 mypy src tests
```

```
Success: no issues found in 36 source files
```

```bash
git diff --cached --check
```

```
(no output — no whitespace errors)
```

## Behaviour the tests now pin

| Property | Test |
|---|---|
| Cold bootstrap creates app, principal, credential, two roles, three assignments, environment, six variables, one secret | `test_cold_bootstrap_creates_identity_roles_and_environment` |
| A second run creates nothing: no app, principal, credential, or assignment; roles take the update path | `test_second_run_reuses_every_created_object` |
| Namespace DataActions land only on `<aks-id>/namespaces/ollama`, the scaler only on the returned `modeleval` id, Cluster User Role only on the cluster | `test_bootstrap_assigns_exactly_three_scoped_roles` |
| Rendered role JSON carries the exact five DataActions, the exact two agent-pool Actions, and the discovered subscription scope | `test_rendered_role_definitions_carry_exact_actions_and_scope` |
| No `__PLACEHOLDER__` ever reaches `az` | `test_no_unexpanded_template_placeholder_reaches_azure` |
| Every rendered payload is mode `600`, and the render directory is gone on exit | `test_rendered_payloads_are_private_files`, `test_render_directory_is_removed_on_exit` |
| An existing application and principal are reused | `test_existing_application_and_principal_are_reused` |
| A matching credential is untouched; a drifted subject or audience is deleted and re-created with the exact subject | `test_matching_federated_credential_is_left_untouched`, `test_drifted_federated_credential_is_replaced`, `test_drifted_federated_audience_is_replaced` |
| The environment PUT carries `reviewers: [{"type": "User", "id": <authenticated user>}]` and precedes every variable and secret write | `test_environment_requires_the_authenticated_user_as_reviewer`, `test_environment_exists_before_variables_and_secrets_are_written` |
| Azure failure aborts before any GitHub mutation; role, app, environment, and secret failures all propagate | `test_azure_failure_stops_before_any_github_mutation` and four siblings |
| Transient failures retry, bounded, then fail | `test_transient_principal_creation_is_retried`, `test_transient_role_assignment_failure_is_retried`, `test_retries_are_bounded_and_then_fail` |
| Unreadable key file, missing `KORVID_APP_ID`, half a reflection pair, or a wrong node-pool id fail before any cloud call or assignment | `test_unreadable_private_key_fails_before_touching_the_cloud` and three siblings |
| No key, credential, subscription, tenant, client, principal, or resource id reaches stdout, stderr, or a command line | `test_no_identifier_or_secret_reaches_the_console` |
| Private key and reflection credential arrive byte-exact on stdin | `test_cold_bootstrap_...`, `test_reflection_credential_is_streamed_from_a_file` |
| `kubectl`/`kubelogin` are required but never invoked — no Kubernetes RBAC object is created | `test_kubernetes_clients_are_never_invoked` |

## CLI surface verified against the installed Azure CLI 2.89.1

- `az role assignment list --assignee-object-id --role --scope` exists, and
  without `--include-inherited` it filters to the exact scope, which is what
  makes the idempotency probe correct.
- `az role assignment create --assignee-object-id --assignee-principal-type`
  exists; the principal type avoids Graph propagation errors.
- `az role definition list --name` matches `roleName`, and
  `az role definition update --role-definition @file` resolves the GUID from
  the `Name` plus `AssignableScopes` in the rendered file.
- `az` expands a leading `@` into the file's contents before parsing, so
  `--role-definition "@file"` and `--parameters "@file"` are both valid.
- `az` hard-codes `type='CustomRole'` when creating a role definition, so the
  template deliberately carries no `IsCustom` key.
- `gh variable set` / `gh secret set` read the value from stdin when `--body`
  is omitted, and accept `--env` with `--repo`.

## Residual concerns

1. **No live run.** Everything is proven against fakes and against the
   installed `az`/`gh` parameter surface. The first real execution still needs
   an operator with Owner (or User Access Administrator) rights on the
   subscription and admin rights on the repository.
2. **Required reviewers need a plan.** GitHub only enforces environment
   reviewers on public repositories or paid plans; on a private free repository
   the PUT succeeds but the protection is not enforced.
3. **Role definition propagation.** A freshly created custom role can take a
   few seconds to become assignable. The bounded retry around
   `az role assignment create` covers the common case; a very slow tenant may
   need `_GROUNDING_RETRY_ATTEMPTS` raised.
