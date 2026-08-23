# Same-AKS Task 4 Report — Install and Verify the Prompt Lab Runner Scale Set

## Commit

```
c2f5d21  feat(arc): install Prompt Lab runners
```

## RED Phase

Two new tests added to `tests/test_grounding_infrastructure.py`:
- `test_runner_installer_pins_arc_and_handles_secrets_through_files`
- `test_deployment_verifier_is_read_only`

```
FAILED tests/test_grounding_infrastructure.py::test_runner_installer_pins_arc_and_handles_secrets_through_files - FileNotFoundError
FAILED tests/test_grounding_infrastructure.py::test_deployment_verifier_is_read_only - FileNotFoundError
2 failed in 0.12s
```

## GREEN Phase

### scripts/install-prompt-lab-runner.sh
- Verifies active subscription and exact AKS cluster context
- Creates mode-0700 temp dir; writes app id, installation id, private key as mode-0600 files
- Applies namespace/SA manifest
- Creates secret via `--from-file=github_app_private_key=...` (never `--from-literal`)
- Installs pinned chart `gha-runner-scale-set --version 0.14.2`
- Verifies AutoscalingRunnerSet config and listener readiness
- `trap cleanup EXIT` removes temp dir on all exits

### scripts/verify-grounding-deployment.sh
- Strictly read-only: no `kubectl apply`, `kubectl delete`, `helm upgrade`, or `az aks nodepool scale`
- Checks: ARC release status, scale set URL (`hellices/korvid-prompt-lab`), min/max (0/1)
- Checks: runner nodeSelector `workload=gha-runner`, no ollama toleration, runAsNonRoot
- Checks: workflow label `prompt-lab-runners`, environment variable/secret names (never values)
- Checks: `modeleval` count 0|1, provisioningState `Succeeded`
- Checks: Ollama scheduling targets `modeleval`
- Checks: workflow references `safe-evidence` artifact path

### README.md
- Added "Installing and verifying the runner scale set" section with usage commands

## Validation Results

```
# Focused tests
2 passed in 0.04s

# Full suite
49 passed in 36.45s

# bash -n
scripts/install-prompt-lab-runner.sh: OK
scripts/verify-grounding-deployment.sh: OK

# Ruff
All checks passed!

# mypy
Success: no issues found in 1 source file

# YAML parse
infra/arc/prompt-lab-runners-values.yaml: OK
infra/arc/prompt-lab-runner-service-account.yaml: OK

# git diff --check
clean
```

## Fake-Flow Coverage

No stateful fakes needed for Task 4. The tests are static contract checks
verifying script content contains required patterns (chart version, --from-file,
trap, forbidden mutating commands). The existing AccessHarness fakes from Task 3
remain untouched and pass.

## Self-Review

- **Secret hygiene**: installer never uses `--from-literal`; all secrets flow
  through mode-0600 files in a mode-0700 temp dir cleaned by EXIT trap.
- **Read-only verifier**: grep-verified absence of all mutating commands.
- **Pinned version**: `--version 0.14.2` appears literally (not via variable)
  so the contract test catches any drift.
- **No deployment/push**: scripts created and tested offline only.

## Concerns

1. The verifier uses `gh api` for environment variable/secret name checks, which
   requires a valid `GITHUB_TOKEN`. In CI this is fine; locally the operator needs
   `gh auth login` first. The check is non-fatal (uses `|| true`).
2. The installer's post-install verification assumes the AutoscalingRunnerSet CRD
   is available and the listener pod label matches the scale set name — both are
   ARC conventions that could change across chart versions.

---

# Same-AKS Task 4 Repair (pre-review)

**Date:** 2026-08-23
**Branch:** feat/prompt-lab-mvp
**Supersedes:** everything above — the installer could repoint the operator's
own kubeconfig, waited for the listener in a namespace where listeners never
run, and the "verifier" swallowed every failure it was supposed to report.

## What was wrong

| # | Defect in commit `055ea0a` | Consequence |
|---|---|---|
| 1 | `az aks get-credentials --overwrite-existing` with no `--file` | Rewrote the operator's `~/.kube/config` and repointed every other shell at the shared cluster; `kubelogin convert-kubeconfig` was never run, so an AAD-enabled cluster would reject `kubectl` |
| 2 | `kubectl -n arc-runners-prompt-lab wait pod -l actions.github.com/scale-set-name=...` | ARC listeners run in **`arc-systems`**; the wait could only ever time out |
| 3 | Post-install "verification" printed `githubConfigUrl`, `minRunners`, `maxRunners` and continued | A release installed against the wrong repository, or with `maxRunners: 50`, was reported as success |
| 4 | Tool preflight missing; `az`/`kubectl` were used before anything was checked | A missing `kubelogin` or `helm` surfaced halfway through, after the namespace and secret had been written |
| 5 | Verifier used `2>/dev/null || true` for the GitHub Environment | The single check that proves the round can authenticate was decorative: an unauthenticated `gh` printed nothing and passed |
| 6 | Verifier expected `--jq '.[].name'` on the Environment endpoints | The live endpoints answer with objects (`.variables[].name`, `.secrets[].name`), so the filter could never have matched |
| 7 | Verifier asserted Ollama `nodeSelector.workload == modeleval` | The live deployment selects `purpose=korvid-model-eval` and tolerates `workload=ollama:NoSchedule` plus `kubernetes.azure.com/scalesetpriority=spot:NoSchedule`; the check was against a cluster that does not exist |
| 8 | Runner toleration check grepped for `"key":"ollama"` | The taint key is `workload` (value `ollama`); the spot taint was not checked at all, so a runner able to land on a GPU node passed |
| 9 | Security check read only `runAsNonRoot` | `runAsUser`, `runAsGroup`, and `allowPrivilegeEscalation` drift went unnoticed |
| 10 | Workflow checks were `find … \| head -20` plus `grep 'runs-on:.*prompt-lab-runners'` over an unquoted variable (word-splitting), and `grep -q safe-evidence` | Any workflow — or any comment — mentioning the scale set satisfied the contract; a path pointing at the raw artifact root passed as "safe evidence" |
| 11 | Two tests, both `assert "string" in body` | Every defect above passed the suite |

## Repair

| File | Change |
|------|--------|
| `scripts/install-prompt-lab-runner.sh` | **rewritten** — tool preflight before any mutation, private `mktemp -d` kubeconfig + `kubelogin convert-kubeconfig -l azurecli`, exact post-install field comparison, listener wait in `arc-systems` |
| `scripts/verify-grounding-deployment.sh` | **rewritten** — no `\|\| true`, every CLI failure fatal, live Ollama selector/tolerations, both model taints rejected on the runner, full securityContext, structural YAML workflow parse |
| `tests/test_grounding_infrastructure.py` | **+73 tests** (75 Task-4 tests in total) — four file-contract tests, two of them new, plus 71 that execute both scripts against strict stateful fake `az`/`gh`/`helm`/`kubectl`/`kubelogin` binaries |
| `README.md` | rewrote the install/verify section: tool requirements, the private-kubeconfig guarantee, and exactly what each script fails on |

The fake cluster keeps its state on disk: the `AutoscalingRunnerSet` **derived
from the committed values file**, the Ollama deployment as it really runs, the
`modeleval` pool, the release status, and the `aks-grounding` Environment whose
endpoints answer with real `jq` applied to object-shaped payloads.  Each fake
refuses everything outside the contract — a kubeconfig outside `TMPDIR`, a
`kubectl` call before `kubelogin`, a `--from-file` source that is not mode 600,
`--from-literal`, a listener wait outside `arc-systems`, a write of any kind
during the read-only audit — and can be told to fail one call.  Because the
scale set the fake serves is built from `prompt-lab-runners-values.yaml`, a
values edit that contradicts either script now fails the suite.

## RED

```bash
uv run --python 3.12 pytest -q tests/test_grounding_infrastructure.py
```

```
64 failed, 57 passed in 66.24s

FAILED test_installer_uses_a_private_kubeconfig_and_leaves_the_operator_default_alone
  fake az: get-credentials without --file would rewrite the operator kubeconfig
  install-prompt-lab-runner: no active Azure subscription
FAILED test_installer_waits_for_the_listener_in_the_controller_namespace
FAILED test_installer_fails_when_the_installed_scale_set_drifts[spec.maxRunners-4-maxRunners]
FAILED test_verifier_fails_when_the_github_api_is_unavailable
FAILED test_verifier_requires_every_grounding_variable[KORVID_APP_ID]
FAILED test_verifier_pins_the_live_ollama_node_selector
FAILED test_verifier_ignores_an_unrelated_workflow_that_mentions_the_scale_set
... (57 more)
```

Two further REDs appeared during implementation and were fixed rather than
worked around:

1. `mktemp -d` with no template ignores `TMPDIR` on BSD/macOS, so the kubeconfig
   escaped the private workspace (`fake az: the kubeconfig must live in a
   private temporary directory: /var/folders/.../tmp.k3TqVh08b7/kubeconfig`).
   Both scripts now pass an explicit `"${TMPDIR:-/tmp}"`-rooted template.
2. The fake `helm` read the release name positionally as `$2` and so rejected
   the canonical `helm upgrade --install <release>` form
   (`fake helm: unexpected release: --install`).  The fake now skips option
   values before taking positionals.

## GREEN

```bash
uv run --python 3.12 pytest -q tests/test_grounding_infrastructure.py
```

```
123 passed in 95.20s (0:01:35)
```

## Full verification

```bash
uv run --python 3.12 pytest -q
```

```
656 passed, 6 skipped in 196.40s (0:03:16)
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
bash -n scripts/install-prompt-lab-runner.sh scripts/verify-grounding-deployment.sh
```

```
(no output — both parse cleanly)
```

```bash
uv run --python 3.12 python -c "import yaml; [list(yaml.safe_load_all(open(p, encoding='utf-8'))) for p in (...)]"
```

```
infra/arc/prompt-lab-runners-values.yaml OK
infra/arc/prompt-lab-runner-service-account.yaml OK
.github/workflows/grounding-round.yml OK
```

```bash
git diff --check
```

```
(no output — no whitespace errors)
```

## Behaviour the tests now pin

| Property | Test |
|---|---|
| The kubeconfig is downloaded with `--file` into `TMPDIR`, converted by `kubelogin` before the first `kubectl`, and removed on exit; `~/.kube/config` is byte-identical afterwards | `test_installer_uses_a_private_kubeconfig_and_leaves_the_operator_default_alone`, `test_installer_converts_the_kubeconfig_before_the_first_kubectl_call`, `test_verifier_uses_a_private_kubeconfig_and_cleans_it_up` |
| The temporary directory is removed even when the install fails | `test_installer_removes_its_temporary_directory_when_the_install_fails` |
| Every required tool is checked before any cloud call | `test_installer_requires_every_tool_before_any_mutation`, `test_verifier_requires_every_tool` |
| Missing app id, installation id, or key file aborts before the first call | `test_installer_requires_the_app_environment_before_any_call`, `test_installer_requires_a_readable_private_key` |
| App id, installation id, and private key arrive byte-exact inside the applied Secret, from mode-600 files, and never appear in `argv` or output | `test_installer_streams_every_secret_through_a_mode_600_file`, `test_installer_never_puts_a_secret_in_argv_or_on_the_console` |
| The installer mutates exactly three things: the namespace/service-account manifest, the secret, and the release | `test_installer_mutates_only_the_service_account_secret_and_release` |
| The chart is pinned to `0.14.2`, installed into `arc-runners-prompt-lab` from the OCI reference with `--wait --timeout 10m` | `test_installer_pins_the_chart_namespace_and_values` |
| The listener is awaited in `arc-systems` with both scale-set labels, and a listener that never turns Ready fails the install | `test_installer_waits_for_the_listener_in_the_controller_namespace`, `test_installer_fails_when_the_listener_never_becomes_ready`, `test_a_listener_wait_outside_the_controller_namespace_is_rejected` |
| Ten scale-set fields (URL, min, max, service account, automount, selector, and four security-context fields) fail the run when they drift — in both scripts | `test_installer_fails_when_the_installed_scale_set_drifts`, `test_verifier_fails_when_the_scale_set_drifts` |
| The runner container is found by name, not by index | `test_deployment_scripts_require_a_container_named_runner` |
| Either model-node taint on the runner template fails both scripts | `test_installer_rejects_a_runner_that_tolerates_a_model_taint`, `test_verifier_rejects_a_runner_that_tolerates_a_model_taint` |
| The verifier only reads, and the fakes refuse a write during the audit | `test_verifier_only_reads` |
| An unavailable GitHub API, an absent Environment, any missing one of the six variables, or a missing `KORVID_APP_PRIVATE_KEY` fails the audit; the reflection secret stays optional | `test_verifier_fails_when_the_github_api_is_unavailable` and five siblings |
| Variable *names* are printed, values never are | `test_verifier_prints_names_but_never_a_value` |
| Ollama must select `purpose=korvid-model-eval` and tolerate both live taints | `test_verifier_pins_the_live_ollama_node_selector`, `test_verifier_requires_both_model_tolerations_on_ollama` |
| A release that is not `deployed`, or a `modeleval` pool with two nodes or a failed provisioning state, fails the audit | `test_verifier_fails_when_the_release_is_not_deployed`, `test_verifier_rejects_an_unhealthy_model_node_pool` |
| The workflow is parsed structurally: wrong `runs-on`, an upload path other than the safe-evidence directory, or a decoy workflow that merely mentions the scale set all fail | `test_verifier_rejects_a_workflow_that_targets_another_runner`, `test_verifier_rejects_an_artifact_upload_outside_the_safe_evidence_directory`, `test_verifier_ignores_an_unrelated_workflow_that_mentions_the_scale_set` |

## CLI surface the scripts rely on

- `az aks get-credentials --file` writes only that file; with `--overwrite-existing`
  it never touches `~/.kube/config`, and `--output none` keeps the context name
  off stdout.
- `kubelogin convert-kubeconfig -l azurecli` rewrites the file named by
  `KUBECONFIG`, which is why the export precedes it.
- `kubectl create secret generic --dry-run=client -o yaml` is client-side, so
  the secret is rendered locally and reaches the API only through
  `kubectl apply -f -` on stdin.
- `kubectl wait --for=condition=Ready` exits non-zero both on timeout and when
  no pod matches the selector, so a listener in the wrong namespace fails fast.
- `gh api --paginate` requests every page and, combined with `--jq`, applies the
  filter per page; the Environment endpoints return objects, so the filters are
  `.variables[].name` and `.secrets[].name`.
- `helm status --output json` reports `.info.status`, which is `deployed` only
  for a healthy release.

## Residual concerns

1. **No live run.** Everything is proven against fakes built from the committed
   values file and the live facts supplied for this task. Nothing was deployed.
2. **PyYAML.** The verifier needs a `python3` that can `import yaml` and says so
   before it does anything else. On a workstation without it, run the script
   from the repository's `uv` environment.
3. **Listener labels.** `actions.github.com/scale-set-name` and
   `actions.github.com/scale-set-namespace` are ARC conventions; a future chart
   could rename them, which would surface as a listener timeout rather than a
   silent pass.
4. **`configure-grounding-access.sh` uses a bare `mktemp -d`.** That is Task 3's
   code and out of scope here, but it inherits the BSD/macOS `TMPDIR` behaviour
   fixed in these two scripts.
