# Same-AKS Task 5 Report: ACR Build Fix — `--kubectl-version` → `--client-version`

## Live Failure

`az acr build` run `ch1r` failed inside the Docker build step with an error from
`az aks install-cli` rejecting the flag `--kubectl-version v1.35.6`.

**Root cause:** The flag `--kubectl-version` does not exist in the installed Azure CLI.
`az aks install-cli --help` reveals the correct flag is `--client-version` for
kubectl, while kubelogin continues to use `--kubelogin-version`.

## TDD Evidence

### RED (before Dockerfile fix)

Test updated first in `tests/test_grounding_infrastructure.py`:

```python
assert "--client-version v1.35.6" in body
assert "--kubectl-version" not in body
assert "--kubelogin-version v0.2.19" in body
```

Run:
```
uv run pytest -q tests/test_grounding_infrastructure.py::test_prompt_lab_runner_image_pins_required_tools_and_non_root_user
```

Result:
```
FAILED tests/test_grounding_infrastructure.py::test_prompt_lab_runner_image_pins_required_tools_and_non_root_user
AssertionError: assert '--client-version v1.35.6' in '...'
1 failed in 0.21s
```

### GREEN (after Dockerfile fix)

`infra/arc/runner/Dockerfile` changed:
```diff
-      --kubectl-version v1.35.6 \
+      --client-version v1.35.6 \
```

Run:
```
uv run pytest -q tests/test_grounding_infrastructure.py::test_prompt_lab_runner_image_pins_required_tools_and_non_root_user
```

Result:
```
1 passed in 0.04s
```

Full suite:
```
123 passed in 94.76s (0:01:34)
```

Ruff + mypy: clean.

## Files Changed

| File | Change |
|------|--------|
| `tests/test_grounding_infrastructure.py` | Require `--client-version`, forbid `--kubectl-version` |
| `infra/arc/runner/Dockerfile` | `--kubectl-version` → `--client-version` |
| `docs/superpowers/plans/2026-08-23-same-aks-runner-grounding.md` | Same flag fix in code block and test snippet |

---

# Same-AKS Task 5 — Pre-merge Validation (2026-08-23)

## Read-only state checks

| Check | Result |
|-------|--------|
| `hellices/korvid-prompt-lab` default branch | `main` |
| admin access | `true` |
| `modeleval` count | `0` |
| `modeleval` provisioningState | `Succeeded` |
| `runner-base:prompt-lab-v1` ACR image digest | `sha256:5c8105400a9f6035a8fb7f7a06e6f81277af45584a148a0af6437bef259bae56` |
| ACR image lastUpdateTime | `2026-08-23T04:13:18Z` |
| `prompt-lab-runners` ARC release | absent |
| `aks-grounding` Environment | absent |
| `korvid-runners` githubConfigUrl | `https://github.com/hellices/korvid` (unchanged) |

## Repository validation

| Check | Result |
|-------|--------|
| pytest (KORVID_SOURCE_ROOT=/Users/hwang-inhwan/workspace/kube) | 656 passed, 6 skipped |
| ruff check . | All checks passed |
| mypy --python-version 3.12 src tests | no issues found in 36 source files |
| bash -n scripts/*.sh | OK |
| YAML parse .github/workflows/*.yml | OK |

## Deployment boundary

- **Deployed:** ACR image `runner-base:prompt-lab-v1` built and pushed (ch1s fix applied: `--client-version` not `--kubectl-version`).
- **Not installed:** `aks-grounding` Environment and `prompt-lab-runners` ARC scale set — all GitHub App env inputs (`KORVID_APP_ID`, `KORVID_APP_PRIVATE_KEY_FILE`, `ARC_GITHUB_APP_ID`, `ARC_GITHUB_APP_INSTALLATION_ID`, `ARC_GITHUB_APP_PRIVATE_KEY_FILE`) are unset.
- **Blocked by design:** Live grounding round waits for default-branch merge (`grounding-round.yml` dispatches from `main` only).

## Remaining prerequisites

1. `KORVID_APP_ID` + `KORVID_APP_PRIVATE_KEY_FILE` → `scripts/configure-grounding-access.sh`
2. `ARC_GITHUB_APP_ID` + `ARC_GITHUB_APP_INSTALLATION_ID` + `ARC_GITHUB_APP_PRIVATE_KEY_FILE` → `scripts/install-prompt-lab-runner.sh`
3. Merge to `main`
4. Dispatch `grounding-round.yml` from `main`

## Verdict

**DONE_WITH_CONCERNS**

All pre-merge, credential-free checks pass. Three remaining prerequisites are operator-credential-gated (Steps 3–4) and a fourth is merge-gated (Steps 7–8). No cloud or GitHub mutation was performed.

---

# Same-AKS Task 5 — Final-Review Follow-up (2026-08-23)

Relevant because Task 5's deliverable is the ACR image itself.

The image `acrpensionguard.azurecr.io/runner-base:prompt-lab-v1` that this task
built and pushed was, until now, asserted only in `infra/arc/prompt-lab-runners-values.yaml`
and in the offline values test.  Neither `scripts/install-prompt-lab-runner.sh`
nor `scripts/verify-grounding-deployment.sh` read `.image` off the live
`AutoscalingRunnerSet`, so a release installed or edited onto any other image
passed both scripts.

Both scripts now assert the runner container — selected by **name**, never by
index — carries exactly
`acrpensionguard.azurecr.io/runner-base:prompt-lab-v1`.  A drift fails the
install and fails the read-only audit.

Evidence, RED/GREEN transcripts, and residual concerns (notably: the pin is by
tag, not by the digest `sha256:5c8105400a9f6035a8fb7f7a06e6f81277af45584a148a0af6437bef259bae56`
recorded above) are in the "Final-Review Fixes" section of
`.superpowers/sdd/same-aks-task-4-report.md`.

No cloud, ACR, or GitHub mutation was performed for this follow-up: the image
and the deployment boundary recorded above are unchanged.
