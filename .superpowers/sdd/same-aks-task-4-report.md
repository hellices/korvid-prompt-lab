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
