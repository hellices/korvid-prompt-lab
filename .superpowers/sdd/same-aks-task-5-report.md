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
