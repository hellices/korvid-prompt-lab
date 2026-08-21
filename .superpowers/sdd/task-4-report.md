# Task 4 Report

## Status
DONE

## Files
- `src/korvid_prompt_lab/aks.py`
- `tests/test_aks.py`

## Commands and results
- `uv run --python /Users/hwang-inhwan/.local/bin/python3.12 python -m pytest tests/test_aks.py -q`
  - RED: failed with `ModuleNotFoundError: No module named 'korvid_prompt_lab.aks'`
- `uv run --python /Users/hwang-inhwan/.local/bin/python3.12 python -m pytest tests/test_aks.py -q`
  - GREEN: `6 passed in 0.06s`
- `uv run --python /Users/hwang-inhwan/.local/bin/python3.12 python -m pytest tests/test_aks.py::test_aks_port_forward_cleans_up_on_base_exception_during_startup -q`
  - REVIEW RED: failed because cleanup did not run on `KeyboardInterrupt`
- `uv run --python /Users/hwang-inhwan/.local/bin/python3.12 python -m pytest tests/test_aks.py -q`
  - REVIEW GREEN: `7 passed in 0.12s`
- `uv run --python /Users/hwang-inhwan/.local/bin/python3.12 python -m pytest -q`
  - FULL: `73 passed in 3.26s`

## Self-review
- Verified all Azure and Kubernetes interactions are injected through command, process, and HTTP callables, so unit tests never touch AKS or Azure.
- Verified cluster, namespace, Service, Ready endpoints, loopback-only binding, and `/v1/models` model identity checks fail closed.
- Verified cleanup removes only the owned port-forward process and the exact generated kubeconfig path, including on startup interruption.
- Verified code and report avoid subscription identifiers and kubeconfig content.

## Code review
- Reviewer found one medium issue: startup cleanup skipped `BaseException` paths such as `KeyboardInterrupt`.
- Fixed by expanding startup cleanup to `except BaseException` and added a regression test.

## Commit
- `feat: add AKS model endpoint backend` with the required Co-authored-by trailer.

## Concerns
- The runtime cleanup currently waits indefinitely for the owned `kubectl port-forward` process after `terminate()`. That keeps behavior simple and exact, but a hung child process would block shutdown until the OS reaps it.
