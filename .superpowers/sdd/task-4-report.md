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
- Addressed by the blocking quality follow-up below.

## Task 4 blocking quality follow-up

### Status
- DONE

### Files
- `src/korvid_prompt_lab/aks.py`
- `tests/test_aks.py`

### Fix summary
- Added `AKSPortForwardTimeoutError`, injected `monotonic_clock`, and enforced a bounded readiness deadline for a live but stalled port-forward process.
- Switched the subprocess adapter from blocking `readline()` semantics to nonblocking output chunk reads so partial port-forward output cannot bypass the readiness deadline.
- Hardened cleanup to terminate, bounded-wait, kill if needed, bounded-wait again, and only do the final unconditional reap once exit is observable.

### Additional commands and results
- `uv run --python 3.12 pytest tests/test_aks.py -q -k 'stalled_live_process or ignores_terminate'`
  - RED/GREEN: `2 passed, 7 deselected in 0.11s`
- `uv run --python 3.12 pytest tests/test_aks.py -q -k 'partial_output_still_honors_timeout'`
  - RED: failed with `AssertionError: read_line should not be used for partial port-forward output`
- `uv run --python 3.12 pytest tests/test_aks.py -q`
  - PASS: `10 passed in 0.16s`
- `uv run --python 3.12 pytest -q`
  - PASS: `76 passed in 3.87s`

### Review follow-up
- Read-only review after the first fix pass found one remaining important issue: readiness still depended on blocking `readline()` behavior and could hang on partial output.
- Fixed by accumulating nonblocking output chunks during readiness parsing and re-verified the focused regression plus the full suite.

### Current concerns
- None.
