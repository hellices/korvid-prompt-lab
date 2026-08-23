# Actions Task 4 Report

## Status

DONE

## Commit

`4ee913c` — `docs: explain remote grounding rounds`

## Files modified

- `README.md`

## Files already committed (verified in place)

- `examples/campaigns/aks-shared-runners.yaml` — `--turn-timeout 300` explicit in serving command
- `tests/test_contracts.py` — `test_load_campaign_from_example_yaml` asserts `("--turn-timeout", "300")` in the AKS serving command tuple

## Changes made

### Corrected inaccurate README wording

Replaced the false claim that `qwen3:4b` tool-enabled turns "took just over two
minutes" with the accurate observation:

- Requests were observed still running at **5m20s and 10m40s**.
- Both the 300 s and 600 s budgets were exhausted without completion.
- Root cause: Ollama generation is unbounded by default.
- `--turn-timeout 300` is pinned for the initial `qwen3:0.6b` rounds only.
- Larger reasoning models require a separate bounded-serving policy (e.g.
  `num_predict` or a vLLM `max_tokens` guard) before selection.

### Added: GitHub Actions Grounding Rounds section

Documents the full operator setup required to dispatch a remote grounding round:

| Topic | Detail |
| --- | --- |
| Environment | `aks-grounding` with required reviewers |
| Repository variables | `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `KORVID_AKS_NAMESPACE`, `KORVID_AKS_SERVICE`, `KORVID_APP_ID`, `GROUNDING_REFLECTION_MODEL` (env-scoped) |
| Secrets | `KORVID_APP_PRIVATE_KEY`, `GROUNDING_REFLECTION_CREDENTIAL` (env-scoped) |
| GitHub App | Read-only install on `hellices/korvid` (`contents: read` only); App id in `vars.KORVID_APP_ID`, private key in `secrets.KORVID_APP_PRIVATE_KEY` |
| ARC runner label | `korvid-runners` (existing ARC scale set) |
| Dispatch inputs | All 12 inputs with defaults and validation rules documented |
| Result surfaces | Job Summary, artifact (`grounding-round-<run-id>`), sticky PR comment |
| Cleanup semantics | `if: always()` step restores exact original `modeleval` count; covers SIGKILL/cancel; idempotent |
| Rerun semantics | Each dispatch is independent; no accumulated state; fix root cause before redispatching |
| Local vs remote path | Remote = normal (ENV gate, OIDC, ARC, cleanup); local = diagnostic only |

### Added: Measured baseline section

```text
model:               qwen3:0.6b / shipped-small
campaign:            aks-shared-runners (5 repetitions × 2 cases)
live runs completed: 10
aggregate score:     0.01
pass^3:              0.0
pass^5:              0.0
hard safety failures: 14
systemic failures:   0
```

Clarifies that all 10 runs completed without systemic failure (bridge, AKS
port-forward, and harness wiring all worked); low scores are model-capability
observations, not infrastructure failures.

### Updated: Model matrix

- Added `qwen3:0.6b` row with baseline figures.
- Corrected `qwen3:4b` row: noted that unbounded serving makes it not a valid
  comparison point; larger models (`qwen3:8b`, `qwen3:14b`) updated similarly
  to note bounded serving prerequisite.

## Verification commands and results

```
KORVID_SOURCE_ROOT=.../feat-307-small-operator-foundation uv run --python 3.12 pytest -q
→ 457 passed in 84.65s

uv run --python 3.12 ruff check .
→ All checks passed!

uv run --python 3.12 mypy --python-version 3.12 src tests
→ Success: no issues found in 33 source files

bash -n scripts/run-grounding-round.sh
→ (no output — syntax OK)
```

## nodepool modeleval count — read-only verification

`test_grounding_workflow.py::test_grounding_workflow_modeleval_record_step_is_read_only`
confirms the `Record original modeleval node count` step contains no
`--node-count` argument (read-only `az aks nodepool show` only). The test is
already part of the 457-test suite above.

## Concerns

None. The fix addresses all inaccurate wording, the full operator setup is
documented, the measured baseline is recorded with safe aggregate-only evidence,
the `--turn-timeout 300` contract is pinned and tested, and all verification
gates pass.
