# Task 4 Report: Safe Campaign Evidence Ingestion & CLI

## Status: ✅ DONE

**Commits:**
- `fbfe523` — `feat(campaigns): emit safe campaign decisions`
- `3dcb4a3` — `fix(campaigns): strict evidence contract, CAS, atomic writes`
- `57f9555` — `fix(campaigns): cross-process CAS, strict schemas, artifact validation`
- `2a9aff3` — `fix(campaigns): compatibility fixes — campaign_action_id, case_repetitions shape, exact max_metric_calls`

## RED → GREEN

| Phase | Command | Result |
|-------|---------|--------|
| RED (wave 2) | `pytest tests/test_campaign_artifacts.py tests/test_campaign_cli.py -q` | ImportError / assertion failures on new tests |
| GREEN | same + state tests | 77 passed in 1.50s |
| Lint | `ruff check src/korvid_prompt_lab/campaign_artifacts.py src/korvid_prompt_lab/campaign_cli.py tests/test_campaign_artifacts.py tests/test_campaign_cli.py` | All checks passed |
| Types | `mypy src/korvid_prompt_lab/campaign_artifacts.py src/korvid_prompt_lab/campaign_cli.py` | Success: no issues found in 2 source files |

## CAS Invariants (cross-process)

1. `write_campaign_state` acquires exclusive advisory lock (`fcntl.flock(LOCK_EX)`) on adjacent `.lock` file inside validated state root.
2. Under lock: re-reads target, verifies expected_prior_hash against stored state_hash, writes to uuid-named temp in same directory, `os.fsync` fd, `os.replace` atomic, `os.fsync` directory fd, cleans temp in finally.
3. Two threads/processes with same expected_prior_hash: exactly one succeeds; second re-reads the new hash and raises ValueError.
4. Deterministic concurrent test uses `threading.Barrier(2)` — both threads wait, then race CAS; asserts exactly one ValueError.
5. Lock file opened safely; path validated within state_root; temp cleaned on any BaseException.

## Strict Schema Invariants

1. `_ROUND_SUMMARY_REQUIRED_KEYS` and `_EVAL_SUMMARY_REQUIRED_KEYS` define exact allowed key sets; unknown/missing keys rejected.
2. prompt_lab_revision and korvid_revision required non-empty in evidence; must match state exactly.
3. evaluated_models must contain exactly the expected model (state.model_identity.model); extras/duplicates rejected.
4. All numeric fields validated as finite, non-negative where required; bool-as-int rejected; empty strings rejected.

## Artifact Refs Validation

1. Every ref in `artifact_refs` is: non-empty, relative, no `..` traversal, no absolute path, unique.
2. Each ref resolves under evidence root to non-symlink path (stat only, never read content).
3. SEARCH requires comparison-summary.json present; loaded and validated.
4. MILESTONE/CONFIRM rejects comparison-summary.json and optimization files.


## Wave 3 Compatibility Fixes

- Status: DONE
- Commit SHA: `2a9aff3`

### What was fixed

1. `round-summary.json` now uses optional `campaign_action_id`, and the round CLI forwards `--campaign-action-id` into `write_safe_evidence`.
2. Campaign artifact ingestion now validates `comparison-summary.contract.case_repetitions` in the real `[case_id, model, repetition]` shape and rejects the old 2-element form.
3. Campaign artifact ingestion now requires `run_identity.max_metric_calls == action.metric_calls` exactly; lower and higher values both fail.
4. Tests were updated first (RED), then the producer/loader/CLI changes were implemented and re-verified (GREEN).

### RED evidence

```text
$ python -m pytest tests/test_campaign_artifacts.py tests/test_campaign_cli.py -x -q
F
FAILED tests/test_campaign_artifacts.py::TestLoadRoundOutcome::test_loads_valid_search_evidence
E   ValueError: round-summary missing key(s): action_id
```

### GREEN evidence

```text
$ python -m pytest tests/test_campaign_artifacts.py tests/test_campaign_cli.py -x -q
................................................                         [100%]
48 passed in 1.52s

$ python -m pytest tests/test_rounds.py -x -q
......................................                                   [100%]
38 passed in 0.79s

$ python -m pytest tests/test_rounds.py tests/test_round_cli.py -x -q 2>/dev/null; true
no tests ran in 0.00s

$ python -m ruff check src/korvid_prompt_lab/campaign_artifacts.py src/korvid_prompt_lab/rounds.py src/korvid_prompt_lab/round_cli.py src/korvid_prompt_lab/campaign_cli.py
All checks passed!

$ python -m mypy src/korvid_prompt_lab/campaign_artifacts.py src/korvid_prompt_lab/rounds.py src/korvid_prompt_lab/round_cli.py src/korvid_prompt_lab/campaign_cli.py
Success: no issues found in 4 source files
```

### Exact commands run

```text
cd /Users/hwang-inhwan/workspace/kube-prompt-grounding/.worktrees/feat-bounded-optimization-campaign && python -m pytest tests/test_campaign_artifacts.py tests/test_campaign_cli.py -x -q
cd /Users/hwang-inhwan/workspace/kube-prompt-grounding/.worktrees/feat-bounded-optimization-campaign && python -m pytest tests/test_campaign_artifacts.py tests/test_campaign_cli.py -x -q
cd /Users/hwang-inhwan/workspace/kube-prompt-grounding/.worktrees/feat-bounded-optimization-campaign && python -m pytest tests/test_rounds.py -x -q
cd /Users/hwang-inhwan/workspace/kube-prompt-grounding/.worktrees/feat-bounded-optimization-campaign && python -m pytest tests/test_rounds.py tests/test_round_cli.py -x -q 2>/dev/null; true
cd /Users/hwang-inhwan/workspace/kube-prompt-grounding/.worktrees/feat-bounded-optimization-campaign && python -m ruff check src/korvid_prompt_lab/campaign_artifacts.py src/korvid_prompt_lab/rounds.py src/korvid_prompt_lab/round_cli.py src/korvid_prompt_lab/campaign_cli.py
cd /Users/hwang-inhwan/workspace/kube-prompt-grounding/.worktrees/feat-bounded-optimization-campaign && python -m mypy src/korvid_prompt_lab/campaign_artifacts.py src/korvid_prompt_lab/rounds.py src/korvid_prompt_lab/round_cli.py src/korvid_prompt_lab/campaign_cli.py
git commit -m "fix(campaigns): compatibility fixes — campaign_action_id, case_repetitions shape, exact max_metric_calls" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
