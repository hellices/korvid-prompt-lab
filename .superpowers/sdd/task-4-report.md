# Task 4 Report: Safe Campaign Evidence Ingestion & CLI

## Status: ✅ DONE

**Commits:**
- `fbfe523` — `feat(campaigns): emit safe campaign decisions`
- `465336a` — `fix(campaigns): strict evidence contract, CAS, atomic writes`

## RED → GREEN

| Phase | Command | Result |
|-------|---------|--------|
| RED | `uv run --python 3.12 pytest tests/test_campaign_artifacts.py tests/test_campaign_cli.py -q` | ModuleNotFoundError |
| GREEN | same | 31 passed in 0.35s |
| Lint | `uv run --python 3.12 ruff check src/korvid_prompt_lab/campaign_artifacts.py src/korvid_prompt_lab/campaign_cli.py tests/test_campaign_artifacts.py tests/test_campaign_cli.py` | All checks passed |
| Types | `uv run --python 3.12 mypy src/korvid_prompt_lab/campaign_artifacts.py src/korvid_prompt_lab/campaign_cli.py` | Success: no issues found in 2 source files |
| Task 3 compat | `uv run --python 3.12 pytest tests/test_campaign_state.py -q` | 34 passed |
| Full suite | all three test files | 65 passed in 0.26s |

## Safe-Boundary Invariants

1. **Allowlisted files only**: `_ALLOWED_FILES` frozenset enforced before any read.
2. **Symlink rejection**: `_reject_symlink()` checks every path and ancestors.
3. **Path containment**: `_resolve_safe_path()` verifies resolved path is under root.
4. **No responses/ traversal**: never opened or listed.
5. **Action/model/revision/case contract**: `load_round_outcome` validates all identity fields against `control` + `state` — action_id, kind, evaluated case IDs (validation for SEARCH, milestone for MILESTONE/CONFIRM), model name, prompt_lab_revision, korvid_revision, candidate fingerprint, execution_modes include "live", repetitions_per_case positive int.
6. **MILESTONE/CONFIRM champion binding**: candidate_fingerprint must equal state.champion_fingerprint.
7. **SEARCH optimization validation**: requires optimization-summary.json + best-candidate.yaml, validates seed against action stage/seed_index, total_metric_calls ≤ action.metric_calls, seed_candidate_fingerprint = champion, train/validation case IDs match control, run_identity exact keys + values.
8. **MILESTONE/CONFIRM rejects optimization files**: presence of optimization-summary.json or best-candidate.yaml raises ValueError.
9. **GitHub output trust boundary**: CLI never reads GITHUB_OUTPUT env; writes only to explicit --github-output.

## CAS (Compare-and-Swap) Invariants

1. `--expected-prior-hash` is **required** on advance CLI (argparse enforced).
2. CLI validates `expected_prior_hash == state_hash(loaded_state)` before any advance attempt.
3. `write_campaign_state` on existing file: rejects if file's `state_hash` ≠ expected_prior_hash.
4. Two workers loading same prior: first succeeds writing output; second's advance attempt fails because the output file now has the first worker's new hash.
5. Temp file (`*.cas_tmp`) cleaned in `finally` block on any BaseException.
6. Original state file preserved intact when write/replace fails.

## Strict Types

- No `str()`, `float()`, `int()` coercion on untrusted values.
- `_require_int`: rejects bool, non-int.
- `_require_finite_float`: rejects bool, str, NaN, Inf.
- `_require_str`: rejects non-str, empty.
- `_require_bool`: rejects non-bool.
- `_require_string_list`: rejects non-list, non-str items, empty items.
- `_ensure_exact_keys`: rejects missing or unknown keys.

## Atomic Write Safety

- `write_campaign_state`: writes to `.cas_tmp`, replaces target in finally-guarded block; unlinks temp on any failure.
- `write_campaign_artifacts`: creates output dir, catches BaseException, removes partial files and attempts rmdir on failure.
- Tests inject `OSError` on `Path.replace` and `Path.write_text`; verify no temp leftovers and original state preservation.

## Self-Review

- All 7 review findings addressed with tests.
- Render uses `next_action()` from the state machine for exact stage name and budget in markdown.
- No defaults/fallbacks for missing required fields — strict `_require_*` validators raise on any mismatch.
- CLI advance passes explicit `state_root` to `write_campaign_state` for path containment.
- Test coverage: wrong action_id, wrong case set, wrong model, wrong revision, symlinks, malformed JSON, bool-as-int, non-finite float, missing optimization-summary, wrong seed, budget exceeded, wrong seed fingerprint, milestone rejects optimization files, CAS stale/concurrent/failure.

## Concerns

- `write_campaign_state` uses `Path.replace()` for atomic rename which is not atomic across filesystems on some platforms; production deployments should ensure state file and temp are on same filesystem.
- For truly concurrent multi-worker safety on shared state, an advisory file lock or server-side compare-and-swap would be stronger than file-based CAS.
- `_load_control` in CLI reconstructs `OptimizationCampaign` without the full `Campaign` evaluation cross-validation (case ID coverage check); this is acceptable because the control file was already validated at campaign initialization time.
