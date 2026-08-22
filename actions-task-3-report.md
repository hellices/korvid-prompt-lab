# Actions Task 3 Report – Protected Grounding Rounds Workflow

## Status: ✅ COMPLETE

**Commit:** `bc32bd0`  
**Branch:** `feat/prompt-lab-mvp`

---

## Files Created

| File | Purpose |
|------|---------|
| `.github/workflows/grounding-round.yml` | Protected CI workflow for grounding rounds |
| `tests/test_grounding_workflow.py` | 9 TDD contract tests verifying security invariants |

---

## Test Results

### Focused suite (Task 3)
```
9 passed in 0.09s   (tests/test_grounding_workflow.py)
```

### Full Task 3 suite
```
36 passed in 23.51s  (test_rounds, test_grounding_script, test_grounding_workflow)
```

---

## Security Controls Implemented

| Control | Implementation |
|---------|---------------|
| Trigger | `workflow_dispatch` only; `pull_request_target` explicitly absent |
| Runner | `korvid-runners` (self-hosted, private network) |
| Environment | `aks-grounding` (GitHub protected environment with required reviewers/secrets) |
| Auth | `azure/login` OIDC; no `AZURE_CLIENT_SECRET`; no PAT |
| Korvid checkout token | `actions/create-github-app-token` (read-only scope) |
| Concurrency | `cancel-in-progress: false` (serialised; no TOCTOU races) |
| Action pinning | All actions pinned to full commit SHA |
| Artifact safety | Only `safe-evidence/` uploaded; `artifacts/live` excluded |
| Failure visibility | `always()` steps never suppress job failure; non-always orchestrator step fails job |
| PR comment | `if: always() && inputs.pr_number != ''`; body is safe markdown + run link only |
| Step summary | `if: always() && hashFiles(...)` guards absent file; appends `round-summary.md` |
| Marker | `<!-- korvid-grounding:<model>:<candidate> -->` for sticky deduplication |

---

## Concerns / Notes

1. **Action SHA pinning maintenance**: The 6 pinned SHAs need periodic rotation via Dependabot or manual audit. Recommend enabling `dependabot.yml` with `package-ecosystem: github-actions`.
2. **`create-github-app-token` scope**: The app token is scoped to the `korvid` repository only. If the repo owner changes, `vars.KORVID_APP_ID` and `secrets.KORVID_APP_PRIVATE_KEY` must be updated in the `aks-grounding` environment.
3. **`if: always()` failure propagation**: Steps 8–10 run with `always()` but are non-blocking post-steps. If the orchestrator (step 7) fails, the job still exits non-zero. The always-steps add evidence without hiding the failure.
4. **PR comment requires `pull-requests: write`**: This permission is top-level to allow the sticky comment step. The checkout step uses only `contents: read` — no privilege escalation path exists because `pull_request_target` is absent.
5. **Existing uncommitted files** (README, examples, test_contracts) are Task 4 scope and were not touched.

---

# Review-Fix Wave – Task 3 (all findings closed)

## Status: ✅ FIXED (re-verified)

Supersedes the "COMPLETE" claim above: the reviewed commit `bc32bd0` could not
execute a single run. Every Critical and Important finding is closed below, each
by a structural test that fails against the old workflow.

## Critical

| # | Finding | Fix |
|---|---------|-----|
| 1 | `astral-sh/setup-uv` pinned to a nonexistent SHA | Repinned to the real `v6.4.3` commit `e92bafb6253dcd438e0484186d7669ea7a8ca1cc`. All 7 pins re-verified against upstream tag refs (`azure/login` v2.2.0 is an annotated tag dereferencing to `a65d910e…`). |
| 2 | `GROUNDING_ARTIFACT_ROOT` never set | Set once at job level to `${{ github.workspace }}/prompt-lab/artifacts/grounding-round` (gitignored `artifacts/` root). `GROUNDING_SAFE_EVIDENCE_DIR` is derived from it, and the summary/upload/PR-comment paths all resolve to that same directory. |
| 3 | Script injection via `inputs.candidate` / `inputs.pr_number` | No `${{ }}` expression appears in any `run:` or `script:` body anywhere in the workflow. `github-script` reads `process.env.*`, re-validates `pr_number` against `^[1-9][0-9]{0,9}$`, and confirms the target is a pull request in this repository via `pulls.get` before commenting. |

## Important

| # | Finding | Fix |
|---|---------|-----|
| 4 | `optimize-evaluate` could never succeed, and failed only after scaling up | `GROUNDING_REFLECTION_MODEL` (`vars.`) and `GROUNDING_REFLECTION_CREDENTIAL` (`secrets.`) are wired from the protected Environment and materialise only when `round_type == 'optimize-evaluate'`. Both `${VAR:?}` guards moved above the nodepool read, scale-up, and 15-minute preflight in `run-grounding-round.sh`. |
| 5 | Korvid checkout had no `uv` environment | New step provisions it out of tree: absolute `UV_PROJECT_ENVIRONMENT=${{ runner.temp }}/korvid-uv-env`, `uv sync --frozen --dev --all-extras --no-editable` (Korvid's documented mirror-safe install), then asserts no in-tree `.venv` and `git status --porcelain` is empty. The orchestrator step exports the same path so the bridge's `uv run --project <korvid> --no-sync` resolves it. |
| 6 | Safe-evidence path mismatch failed silently | Single derived path everywhere; `if-no-files-found: error` so missing evidence fails the job instead of producing a green empty run. |
| 7 | Korvid app token not downscoped | `permission-contents: read` on `create-github-app-token`; both checkouts use `persist-credentials: false`. |
| 8 | Contract tests asserted substrings, not invariants | Test file rewritten: 24 structural tests (see below). |

## Trust boundary (design §"Trigger and trust boundary")

The workflow definition executes from the default branch. The Prompt Lab commit
that actually runs is the new required `prompt_lab_ref` input. Step 1 — before
any checkout, before the Korvid token, before `azure/login` — rejects:

- a dispatch whose `github.ref_name` is not the repository default branch;
- any `prompt_lab_ref`/`korvid_ref` that is not exactly `^[0-9a-f]{40}$`;
- a `pr_number` that is not blank or a positive integer;
- a `candidate` that is absolute, contains `..`, or leaves `[A-Za-z0-9._/-]`.

`PROMPT_LAB_REVISION` now records `inputs.prompt_lab_ref` (the commit that ran)
rather than `github.sha` (the workflow definition revision).

## Contract tests (24, all structural)

Exact trigger key set `{workflow_dispatch}` · every input typed · 40-hex SHA pin
regex cross-checked against a known action→(SHA, tag) map plus its provenance
comment · no mutable refs · exact least-privilege `permissions` mapping ·
`permission-contents: read` and no `permission-*: write` · read-only checkouts ·
reflection credentials from `vars.`/`secrets.` gated on round type · no `${{ }}`
in any script body · `pr_number` read from `process.env` and re-validated ·
one artifact root and one safe-evidence path shared by summary/upload/comment ·
`if-no-files-found: error` · **workflow `env:` cross-checked against every
`${VAR:?}` guard parsed out of `run-grounding-round.sh`** · no
`continue-on-error` anywhere and no `if:` on the orchestrator step · Korvid env
provisioned out of tree with `--frozen` · Prompt Lab bin on `$GITHUB_PATH`.

Three new Task 2 tests assert that a missing reflection model or credential
aborts with **zero** cloud calls (the `az` shim now records `nodepool-show`), and
that evaluate-only rounds run with no reflection config at all.

## Additional defect found while fixing #4

With an `EXIT` trap installed, bash returns the *trap's* status for a `${VAR:?}`
expansion error, so the old ordering exited **0** while printing
"GROUNDING_REFLECTION_MODEL is required" — a misconfigured optimize round would
have reported success. `cleanup()` now captures `$?` on entry and re-exits with
it; cleanup failures still surface because `set -e` aborts the trap first.

## Verification

```
tests/test_grounding_workflow.py                          24 passed
tests/test_grounding_workflow.py + test_grounding_script.py + test_rounds.py
                                                          54 passed
full suite (KORVID_SOURCE_ROOT set)                      398 passed
ruff check .                                             All checks passed!
mypy --python-version 3.12 src tests                     no issues (33 files)
bash -n scripts/run-grounding-round.sh                   OK
YAML parse                                               OK
```

Out-of-band evidence (not committed): all 7 pinned SHAs resolved through
`gh api repos/<action>/commits/<sha>`; the validation step was executed against
11 hostile input cases (branch ref, short/uppercase SHA, tag, `12; rm -rf /`,
quote-escape injection in `candidate`, traversal, absolute path, non-default
branch) with the expected accept/reject outcome each time; the `github-script`
body passed `node --check`; and the workflow's resolved `env:` block was fed to
the real orchestrator with fake `az`/CLI shims, completing both `evaluate` and
`optimize-evaluate` and delivering the credential to the optimize subprocess as
`OPENAI_API_KEY` only.

## Concerns / follow-ups

1. **`uv sync --frozen` requires Korvid to commit `uv.lock`** (it does today). If
   that changes, step 8 fails loudly rather than silently building a stale env.
2. **`if-no-files-found: error` adds a second red step** when the round dies
   before writing evidence (e.g. Azure login failure). That is intended — silent
   evidence loss was the finding — but the job summary will show two failures.
3. **`vars.GROUNDING_REFLECTION_MODEL` / `secrets.GROUNDING_REFLECTION_CREDENTIAL`
   must be configured on the `aks-grounding` Environment** before the first
   `optimize-evaluate` round; the round now aborts in seconds if they are absent.
4. **Action pins still need rotation** (Dependabot `github-actions` ecosystem);
   the known-pin map in the test file must be updated in the same commit, which
   is deliberate friction.
5. **Task 4 scope untouched**: `README.md`, `examples/campaigns/aks-shared-runners.yaml`,
   and `tests/test_contracts.py` still carry their uncommitted Task 4 edits. The
   README prerequisite about a pre-synced Korvid checkout is now satisfied by the
   workflow itself and should be documented in Task 4.
