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

---

# Task 3 Review Fix — Round 3 (executable pipeline)

## Status: ✅ COMPLETE

**Branch:** `feat/prompt-lab-mvp`

The previous round closed every trust-boundary finding but left the pipeline
**non-executable**: all three subcommand invocations were argv-incompatible with
the installed CLI, no campaign or `KORVID_AKS_*` variable was supplied, and the
design's `if: always()` cleanup step did not exist. All three findings are now
closed, and — critically — the tests can no longer pass without them.

## Finding 1 — orchestrator argv did not match the real CLI

`aks-check`, `evaluate`, and `optimize` all passed `--korvid-source-root` and
`--model` (which the parser never defines) and omitted `--campaign` (required on
all three) and `--max-metric-calls` (required on `optimize`). Every invocation
exited 2 before doing work, and because `aks-check` sat inside the retry loop the
argparse failure was indistinguishable from "pool not ready yet": each round
scaled a GPU node to 1 and spun for the full 15-minute deadline.

**Fix** (`scripts/run-grounding-round.sh`):

| subcommand | argv now emitted |
|---|---|
| `aks-check` | `--campaign`, `--artifact-root` |
| `optimize` | `--candidate`, `--campaign`, `--artifact-root`, `--max-metric-calls`, `--seed`, `--reflection-model`, `--train-case-id`, `--validation-case-id` |
| `evaluate` | `--candidate`, `--campaign`, `--artifact-root`, `--train-case-id`, `--validation-case-id`, repeated `--milestone-case-id` |

`KORVID_SOURCE_ROOT` stays runtime policy read from the environment — it is never
argv. The model comes from the campaign, not a flag.

The preflight loop now distinguishes failure classes: **exit 1 is retryable**
("not ready yet"); **any other non-zero exit aborts immediately** and propagates
its status, because argparse usage errors and campaign/config failures both exit
2 and can never resolve themselves.

## Finding 2 — no campaign, and none of the variables the campaign resolves

`examples/campaigns/aks-shared-runners.yaml` resolves `models`,
`serving.namespace`, `serving.service`, and `serving.model` through
`env:KORVID_AKS_{MODEL,NAMESPACE,SERVICE}`; `_resolve_env_string` raises when any
is absent, so campaign loading failed even after the argv fix.

**Fix** (`.github/workflows/grounding-round.yml`): six new dispatch inputs —
`campaign`, `train_case_id`, `validation_case_id`, `milestone_case_ids`
(comma-separated), `max_metric_calls` (number), `seed` (number) — all validated
in step 1, **before** the Korvid app token or Azure OIDC login. The orchestrator
step now also exports `KORVID_AKS_MODEL` from the allowlisted `model` input and
`KORVID_AKS_NAMESPACE`/`KORVID_AKS_SERVICE` from protected Environment `vars`.

Splitting is injection-safe: `IFS=',' read -r -a` splits on commas only — no
`eval`, no whitespace word-splitting, no pathname expansion — and the *whole*
list is matched against a closed vocabulary first, because `read -a` silently
drops empty leading/trailing fields (`a,` and `,a` would otherwise slip through).
Train/validation disjointness is enforced in the workflow *and* the script, and
`KORVID_AKS_MODEL` must equal `GROUNDING_MODEL`, so the allowlist actually binds
the model that gets served rather than only the one named in the report.

## Finding 3 — no `if: always()` restore step, no timeouts

On cancellation the runner SIGKILLs about ten seconds in, while
`az aks nodepool scale` takes minutes — so the trap was guaranteed to die
mid-flight and leak the GPU node. Job eviction and the 360-minute default timeout
never ran the trap at all.

**Fix:** `timeout-minutes: 180` on the job and `150` on the orchestrator step (so
the step expires first and cleanup still runs); a new read-only step records the
original count as `steps.modeleval.outputs.original-count` before the round; and
a final `if: always()` step restores it idempotently — original `1` is left
untouched without even querying, original `0` re-reads the current count and
requests an exact scale to `0` only when it is not already there. The step is
last, so summary, upload, and PR comment publish evidence first, and it carries
no `continue-on-error`, so cleanup failure stays visible. The shell trap is
retained as the fast path.

## Why the tests could not catch this before — and now can

The old fake `korvid-prompt-lab` shim accepted **any** argv, and the workflow
contract test derived "required" from `${VAR:?}` guards in the shell script,
which is strictly weaker than what the CLI and campaign loader need. Both holes
are closed:

- **Strict fake parser.** The shim now mirrors `build_parser()`: unknown options
  and missing required options exit 2 with an argparse-shaped usage error.
  `test_fake_parser_spec_matches_the_real_cli_parser` proves the shim's option
  table against the real parser, so it cannot drift.
- **Real-parser replay.** `test_round_script_argv_is_accepted_by_the_real_cli_parser`
  runs the orchestrator, then feeds every recorded argv to the real
  `build_parser().parse_args()`. `test_real_cli_parser_rejects_the_previous_orchestrator_argv`
  pins the old argv as a usage error.
- **Real campaign loader.** `test_campaign_loads_only_with_the_variables_the_workflow_exports`
  calls the real `load_campaign()` with the values the workflow's expressions
  resolve to, and asserts that deleting *any one* of them raises
  "references missing environment variable".
- **Executable cleanup logic.** The record and restore step bodies are extracted
  from the workflow YAML and run against a fake `az`, covering leaked-node
  restore, idempotent no-op, pre-existing capacity, never-recorded output, and
  a failing `az` (which must fail the step).

## Verification

```
tests/test_grounding_workflow.py                          40 passed
tests/test_grounding_script.py                            37 passed
Task 2 + Task 3 (workflow + script + rounds)              89 passed
full suite (KORVID_SOURCE_ROOT set)                      433 passed
ruff check .                                             All checks passed!
mypy --python-version 3.12 src tests                     no issues (33 files)
bash -n scripts/run-grounding-round.sh                   OK
bash -n on all 7 workflow run: bodies                    OK
node --check on the github-script body                   OK
YAML parse                                               OK
```

Out-of-band evidence (not committed):

- The **real** `korvid-prompt-lab` binary was run with the exact new argv:
  `aks-check --campaign examples/campaigns/aks-shared-runners.yaml --artifact-root …`
  now reaches the live AKS probe and exits **1**
  (`aks-check failed: AKS Service must expose Ready endpoints`) — the retryable
  signal. With `KORVID_AKS_MODEL` unset it exits **2**
  (`campaign.models references missing environment variable KORVID_AKS_MODEL`) —
  the immediate-abort signal. The old argv still exits 2 with
  `the following arguments are required: --campaign`.
- The validation step was executed against **26** input cases (accept baseline
  plus absolute/traversal/non-YAML/injected campaign, whitespace, command
  substitution, option-smuggling, glob and empty case ids, overlapping splits,
  leading/trailing/double-comma milestone lists, zero/negative/float/injected
  budgets, negative and alphabetic seeds, non-default branch, short SHA, tag ref,
  injected `pr_number`); every case produced the expected accept/reject outcome
  and no `pwned` marker was ever created.

An empty `qwen3:1.7b/` directory left in the worktree root by the old broken
argv was removed.

## Concerns / follow-ups

1. **`vars.KORVID_AKS_NAMESPACE` and `vars.KORVID_AKS_SERVICE` must be configured
   on the `aks-grounding` Environment** before the first round. They are blank by
   default, and the script aborts in seconds with a named variable rather than
   failing deep inside campaign loading.
2. **The shipped case-id defaults are campaign-specific.** They match
   `examples/campaigns/aks-shared-runners.yaml`; dispatching a different campaign
   requires overriding all three case-id inputs, and the CLI rejects case ids that
   are not drawn from the campaign's evaluated cases.
3. **`timeout-minutes: 150` on the orchestrator is a first estimate.** It has to
   cover a 15-minute preflight plus optimization at the chosen metric-call budget;
   raise both numbers together if a legitimate round is ever truncated.
4. **A pre-existing node (`original-count: 1`) is never released**, by design, but
   nothing in this workflow can distinguish "another round is using it" from "a
   previous round leaked it". That remains an operator responsibility.
5. **Task 4 scope untouched**: `README.md`,
   `examples/campaigns/aks-shared-runners.yaml`, and `tests/test_contracts.py`
   still carry their uncommitted Task 4 edits and are not part of this commit.
