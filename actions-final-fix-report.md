# Actions Final Fix Report

## Status

DONE — three verified whole-branch review findings closed (TDD: RED → GREEN).

## Files modified

- `.github/workflows/grounding-round.yml`
- `src/korvid_prompt_lab/rounds.py`
- `tests/test_grounding_workflow.py`
- `tests/test_rounds.py`
- `README.md`

## Finding 3 — `korvid_ref` provenance not proven (final review finding)

### Root cause

`korvid_ref` was validated only for shape (40-hex SHA). There was no proof that
the SHA was a commit already reachable from the default branch of the
authoritative Korvid repository. A dispatcher could supply an arbitrary commit
from an unmerged branch or fork, which would then be checked out beside the
Azure OIDC session and Korvid app token — the same exposure the `prompt_lab_ref`
trust step was designed to prevent.

### Fix

The existing pre-credential `actions/github-script` step (step 2,
`Verify the Prompt Lab commit is trusted`) is extended: after
`prompt_lab_ref` provenance is established it also verifies `korvid_ref` using
the same read-only `GITHUB_TOKEN` against the **public** repository
`{github.repository_owner}/korvid` (derived, not user-controlled) — before the
Korvid GitHub App token, Azure login, or any checkout.

Two new `env:` bindings are added to the step:
- `KORVID_REF: ${{ inputs.korvid_ref }}` — read via `process.env.KORVID_REF`
- `KORVID_REPO: ${{ github.repository_owner }}/korvid` — never user-controlled

The script calls `repos.get` on the Korvid repo to obtain its current default
branch, then calls `compareCommitsWithBasehead(<korvid_ref>...<default_branch>)`
(falling back to `compareCommits` on older octokit builds). Status `identical`
or `ahead` → accepted; `behind`/`diverged`, API errors, unknown SHAs → rejected.

| `korvid_ref` | Accepted | Rejected |
| --- | --- | --- |
| Any 40-hex SHA | `identical` (tip) or `ahead` (ancestor) against Korvid default branch | `behind`/`diverged` (unmerged/fork), API unreachable, unknown SHA |

### Tests (structural + executable)

**Structural (RED → GREEN)**:
- `test_grounding_workflow_trust_check_binds_korvid_ref_env` — asserts `KORVID_REF == "${{ inputs.korvid_ref }}"` and `process.env.KORVID_REF` in script
- `test_grounding_workflow_trust_check_binds_korvid_repo_env` — asserts `KORVID_REPO == "${{ github.repository_owner }}/korvid"` and `process.env.KORVID_REPO` in script
- `test_grounding_workflow_trust_check_verifies_korvid_provenance_before_app_token` — asserts both env keys present in pre-credential trust step
- `test_grounding_workflow_trust_check_script_reads_korvid_ref_from_env` — asserts `process.env.KORVID_REF`, `process.env.KORVID_REPO`, no `${{` in script

**Executable (RED → GREEN)**:
- `test_trust_script_korvid_accepts_default_branch_ancestor[ahead/identical]` — korvid ancestor accepted; compare call made against `repo=korvid`
- `test_trust_script_korvid_rejects_unmerged_ref[diverged/behind]` — `setFailed` with korvid message
- `test_trust_script_korvid_rejects_api_failure` — API error for korvid compare → `setFailed`

**Harness extension** (`run_trust_script`): compare routing now branches by `params.repo === 'korvid'` so prompt-lab and korvid comparisons use separate fixture slots; `repos.get` handler added; default `korvid_compare={"status":"ahead"}` keeps all existing tests green.

**Updated existing assertions**: `test_trust_script_accepts_a_same_repository_pull_request_head`, `test_trust_script_accepts_a_default_branch_ancestor`, and `test_trust_script_still_compares_without_the_basehead_endpoint` now account for the additional korvid API calls (`repos.get` + `compareCommitsWithBasehead/compareCommits`) that follow a successful prompt_lab check.

**RED count**: 9 failures before implementation.  **GREEN count**: 70 passed.

## Finding 1 — trust boundary before checkout, app token, and Azure login

### Root cause

`RoundReport`, `round-summary.json`, and `round-summary.md` omitted four items
the design's result model requires: per-model scores, per-run elapsed duration,
artifact names, and the reproduction command (the command was in the JSON only,
unquoted and never displayed).

### Fix (`src/korvid_prompt_lab/rounds.py`)

- `RoundReport` gains `model_scores`, `artifact_refs`, `reproduction_command`;
  `CaseRunSummary` gains `elapsed_seconds`, sourced from the already-safe
  `usage.wall_time_seconds` projection (`None` for model failures).
- Markdown gains `## Per-model scores`, an `Elapsed (s)` column, `## Artifacts`,
  and `## Reproduction command` rendered through `shlex.join` so a copied command
  cannot re-split or re-interpret an argument. Rendering stays deterministic.
- `round-summary.json` gains `model_scores`, per-run `elapsed_seconds`, and
  `evaluation_artifact_refs`; `artifact_refs` still names the uploaded package.
- Validation: artifact refs must be relative, control-character-free, in-tree
  POSIX paths (absolute, `~`, `\`, `..`, and empty refs raise);
  `reproduction_command` tokens reject control characters; `_require_numeric`
  now rejects non-finite values and durations reject negatives.
- Display allowlist/denylist: only `.json`/`.yaml`/`.yml`/`.md` names may be
  shown, and any ref naming a request payload, audit record, kubeconfig,
  credential/secret/token, `.env`, GEPA state, or answer is dropped — a real
  `evaluation-summary.json` lists `runs/*/request.json`, which never reaches the
  report, the Markdown, or the JSON.

### Tests

`tests/test_rounds.py` gained realistic protocol-v2 fixtures (`qwen3:0.6b`, the
two shipped AKS cases, two repetitions, real wall times, `request.json`
artifacts on disk and in `artifact_refs`) plus tests for the report contract,
Markdown rendering, forbidden-artifact filtering, unsafe path rejection, unsafe
command tokens, unusable durations (`-1`, `inf`, `nan`), model-failure `n/a`
duration, and the full safe `round-summary.json` contract.

## README

- New “Trust boundary for `prompt_lab_ref`” section with the accept/reject table.
- New "Trust boundary for `korvid_ref`" section with the accept/reject table.
- `prompt_lab_ref` / `pr_number` / `korvid_ref` input rows describe the provenance requirement.
- Job Summary row lists per-model scores, elapsed duration, artifact names, and
  the shell-quoted reproduction command.
- Artifact name corrected from `grounding-round-<run-id>` to the actual
  **`safe-evidence`**, and `artifact_refs` vs `evaluation_artifact_refs`
  explained.

## Verification commands and results

```
KORVID_SOURCE_ROOT=…/feat-307-small-operator-foundation uv run --python 3.12 \
  pytest -q tests/test_grounding_workflow.py
→ 70 passed in 3.84s   (RED first: 9 failures before implementation)

KORVID_SOURCE_ROOT=…/feat-307-small-operator-foundation uv run --python 3.12 pytest -q
→ 492 passed, 6 skipped in 75s

uv run --python 3.12 ruff check .
→ All checks passed!

uv run --python 3.12 mypy --python-version 3.12 src tests
→ Success: no issues found in 33 source files

bash -n scripts/run-grounding-round.sh
→ (no output — syntax OK)

python -c "import yaml; yaml.safe_load(open('.github/workflows/grounding-round.yml'))"
→ parsed OK; steps = 17
```

## Node-pool read-only validation

- Workflow contract tests: `Record original modeleval node count` contains no
  `--node-count`, the executable fake-`az` run records `calls == ["show"]`, and
  restore only ever scales back to the recorded original count
  (11 selected tests passed).
- Live read-only confirmation (no mutation):
  `az aks nodepool show -g rg-pension-guard --cluster-name aks-shared-runners --name modeleval`
  → `count: 0`, `mode: User`, `provisioningState: Succeeded`.

## Concerns

- `tests/test_grounding_script.py::test_round_script_sigterm_exits_143_and_scales_down_exactly_once`
  failed once during a full-suite run and then passed on a full re-run and 3/3
  in isolation. It signals a process group and races the shell's trap, so it is
  load-sensitive. Nothing in this change touches `scripts/run-grounding-round.sh`
  or that test; it is a pre-existing flake.
- The trust check depends on the GitHub API being reachable from the runner; an
  API outage fails the round closed (no checkout, no credentials), which is the
  intended direction — this applies to both `prompt_lab_ref` and `korvid_ref`.
- The `repos.get` call to fetch the Korvid default branch adds one extra API
  request before the round. At one round per dispatch on a public repo this is
  well within any rate limit.

---

# Review Round 2 — the shipped `korvid_ref` default could not pass the shipped gate

## Status

DONE — one verified finding closed (TDD: RED → GREEN).

## Finding

Round 1 added a pre-credential provenance gate that accepts `korvid_ref` only
when the Korvid **default branch** contains it. The workflow's own default,
`fc7eece2adb66a5b2a18d378bdfd7503ddbdd2ca`, is not such a commit:

```
gh api repos/hellices/korvid/compare/fc7eece2adb66a5b2a18d378bdfd7503ddbdd2ca...main
→ {"status":"diverged","ahead_by":4,"behind_by":41}
```

Every default dispatch was therefore rejected before it began. The gate and the
default were reviewed in separate changes and nothing bound them together.

## Root cause (deeper than a stale pin)

`fc7eece2` is on `hellices/korvid` branch `feat/307-small-operator-foundation`,
the head branch of **open PR #312** (`base: main`, head repo `hellices/korvid` —
not a fork). Its 41 commits carry the entire operation-journey harness.

That harness **has never existed on `main`**:

```
git log origin/main -- src/korvid/evals/operation.py tests/evals/operation_app.py \
                       tests/evals/operation_campaign.py tests/evals/operation_scripts.py
→ (empty)
git grep LIFECYCLE_CHECKPOINTS origin/main -- src tests
→ (empty)
```

So **no `main`-reachable commit can satisfy the bridge contract.** Repinning to
today's `main` tip (`aaf07ff`) would have cleared the gate and then failed at run
time with "korvid operation harness is not importable" — *after* the Korvid app
token, the Azure OIDC session, and the GPU node pool were spent. That trades a
cheap pre-credential rejection for an expensive post-credential failure.

The defect is the gate's *policy*, not the pin.

## Compatibility evidence

`bridge_worker._import_korvid` requires exactly:

| Module | Names | Present at `fc7eece2` |
| --- | --- | --- |
| `korvid.agent.profiles` | `PromptOverrides` | yes |
| `korvid.evals.operation` | `LIFECYCLE_CHECKPOINTS`, `bundled_operations_dir`, `load_operation_journeys` | yes |
| `korvid.evals.scripted` | `ScriptedProvider` | yes |
| `korvid.providers.openai_compat` | `OpenAICompatProvider`, `ProviderError` | yes |
| `korvid.providers.static_creds` | `StaticHeaderSource` | yes |
| `tests.evals` | `operation_app` (+ its `build_profile` attribute the bridge rebinds) | yes |
| `tests.evals.operation_campaign` | `approval_timeout_for` | yes |
| `tests.evals.operation_scripts` | `OPERATION_SCRIPTS` | yes |
| `tests.ui.waits` | `WaitTimeout` | yes |

Why the pin does **not** move forward to the branch head:

- Every bridge-imported file is **byte-identical** (same blob SHA) between
  `fc7eece2` and the later branch heads `bff8faff` and `525378f0` — moving
  forward buys the bridge nothing.
- The 7+ newer commits touch only `operation_outcome.py`, release scripts,
  packaging and docs. The packaging change
  (`[tool.hatch.build] exclude = ["/src/korvid/evals", "/tests/evals"]`) affects
  built wheels/sdists only; the bridge runs `uv run --no-sync` against a source
  checkout with `PYTHONPATH=$KORVID_SOURCE_ROOT`, so it is unaffected.
- **CI**: `fc7eece2` is the newest fully green commit on the branch
  (`test 3.11/3.12/3.13`, `windows-test`, `pre-commit`, `security`, `analyze` all
  `success`). `4d64ddc` fails `test (3.11)` and `windows-test`; `bff8faff` fails
  `windows-test`; `525378f0` is still running. `fc7eece2` is therefore the newest
  **stable** commit that carries every bridge dependency.

## Fix

1. **Gate policy** — `korvid_ref` is now proven authoritative by either route:
   containment in the Korvid default branch, **or** containment in the head of an
   **open** pull request of `hellices/korvid` whose head repository is that same
   repository and whose base is its default branch. This reuses the trust
   argument the workflow already accepts for `prompt_lab_ref` PR heads (a
   same-repository branch requires write access). Fork heads, closed PRs, other
   bases, malformed head SHAs, and API failures all still fail closed.
2. **Single reviewed declaration** — `src/korvid_prompt_lab/korvid_pin.py` holds
   the approved SHA, a dated provenance snapshot (including the *failing*
   `diverged` status, so a silent regression to a default-branch-only rule is
   impossible), and the bridge's Korvid import contract.
3. **Drift is now a test failure** — contract tests bind the workflow default,
   the README, and `bridge_worker._import_korvid` (derived via AST) to that one
   declaration.
4. **Live re-verification** — `scripts/verify-korvid-pin.sh` re-proves provenance
   and compatibility through `gh api` (no new dependencies), so the offline tests
   stay deterministic.

The pin stays `fc7eece2adb66a5b2a18d378bdfd7503ddbdd2ca`.

## RED → GREEN

RED (before the fix), `tests/test_korvid_pin.py`: **12 failed, 15 passed**. The
finding itself:

```
FAILED test_shipped_korvid_default_passes_the_shipped_trust_gate
E  AssertionError: the shipped korvid_ref default must clear the shipped
   provenance gate; it was rejected with ['korvid_ref is neither the tip of main
   in hellices/korvid nor an ancestor of it (compare status: diverged); only
   commits already merged/reachable from the Korvid default branch are accepted']
```

Also RED: the open-PR route (`pulls.list`, `state: open`, `base`), fork-PR
rejection, PR-API-failure closure, README SHA/PR/route drift guards, and the
verify script's existence and contents.

GREEN: `tests/test_korvid_pin.py` 27 passed; `tests/test_grounding_workflow.py`
70 passed.

## Verification

```
KORVID_SOURCE_ROOT=/Users/hwang-inhwan/workspace/kube pytest -q
→ 519 passed, 6 skipped

ruff check .            → All checks passed!
mypy src tests          → Success: no issues found in 35 source files
bash -n scripts/*.sh    → OK (both scripts)
yaml.safe_load(workflow)→ parsed OK; steps = 17; korvid_ref default = fc7eece2…
node --check <trust script body> → parses
```

Live proof through the new script (`scripts/verify-korvid-pin.sh`, exit 0):

```
provenance: compare fc7eece2…...main => diverged
provenance: compare fc7eece2…...7830543f… (PR #314) => diverged
provenance: compare fc7eece2…...525378f0… (PR #312) => ahead
provenance: PROVEN via open pull request #312
compatibility: present  … 9/9 required Korvid source paths
OK: fc7eece2… is authoritative hellices/korvid code and carries every bridge dependency.
```

## Concerns

- **The pin is on an unmerged pull request by necessity.** When PR #312 merges,
  the default-branch route will start succeeding and the pin should move to a
  `main` commit; `korvid_pin.py` must be updated then. The declaration test
  (`test_pin_records_why_the_default_branch_route_is_insufficient`) forces that
  edit to be deliberate.
- **A vouching PR head moves.** PR #312's head advanced from `bff8faff` to
  `525378f0` during this session. The gate re-derives the head live, so this is
  correct at dispatch time; only the recorded snapshot ages, which is why the
  live script exists.
- **The open-PR route is a real widening**, from "reviewed and merged" to "pushed
  by someone with write access and under review". It is bounded to non-fork heads
  of PRs targeting the default branch of the authoritative repo, and it is the
  same boundary already accepted for `prompt_lab_ref`. It cannot be narrowed
  further without making Prompt Lab unrunnable.
- `pulls.list` is capped at `per_page: 100` and is only reached when the default
  branch does not contain the ref; a repo with >100 open PRs targeting `main`
  could miss a vouching PR and fail closed (safe direction).
- Two timing/subprocess tests (`test_optimize.py::…records_how_its_evidence_was_produced`,
  `test_runner.py::…kills_a_stubborn_worker_behind_the_real_launcher`) failed once
  under full-suite load and passed in isolation and on a clean full re-run. They
  touch nothing in this change; load-sensitive pre-existing flakes.
