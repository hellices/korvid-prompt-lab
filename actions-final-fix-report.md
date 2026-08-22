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
