# Actions Final Fix Report

## Status

DONE — both verified whole-branch review findings closed (TDD: RED → GREEN).

## Files modified

- `.github/workflows/grounding-round.yml`
- `src/korvid_prompt_lab/rounds.py`
- `tests/test_grounding_workflow.py`
- `tests/test_rounds.py`
- `README.md`

## Finding 1 — trust boundary before checkout, app token, and Azure login

### Root cause

The dispatch validation step only proved that `prompt_lab_ref` had the *shape* of
a 40-hex SHA. `actions/checkout` can fetch any commit the repository can reach,
including the head of a **fork** pull request through `refs/pull/<n>/head`, so a
dispatcher could point the round at unreviewed third-party code, which then ran
beside the Korvid GitHub App token and the Azure OIDC session.

### Fix

A new step 2, `Verify the Prompt Lab commit is trusted`
(`actions/github-script`, pinned `60a0d83…` / v7.0.1), runs **after** input
validation and **before** the first checkout, the app token, and `azure/login`.
It uses only the job's own read-only `GITHUB_TOKEN` (`github-token: ${{ github.token }}`)
and reads every value from `env:` (`PROMPT_LAB_REF`, `PR_NUMBER`,
`EXPECTED_REPOSITORY`, `DEFAULT_BRANCH`) — no `${{ }}` appears in any script or
shell body.

| Dispatch | Accepted | Rejected |
| --- | --- | --- |
| `pr_number` supplied | `pulls.get` resolves a PR **in this repository**, its `head.repo.full_name` equals `github.repository`, and `head.sha` equals the exact 40-hex `prompt_lab_ref` | not a PR here, fork head (including a deleted fork whose `head.repo` is null), or a ref that is not that PR's head |
| `pr_number` blank | `compareCommitsWithBasehead(<ref>...<default_branch>)` returns `identical` or `ahead` — the ref is the default-branch tip or an ancestor of it | `behind`/`diverged` (unmerged), or a commit the repository cannot resolve |

Same-repository PR evaluation is preserved (a same-repository branch already
requires write access). A defensive fallback to `repos.compareCommits` keeps the
check fail-closed on octokit builds without the `basehead` endpoint.

### Tests (structural + executable)

Structural: step ordering before every checkout / `create-github-app-token` /
`azure/login`; no `if:`/`continue-on-error`; exact `env:` bindings; `process.env`
reads; 40-hex re-validation; `github-token` is the job token; `node --check`.

Executable (the workflow's own script body run under `node` against a scripted
GitHub API): same-repo PR accepted, fork PR rejected, deleted-fork head
rejected, head-SHA mismatch rejected, non-PR number rejected, default-branch
ancestor accepted (`ahead` and `identical`), unmerged SHA without `pr_number`
rejected (`diverged` and `behind`), unknown commit rejected, non-SHA ref and
malformed `pr_number` rejected without any API call, and the legacy
`compareCommits` fallback path.

## Finding 2 — summary/result contract

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
- `prompt_lab_ref` / `pr_number` input rows describe the provenance requirement.
- Job Summary row lists per-model scores, elapsed duration, artifact names, and
  the shell-quoted reproduction command.
- Artifact name corrected from `grounding-round-<run-id>` to the actual
  **`safe-evidence`**, and `artifact_refs` vs `evaluation_artifact_refs`
  explained.

## Verification commands and results

```
KORVID_SOURCE_ROOT=…/feat-307-small-operator-foundation uv run --python 3.12 \
  pytest -q tests/test_grounding_workflow.py tests/test_rounds.py
→ 88 passed in 3.82s   (RED first: 16 then 14 failures before implementation)

KORVID_SOURCE_ROOT=…/feat-307-small-operator-foundation uv run --python 3.12 pytest -q
→ 489 passed in 88.97s

uv run --python 3.12 ruff check .
→ All checks passed!

uv run --python 3.12 mypy --python-version 3.12 src tests
→ Success: no issues found in 33 source files

bash -n scripts/run-grounding-round.sh
→ (no output — syntax OK)

python -c "yaml.safe_load(open('.github/workflows/grounding-round.yml'))"
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
  failed once during a full-suite run and then passed on a full re-run (489
  passed) and 3/3 in isolation. It signals a process group and races the shell's
  trap, so it is load-sensitive. Nothing in this change touches
  `scripts/run-grounding-round.sh` or that test; it is a pre-existing flake.
- The trust check depends on the GitHub API being reachable from the runner; an
  API outage fails the round closed (no checkout, no credentials), which is the
  intended direction.
- `korvid_ref` provenance is unchanged: it is a pinned SHA in a separate
  repository fetched with a read-only, single-repository app token.
