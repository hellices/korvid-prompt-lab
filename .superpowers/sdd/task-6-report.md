# Task 6 Report

## Status

READY — implemented and locally verified. Live dispatch remains Task 7.

## Files

- `.github/workflows/optimization-campaign.yml`
- `tests/test_optimization_campaign_workflow.py`
- `README.md`
- `.superpowers/sdd/task-6-report.md`

## RED / GREEN

- RED: `uv run --python 3.12 pytest tests/test_optimization_campaign_workflow.py -q`
  - `10 failed`: the workflow did not exist.
- GREEN: the same command passed after adding exact trigger, permissions, trust,
  artifact, cleanup, and dispatch structure.
- Regression RED: the continuation preflight test exposed that downloading into
  `CAMPAIGN_ROOT/prior` creates the parent before state preparation.
- Regression GREEN: preparation now permits that fixed parent and rejects
  pre-existing state/candidate files.
- Regression RED: upstream `upload-artifact@v4.6.2` validation proved that the
  canonical `sha256:<hex>` state hash contains an invalid artifact-name colon.
- Regression GREEN: artifact names use the reversible `sha256-<hex>` form while
  state content and continuation inputs retain exact canonical hashes.
- RED: the README contract failed because bounded campaign semantics were absent.
- GREEN: the documented canary, statuses, budgets, stops, tier isolation, and
  approval gate all pass.
- Regression RED: review found a successor could inspect its predecessor before
  the predecessor had reached a successful conclusion.
- Regression GREEN: repository-wide dispatch serialization now holds the
  successor until the predecessor finishes; the expensive job also retains the
  immutable validated campaign-ID concurrency group.

## Invariants

### Permissions and trust

- Workflow permissions are exactly `actions: write`, `contents: read`, and
  `id-token: write`.
- Input shapes and continuation half-pairs are rejected before checkout.
- Prompt Lab and Korvid SHAs are proven against authoritative history before
  privileged credentials; exact-SHA checkouts disable persisted credentials.
- Prior runs must be positive safe integers, successful `workflow_dispatch` runs
  of this workflow, on this repository's default branch.
- The protected `aks-grounding` environment remains on the expensive job, and
  Korvid import preflight precedes Azure OIDC and reflection credentials.

### Idempotency and artifacts

- Validated manifest content produces the campaign ID and SHA-256 identity used
  by concurrency and continuation verification.
- Continuations download one exact artifact by repository, run ID, campaign ID,
  and state-hash name.
- State hash, embedded hash, campaign ID, revisions, manifest identity, package
  allowlist, symlinks, and champion candidate fingerprint are checked before the
  wrapper plans.
- One wrapper invocation performs one action and one CAS advance.
- Upload contains only `safe-campaign/` and `safe-round/`; raw evidence paths are
  rejected. The exact generated `campaign-summary.md` is appended.

### Dispatch and publication

- Dispatch uses `GH_TOKEN` from `github.token`; no token appears in argv or the
  shell body and no GitHub App private key is used for dispatch.
- Dispatch occurs only for lowercase controller status `running`, after upload
  and cleanup success, and passes the same manifest/revisions plus current run
  ID and exact new state hash.
- `qualified`, `not_converged`, and `system_error` cannot dispatch.
- There is no publication step or write permission.

### Cleanup

- `if: always()` restores `modeleval` only when this run observed an original
  count of zero; pre-owned capacity is untouched.
- ARC verification reads only the dedicated scale set, allows the active runner,
  fails on stale terminal/deleting runner pods, and performs no delete/kill.

## Exact verification commands

```text
uv run --python 3.12 pytest tests/test_optimization_campaign_workflow.py tests/test_grounding_workflow.py tests/test_optimization_campaign_script.py -q
# 94 passed

uv run --python 3.12 python - <<'PY'
# Parsed workflow, bash -n checked every run block, compiled four embedded Python blocks.
PY
# workflow syntax valid: shell blocks and 4 embedded Python blocks

git diff --check
# clean
```

## Self-review

- Traced initial and continuation data flow from immutable input through trusted
  checkout, manifest resolution, state preparation, one wrapper call, safe
  packaging, upload, cleanup, and dispatch.
- Checked terminal exit behavior: qualification succeeds without continuation;
  non-convergence returns 1; system/persistence failures return 70.
- Checked queued continuation timing and added whole-run serialization so the
  trusted prior-run success check cannot race its predecessor.
- Confirmed all third-party actions are exact 40-hex pins and tests parse YAML
  jobs, steps, `with`, `env`, ordering, conditions, permissions, and paths
  structurally rather than accepting a decoy substring.

## Concerns

- GitHub forbids `:` in artifact names, so the name uses `sha256-<hex>` while
  state JSON and workflow inputs keep canonical `sha256:<hex>`.
- Whole-run dispatch serialization is intentionally repository-wide to close
  predecessor-completion races; the expensive job is additionally serialized by
  validated campaign ID.
- No live workflow was dispatched in Task 6. Task 7 must validate environment
  approval, external APIs, artifact transfer, ARC observation, and self-dispatch.
