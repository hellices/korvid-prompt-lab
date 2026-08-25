# Bounded Optimization Campaign — Final-Review Fix Report

Branch: `feat-bounded-optimization-campaign`
Baseline reviewed: `81c444d..efd1bcb`
Reviewer verdict at entry: **Ready to merge / live canary: NO** — 7 findings.

All 7 findings are fixed in this wave. Nothing was partially landed.

---

## Finding 1 — Safe round projections were unconditionally rejected

**Symptom.** Both the workflow prior-artifact check and the `package` step hard-failed
on any path component in `{"responses", "raw", "transcripts"}`. Because
`scripts/run-optimization-campaign-step.sh` copies the entire `write_safe_evidence`
package into `next/round-evidence`, and `write_safe_evidence` *always* emits
`responses/`, every evidence-producing run aborted before upload. A changed candidate
additionally emits `before-responses/`.

**Fix.** Replaced the directory-name heuristic with an explicit safe package allowlist,
`korvid_prompt_lab.campaign_artifacts.validate_safe_round_package()`:

- top-level files must be a subset of `SAFE_ROUND_PACKAGE_FILES`
  (`round-summary.json`, `round-summary.md`, `evaluation-summary.json`,
  `optimization-summary.json`, `best-candidate.yaml`, `comparison-summary.json`,
  `before-evaluation-summary.json`);
- `SAFE_ROUND_PACKAGE_REQUIRED_FILES` must all be present;
- the only permitted directories are `responses/` and `before-responses/`, each
  containing exclusively regular `*.json` files matching
  `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json$` (no nesting);
- `before-responses/` requires the comparison projections that produce it;
- symlinks are rejected at the root, at top level and inside the projections;
- non-regular entries are rejected.

The workflow `package` step now calls `validate_safe_round_package(round_evidence)`
and the `prepare` step validates each `safe-round/<action_id>/` projection of a
downloaded prior artifact, plus a strict top-level shape
(`{"safe-campaign", "safe-round"}`) and a UUID-shaped action id.

**Tests.** New `tests/test_safe_round_package.py` (18 tests):
`TestSafeRoundAllowlist` runs the predicate against genuine `write_safe_evidence`
output (with `responses/` and a changed-candidate `before-responses/`) and against
raw artifact roots, injected `runs/`, `transcripts/`, `audit/`, `kubeconfig`,
`.kube/config`, `credentials.json`, `gepa-state/`, nested/non-JSON projections,
unexpected files and symlinks. `TestWorkflowPackagingAcceptsSafeProjections` is an
integration/process test: it runs the real `prepare` embedded Python, builds a real
safe package, runs the real `package` embedded Python as a subprocess, asserts
`responses/` and `before-responses/` survive into `safe-round/<action_id>/`, and then
re-runs the real `prepare` predicate in continuation mode against the uploaded
package to prove the run stays resumable. A second process test proves an injected
`transcripts/` directory still fails packaging.

RED: 16 failed / 2 passed. GREEN: 18 passed.

---

## Finding 2 — `CampaignScore.core_regression` was hardcoded `False`

**Fix.** `_validate_comparison_summary` now returns the derived core-regression flag.
It collects every `core: true` metric result and `_derive_core_regression()`
re-derives the expected summary `outcome` from those results
(`regressed` > `improved` > `unchanged`), raising a `ValueError` mentioning
`contradicts` when the producer's `outcome` disagrees, and rejecting
`status: unchanged` combined with a moving core metric. `load_round_outcome` threads
the flag into the ingested `CampaignScore`. `campaigns._is_strictly_better` gained an
unconditional `if candidate.core_regression: return False` guard, so a core
regression can never promote even if the aggregate rose.

**Tests.** `tests/test_campaign_artifacts.py::TestCoreRegressionDerivation` (5 tests):
derivation from core metrics, non-core regressions ignored, contradictory `outcome`
rejected, contradictory `unchanged` status rejected, and a real ingestion→advance
regression that feeds a higher-aggregate core-regressed candidate through
`load_round_outcome` + `advance_state` and asserts the champion did not change.
`_write_search_evidence` and the CLI evidence fixture were made self-consistent
(previously `outcome: improved` with no improving core metric).

RED: 4 failed / 47 passed. GREEN: 135 passed across the four campaign test modules.

---

## Finding 3 — `infrastructure_retry_limit: 1` granted zero retries

**Fix.** `advance_state` now terminates only when `new_retries > limit` (was `>=`), so
the limit is genuinely "retries allowed per action". `retries_used` is reset to `0` on
every evidence-outcome return (search, stagnation, milestone, confirm, tier
exhaustion), making the bound per attempt rather than per campaign lifetime. The
wall-clock check is preserved and still evaluated at the same point, so a system error
past the wall-clock limit terminates regardless of the remaining retry allowance.

**Tests.** `tests/test_campaign_state.py`:
`test_retry_exhaustion_terminates` (limit 1: first system error stays `RUNNING` with
`retries_used == 1` and re-plans the identical logical action — same kind, stage,
seed, tier and metric calls; second consecutive system error terminates with
`stop_reason == "infrastructure_retry_limit_exhausted"`),
`test_retry_counter_resets_after_valid_evidence`, and
`test_system_error_wall_clock_crossing_terminates_before_retry_allowance`.

---

## Finding 4 — Unrelated ARC runner pods failed the campaign

**Symptom.** A single unrelated `Succeeded`/terminating pod on the *shared* scale set
made the `cleanup` step exit non-zero, which suppressed `dispatch` and forced
`exit 70` — a healthy uploaded `RUNNING` state became unresumable because of another
run's lifecycle.

**Fix.** The step is now split into two clearly separated halves:

- **Owned capacity restoration** (`modeleval` nodepool scale-back) remains fatal and
  is the only contributor to `cleanup_status`.
- **ARC observation** (kubeconfig acquisition, `kubelogin`, `kubectl`, and the
  embedded Python pod scan) is advisory: it accumulates into a separate `arc_status`,
  emits `::warning::` for unrelated stale pods and for an inconclusive observation,
  and never affects the step's exit code.

**Tests.** `tests/test_optimization_campaign_workflow.py`:
`test_unrelated_arc_runner_pods_are_advisory_only` executes the real embedded ARC
Python as a subprocess with an unrelated `Succeeded` pod and an unrelated
terminating pod, asserting exit 0 and both names in a `::warning::`;
`test_clean_arc_observation_reports_success`; and
`test_only_owned_capacity_restoration_can_fail_cleanup`, a structural test proving
nothing in the observation half touches `cleanup_status`.

---

## Finding 5 — Cross-run lineage CAS was local-file only

**Symptom.** `write_campaign_state`'s file CAS cannot serialize two runs that each
downloaded their own copy of the same prior state, so two successors of one state
could both do expensive work and both persist.

**Fix.** A durable, same-repository GitHub artifact marker protocol:

1. `prepare` derives a deterministic lineage key from the *validated* campaign id and
   the prior state hash — `initial` when there is no prior run, otherwise
   `sha256-<64hex>` — and emits `lineage-from-key` and
   `lineage-marker-name` (`campaign-lineage-<campaign_id>-<from_key>`) as step
   outputs. Continuations must present a canonical `sha256:<64 lowercase hex>` hash.
2. `lineage-scan` (github-script) queries `listArtifactsForRepo` filtered by that exact
   name, ignoring expired artifacts and the current run's own artifacts, and fails
   closed if the marker name does not match the strict pattern.
3. `lineage-download` + `lineage-reject` run **before** `korvid-token`, `azure` and
   `attempt`. `lineage-reject` validates the downloaded marker's exact shape
   (schema version, campaign id, from key, from hash, `sha256:` to-hash, producer run
   id, 40-hex producer revisions) and then stops the run with
   "campaign lineage input was already consumed by run …". Any malformed marker also
   stops the run (fail closed).
4. After a successful evidence `upload`, `lineage-write` builds and round-trips the
   marker and `lineage-claim` uploads it (`retention-days: 90`,
   `if-no-files-found: error`) **before** `dispatch`.
5. `dispatch` now additionally requires `steps.lineage-claim.outcome == 'success'`,
   and `Enforce terminal and persistence result` fails the run when the claim did not
   succeed — a marker upload failure therefore prevents dispatch.

No token is passed in argv or printed: `github-token` is supplied through the action
inputs of `github-script`/`download-artifact` only, and no lineage step body
references `GH_TOKEN`, `github.token` or `secrets.*`.

**Tests.** New `tests/test_campaign_lineage_marker.py` (21 tests). `TestLineageScan`
executes the real `lineage-scan` JavaScript under Node with a stubbed Octokit and
proves: a duplicate prior-state hash is detected, a duplicate **initial** state is
detected, a successor with a new hash proceeds (`marker-run-id == ''`), the run's own
marker and expired markers are ignored, unrelated artifacts are ignored, and an
invalid marker name fails closed. `TestLineageReject` runs the real embedded Python
against valid markers (stop) and seven malformed markers (fail closed).
`TestLineageWorkflowStructure` pins ordering (scan/download/reject strictly before
`korvid-token`, `azure` and `attempt`; write/claim strictly between `upload` and
`dispatch`), the dispatch and terminal gating, the deterministic marker naming, and
the absence of any token exposure.

**Known trade-off.** Because the initial key is the constant `initial`, a campaign id
can only be *started* once per repository while the marker artifact lives
(90 days). This is the repository-wide serialization the review asked for; restarting
a campaign from scratch requires a new campaign id or an expired/deleted marker.

---

## Finding 6 — Tier roll could produce a RUNNING state with no affordable action

**Fix.** `_handle_tier_exhaustion` now builds the candidate next-tier state and only
returns it when `_next_tier_action_fits()` holds: strictly positive remaining
wall-clock budget, a non-`None` planned action, and
`metric_calls_used + action.metric_calls <= total_metric_call_limit`. Otherwise the
campaign terminates `NOT_CONVERGED` with
`stop_reason == "next_tier_budget_exhausted"` and the exhausted tier's `TierResult`
appended.

**Tests.** `tests/test_campaign_state.py`:
`test_tier_roll_without_metric_budget_terminates_not_converged` (24 used of a 25 limit
vs. a 12-call next action),
`test_tier_roll_with_exact_metric_budget_still_rolls` (24 + 12 == 36 limit exactly ⇒
must still roll), and
`test_tier_roll_without_wall_clock_budget_terminates_not_converged` (elapsed exactly
equal to the wall-clock limit ⇒ no remaining budget).

---

## Finding 7 — CLI loaded control manifests without validation or binding

**Fix.**

- `campaign_cli._load_control` now safe-resolves the manifest (rejecting symlinks at
  the path and anywhere in its resolved ancestry, and non-regular files), reads the
  declared `evaluation_campaign`, validates it as a bare campaign id against
  `^[a-z0-9][a-z0-9._-]{0,62}$` (so traversal and separators are rejected *before* any
  filesystem lookup), resolves it as a sibling file or under an ancestor's
  `examples/campaigns/`, and then calls the strict
  `load_optimization_campaign(control_path, load_campaign(evaluation_path))`. That
  restores every strict check to the CLI: pairwise-disjoint and exactly-covering case
  sets, canonical `sha256:<64 hex>` model digests, positive limits, unique stage seeds
  and unique tier identities.
- New `campaigns.validate_state_binding()` binds a state to its control inside the
  **pure** planning and advancement path: `campaign_id` equality, tier index within
  the declared tiers, stage index within the declared stages, and model identity. It
  is called first in `next_action` and first in `_validate_action` — before the state
  hash comparison — so identity mismatches are reported as such.
- New optional CLI bindings `--expected-manifest-sha256`,
  `--expected-prompt-lab-revision` and `--expected-korvid-revision` on `plan`,
  `advance` and `validate-evidence`, enforced by `_enforce_identity_bindings` in every
  command. `scripts/run-optimization-campaign-step.sh` forwards them from
  `CAMPAIGN_MANIFEST_SHA256`, `PROMPT_LAB_REVISION` and `KORVID_REVISION`, and the
  workflow `attempt` step now exports `CAMPAIGN_MANIFEST_SHA256` from the trusted
  identity job output.

**Tests.** `tests/test_campaign_cli.py::TestStrictControlLoading` (13 tests):
overlapping case sets, case sets that do not cover the evaluation campaign, mutable
`latest` digest, non-positive limits, duplicate stage seeds, missing evaluation
campaign, `evaluation_campaign: ../outside` traversal, symlinked control, state from a
different campaign (rejected by both `plan` and `advance`, and no state written),
out-of-range tier index, manifest identity mismatch, matching manifest identity plus
revisions accepted, and wrong expected revisions. The test helper now writes a real
sibling evaluation campaign covering exactly `case-a..case-d`.
`tests/test_optimization_campaign_script.py` adds three wrapper tests proving the
bindings actually reach the CLI (wrong manifest digest and wrong Korvid revision abort
with exit 70 before any round and produce no output; the exact digest is accepted and
the round runs).

---

## Commits

| Commit | Contents |
|---|---|
| `6868425` | All seven findings, their tests, the README update and this report |

Single coherent commit on `feat-bounded-optimization-campaign`, parent `efd1bcb`.
No finding was partially landed.

## Verification

All commands run from the worktree
`/Users/hwang-inhwan/workspace/kube-prompt-grounding/.worktrees/feat-bounded-optimization-campaign`.

| Check | Result |
|---|---|
| `uv run --frozen pytest -q` (baseline, before changes) | 881 passed, 6 skipped (245.92 s) |
| `uv run --frozen pytest -q` (final) | **950 passed, 6 skipped (240.95 s)** |
| `uv run --frozen ruff check .` | **All checks passed!** |
| `uv run --frozen mypy --python-version 3.12 src tests` | **Success: no issues found in 49 source files** |
| `bash -n` on every tracked `scripts/**/*.sh` | clean |
| Every workflow `run:` body through `bash -n` | clean |
| Every workflow embedded `python3 … <<'PY'` heredoc compiled | clean |
| `python3 -c "yaml.safe_load(...)"` on the workflow | parses; jobs `identity`, `campaign` |
| `git diff --check` | clean |

Focused suites used during TDD:
`tests/test_campaign_state.py`, `tests/test_campaign_artifacts.py`,
`tests/test_campaign_cli.py`, `tests/test_campaigns.py`,
`tests/test_optimization_campaign_workflow.py`,
`tests/test_optimization_campaign_script.py`,
`tests/test_safe_round_package.py`, `tests/test_campaign_lineage_marker.py`.

New test count: 69 tests added (881 → 950).

### Pre-existing type debt repaired

`mypy --python-version 3.12 src tests` reported **33 pre-existing errors** at
`efd1bcb` (verified by stashing this wave's changes and re-running). They were all in
test modules touched by this feature (`CampaignAction | None` passed to
`advance_state`, an over-narrow `run_step` return annotation, and an
`object`-annotated monkeypatch signature). They are fixed here so the honest claim is
now "mypy clean", not "clean except known noise".

## Self-review

- **Behaviour preserved where it should be.** The safe allowlist is strictly narrower
  than "no `responses`/`raw`/`transcripts` component" for everything except the two
  sanitized projection directories: unexpected top-level files, nested projection
  directories, non-JSON projections, non-regular entries and symlinks are all still
  rejected, and the raw evaluator artifact root is rejected outright.
- **Retry semantics.** A retry deliberately yields a *different* `action_id` (because
  `retries_used` participates in `state_hash`, which the CAS requires) while the
  logical action — kind, tier, stage, seed and metric calls — is identical; the test
  asserts exactly that rather than action id equality.
- **Tier budget boundary.** The metric test pins both sides of the boundary
  (exact fit rolls, one short terminates), and the wall-clock test uses elapsed
  *exactly* equal to the limit, which is the only reachable zero-remaining case
  because a strictly greater elapsed terminates earlier in `advance_state`.
- **ARC advisory scope.** Only the observation half was made advisory. Owned
  `modeleval` restoration failure is still fatal and still gates `dispatch` through
  `steps.cleanup.outcome`.
- **Lineage marker fail-closed.** Every validation failure in `lineage-reject` exits
  non-zero, so a corrupted or forged marker stops the run rather than silently
  allowing it. `lineage-scan` also `setFailed`s on an invalid marker name.
- **Token hygiene.** Verified by test: no lineage step body references `GH_TOKEN`,
  `github.token` or `secrets.*`, and no lineage step `env` value carries a secret.
- **Pure vs. wrapper binding.** Campaign id, tier index, stage index and model
  identity are now enforced inside `next_action`/`_validate_action` (pure), while
  manifest digest and revision expectations are explicit CLI inputs enforced in every
  command — the wrapper only supplies them.

## Residual concerns

1. **Not executed against live GitHub Actions.** The lineage protocol, the ARC
   advisory path and the packaging changes are verified by executing the real embedded
   Python/JavaScript as subprocesses with stubbed Octokit and fixture data, not by a
   live canary run. A live canary is still required before merge.
2. **`listArtifactsForRepo` name filtering.** The scan passes `name` as a query
   parameter *and* re-filters client-side (`artifact.name === markerName`), so it is
   correct whether or not the server honours the filter; only pagination cost differs.
3. **Initial-state serialization is repository-wide and durable for 90 days.** See the
   trade-off note under Finding 5.
4. **Marker artifact deletion is not defended against.** An operator with artifact
   delete permission can remove a lineage marker and re-run a consumed lineage. That
   is the same trust level already required to rewrite campaign evidence.

---

# Wave 2 — re-review fix wave

The wave 1 re-review raised four new findings. Every one is fixed at its root and
proven by a failing-first test.

| Commit | Scope |
| --- | --- |
| `7339106` | All four wave 2 findings, tests, README recovery docs, this report |

## Finding 1 (Critical) — `KORVID_AKS_MODEL` was never exported to `attempt`

**Root cause.** The wrapper and the round script need `KORVID_AKS_MODEL`, but the
`attempt` step's `env:` never set it, so the very first `korvid-campaign` call in
every run would have died on an unset variable. The existing script tests hid this
because they built the subprocess environment with `dict(os.environ)` and the
developer shell happened to export the variable.

**Fix.**
- `scripts/run-optimization-campaign-step.sh` now requires the variable in its
  required-env preamble (`: "${KORVID_AKS_MODEL:?...}"`) *before* any planning, so a
  missing model fails in milliseconds instead of after a download.
- The `identity` job publishes a new validated `model` output taken from tier 0 of
  the validated manifest, and the `attempt` step binds
  `KORVID_AKS_MODEL: ${{ needs.identity.outputs.model }}`.
- `tests/test_optimization_campaign_script.py::run_step` was rewritten to build the
  subprocess environment explicitly from a small `_INHERITED_ENV` allowlist plus the
  variables under test, with an `omit_env` parameter. No test can now pass because of
  a leaked ambient `KORVID_AKS_MODEL`.

**RED evidence.** With the explicit environment and the new tests in place, before the
wrapper/workflow change: `3 failed, 1 passed`
(`test_missing_model_env_fails_before_planning`,
`test_attempt_supplies_every_environment_the_wrapper_requires`,
`test_attempt_binds_the_validated_identity_model`).
**GREEN.** `uv run --frozen pytest -q tests/test_optimization_campaign_script.py
tests/test_optimization_campaign_workflow.py` → `33 passed`.

## Finding 2 (High) — multi-tier rollover wrote a *path* into `champion_fingerprint`

**Root cause.** `_make_fresh_tier_state` reset the champion to
`control.initial_candidate`, which is a manifest **path** (`seed.yaml`), not a
candidate fingerprint. The next `validate_state_binding` / packaging step would reject
it, so every campaign that survived tier 0 hard-failed on rollover.

**Fix — carry the real seed fingerprint in state.**
- `CampaignState` gained a required `seed_candidate_fingerprint: str` (bare 64-hex),
  validated by a new public `validate_seed_candidate_fingerprint`.
- It participates in `state_hash`, in `_serialize_state`/`_load_state`, and in
  `validate_state_binding`'s tamper check, so it cannot be edited in a downloaded
  artifact.
- `initial_state(...)` now *requires* it; the workflow `prepare` step computes it once
  as `load_candidate(initial_candidate_path).fingerprint` and passes it in. The
  continuation path re-derives it from the same manifest candidate and rejects a state
  whose stored value disagrees.
- `_make_fresh_tier_state` seeds the new tier's `champion_fingerprint`,
  `champion_score.fingerprint` and `seed_candidate_fingerprint` from that validated
  real fingerprint. The initial candidate **path** survives only as control config.
- The value is bound end to end: `prepare` emits `seed-candidate-fingerprint`,
  `attempt` passes `CAMPAIGN_SEED_FINGERPRINT`, the wrapper forwards
  `--expected-seed-fingerprint`, and `_enforce_identity_bindings` compares it. The
  `package` step re-checks it and adds the seed candidate to the champion-source set
  so a rolled tier can actually resolve its champion.
- The dead `replace(...)`/`CampaignScore` patch-up in `prepare` was deleted.

**RED evidence.** `tests/test_campaign_state.py` → `40 failed, 4 passed` before the
field existed. The dedicated end-to-end proof
`tests/test_campaign_tier_rollover.py` (2 tests: run the real `prepare` step, roll to
tier 1 through the real state machine, assert the rolled fingerprint is the canonical
`load_candidate(...).fingerprint`, then run the real `package` step) was proven RED by
temporarily restoring `control.initial_candidate` in `_make_fresh_tier_state` →
`2 failed`.
**GREEN.** `2 passed`; `tests/test_campaign_artifacts.py tests/test_campaign_cli.py` →
`78 passed`; workflow/package/script suites → `51 passed`.

## Finding 3 (High) — marker consumption was irreversible

**Root cause.** Wave 5's durable marker is claimed *after* the state upload and
*before* dispatch. Any failure in that window (cleanup, dispatch, runner loss) left the
lineage consumed with no legal continuation, permanently wedging the campaign — and a
naive re-run would redo hours of GPU work only to be refused at claim time.

**Fix — a narrow, proof-carrying recovery path.**
- The `identity` trust check now also admits a prior run whose conclusion is
  `failure`, and publishes `prior-run-conclusion`. `success` and `failure` are the only
  admissible conclusions.
- Three new steps run **before `prepare`** and therefore before the Azure login, the
  Korvid token and the `attempt` step:
  - `recovery-scan` (github-script) lists the failed run's own artifacts, applies the
    same producer-trust predicate as Finding 4, and requires **exactly one** matching
    `campaign-lineage-<id>-(initial|sha256-…)` artifact.
  - `recovery-download` fetches it with `run-id: inputs.prior_run_id`.
  - `recovery-verify` validates the package shape (a single regular
    `lineage-marker.json`, no symlink), the exact 8-key schema, `schema_version == 1`,
    the campaign id, that the artifact name matches the marker content, that
    `producer_run_id == prior_run_id`, that the revisions are 40-hex and equal the
    dispatched refs, that the transition is forward, and the decisive invariant
    **`to_state_hash == expected_state_hash`**. Only then does it emit `verified=true`.
- `prepare` receives `PRIOR_RUN_CONCLUSION` and `RECOVERY_VERIFIED` and raises
  `SystemExit` when a `failure`-conclusion prior run lacks the proof. Every other failed
  prior run therefore remains rejected exactly as before.
- Re-running the failed job itself is now caught: `lineage-scan` reports a
  `marker-scope` of `current` when the only trusted marker belongs to this run, and
  `lineage-reject` stops **before the wrapper** with a bounded recovery instruction
  naming the produced `to_state_hash` and the producer run id. The foreign-duplicate
  message names the same recovery dispatch.
- Owned cleanup failure is **not** made success-shaped: the `Enforce terminal` step
  still requires `CLEANUP_OUTCOME == success` (asserted by
  `test_owned_cleanup_failure_is_never_success_shaped`).
- `README.md` documents the operator procedure and its trust implications: a campaign
  id is single-use per lineage step for the 90-day retention window, and deleting a
  marker deliberately re-opens that step.

## Finding 4 (Medium) — `lineage-scan` trusted any repository artifact by name

**Root cause.** `listArtifactsForRepo` is repository-wide. Any workflow — including a
fork PR run or an unrelated workflow — could upload an artifact with the marker name
and permanently consume a lineage, a cheap denial of service.

**Fix.** `lineage-scan` now requires, for every candidate artifact:
- artifact-level: name match, not expired, a producing run id, and
  `workflow_run.repository_id == head_repository_id == context.payload.repository.id`
  with `head_branch == default_branch`;
- run-level (memoised `getWorkflowRun`): `path` exactly
  `.github/workflows/optimization-campaign.yml`, `event == 'workflow_dispatch'`,
  `head_branch == default_branch`, and both `repository.id` and `head_repository.id`
  equal to the authoritative repository id with a case-insensitive `full_name` match.

An untrusted artifact is `core.warning`ed and skipped — it never masks a trusted one
later in the list, and never fails the run. A `getWorkflowRun` error means untrusted,
not fatal. An incomplete repository context fails closed. Once trusted, malformed
marker **content** still fails closed in `lineage-reject`. Pagination is exercised
(`github.paginate` with `per_page: 100` and the `name` filter).

**RED evidence (findings 3 + 4 together).** With the new
`tests/test_campaign_lineage_marker.py` harness (explicit `context.payload.repository`,
paginating `github.paginate`, `getWorkflowRun` stub, `repository_id` /
`head_repository_id` / `head_branch` on artifacts) and the pre-wave-2 workflow:
`36 failed, 20 passed`.
**GREEN.** `uv run --frozen pytest -q tests/test_campaign_lineage_marker.py` →
`56 passed`, covering: foreign workflow path, fork head repository, non-default branch,
untrusted event, unresolvable run, foreign repository id, untrusted-does-not-mask-
trusted, pagination, `marker-scope` foreign/current, incomplete repository context,
recovery marker accept/reject matrix, recovery step ordering and `if:` gating, the
`prepare` recovery gate, the rerun instruction, and the cleanup terminal contract.

## Wave 2 verification

| Check | Result |
| --- | --- |
| `uv run --frozen pytest -q` | `995 passed, 6 skipped` (plus the identity-outputs assertion updated to include `prior-run-conclusion`) |
| `uv run --frozen ruff check .` | All checks passed |
| `uv run --frozen mypy --python-version 3.12 src tests` | Success: no issues found in 50 source files |
| `bash -n` on every `git ls-files '*.sh'` | clean |
| `bash -n` on every workflow `run:` body | 26 bodies clean |
| `compile()` on every embedded `<<'PY'` heredoc | 6 bodies clean |
| `git diff --check` | clean |

## Wave 2 self-review

- **Ordering.** `recovery-scan` → `recovery-download` → `recovery-verify` → `prepare`
  all precede `korvid-token`, `azure` and `attempt`; `download` still precedes
  `recovery-scan` so the artifact and the marker are checked together. Asserted by
  `test_recovery_proof_precedes_any_expensive_work`.
- **Claim/dispatch ordering is unchanged.** The marker is still claimed after the
  evidence upload and before dispatch; recovery makes that window survivable rather
  than moving it, which keeps the "no duplicate lineage" invariant intact.
- **Recovery cannot widen the duplicate window.** A recovery continuation still has to
  claim the *next* marker (`sha256-<to_state_hash>`), so a second recovery from the same
  failed run is refused at claim time exactly like any duplicate.
- **No secret exposure.** The recovery steps use `github.token` only through
  `github-token:` inputs, never in argv or logs; the token-hygiene test still passes.
- **Fail-closed everywhere.** Zero, multiple, expired or untrusted recovery markers all
  `setFailed`; every content mismatch raises `SystemExit`.

## Wave 2 residual concerns

1. **Still not executed against live GitHub Actions.** As in wave 1, the JavaScript and
   embedded Python are executed as real subprocesses with stubbed Octokit and fixture
   data. A live canary — including one deliberate post-upload failure followed by a
   recovery dispatch — is still required before merge.
2. **`actions/download-artifact` with `run-id` for the recovery marker** is exercised
   only structurally (name, path, `run-id`, `repository`, `github-token`), not against
   the live API.
3. **Marker artifact deletion remains undefended**, and now additionally means an
   operator can re-open a lineage step. This is documented in `README.md` as a trusted
   operator action; it is the same permission level that can already rewrite evidence.
