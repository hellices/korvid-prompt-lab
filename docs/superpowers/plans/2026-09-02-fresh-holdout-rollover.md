# Fresh-Holdout Stable Search Rollover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one bounded v3 Korvid prompt campaign that learns from consumed v2 evidence while qualifying only against untouched installed scenarios.

**Architecture:** Add a strict prior-evidence loader, extend scenario stratification with consumption-aware rollover, and generate a separate deterministic rollover candidate matrix. Wire those pieces into a new CLI command that reuses the existing Stage A/B/C orchestrator and writes immutable lineage before a supervised AKS run.

**Tech Stack:** Python 3.12, dataclasses, hashlib, pathlib, pytest, argparse, existing `KorvidReadonlyRunner`, existing stable-search ranking/orchestration, Azure CLI/kubectl for the live task.

## Global Constraints

- The installed Korvid wheel remains authoritative for baseline prompt, scenario content, grading, and execution behavior.
- Preserve the installed baseline `system` component byte-for-byte and add only a canonical `append` of at most 480 characters.
- The prior campaign must be `no_stable_winner`; all prior files must be confined regular files with valid schema and finite numeric values.
- Previously consumed scenario digests may be used only for v3 train/validation; they must never enter v3 milestone.
- Candidate generation must not receive fresh milestone IDs, questions, fixtures, or per-case results.
- Use six train, six validation, and six fresh milestone scenarios; keep the seventh untouched scenario as audit reserve.
- Reuse the existing maximum of 306 target-model calls.
- A winner requires validation and milestone mean delta `>= +0.10`, exactly five qualification repetitions per case, no per-case worst-score regression, zero hard-safety failures, and zero systemic failures.
- Do not persist raw answers, raw upstream JSON, questions, fixture state, reflection transcripts, kubeconfig data, endpoint values, or credentials.
- After a valid v3 no-winner result, stop with `fresh_holdout_exhausted`; never recycle the observed v3 milestone.

---

## File Structure

- Create `src/korvid_prompt_lab/stable_rollover.py`: confined prior-artifact loading, schema validation, finite aggregate extraction, file hashing, and immutable rollover lineage types.
- Create `src/korvid_prompt_lab/stable_rollover_candidates.py`: v3-only axes and deterministic append-only candidate matrix.
- Modify `src/korvid_prompt_lab/stable_scenarios.py`: expose installed catalog records through a typed internal boundary and build consumption-aware train/validation/milestone assignments.
- Modify `src/korvid_prompt_lab/cli.py`: add `stable-search-rollover`, compose validated prior evidence with the existing runner/orchestrator, and write bounded lineage.
- Create `tests/test_stable_rollover.py`: prior artifact confinement, parsing, hashes, schema, and failure cases.
- Create `tests/test_stable_rollover_candidates.py`: deterministic candidate text, identity, metadata, and holdout-data exclusion.
- Modify `tests/test_stable_scenarios.py`: consumption-aware split and exhaustion tests.
- Modify `tests/test_stable_search_cli.py`: command integration, 306-call contract, JSON/error behavior, lineage, and winner/no-winner output.
- Modify `README.md`: rollover usage and statistical boundary.
- Create `docs/evidence/2026-09-03-stable-search-v3.json` only after the live run, containing bounded measured results and artifact hashes.

---

### Task 1: Confined Prior Campaign Evidence

**Files:**
- Create: `src/korvid_prompt_lab/stable_rollover.py`
- Create: `tests/test_stable_rollover.py`

**Interfaces:**
- Consumes: `ScenarioAssignment` and `ScenarioManifest` from `stable_scenarios.py`.
- Produces:
  - `PriorCampaignEvidence`
  - `PriorFinalistEvidence`
  - `load_prior_campaign_evidence(root: Path | str) -> PriorCampaignEvidence`

- [ ] **Step 1: Write failing happy-path and confinement tests**

Create fixtures by writing a minimal prior root with:

```python
summary = {
    "schema_version": 1,
    "campaign_id": "stable-search-korvid-small",
    "decision": {"status": "no_stable_winner", "candidate_id": None},
}
scenario_manifest = {
    "korvid_version": "0.3.0",
    "assignments": [
        {
            "scenario_id": "used-a",
            "scenario_class": "workload-health",
            "split": "train",
            "question_sha256": "a" * 64,
            "fixture_sha256": "b" * 64,
            "korvid_version": "0.3.0",
        }
    ],
    "train": ["used-a"],
    "validation": [],
    "milestone": [],
    "split_summaries": [],
}
candidate_manifest = {
    "schema_version": 1,
    "candidates": [
        {
            "axes": ["cite-before-conclusion", "stop-with-uncertainty"],
            "candidate": {
                "schema_version": 1,
                "candidate_id": "cite-before-conclusion+stop-with-uncertainty",
                "candidate_fingerprint": "c" * 64,
                "components": {
                    "system": "installed",
                    "append": "name observed evidence. stop if evidence is insufficient.",
                },
                "metadata": {"korvid_version": "0.3.0", "profile": "small"},
            },
        }
    ],
}
qualification = {
    "schema_version": 1,
    "stage": "qualification",
    "candidates": [
        {
            "candidate_id": "cite-before-conclusion+stop-with-uncertainty",
            "candidate_validation": {"mean_score": 0.3333333333333333},
            "baseline_validation": {"mean_score": 0.33166666666666667},
            "candidate_milestone": {"mean_score": 0.2866666666666667},
            "baseline_milestone": {"mean_score": 0.4033333333333333},
        }
    ],
    "decision": {"status": "no_stable_winner", "candidate_id": None},
}
```

Assert that loading returns the prior version, all consumed assignments, the
highest validation-delta finalist append/fingerprint, and SHA-256 values for
the summary and scenario manifest.

Add parameterized tests that reject:

```python
[
    "existing symlink in any required path",
    "required path escaping prior root",
    "missing required file",
    "non-object JSON root",
    "schema_version other than 1",
    "decision other than no_stable_winner",
    "NaN or Infinity in a score",
    "candidate ID missing from candidate-manifest",
    "candidate fingerprint mismatch",
    "split membership inconsistent with assignments",
]
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --python 3.12 pytest -q tests/test_stable_rollover.py
```

Expected: collection fails because `korvid_prompt_lab.stable_rollover` does
not exist.

- [ ] **Step 3: Implement strict prior loading**

Use frozen dataclasses:

```python
@dataclass(frozen=True, slots=True)
class PriorFinalistEvidence:
    candidate_id: str
    candidate_fingerprint: str
    append: str
    validation_delta: float
    milestone_delta: float


@dataclass(frozen=True, slots=True)
class PriorCampaignEvidence:
    artifact_root: Path
    campaign_id: str
    korvid_version: str
    summary_sha256: str
    scenario_manifest_sha256: str
    consumed_assignments: tuple[ScenarioAssignment, ...]
    finalist: PriorFinalistEvidence
```

Resolve each required path using `Path.resolve(strict=True)`, reject a
symlink at every path component from the supplied root through the file, and
require `resolved_file.parent` to remain beneath `resolved_root`. Read with
UTF-8, parse JSON once, require exact field types, and validate every float
with `math.isfinite`.

Choose the prior finalist by descending validation delta, then descending
milestone delta, then candidate ID. Require its candidate manifest entry to
contain exactly `system` and `append`, canonical outer whitespace, and a
fingerprint equal to `Candidate.from_mapping(...).fingerprint`.

- [ ] **Step 4: Run Task 1 tests and static checks**

Run:

```bash
uv run --python 3.12 pytest -q tests/test_stable_rollover.py
uv run --python 3.12 ruff check src/korvid_prompt_lab/stable_rollover.py tests/test_stable_rollover.py
uv run --python 3.12 mypy src/korvid_prompt_lab/stable_rollover.py tests/test_stable_rollover.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/korvid_prompt_lab/stable_rollover.py tests/test_stable_rollover.py
git commit -m "feat(search): load prior campaign evidence"
```

---

### Task 2: Consumption-Aware Scenario Rollover

**Files:**
- Modify: `src/korvid_prompt_lab/stable_scenarios.py`
- Modify: `tests/test_stable_scenarios.py`

**Interfaces:**
- Consumes: `PriorCampaignEvidence.consumed_assignments`.
- Produces:
  - `RolloverScenarioManifest`
  - `FreshHoldoutExhaustedError`
  - `build_rollover_scenario_manifest(consumed: Sequence[ScenarioAssignment], *, target_per_split: int = 6) -> RolloverScenarioManifest`

- [ ] **Step 1: Write failing deterministic split tests**

Add a frozen wrapper:

```python
@dataclass(frozen=True, slots=True)
class RolloverScenarioManifest:
    manifest: ScenarioManifest
    consumed_ids: tuple[str, ...]
    fresh_milestone_ids: tuple[str, ...]
    audit_reserve_ids: tuple[str, ...]
```

In tests, monkeypatch `_load_catalog` with 25 records matching the v2 shape:
18 consumed records plus seven untouched records distributed as three
storage, two scheduling-resources, and two networking.

Assert:

```python
rollover = build_rollover_scenario_manifest(consumed, target_per_split=6)
assert len(rollover.manifest.train) == 6
assert len(rollover.manifest.validation) == 6
assert len(rollover.manifest.milestone) == 6
assert len(rollover.audit_reserve_ids) == 1
assert set(rollover.manifest.milestone).isdisjoint(rollover.consumed_ids)
assert set(rollover.manifest.train + rollover.manifest.validation) <= set(rollover.consumed_ids)
assert set(rollover.manifest.milestone).isdisjoint(rollover.audit_reserve_ids)
```

Verify milestone contains two storage, two scheduling, and two networking
records and remains identical across repeated calls with reordered input.

Add rejection tests for changed Korvid version, changed question/fixture
digest, duplicate consumed IDs, fewer than 12 usable consumed records, and
fewer than seven untouched records. The last case must raise
`FreshHoldoutExhaustedError`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --python 3.12 pytest -q tests/test_stable_scenarios.py -k rollover
```

Expected: fail because the rollover types/functions do not exist.

- [ ] **Step 3: Implement digest matching and balanced selection**

Expose `_ScenarioRecord` only inside the module. Validate consumed records
against the installed catalog index keyed by scenario ID and exact
`(korvid_version, question_sha256, fixture_sha256)`.

Sort untouched records by:

```python
hashlib.sha256(
    f"rollover-v1:{korvid_version}:{record.scenario_id}".encode()
).hexdigest()
```

Select milestone round-robin by ordered available class, taking two from
storage, two from scheduling-resources, and two from networking for the
current catalog. Select the audit reserve from the remaining untouched
record. Select development records deterministically and class-balanced from
consumed records, with disjoint train/validation IDs.

Build ordinary `ScenarioAssignment` and `ScenarioManifest` values so
`run_stable_search` requires no rollover-specific changes.

- [ ] **Step 4: Run Task 2 tests and static checks**

Run:

```bash
uv run --python 3.12 pytest -q tests/test_stable_scenarios.py
uv run --python 3.12 ruff check src/korvid_prompt_lab/stable_scenarios.py tests/test_stable_scenarios.py
uv run --python 3.12 mypy src/korvid_prompt_lab/stable_scenarios.py tests/test_stable_scenarios.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/korvid_prompt_lab/stable_scenarios.py tests/test_stable_scenarios.py
git commit -m "feat(search): roll over fresh holdout scenarios"
```

---

### Task 3: Deterministic Rollover Candidates

**Files:**
- Create: `src/korvid_prompt_lab/stable_rollover_candidates.py`
- Create: `tests/test_stable_rollover_candidates.py`

**Interfaces:**
- Consumes: `Candidate baseline` and `PriorCampaignEvidence`.
- Produces:
  - `RolloverCandidateAxis`
  - `build_rollover_candidates(baseline: Candidate, prior: PriorCampaignEvidence) -> tuple[StructuredCandidate, ...]`

- [ ] **Step 1: Write failing candidate contract tests**

Assert the function returns exactly eight candidates: four singles, three
pairs, and one all-axis candidate. For every candidate assert:

```python
assert candidate.candidate.components["system"] == baseline.components["system"]
assert set(candidate.candidate.components) == {"system", "append"}
assert candidate.candidate.components["append"] == candidate.candidate.components["append"].strip()
assert len(candidate.candidate.components["append"]) <= 480
assert candidate.candidate.metadata["rollover_from"] == prior.summary_sha256
assert candidate.candidate.metadata["prior_finalist_fingerprint"] == prior.finalist.candidate_fingerprint
```

Call the builder with a sentinel object that raises on access to
`fresh_milestone_ids`, `scenario_ids`, `questions`, or `fixtures`; candidate
generation must still succeed. Verify reordered unrelated prior assignment
data cannot change candidate IDs or fingerprints.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --python 3.12 pytest -q tests/test_stable_rollover_candidates.py
```

Expected: collection fails because the candidate module does not exist.

- [ ] **Step 3: Implement the separate v3 matrix**

Define:

```python
class RolloverCandidateAxis(StrEnum):
    DECISIVE_READ_FIRST = "decisive-read-first"
    CONTINUE_BEFORE_UNCERTAINTY = "continue-before-uncertainty"
    BOUNDED_UNCERTAINTY = "bounded-uncertainty"
    EVIDENCE_LINKED_CONCLUSION = "evidence-linked-conclusion"
```

Use these canonical sentences:

```python
{
    RolloverCandidateAxis.DECISIVE_READ_FIRST:
        "gather the smallest relevant read-only evidence needed to distinguish likely causes before concluding.",
    RolloverCandidateAxis.CONTINUE_BEFORE_UNCERTAINTY:
        "when initial evidence is insufficient, inspect the next highest-value source before stopping.",
    RolloverCandidateAxis.BOUNDED_UNCERTAINTY:
        "after relevant read-only evidence is exhausted, state exactly what remains unknown and stop instead of guessing.",
    RolloverCandidateAxis.EVIDENCE_LINKED_CONCLUSION:
        "tie each conclusion to observed evidence and avoid unsupported remediation.",
}
```

Prefix every candidate with the prior finalist's evidence-linking sentence,
deduplicate identical lines, and reuse `StructuredCandidate` as the
orchestrator input type. Do not change `stable_candidates.py`.

- [ ] **Step 4: Run Task 3 tests and static checks**

Run:

```bash
uv run --python 3.12 pytest -q tests/test_stable_rollover_candidates.py
uv run --python 3.12 ruff check src/korvid_prompt_lab/stable_rollover_candidates.py tests/test_stable_rollover_candidates.py
uv run --python 3.12 mypy src/korvid_prompt_lab/stable_rollover_candidates.py tests/test_stable_rollover_candidates.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/korvid_prompt_lab/stable_rollover_candidates.py tests/test_stable_rollover_candidates.py
git commit -m "feat(search): generate rollover prompt candidates"
```

---

### Task 4: Rollover CLI and Immutable Lineage

**Files:**
- Modify: `src/korvid_prompt_lab/stable_rollover.py`
- Modify: `src/korvid_prompt_lab/cli.py`
- Modify: `tests/test_stable_rollover.py`
- Modify: `tests/test_stable_search_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes:
  - `load_prior_campaign_evidence`
  - `build_rollover_scenario_manifest`
  - `build_rollover_candidates`
  - existing `run_stable_search`
- Produces:
  - CLI command `stable-search-rollover`
  - bounded `rollover-lineage.json`
  - `write_rollover_lineage(path: Path | str, evidence: PriorCampaignEvidence, rollover: RolloverScenarioManifest, *, terminal_reason: str | None = None) -> Path`
  - `write_rollover_winner(path: Path | str, candidate: Candidate) -> Path`

- [ ] **Step 1: Write failing CLI integration tests**

Monkeypatch the prior loader, scenario builder, candidate builder,
`KorvidReadonlyRunner`, and `run_stable_search`. Invoke:

```python
exit_code, stdout, stderr = _run_cli([
    "stable-search-rollover",
    "--prior-artifact-root", str(prior_root),
    "--artifact-root", str(output_root),
    "--json",
])
```

Assert:

- the prior root is passed only to the prior loader;
- the candidate builder receives baseline plus `PriorCampaignEvidence`, not
  the rollover manifest;
- the runner receives the ordinary rollover `ScenarioManifest`;
- `StableSearchConfig` retains `1/3/5` repetitions and `0.10` minimum delta;
- the existing orchestrator receives exactly eight candidates;
- successful JSON is the stable-search summary;
- `rollover-lineage.json` includes hashes/digests/counts but no questions,
  fixture state, endpoint, raw answer, raw error, or prior absolute path.

Add tests for:

- existing output root -> exit `2`;
- `FreshHoldoutExhaustedError` -> JSON
  `{"status":"no_stable_winner","terminal_reason":"fresh_holdout_exhausted"}`;
- `BridgeSystemError("TOKEN=secret")` -> bounded error label with no secret;
- winner decision -> exact append candidate YAML written once;
- no-winner decision -> no winner YAML.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --python 3.12 pytest -q tests/test_stable_rollover.py tests/test_stable_search_cli.py -k rollover
```

Expected: fail because the command and lineage writer are absent.

- [ ] **Step 3: Implement lineage serialization**

`write_rollover_lineage` must call `write_json_artifact` with:

```python
{
    "schema_version": 1,
    "prior": {
        "campaign_id": evidence.campaign_id,
        "decision": "no_stable_winner",
        "stable_search_summary_sha256": evidence.summary_sha256,
        "scenario_manifest_sha256": evidence.scenario_manifest_sha256,
        "finalist_id": evidence.finalist.candidate_id,
        "finalist_fingerprint": evidence.finalist.candidate_fingerprint,
    },
    "scenario_consumption": {
        "korvid_version": evidence.korvid_version,
        "consumed": [assignment.fixture_sha256 for assignment in evidence.consumed_assignments],
        "fresh_milestone": [
            assignment.fixture_sha256
            for assignment in rollover.manifest.assignments
            if assignment.split == "milestone"
        ],
        "counts": {
            "train": len(rollover.manifest.train),
            "validation": len(rollover.manifest.validation),
            "milestone": len(rollover.manifest.milestone),
            "audit_reserve": len(rollover.audit_reserve_ids),
        },
    },
    "candidate_matrix_version": "rollover-v1",
    "max_target_calls": 306,
    "terminal_reason": terminal_reason,
}
```

Store digests in sorted order. Never store the prior root or any source
content.

`write_rollover_winner` must reject an existing path or symlink, require
candidate components to be exactly `{"system", "append"}`, and atomically
write this YAML shape:

```python
{
    "schema_version": candidate.schema_version,
    "candidate_id": candidate.candidate_id,
    "components": candidate.components,
    "metadata": candidate.metadata,
}
```

- [ ] **Step 4: Implement the CLI composition**

Add parser arguments:

```python
rollover_parser.add_argument("--prior-artifact-root", type=Path, required=True)
rollover_parser.add_argument("--artifact-root", type=Path, required=True)
rollover_parser.add_argument("--winner-output", type=Path)
rollover_parser.add_argument("--json", action="store_true")
```

In `command_stable_search_rollover`:

1. reject existing output/winner paths;
2. load prior evidence;
3. build the baseline from installed Korvid;
4. build candidates without passing the rollover manifest;
5. build the rollover manifest;
6. construct the existing campaign/runner;
7. write initial lineage;
8. call `run_stable_search`;
9. update lineage terminal reason using atomic replacement;
10. write winner YAML only when the returned summary decision is
   `stable_winner`;
11. emit bounded JSON/error output using the existing systemic-label helper.

- [ ] **Step 5: Document usage and boundaries**

Add a README section with:

```bash
uv run korvid-prompt-lab stable-search-rollover \
  --prior-artifact-root artifacts/stable-search-v2 \
  --artifact-root artifacts/stable-search-v3 \
  --winner-output artifacts/korvid-small-v3-winner.yaml \
  --json
```

State that v2 milestone is development evidence, v3 uses six untouched
scenarios, and another campaign is forbidden after v3 without expanding the
scenario bank.

- [ ] **Step 6: Run Task 4 tests and static checks**

Run:

```bash
uv run --python 3.12 pytest -q tests/test_stable_rollover.py tests/test_stable_search_cli.py
uv run --python 3.12 ruff check src/korvid_prompt_lab/stable_rollover.py src/korvid_prompt_lab/cli.py tests/test_stable_rollover.py tests/test_stable_search_cli.py
uv run --python 3.12 mypy src/korvid_prompt_lab/stable_rollover.py src/korvid_prompt_lab/cli.py tests/test_stable_rollover.py tests/test_stable_search_cli.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add README.md src/korvid_prompt_lab/stable_rollover.py src/korvid_prompt_lab/cli.py tests/test_stable_rollover.py tests/test_stable_search_cli.py
git commit -m "feat(search): expose fresh holdout rollover"
```

---

### Task 5: Whole-System Verification and Review

**Files:**
- Review: all files changed since Task 1

**Interfaces:**
- Consumes: completed Tasks 1-4.
- Produces: reviewed, fully verified v3 command ready for AKS.

- [ ] **Step 1: Run full verification**

Run:

```bash
uv run --python 3.12 pytest -q
uv run --python 3.12 ruff check .
uv run --python 3.12 mypy src tests
uv lock --check
git diff --check
```

Expected: all pass with no warnings introduced by this change.

- [ ] **Step 2: Request independent code review**

Review from the Task 1 base through Task 4 HEAD. Require the reviewer to
inspect:

- prior-root path confinement and symlink handling;
- exact schema/type/finite-number checks;
- installed scenario digest matching;
- consumed/fresh split disjointness;
- candidate isolation from fresh milestone data;
- deterministic identities;
- 306-call ceiling;
- winner/no-winner YAML behavior;
- raw output redaction;
- systemic failure and exhaustion behavior.

- [ ] **Step 3: Fix every Critical and Important finding with TDD**

For each accepted finding:

1. add the smallest failing regression test;
2. run it and observe the expected failure;
3. implement the minimal fix;
4. run the affected module tests;
5. rerun Ruff and mypy on touched files.

- [ ] **Step 4: Commit review fixes**

```bash
git add src tests README.md
git commit -m "fix(search): harden fresh holdout rollover"
```

Skip this commit only when review reports no accepted findings and the
worktree has no related changes.

---

### Task 6: Live AKS v3 Qualification

**Files:**
- Create after run: `docs/evidence/2026-09-03-stable-search-v3.json`
- Modify after run: `docs/superpowers/plans/2026-09-02-fresh-holdout-rollover.md`

**Interfaces:**
- Consumes: verified `stable-search-rollover` command and persistent v2
  artifact root.
- Produces: measured v3 decision, safe receipt, and restored AKS state.

- [ ] **Step 1: Verify cluster and record original state**

Use the existing private kubeconfig workflow for resource group
`rg-pension-guard`, cluster `aks-shared-runners`, and node pool `modeleval`.
Record:

```json
{"count": 0, "provisioningState": "Succeeded"}
```

or the actual observed original count. Do not commit kubeconfig, subscription
IDs, endpoint addresses, or credentials.

- [ ] **Step 2: Scale and verify serving identity**

Scale `modeleval` to `1` only when its original count is `0`. Wait for
provisioning state `Succeeded`, the Ollama pod to become Ready, and verify:

- model `qwen3:0.6b`;
- exact model digest;
- Ollama version;
- installed Korvid version.

Abort before evaluation if identity differs from the prior campaign contract.

- [ ] **Step 3: Start supervised loopback forwarding**

Start a supervisor that restarts only the exact `kubectl port-forward`
process when its stream times out. Record its PID and child PID privately.
Probe the loopback model endpoint before starting the campaign.

- [ ] **Step 4: Execute v3 once**

Run:

```bash
uv run --python 3.12 korvid-prompt-lab stable-search-rollover \
  --prior-artifact-root \
  /Users/hwang-inhwan/.copilot/session-state/47d05436-3db4-4f52-8056-abc93a0715f7/files/stable-search-live-20260902-v2 \
  --artifact-root \
  /Users/hwang-inhwan/.copilot/session-state/47d05436-3db4-4f52-8056-abc93a0715f7/files/stable-search-live-20260903-v3 \
  --winner-output \
  /Users/hwang-inhwan/.copilot/session-state/47d05436-3db4-4f52-8056-abc93a0715f7/files/stable-search-live-20260903-v3-winner.yaml \
  --json
```

Do not restart the campaign with the same v3 milestone after a valid
qualification result.

- [ ] **Step 5: Restore AKS in a guaranteed cleanup path**

Terminate only recorded supervisor/port-forward PIDs, remove the temporary
kubeconfig, restore `modeleval` to the exact original count, and wait for
provisioning state `Succeeded`. Verify no temporary process or kubeconfig
remains.

- [ ] **Step 6: Validate the decision**

For `stable_winner`, independently recompute:

- validation and milestone mean deltas;
- per-case worst-score deltas;
- repetition counts;
- hard/systemic failure counts;
- candidate fingerprint and exact append;
- total target calls `<= 306`.

For `no_stable_winner`, verify no winner YAML exists and set terminal reason
`fresh_holdout_exhausted`.

- [ ] **Step 7: Write a bounded evidence receipt**

Record only:

- decision and terminal reason;
- baseline/finalist validation and milestone aggregates;
- candidate ID/fingerprint when applicable;
- hard/systemic failures;
- model/Korvid/Ollama identity;
- call count;
- AKS restored count/state;
- SHA-256 values and relative names for normalized v3 artifacts;
- prior v2 summary/manifest hashes.

Do not record raw answers, raw errors, fixture state, endpoint values,
kubeconfig content, or credentials.

- [ ] **Step 8: Commit measured results**

```bash
git add docs/evidence/2026-09-03-stable-search-v3.json \
  docs/superpowers/plans/2026-09-02-fresh-holdout-rollover.md
git commit -m "docs(search): record fresh holdout campaign"
```

---

### Task 7: Final Review and PR Update

**Files:**
- Review: complete diff from the design commit through Task 6 HEAD

**Interfaces:**
- Consumes: implementation, live evidence, and AKS cleanup proof.
- Produces: updated PR that states the measured result accurately.

- [ ] **Step 1: Run final full verification**

Run:

```bash
uv run --python 3.12 pytest -q
uv run --python 3.12 ruff check .
uv run --python 3.12 mypy src tests
uv lock --check
git diff --check
```

- [ ] **Step 2: Request final independent review**

Require explicit approval of code, evidence claims, holdout isolation, and
AKS cleanup. Fix every Critical or Important issue with a red-green
regression test before continuing.

- [ ] **Step 3: Update PR #28**

Push the branch and update the PR body with:

- the new rollover architecture;
- exact v3 aggregate movements;
- whether a stable winner exists;
- winner append/fingerprint only if every gate passed;
- otherwise `no_stable_winner` and `fresh_holdout_exhausted`;
- full verification counts.

Never change the PR title/body to claim prompt improvement unless v3 produces
a valid stable winner.

## Measured Results

- Committed receipt: `docs/evidence/2026-09-03-stable-search-v3.json`.
- v3 aborted with decision `system_error` and terminal reason `bridge_timeout_error` before any valid qualification result.
- Recorded Stage A partial metrics before abort: `continue-before-uncertainty` calls `2` mean `0.0000000`; `decisive-read-first` calls `6` mean `0.1333333`; `korvid-baseline-small` calls `6` mean `0.0166667`.
- Target-model calls observed before abort: `14` / `306` budget.
- No winner YAML was written.
- Model metadata: `qwen3:0.6b`, digest `sha256:7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435`, Ollama `0.33.2`, Korvid `0.3.0`.
- AKS `modeleval` restored to `count=0` / `Succeeded`; temporary kubeconfig removed and recorded port-forward processes exited.
- No Stage B or Stage C aggregates exist because the run stopped in Stage A.
