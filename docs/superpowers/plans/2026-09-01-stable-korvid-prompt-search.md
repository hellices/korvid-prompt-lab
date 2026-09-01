# Stable Korvid Prompt Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a bounded structured-first campaign that either produces a repeatedly verified Korvid `small` prompt append or records `no_stable_winner`.

**Architecture:** Add focused modules for deterministic append candidates, installed-scenario stratification, staged ranking/qualification, and immutable search artifacts. Reuse the merged `KorvidReadonlyRunner` for every live score and keep the optional DSPy proposer behind the structured validation gate.

**Tech Stack:** Python 3.12, existing `korvid[agent]>=0.3`, DSPy/GEPA, PyYAML, pytest, Ruff, mypy, existing AKS port-forward tooling.

## Global Constraints

- The installed Korvid wheel remains the source of baseline prompt, scenarios, tool arm, grading, and execution behavior.
- The target model is AKS-hosted `qwen3:0.6b`.
- Candidates may add only a canonical `append`; they must not replace the installed system prompt.
- A winner needs validation and milestone mean delta `>= 0.10`, five repetitions per case, no worst-case regression, and zero safety/systemic failures.
- Raw answers, raw Korvid JSON, fixture state, kubeconfig, credentials, endpoints, and reflection transcripts must not persist.
- Default live evaluation maximum is 306 target-model runs.

---

### Task 1: Deterministic Candidate Matrix

**Files:**
- Create: `src/korvid_prompt_lab/stable_candidates.py`
- Create: `tests/test_stable_candidates.py`

**Interfaces:**
- Consumes: `Candidate` from `korvid_prompt_lab.contracts`.
- Produces: `CandidateAxis`, `StructuredCandidate`, and `build_structured_candidates(baseline: Candidate) -> tuple[StructuredCandidate, ...]`.

- [ ] **Step 1: Write failing matrix tests**

```python
def test_structured_matrix_has_eight_unique_append_candidates() -> None:
    candidates = build_structured_candidates(_baseline())
    assert len(candidates) == 8
    assert len({item.candidate.fingerprint for item in candidates}) == 8
    assert all(set(item.candidate.components) == {"system", "append"} for item in candidates)
    assert all(len(item.candidate.components["append"]) <= 480 for item in candidates)


def test_matrix_preserves_exact_baseline_system_prompt() -> None:
    baseline = _baseline(system="  exact installed prompt  ")
    assert {
        item.candidate.components["system"] for item in build_structured_candidates(baseline)
    } == {"  exact installed prompt  "}
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run --python 3.12 pytest -q tests/test_stable_candidates.py`

Expected: import failure for `korvid_prompt_lab.stable_candidates`.

- [ ] **Step 3: Implement immutable axes and matrix**

```python
class CandidateAxis(StrEnum):
    EVIDENCE_FIRST = "evidence-first"
    ONE_TOOL_AT_A_TIME = "one-tool-at-a-time"
    CITE_BEFORE_CONCLUSION = "cite-before-conclusion"
    STOP_WITH_UNCERTAINTY = "stop-with-uncertainty"


@dataclass(frozen=True, slots=True)
class StructuredCandidate:
    axes: tuple[CandidateAxis, ...]
    candidate: Candidate


_MATRIX = (
    (CandidateAxis.EVIDENCE_FIRST,),
    (CandidateAxis.ONE_TOOL_AT_A_TIME,),
    (CandidateAxis.CITE_BEFORE_CONCLUSION,),
    (CandidateAxis.STOP_WITH_UNCERTAINTY,),
    (CandidateAxis.EVIDENCE_FIRST, CandidateAxis.ONE_TOOL_AT_A_TIME),
    (CandidateAxis.EVIDENCE_FIRST, CandidateAxis.CITE_BEFORE_CONCLUSION),
    (CandidateAxis.CITE_BEFORE_CONCLUSION, CandidateAxis.STOP_WITH_UNCERTAINTY),
    tuple(CandidateAxis),
)
```

Render each axis from a fixed concise sentence, join with newlines, reject outer whitespace and lengths above 480, and derive deterministic IDs from axis values.

- [ ] **Step 4: Run tests and confirm GREEN**

Run: `uv run --python 3.12 pytest -q tests/test_stable_candidates.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/korvid_prompt_lab/stable_candidates.py tests/test_stable_candidates.py
git commit -m "feat(search): add structured prompt candidates"
```

### Task 2: Installed Scenario Stratification

**Files:**
- Create: `src/korvid_prompt_lab/stable_scenarios.py`
- Create: `tests/test_stable_scenarios.py`

**Interfaces:**
- Consumes: `bundled_scenarios_dir()` and `load_scenario()` from installed Korvid.
- Produces: `ScenarioManifest`, `ScenarioAssignment`, and `build_scenario_manifest(target_per_split: int = 6) -> ScenarioManifest`.

- [ ] **Step 1: Write failing split tests**

```python
def test_manifest_builds_disjoint_stratified_splits() -> None:
    manifest = build_scenario_manifest()
    train = set(manifest.train)
    validation = set(manifest.validation)
    milestone = set(manifest.milestone)
    assert len(train) == len(validation) == len(milestone) == 6
    assert not train & validation
    assert not train & milestone
    assert not validation & milestone
    assert all(len(split.classes) >= 2 for split in manifest.split_summaries)


def test_manifest_is_stable_for_the_same_installed_catalog() -> None:
    assert build_scenario_manifest() == build_scenario_manifest()
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run --python 3.12 pytest -q tests/test_stable_scenarios.py`

Expected: module import failure.

- [ ] **Step 3: Implement catalog classification and stable assignment**

Use scenario ID prefixes and authored fields to map into a closed vocabulary:

```python
ScenarioClass = Literal[
    "workload-health",
    "image-config",
    "scheduling-resources",
    "networking",
    "storage",
    "healthy-control",
]
```

Sort each class by `sha256(f"{korvid_version}:{scenario.id}")`, then round-robin into train, validation, and milestone. Reduce all split sizes together when fewer than eighteen eligible scenarios exist; reject fewer than twelve.

Persist only scenario ID, class, question SHA-256, fixture SHA-256, and installed Korvid version.

- [ ] **Step 4: Run tests and confirm GREEN**

Run: `uv run --python 3.12 pytest -q tests/test_stable_scenarios.py`

Expected: all tests pass against the installed 25-scenario catalog.

- [ ] **Step 5: Commit**

```bash
git add src/korvid_prompt_lab/stable_scenarios.py tests/test_stable_scenarios.py
git commit -m "feat(search): stratify installed Korvid scenarios"
```

### Task 3: Staged Ranking and Qualification

**Files:**
- Create: `src/korvid_prompt_lab/stable_ranking.py`
- Create: `tests/test_stable_ranking.py`

**Interfaces:**
- Consumes: normalized per-run score/status/grade/journal records.
- Produces: `CandidateMeasurement`, `StageDecision`, `rank_screening(...)`, `select_finalists(...)`, and `qualify_winner(...)`.

- [ ] **Step 1: Write failing ranking tests**

```python
def test_screening_rejects_safety_and_systemic_failures() -> None:
    decisions = rank_screening(_baseline(), [_safe_gain(), _unsafe_gain(), _systemic_gain()])
    assert [item.candidate_id for item in decisions.survivors] == ["safe-gain"]


def test_qualification_requires_both_split_deltas() -> None:
    decision = qualify_winner(
        baseline_validation=_measurement(mean=0.40, repetitions=5),
        candidate_validation=_measurement(mean=0.52, repetitions=5),
        baseline_milestone=_measurement(mean=0.50, repetitions=5),
        candidate_milestone=_measurement(mean=0.58, repetitions=5),
    )
    assert decision.status == "no_stable_winner"
    assert "milestone_delta_below_0_10" in decision.reasons
```

Also cover worst-case regression, variance tie-break, pass-at-3, missing repetitions, and exact `0.10` boundary.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run --python 3.12 pytest -q tests/test_stable_ranking.py`

Expected: module import failure.

- [ ] **Step 3: Implement pure ranking functions**

```python
@dataclass(frozen=True, slots=True)
class CandidateMeasurement:
    candidate_id: str
    split: str
    mean_score: float
    score_variance: float
    worst_case_mean: float
    pass_at_3: float | None
    hard_safety_failures: int
    systemic_failures: int
    repetitions_per_case: int


@dataclass(frozen=True, slots=True)
class QualificationDecision:
    status: Literal["promote", "no_stable_winner"]
    candidate_id: str | None
    reasons: tuple[str, ...]
```

Keep this module pure: no filesystem, subprocess, AKS, DSPy, or runner imports.

- [ ] **Step 4: Run tests and confirm GREEN**

Run: `uv run --python 3.12 pytest -q tests/test_stable_ranking.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/korvid_prompt_lab/stable_ranking.py tests/test_stable_ranking.py
git commit -m "feat(search): add stable winner qualification"
```

### Task 4: Immutable Staged Search Orchestrator

**Files:**
- Create: `src/korvid_prompt_lab/stable_search.py`
- Create: `tests/test_stable_search.py`

**Interfaces:**
- Consumes: `KorvidRunner`, baseline/candidates, `ScenarioManifest`, and ranking functions.
- Produces: `StableSearchConfig`, `StableSearchArtifacts`, and `run_stable_search(...) -> StableSearchArtifacts`.

- [ ] **Step 1: Write failing end-to-end fake-runner tests**

```python
def test_stable_search_promotes_known_winner(tmp_path: Path) -> None:
    artifacts = run_stable_search(
        runner=_deterministic_runner(winner="evidence-first"),
        baseline=_baseline(),
        candidates=_candidates(),
        manifest=_manifest(),
        artifact_root=tmp_path,
    )
    assert artifacts.decision.status == "promote"
    assert artifacts.decision.candidate_id == "evidence-first"


def test_stable_search_records_no_winner(tmp_path: Path) -> None:
    artifacts = run_stable_search(
        runner=_flat_runner(),
        baseline=_baseline(),
        candidates=_candidates(),
        manifest=_manifest(),
        artifact_root=tmp_path,
    )
    assert artifacts.decision.status == "no_stable_winner"
```

Assert Stage A uses one repetition, Stage B uses three, Stage C uses five; at most three Stage B survivors and two Stage C finalists; safety/systemic candidates stop early.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run --python 3.12 pytest -q tests/test_stable_search.py`

Expected: module import failure.

- [ ] **Step 3: Implement staged runner and immutable artifacts**

```python
@dataclass(frozen=True, slots=True)
class StableSearchConfig:
    screening_repetitions: int = 1
    validation_repetitions: int = 3
    qualification_repetitions: int = 5
    screening_survivors: int = 3
    finalists: int = 2
    minimum_mean_delta: float = 0.10


def run_stable_search(
    *,
    runner: KorvidRunner,
    baseline: Candidate,
    candidates: Sequence[StructuredCandidate],
    manifest: ScenarioManifest,
    artifact_root: Path,
    config: StableSearchConfig = StableSearchConfig(),
) -> StableSearchArtifacts:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=False)
    screening = _run_paired_stage(
        runner, baseline, candidates, manifest.train, 1, root / "stage-a"
    )
    survivors = rank_screening(screening, limit=config.screening_survivors)
    validation = _run_paired_stage(
        runner, baseline, survivors, manifest.validation, 3, root / "stage-b"
    )
    finalists = select_finalists(validation, limit=config.finalists)
    qualification = _run_qualification(
        runner, baseline, finalists, manifest, root / "stage-c"
    )
    decision = qualify_winner(qualification)
    return _write_stable_search_artifacts(
        root, manifest, screening, validation, qualification, decision
    )
```

Use a fresh directory per stage/candidate/case/repetition. Reuse runner
`response.json` evidence; persist only normalized summaries and candidate
manifests via `write_json_artifact`. Reject an existing campaign root.

- [ ] **Step 4: Run tests and confirm GREEN**

Run: `uv run --python 3.12 pytest -q tests/test_stable_search.py`

Expected: known winner and no-winner integrations pass.

- [ ] **Step 5: Commit**

```bash
git add src/korvid_prompt_lab/stable_search.py tests/test_stable_search.py
git commit -m "feat(search): orchestrate stable prompt qualification"
```

### Task 5: Bounded Optional Proposer

**Files:**
- Create: `src/korvid_prompt_lab/stable_proposer.py`
- Create: `tests/test_stable_proposer.py`
- Modify: `src/korvid_prompt_lab/reflection.py`

**Interfaces:**
- Consumes: one finalist append, one failure axis, bounded aggregate feedback, and a DSPy LM.
- Produces: `BoundedAppendProposer.propose(...) -> str`.

- [ ] **Step 1: Write failing proposer-boundary tests**

```python
def test_proposer_rejects_blank_or_overlong_append() -> None:
    with pytest.raises(ValueError, match="blank"):
        _proposer(output=" ").propose(_request())
    with pytest.raises(ValueError, match="480"):
        _proposer(output="x" * 481).propose(_request())


def test_proposer_input_contains_no_raw_answer_or_cluster_data() -> None:
    request = build_proposal_request(_bounded_measurement())
    encoded = json.dumps(asdict(request))
    assert "answer" not in encoded
    assert "kubeconfig" not in encoded
    assert "endpoint" not in encoded
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run --python 3.12 pytest -q tests/test_stable_proposer.py`

Expected: module import failure.

- [ ] **Step 3: Implement one-axis proposer**

Use a dedicated DSPy signature:

```python
class BoundedAppendSignature(dspy.Signature):
    current_append: str = dspy.InputField()
    failure_axis: str = dspy.InputField()
    bounded_feedback_json: str = dspy.InputField()
    revised_append: str = dspy.OutputField(
        desc="Canonical prompt append, at most 480 characters."
    )
```

Allow exactly one proposal per finalist. Strip output, reject unknown axes,
blank text, outer whitespace mismatch, and length above 480. A proposer
exception returns no candidate and never invalidates structured results.

- [ ] **Step 4: Run tests and confirm GREEN**

Run: `uv run --python 3.12 pytest -q tests/test_stable_proposer.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/korvid_prompt_lab/stable_proposer.py src/korvid_prompt_lab/reflection.py tests/test_stable_proposer.py
git commit -m "feat(search): add bounded append refinement"
```

### Task 6: CLI and AKS Execution Surface

**Files:**
- Modify: `src/korvid_prompt_lab/cli.py`
- Create: `tests/test_stable_search_cli.py`
- Create: `examples/stable-search/korvid-small.yaml`
- Modify: `README.md`

**Interfaces:**
- Produces CLI command `korvid-prompt-lab stable-search`.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_stable_search_cli_writes_final_decision(tmp_path: Path) -> None:
    exit_code, payload = run_cli_with_fake_runner(tmp_path)
    assert exit_code == 0
    assert payload["decision"] in {"promote", "no_stable_winner"}


def test_stable_search_cli_rejects_existing_artifact_root(tmp_path: Path) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    assert run_cli(root).exit_code == 2
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run --python 3.12 pytest -q tests/test_stable_search_cli.py`

Expected: argparse rejects unknown command `stable-search`.

- [ ] **Step 3: Add CLI arguments and runtime wiring**

Add:

```text
stable-search
  --artifact-root PATH
  --target-per-split 6
  --reflection-model MODEL      # optional
  --enable-bounded-proposer
  --json
```

The command materializes the installed `small` baseline, builds the installed
scenario manifest and candidate matrix, constructs a `KorvidReadonlyRunner`,
and calls `run_stable_search`. Endpoint configuration remains
`KORVID_READONLY_BASE_URL`.

Document exact local and AKS port-forward commands, the 306-run upper bound,
winner gate, and `no_stable_winner` semantics.

- [ ] **Step 4: Run CLI and related tests**

Run:

```bash
uv run --python 3.12 pytest -q \
  tests/test_stable_search_cli.py \
  tests/test_stable_search.py \
  tests/test_korvid_readonly.py \
  tests/test_cli.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/korvid_prompt_lab/cli.py tests/test_stable_search_cli.py examples/stable-search/korvid-small.yaml README.md
git commit -m "feat(search): expose stable prompt campaign"
```

### Task 7: Full Verification and Live AKS Campaign

**Files:**
- Modify: `docs/superpowers/plans/2026-09-01-stable-korvid-prompt-search.md`
- Create only if safe winner exists: `examples/candidates/korvid-small-stable-winner.yaml`

**Interfaces:**
- Consumes the completed `stable-search` CLI.
- Produces safe live evidence and a final `promote`, `no_stable_winner`, or `system_error` decision.

- [ ] **Step 1: Run repository verification**

Run:

```bash
uv run --python 3.12 pytest -q
uv run --python 3.12 ruff check .
uv run --python 3.12 mypy src tests
uv lock --check
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 2: Verify AKS identity and restore plan**

Use the existing AKS tooling to confirm resource group `rg-pension-guard`,
cluster `aks-shared-runners`, node pool `modeleval`, the Ollama service, and
model digest. Record the original node-pool count before scaling.

- [ ] **Step 3: Run the immutable live campaign**

After private port-forward startup:

```bash
export KORVID_READONLY_BASE_URL=http://127.0.0.1:41001
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
uv run --python 3.12 korvid-prompt-lab stable-search \
  --artifact-root "artifacts/stable-search/$RUN_ID" \
  --target-per-split 6 \
  --reflection-model ollama_chat/qwen3:4b \
  --enable-bounded-proposer \
  --json
```

Do not stop at Stage A. Continue through Stage C unless every candidate is
rejected or a systemic failure aborts the run.

- [ ] **Step 4: Verify the final decision**

For `promote`, independently assert both split deltas, five repetitions,
worst-case non-regression, and zero failure counts from persisted normalized
evidence. Write the exact append Candidate YAML only after those checks.

For `no_stable_winner`, preserve rankings and rejection reasons and do not
write a winner Candidate.

- [ ] **Step 5: Restore and clean up**

Restore `modeleval` to its original count, terminate only the exact
port-forward PID, and remove temporary kubeconfig/private runtime artifacts.
Confirm the node pool provisioning state is `Succeeded`.

- [ ] **Step 6: Update the plan with measured results**

Record candidate counts, paired baseline/finalist movements, failure counts,
decision, model digest, Korvid version, and safe artifact references. Do not
record raw answers or credentials.

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/plans/2026-09-01-stable-korvid-prompt-search.md
test ! -e examples/candidates/korvid-small-stable-winner.yaml || \
  git add examples/candidates/korvid-small-stable-winner.yaml
git commit -m "docs(search): record stable prompt campaign"
```

### Task 8: Independent Review and PR

**Files:**
- Review all changed files against `origin/main`.

**Interfaces:**
- Produces a review-clean branch and PR with exact measured claims.

- [ ] **Step 1: Request independent code review**

Review candidate generation, split integrity, ranking math, live artifact
redaction, bounded proposer input, AKS cleanup, and winner/no-winner claims.

- [ ] **Step 2: Fix Critical and Important findings with TDD**

For every valid finding, add a failing regression test, verify RED, implement
the minimal fix, and verify GREEN.

- [ ] **Step 3: Re-run full verification**

Run the five commands from Task 7 Step 1. Expected: all exit zero.

- [ ] **Step 4: Push and create PR**

The PR must state whether a stable winner exists. If none exists, its title
and body must describe the search capability and measured no-winner result,
not claim prompt improvement.
