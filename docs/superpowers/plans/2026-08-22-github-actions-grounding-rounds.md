# GitHub Actions Grounding Rounds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run protected, manually dispatched Korvid grounding rounds on the existing AKS ARC runner and publish redacted results as a GitHub Job Summary, safe artifact, and optional PR comment.

**Architecture:** A pure Python reporting module validates evaluation evidence and builds allowlisted round artifacts. A shell orchestrator owns the temporary `modeleval` scale-up, preflight, evaluate or optimize/evaluate sequence, and exact scale restoration. A `workflow_dispatch` workflow runs that orchestrator on `korvid-runners` with OIDC and a protected Environment.

**Tech Stack:** Python 3.12, pytest, Bash, GitHub Actions, Azure CLI, kubectl, uv, GitHub OIDC, Actions Runner Controller.

## Global Constraints

- Trigger remotely with `workflow_dispatch`; never use `pull_request_target`.
- Run on the existing `korvid-runners` ARC scale set.
- Require the protected `aks-grounding` GitHub Environment before model compute.
- Use Azure OIDC; never store or print Azure client secrets, tokens, kubeconfigs, tenant IDs, client IDs, or subscription IDs.
- Scale only the existing `modeleval` pool, from its recorded original count to at most one, and restore the exact original count.
- Keep AKS model access loopback-only through the reviewed `AKSPortForward` backend.
- Korvid remains the authoritative evaluator.
- Systemic bridge failures abort; model failures score; any hard safety failure blocks promotion.
- Never upload raw answers, request JSON, audit JSONL, manifests, credentials, kubeconfigs, unrestricted tool output, process logs, or GEPA state.
- Safe evidence is limited to round summaries, evaluation/optimization summaries, best-candidate YAML, and protocol-v2 bridge responses.
- Concurrent grounding rounds serialize and do not cancel in-progress cleanup.
- Python remains `>=3.11`; workflow verification uses Python 3.12.

---

### Task 1: Safe round report and evidence package

**Files:**
- Create: `src/korvid_prompt_lab/rounds.py`
- Create: `src/korvid_prompt_lab/round_cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_rounds.py`

**Interfaces:**
- Consumes: `evaluation-summary.json`, optional `optimization-summary.json`, optional `best-candidate.yaml`, and `runs/*/response.json`.
- Produces: `RoundReport`, `build_round_report(...)`, `render_round_markdown(...)`, `write_safe_evidence(...)`, and the `korvid-grounding-report` console command.

- [ ] **Step 1: Write failing validation and rendering tests**

```python
def test_build_round_report_groups_safe_failures_without_raw_payloads(tmp_path: Path) -> None:
    artifact_root = write_live_fixture(
        tmp_path,
        aggregate_score=0.01,
        pass_at_3=0.0,
        pass_at_5=0.0,
        responses=[
            response("completed", completion=0.0, hard_failures=["wrong_target_write"]),
            response("model_failure", error="turn timeout"),
        ],
    )

    report = build_round_report(artifact_root)
    markdown = render_round_markdown(report)

    assert report.hard_failure_counts == {"wrong_target_write": 1}
    assert report.status_counts == {"completed": 1, "model_failure": 1}
    assert "wrong_target_write" in markdown
    assert "raw answer" not in markdown
    assert "audit" not in markdown.lower()
```

```python
@pytest.mark.parametrize("forbidden", ["audit.jsonl", "request.json", ".kubeconfig-x.yaml", "gepa_state.bin"])
def test_write_safe_evidence_never_copies_forbidden_files(tmp_path: Path, forbidden: str) -> None:
    artifact_root = write_live_fixture(tmp_path)
    path = artifact_root / "runs" / "case-r01" / forbidden
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("SECRET", encoding="utf-8")

    output = write_safe_evidence(artifact_root, tmp_path / "safe")

    assert not any(item.name == forbidden for item in output.rglob("*"))
    assert "SECRET" not in "\n".join(
        item.read_text(encoding="utf-8")
        for item in output.rglob("*")
        if item.is_file()
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --python 3.12 pytest tests/test_rounds.py -q
```

Expected: collection fails because `korvid_prompt_lab.rounds` does not exist.

- [ ] **Step 3: Implement strict report types and renderer**

Implement frozen/slotted types:

```python
@dataclass(frozen=True, slots=True)
class CaseRunSummary:
    run_id: str
    case_id: str
    model: str
    status: str
    completion: float | None
    verification: float | None
    efficiency: float | None
    hard_failures: tuple[str, ...]
    execution_mode: str


@dataclass(frozen=True, slots=True)
class RoundReport:
    campaign_id: str
    candidate_id: str
    candidate_fingerprint: str
    models: tuple[str, ...]
    aggregate_score: float
    pass_at_3: float | None
    pass_at_5: float | None
    promotion_eligible: bool
    promotion_blockers: tuple[str, ...]
    status_counts: Mapping[str, int]
    hard_failure_counts: Mapping[str, int]
    runs: tuple[CaseRunSummary, ...]
```

`build_round_report` must:

- require `execution_modes == ["live"]`;
- require protocol-v2 response fields;
- verify every response fingerprint matches the summary;
- reject missing, duplicate, or extra case/model/repetition evidence;
- reject unknown response keys by reusing runner parsing helpers where practical;
- compute promotion blockers from safety, systemic failures, milestone result, and pass metrics;
- expose only closed-vocabulary fields.

`render_round_markdown` must produce deterministic tables sorted by model,
case, and repetition.

- [ ] **Step 4: Implement allowlisted evidence packaging**

`write_safe_evidence` creates a new directory and copies only:

```text
round-summary.json
round-summary.md
evaluation-summary.json
optimization-summary.json
best-candidate.yaml
responses/<run-id>.json
```

Resolve every source and destination path and reject traversal outside the
artifact root or safe output root. Parse each response before copying it. Write
JSON and Markdown atomically.

- [ ] **Step 5: Add the console command**

Add:

```toml
[project.scripts]
korvid-grounding-report = "korvid_prompt_lab.round_cli:main"
```

The CLI is:

```text
korvid-grounding-report
  --artifact-root PATH
  --safe-output PATH
  --prompt-lab-revision SHA
  --korvid-revision SHA
  --workflow-run-url URL
```

It writes the safe package and prints only the path to `round-summary.md`.

- [ ] **Step 6: Verify and commit**

Run:

```bash
uv run --python 3.12 pytest tests/test_rounds.py -q
uv run --python 3.12 ruff check src/korvid_prompt_lab/rounds.py src/korvid_prompt_lab/round_cli.py tests/test_rounds.py
uv run --python 3.12 mypy --python-version 3.12 src tests
```

Commit:

```bash
git add pyproject.toml src/korvid_prompt_lab/rounds.py src/korvid_prompt_lab/round_cli.py tests/test_rounds.py
git commit -m "feat: add safe grounding round reports"
```

---

### Task 2: Grounding lifecycle orchestrator

**Files:**
- Create: `scripts/run-grounding-round.sh`
- Create: `tests/test_grounding_script.py`

**Interfaces:**
- Consumes: environment variables validated at startup and the two repository checkouts.
- Produces: an evaluation artifact root, `safe-evidence/`, and the exact CLI exit status.

- [ ] **Step 1: Write failing process-boundary tests**

Create fake `az`, `kubectl`, `korvid-prompt-lab`, and
`korvid-grounding-report` executables in a temporary `PATH`.

```python
def test_round_script_restores_zero_count_after_unsafe_evaluation(tmp_path: Path) -> None:
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=1,
    )

    assert result.returncode == 1
    assert calls.index("scale:1") < calls.index("evaluate")
    assert calls[-1] == "scale:0"
    assert "report" in calls
```

```python
def test_round_script_never_scales_down_preexisting_capacity(tmp_path: Path) -> None:
    result, calls = run_script(tmp_path, original_count=1, evaluation_exit=0)

    assert result.returncode == 0
    assert "scale:1" not in calls
    assert "scale:0" not in calls
```

```python
def test_round_script_restores_pool_on_signal_or_systemic_failure(tmp_path: Path) -> None:
    result, calls = run_script(tmp_path, original_count=0, preflight_exit=1)

    assert result.returncode != 0
    assert calls[-1] == "scale:0"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --python 3.12 pytest tests/test_grounding_script.py -q
```

Expected: failure because `scripts/run-grounding-round.sh` is absent.

- [ ] **Step 3: Implement strict input validation and cleanup**

The script starts with:

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

: "${GROUNDING_MODEL:?GROUNDING_MODEL is required}"
: "${GROUNDING_CANDIDATE:?GROUNDING_CANDIDATE is required}"
: "${GROUNDING_ROUND_TYPE:?GROUNDING_ROUND_TYPE is required}"
: "${KORVID_SOURCE_ROOT:?KORVID_SOURCE_ROOT is required}"
: "${GROUNDING_ARTIFACT_ROOT:?GROUNDING_ARTIFACT_ROOT is required}"

case "$GROUNDING_MODEL" in
  qwen3:0.6b|qwen3:1.7b|qwen3:4b|qwen3:8b|qwen3:14b) ;;
  *) echo "unsupported grounding model" >&2; exit 2 ;;
esac
```

Read the original node count once. A cleanup function restores zero only when
this invocation changed zero to one:

```bash
original_count="$(
  az aks nodepool show \
    --resource-group rg-pension-guard \
    --cluster-name aks-shared-runners \
    --name modeleval \
    --query count \
    --output tsv
)"
scaled_by_round=false

cleanup() {
  if [[ "$scaled_by_round" == true ]]; then
    az aks nodepool scale \
      --resource-group rg-pension-guard \
      --cluster-name aks-shared-runners \
      --name modeleval \
      --node-count 0
  fi
}
trap cleanup EXIT INT TERM
```

Reject original counts other than `0` or `1`.

- [ ] **Step 4: Implement preflight and evaluation**

When original count is zero, scale to one. Retry `korvid-prompt-lab aks-check`
with a bounded 15-minute deadline. Then run `evaluate` with explicit disjoint
train/validation and explicit milestone cases.

Capture exit `0` or `1`; exit `1` still runs `korvid-grounding-report`.
Any other exit or missing summary is systemic.

- [ ] **Step 5: Implement optimize-evaluate**

For `GROUNDING_ROUND_TYPE=optimize-evaluate`:

1. require `GROUNDING_REFLECTION_MODEL` and the provider credential;
2. run bounded `optimize`;
3. resolve exactly one new `best-candidate.yaml`;
4. evaluate that candidate into a separate artifact root;
5. pass the optimization root to the report CLI.

Never fall back to the seed when optimization fails.

- [ ] **Step 6: Verify and commit**

Run:

```bash
uv run --python 3.12 pytest tests/test_grounding_script.py -q
bash -n scripts/run-grounding-round.sh
```

Commit:

```bash
git add scripts/run-grounding-round.sh tests/test_grounding_script.py
git commit -m "feat: orchestrate protected grounding rounds"
```

---

### Task 3: Protected GitHub Actions workflow

**Files:**
- Create: `.github/workflows/grounding-round.yml`
- Create: `tests/test_grounding_workflow.py`

**Interfaces:**
- Consumes: the orchestrator from Task 2 and report artifact from Task 1.
- Produces: one serialized Actions run, Job Summary, uploaded safe artifact, and optional sticky PR comment.

- [ ] **Step 1: Write failing workflow contract tests**

```python
def test_grounding_workflow_has_protected_manual_arc_contract() -> None:
    workflow = load_workflow()
    assert "workflow_dispatch" in workflow["on"]
    job = workflow["jobs"]["grounding"]
    assert job["runs-on"] == "korvid-runners"
    assert job["environment"] == "aks-grounding"
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert workflow["permissions"]["id-token"] == "write"
    assert "pull_request_target" not in workflow["on"]
```

```python
def test_grounding_workflow_always_uploads_only_safe_evidence_and_cleans_up() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "if: always()" in text
    assert "safe-evidence" in text
    assert "artifacts/live" not in upload_artifact_path(load_workflow())
    assert "AZURE_CLIENT_SECRET" not in text
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --python 3.12 pytest tests/test_grounding_workflow.py -q
```

Expected: failure because the workflow file is absent.

- [ ] **Step 3: Implement workflow security and checkout**

Use:

```yaml
name: Grounding Round

on:
  workflow_dispatch:
    inputs:
      model:
        type: choice
        options: ["qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]
      round_type:
        type: choice
        options: ["evaluate", "optimize-evaluate"]
      korvid_ref:
        required: true
        default: "fc7eece2adb66a5b2a18d378bdfd7503ddbdd2ca"
      candidate:
        required: true
        default: "examples/candidates/shipped-small.yaml"
      pr_number:
        required: false

permissions:
  contents: read
  id-token: write
  pull-requests: write

concurrency:
  group: aks-grounding-${{ github.repository }}
  cancel-in-progress: false
```

The job uses `runs-on: korvid-runners` and `environment: aks-grounding`.
Check out the Prompt Lab revision and pinned Korvid revision. Obtain the Korvid
checkout token with `actions/create-github-app-token`.

- [ ] **Step 4: Run the orchestrator and publish results**

Use `azure/login`, `actions/setup-python`, and `astral-sh/setup-uv`.
Invoke the orchestrator with repository/Environment variables.

Append `safe-evidence/round-summary.md` to `$GITHUB_STEP_SUMMARY` under
`if: always() && hashFiles('safe-evidence/round-summary.md') != ''`.

Upload only `safe-evidence/` with `actions/upload-artifact`, retention 30 days.

Use `actions/github-script` to create or update a comment containing marker:

```text
<!-- korvid-grounding:<model>:<candidate> -->
```

The comment body is the safe Markdown plus the workflow run URL.

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run --python 3.12 pytest tests/test_grounding_workflow.py -q
uv run --python 3.12 pytest tests/test_rounds.py tests/test_grounding_script.py tests/test_grounding_workflow.py -q
```

Commit:

```bash
git add .github/workflows/grounding-round.yml tests/test_grounding_workflow.py
git commit -m "ci: add protected grounding rounds"
```

---

### Task 4: Documentation, full verification, and PR update

**Files:**
- Modify: `README.md`
- Modify: `examples/campaigns/aks-shared-runners.yaml`
- Modify: `tests/test_contracts.py`

**Interfaces:**
- Consumes: the complete workflow.
- Produces: operator setup instructions and verified PR changes.

- [ ] **Step 1: Document required GitHub configuration**

Document:

- Environment `aks-grounding` with required reviewers;
- repository/Environment variables for Azure OIDC identifiers;
- GitHub App credentials for read-only `hellices/korvid`;
- optional reflection-model credential;
- ARC runner label `korvid-runners`;
- manual dispatch instructions;
- result locations: Summary, artifact, PR comment;
- cleanup and rerun behavior.

- [ ] **Step 2: Document the measured baseline**

Record only safe aggregate evidence:

```text
qwen3:0.6b / shipped-small / 10 live runs
aggregate: 0.01
pass^3: 0.0
pass^5: 0.0
hard safety failures: 14
systemic failures: 0
```

Document that `qwen3:4b` exceeded the current unbounded-reasoning turn budget
and is not yet a valid comparison point.

- [ ] **Step 3: Verify the live campaign timeout contract**

Keep the explicit `--turn-timeout 300` campaign argument for the initial
`qwen3:0.6b` remote rounds and pin it in `test_load_campaign_from_example_yaml`.
Document that larger reasoning models require a separate bounded-serving policy
before selection.

- [ ] **Step 4: Run complete verification**

Run:

```bash
KORVID_SOURCE_ROOT=/Users/hwang-inhwan/workspace/kube/.worktrees/feat-307-small-operator-foundation uv run --python 3.12 pytest -q
uv run --python 3.12 ruff check .
uv run --python 3.12 mypy --python-version 3.12 src tests
bash -n scripts/run-grounding-round.sh
```

Expected: all tests pass, Ruff clean, mypy clean, Bash syntax valid.

- [ ] **Step 5: Final review and PR update**

Request a whole-branch review against the design. Fix all Critical/Important
findings, push `feat/prompt-lab-mvp`, and update Draft PR #1 with:

- workflow security model;
- required repository configuration;
- safe result surfaces;
- measured baseline;
- verification commands and counts.

Commit:

```bash
git add README.md examples/campaigns/aks-shared-runners.yaml tests/test_contracts.py
git commit -m "docs: explain remote grounding rounds"
```
