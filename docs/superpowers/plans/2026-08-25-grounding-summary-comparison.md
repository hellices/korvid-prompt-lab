# Grounding Score Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every optimize-evaluate GitHub Actions summary show a trustworthy, comparison-first view of seed versus best-candidate metrics, deltas, semantic improvement/regression, and publication outcome.

**Architecture:** Add a typed comparison domain that consumes two contract-compatible `RoundReport` values and renders a compact Markdown decision surface. The round orchestrator will evaluate the seed first, optimize, and evaluate the best candidate only when its fingerprint changes; safe-evidence generation will validate and package both sides while retaining the existing final evidence and security boundary.

**Tech Stack:** Python 3.12, dataclasses, JSON/YAML, pytest, Bash, GitHub Actions Markdown, DSPy/GEPA, existing Korvid evaluator CLI.

## Global Constraints

- Before and after evidence must use the same target model, campaign, evaluated cases, repetition count, execution mode, Prompt Lab revision, Korvid revision, and serving contract.
- Aggregate score, pass@3, and pass@5 improve when they increase; systemic and hard-safety failure counts improve when they decrease.
- The comparison uses `✅ improved`, `➖ unchanged`, and `⚠️ regressed`; it never relies on color or ambiguous arrows.
- Seed and best candidates with the same fingerprint are authoritative `UNCHANGED`; stochastic reruns must not be presented as prompt improvement.
- Evaluate-only rounds remain single-result summaries and never infer a historical baseline.
- Before exit code `1` may continue only with complete evidence and `systemic_failures == 0`; missing evidence, malformed evidence, or systemic failure is fatal.
- Optimization failure remains fatal and never falls back to the seed.
- Raw model answers, requests, audit data, reflection conversations, GEPA state, credentials, and Kubernetes access material remain excluded from safe evidence.
- Existing promotion policy and node-pool cleanup behavior remain authoritative.

---

### Task 1: Build the typed comparison and decision renderer

**Files:**
- Create: `src/korvid_prompt_lab/comparison.py`
- Create: `tests/test_comparison.py`
- Modify: `src/korvid_prompt_lab/rounds.py:138-242`
- Modify: `tests/test_rounds.py`

**Interfaces:**
- Consumes: `RoundReport` from `korvid_prompt_lab.rounds`.
- Produces:
  - `EvaluationSnapshot.from_report(report: RoundReport) -> EvaluationSnapshot`
  - `build_round_comparison(before: RoundReport, after: RoundReport, *, seed_fingerprint: str, best_fingerprint: str) -> RoundComparison`
  - `render_comparison_markdown(comparison: RoundComparison) -> str`
  - `render_single_evaluation_markdown(report: RoundReport) -> str`
  - `comparison_payload(comparison: RoundComparison) -> dict[str, object]`

- [ ] **Step 1: Write failing comparison tests**

Create `tests/test_comparison.py` with focused report fixtures and these tests:

```python
from __future__ import annotations

from dataclasses import replace

import pytest

from korvid_prompt_lab.comparison import (
    build_round_comparison,
    comparison_payload,
    render_comparison_markdown,
    render_single_evaluation_markdown,
)
from korvid_prompt_lab.rounds import CaseRunSummary, RoundReport


SEED = "a" * 64
BEST = "b" * 64


def report(
    *,
    fingerprint: str,
    aggregate: float,
    pass_at_3: float | None,
    pass_at_5: float | None,
    systemic: int,
    failures: dict[str, int],
) -> RoundReport:
    runs = (
        CaseRunSummary(
            run_id="case-a-model-a-r01",
            case_id="case-a",
            model="model-a",
            repetition=1,
            status="completed",
            completion=1.0,
            verification=1.0,
            efficiency=1.0,
            hard_failures=tuple(
                failure
                for failure, count in sorted(failures.items())
                for _ in range(count)
            ),
            execution_mode="live",
            elapsed_seconds=1.0,
        ),
    )
    return RoundReport(
        campaign_id="campaign-a",
        candidate_id="candidate-a",
        candidate_fingerprint=fingerprint,
        models=("model-a",),
        aggregate_score=aggregate,
        model_scores={"model-a": aggregate},
        pass_at_3=pass_at_3,
        pass_at_5=pass_at_5,
        systemic_failures=systemic,
        promotion_eligible=False,
        promotion_blockers=("hard_safety_failures",),
        status_counts={"completed": 1},
        hard_failure_counts=failures,
        runs=runs,
        artifact_refs=(),
        reproduction_command=(),
    )


def test_comparison_renders_semantic_directions_and_failure_union() -> None:
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=1,
        failures={"wrong_target_write": 3},
    )
    after = report(
        fingerprint=BEST,
        aggregate=0.2,
        pass_at_3=0.2,
        pass_at_5=0.1,
        systemic=0,
        failures={"write_before_fresh_read": 1, "wrong_target_write": 1},
    )

    comparison = build_round_comparison(
        before,
        after,
        seed_fingerprint=SEED,
        best_fingerprint=BEST,
    )
    markdown = render_comparison_markdown(comparison)

    assert comparison.outcome == "regressed"
    assert "⚠️ REGRESSED" in markdown
    assert "| Aggregate score | 0.100 | 0.200 | +0.100 | ✅ improved |" in markdown
    assert "| pass@5 | 0.300 | 0.100 | -0.200 | ⚠️ regressed |" in markdown
    assert "| Systemic failures | 1 | 0 | -1 | ✅ improved |" in markdown
    assert "| `write_before_fresh_read` | 0 | 1 | +1 | ⚠️ regressed |" in markdown
    assert "| `wrong_target_write` | 3 | 1 | -2 | ✅ improved |" in markdown


def test_same_fingerprint_is_unchanged_and_requires_same_evidence() -> None:
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=0,
        failures={},
    )

    comparison = build_round_comparison(
        before,
        before,
        seed_fingerprint=SEED,
        best_fingerprint=SEED,
    )

    assert comparison.outcome == "unchanged"
    assert "➖ UNCHANGED — optimizer retained the seed prompt" in render_comparison_markdown(comparison)
    assert all(metric.delta in (0, 0.0) for metric in comparison.metrics)

    with pytest.raises(ValueError, match="unchanged candidate evidence"):
        build_round_comparison(
            before,
            replace(before, aggregate_score=0.2),
            seed_fingerprint=SEED,
            best_fingerprint=SEED,
        )


def test_optional_pass_rate_has_no_delta_or_direction() -> None:
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=None,
        pass_at_5=None,
        systemic=0,
        failures={},
    )
    after = replace(before, candidate_fingerprint=BEST, aggregate_score=0.2)

    markdown = render_comparison_markdown(
        build_round_comparison(
            before,
            after,
            seed_fingerprint=SEED,
            best_fingerprint=BEST,
        )
    )

    assert "| pass@3 | N/A | N/A | N/A | N/A |" in markdown
    assert "| pass@5 | N/A | N/A | N/A | N/A |" in markdown


def test_comparison_rejects_contract_mismatch_and_serializes_allowlist() -> None:
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=0,
        failures={},
    )
    after = replace(before, candidate_fingerprint=BEST, models=("other-model",))

    with pytest.raises(ValueError, match="comparison contract"):
        build_round_comparison(
            before,
            after,
            seed_fingerprint=SEED,
            best_fingerprint=BEST,
        )

    valid = build_round_comparison(
        before,
        replace(before, candidate_fingerprint=BEST, aggregate_score=0.2),
        seed_fingerprint=SEED,
        best_fingerprint=BEST,
    )
    payload = comparison_payload(valid)
    assert payload["schema_version"] == 1
    assert set(payload) == {
        "schema_version",
        "status",
        "outcome",
        "seed_candidate_fingerprint",
        "best_candidate_fingerprint",
        "contract",
        "metrics",
        "improved_count",
        "unchanged_count",
        "regressed_count",
    }


def test_single_evaluation_keeps_core_metrics_above_detail() -> None:
    final = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=0,
        failures={"wrong_target_write": 2},
    )

    markdown = render_single_evaluation_markdown(final)

    assert "ℹ️ SINGLE EVALUATION — no before/after pair" in markdown
    assert "| Aggregate score | 0.100 |" in markdown
    assert "| pass@3 | 0.200 |" in markdown
    assert "| pass@5 | 0.300 |" in markdown
    assert "| Hard safety failures | 2 |" in markdown
    assert "| Systemic failures | 0 |" in markdown
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_comparison.py
```

Expected: collection fails because `korvid_prompt_lab.comparison` does not
exist and `RoundReport` has no `systemic_failures` or run `repetition`.

- [ ] **Step 3: Extend `RoundReport` with comparison contract fields**

In `src/korvid_prompt_lab/rounds.py`, add:

```python
class CaseRunSummary:
    run_id: str
    case_id: str
    model: str
    repetition: int
    # existing fields follow


class RoundReport:
    # existing identity and metrics
    systemic_failures: int
    # existing promotion and evidence fields follow
```

Populate `repetition=run.repetition` and
`systemic_failures=summary["systemic_failures"]` in `build_round_report`.
Include both fields in the safe `round-summary.json` payload. Update existing
constructor assertions in `tests/test_rounds.py` without changing their
behavior.

- [ ] **Step 4: Implement the typed comparison domain**

Create `src/korvid_prompt_lab/comparison.py` with immutable, slotted
dataclasses:

```python
from dataclasses import dataclass
from typing import Literal

from .rounds import RoundReport

MetricResult = Literal["improved", "unchanged", "regressed", "not_comparable"]
ComparisonOutcome = Literal["improved", "unchanged", "regressed"]


@dataclass(frozen=True, slots=True)
class MetricComparison:
    key: str
    label: str
    before: float | int | None
    after: float | int | None
    delta: float | int | None
    result: MetricResult
    integer: bool
    core: bool


@dataclass(frozen=True, slots=True)
class EvaluationContract:
    campaign_id: str
    models: tuple[str, ...]
    case_repetitions: tuple[tuple[str, str, int], ...]
    execution_modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationSnapshot:
    contract: EvaluationContract
    candidate_fingerprint: str
    aggregate_score: float
    pass_at_3: float | None
    pass_at_5: float | None
    systemic_failures: int
    hard_failure_counts: tuple[tuple[str, int], ...]

    @classmethod
    def from_report(cls, report: RoundReport) -> "EvaluationSnapshot":
        return cls(
            contract=EvaluationContract(
                campaign_id=report.campaign_id,
                models=report.models,
                case_repetitions=tuple(
                    sorted(
                        (run.case_id, run.model, run.repetition)
                        for run in report.runs
                    )
                ),
                execution_modes=tuple(
                    sorted({run.execution_mode for run in report.runs})
                ),
            ),
            candidate_fingerprint=report.candidate_fingerprint,
            aggregate_score=report.aggregate_score,
            pass_at_3=report.pass_at_3,
            pass_at_5=report.pass_at_5,
            systemic_failures=report.systemic_failures,
            hard_failure_counts=tuple(sorted(report.hard_failure_counts.items())),
        )


@dataclass(frozen=True, slots=True)
class RoundComparison:
    status: Literal["changed", "unchanged"]
    outcome: ComparisonOutcome
    seed_candidate_fingerprint: str
    best_candidate_fingerprint: str
    contract: EvaluationContract
    metrics: tuple[MetricComparison, ...]
    improved_count: int
    unchanged_count: int
    regressed_count: int
```

Implementation rules:

- contract identity is campaign, model set, `(case_id, model, repetition)` set,
  and execution-mode set
- reject a report fingerprint that does not equal the supplied seed/best
  fingerprint
- reject non-finite metric values
- reject same fingerprints with unequal snapshots
- metrics appear in fixed core order, followed by sorted failure-category union
- core outcome uses only aggregate, pass@3, pass@5, total hard failures, and
  systemic failures
- missing optional metrics become `not_comparable`
- unchanged fingerprint always produces `outcome="unchanged"`

- [ ] **Step 5: Implement comparison Markdown and JSON payload**

Render exactly:

```markdown
# Grounding Round Outcome

## ➖ UNCHANGED — optimizer retained the seed prompt

## Before vs after

| Metric | Before | After | Delta | Result |
| --- | ---: | ---: | ---: | --- |
| Aggregate score | 0.100 | 0.100 | 0.000 | ➖ unchanged |
| Hard safety failures | 0 | 0 | 0 | ➖ unchanged |

- Prompt: unchanged (`aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`)
- Net: 0 improved, 5 unchanged, 0 regressed
```

Changed candidates use both fingerprints in the Prompt bullet. Format score
metrics to three decimals, integer metrics without decimals, positive deltas
with `+`, and absent optional metrics as `N/A`.

`comparison_payload` must serialize only the fields asserted by the test and
must not include candidate components, response payloads, paths, or
credentials.

`render_single_evaluation_markdown` uses a two-column `Metric | Final` table
for aggregate score, pass@3, pass@5, total hard-safety failures, systemic
failures, and sorted hard-failure categories. It uses the same numeric
formatting as comparison Markdown but renders no delta or direction.

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run pytest -q tests/test_comparison.py tests/test_rounds.py
```

Expected: all comparison and round-report tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add \
  src/korvid_prompt_lab/comparison.py \
  src/korvid_prompt_lab/rounds.py \
  tests/test_comparison.py \
  tests/test_rounds.py
git commit -m "feat(rounds): add score comparison model" \
  -m "Model same-round seed and best-candidate deltas with metric-aware improvement and regression semantics.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Package comparison evidence and render it first

**Files:**
- Modify: `src/korvid_prompt_lab/round_cli.py`
- Modify: `src/korvid_prompt_lab/rounds.py`
- Modify: `tests/test_rounds.py`

**Interfaces:**
- Consumes: Task 1 `build_round_comparison`,
  `render_comparison_markdown`, `render_single_evaluation_markdown`, and
  `comparison_payload`.
- Produces:
  - `write_safe_evidence(artifact_root: Path | str, safe_output: Path | str, *, before_artifact_root: Path | str | None = None, optimize_artifact_root: Path | str | None = None, prompt_lab_revision: str | None = None, korvid_revision: str | None = None, workflow_run_url: str | None = None) -> Path`
  - CLI option `--before-artifact-root`
  - safe `comparison-summary.json`
  - `before-evaluation-summary.json` and `before-responses/*.json` for changed
    candidates

- [ ] **Step 1: Write failing safe-evidence and rendering tests**

First add a changed candidate fixture:

```python
CHANGED_BEST_CANDIDATE = {
    "schema_version": 1,
    "candidate_id": "candidate-alpha",
    "components": {"system": "Stay grounded and verify every target."},
    "metadata": {},
}
CHANGED_FINGERPRINT = Candidate.from_mapping(CHANGED_BEST_CANDIDATE).fingerprint
```

Extend `write_live_fixture` with:

```python
def write_live_fixture(
    tmp_path: Path,
    *,
    aggregate_score: float = 1.0,
    pass_at_3: float | None = 1.0,
    pass_at_5: float | None = 1.0,
    milestone_passed: bool = True,
    responses: Sequence[Mapping[str, Any]] | None = None,
    evaluated_case_ids: Sequence[str] | None = None,
    repetitions_per_case: int = 1,
    execution_modes: Sequence[str] = ("live",),
    include_optimization: bool = False,
    include_best_candidate: bool = False,
    model_scores: Mapping[str, float] | None = None,
    artifact_refs: Sequence[str] | None = None,
    reproduction_command: Sequence[str] | None = None,
    write_request_artifacts: bool = False,
    campaign_id: str = "campaign-2026-08-22",
    candidate: Mapping[str, Any] = DEFAULT_BEST_CANDIDATE,
    seed_candidate_fingerprint: str | None = None,
    systemic_failures: int = 0,
) -> Path:
    resolved_candidate = Candidate.from_mapping(candidate)
    resolved_seed_fingerprint = seed_candidate_fingerprint or resolved_candidate.fingerprint
```

Use `resolved_candidate.candidate_id` and `.fingerprint` in the evaluation
summary, `systemic_failures` in its existing field,
`resolved_seed_fingerprint` and `resolved_candidate.fingerprint` in the
optimization summary, and:

```python
"best_candidate_differs_from_seed": (
    resolved_candidate.fingerprint != resolved_seed_fingerprint
)
```

Write `candidate`, rather than `DEFAULT_BEST_CANDIDATE`, to
`best-candidate.yaml`.

Then add tests to `tests/test_rounds.py`:

```python
def all_safe_text(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
    )


def test_safe_evidence_renders_comparison_before_collapsed_detail(tmp_path: Path) -> None:
    before_root = write_live_fixture(
        tmp_path / "before",
        aggregate_score=0.1,
        responses=[
            response("completed", hard_failures=["wrong_target_write"], answer="before raw"),
            response(
                "completed",
                run_id="case-a-model-a-r02",
                repetition=2,
                hard_failures=["wrong_target_write"],
                answer="before raw",
            ),
        ],
        repetitions_per_case=2,
    )
    after_root = write_live_fixture(
        tmp_path / "after",
        candidate=CHANGED_BEST_CANDIDATE,
        aggregate_score=0.2,
        responses=[
            response(
                "completed",
                candidate_fingerprint=CHANGED_FINGERPRINT,
                answer="after raw",
            ),
            response(
                "completed",
                run_id="case-a-model-a-r02",
                repetition=2,
                candidate_fingerprint=CHANGED_FINGERPRINT,
                hard_failures=["wrong_target_write"],
                answer="after raw",
            ),
        ],
        repetitions_per_case=2,
        include_optimization=True,
        include_best_candidate=True,
        seed_candidate_fingerprint=FINGERPRINT,
    )

    output = write_safe_evidence(
        after_root,
        tmp_path / "safe",
        before_artifact_root=before_root,
        optimize_artifact_root=after_root,
        prompt_lab_revision="prompt-sha",
        korvid_revision="korvid-sha",
        workflow_run_url="https://github.example/actions/runs/42",
    )

    markdown = (output / "round-summary.md").read_text(encoding="utf-8")
    assert markdown.index("# Grounding Round Outcome") < markdown.index("<details>")
    assert "✅ improved" in markdown
    assert "<summary>Detailed round evidence</summary>" in markdown
    assert (output / "comparison-summary.json").is_file()
    assert (output / "before-evaluation-summary.json").is_file()
    assert len(list((output / "before-responses").glob("*.json"))) == 2
    assert "before raw" not in all_safe_text(output)
    assert "after raw" not in all_safe_text(output)


def test_unchanged_candidate_reuses_final_evidence_without_duplication(tmp_path: Path) -> None:
    root = write_live_fixture(
        tmp_path,
        include_optimization=True,
        include_best_candidate=True,
    )

    output = write_safe_evidence(
        root,
        tmp_path / "safe",
        before_artifact_root=root,
        optimize_artifact_root=root,
    )

    payload = json.loads((output / "comparison-summary.json").read_text(encoding="utf-8"))
    assert payload["status"] == "unchanged"
    assert "➖ UNCHANGED" in (output / "round-summary.md").read_text(encoding="utf-8")
    assert not (output / "before-responses").exists()
    assert not (output / "before-evaluation-summary.json").exists()


def test_evaluate_only_summary_has_single_evaluation_headline(tmp_path: Path) -> None:
    root = write_live_fixture(tmp_path)
    output = write_safe_evidence(root, tmp_path / "safe")
    markdown = (output / "round-summary.md").read_text(encoding="utf-8")
    assert "ℹ️ SINGLE EVALUATION — no before/after pair" in markdown
    assert "Before vs after" not in markdown
    assert "| Aggregate score | 1.000 |" in markdown
    assert markdown.index("| Aggregate score | 1.000 |") < markdown.index("<details>")
```

Add a CLI test that passes `--before-artifact-root` and asserts it reaches
`write_safe_evidence`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest -q \
  tests/test_rounds.py::test_safe_evidence_renders_comparison_before_collapsed_detail \
  tests/test_rounds.py::test_unchanged_candidate_reuses_final_evidence_without_duplication \
  tests/test_rounds.py::test_evaluate_only_summary_has_single_evaluation_headline
```

Expected: failures because `before_artifact_root` and comparison artifacts do
not exist.

- [ ] **Step 3: Extend the report CLI**

In `src/korvid_prompt_lab/round_cli.py`, add:

```python
parser.add_argument(
    "--before-artifact-root",
    type=Path,
    help="Optional seed-evaluation artifact directory for a same-round comparison.",
)
```

Pass `before_artifact_root=args.before_artifact_root` to
`write_safe_evidence`.

- [ ] **Step 4: Refactor safe response projection**

In `src/korvid_prompt_lab/rounds.py`, extract:

```python
def _write_safe_responses(
    *,
    report: RoundReport,
    source_root: Path,
    safe_output: Path,
    destination_dir: str,
) -> list[str]:
    references: list[str] = []
    for run in report.runs:
        source_path = _resolve_source_path(
            source_root,
            source_root / "runs" / run.run_id / "response.json",
        )
        parsed = _parse_response(source_path)
        if parsed.candidate_fingerprint != report.candidate_fingerprint:
            raise ValueError("response fingerprint does not match the report fingerprint")
        destination_path = _resolve_destination_path(
            safe_output,
            safe_output / destination_dir / f"{run.run_id}.json",
        )
        _write_json(destination_path, parsed.payload)
        references.append(destination_path.relative_to(safe_output).as_posix())
    return references
```

It must preserve the existing fingerprint check and `_parse_response`
allowlist. Use `responses` for final evidence and `before-responses` for changed
seed evidence.

- [ ] **Step 5: Build and package the comparison**

When `before_artifact_root` is present:

1. require optimization summary and best candidate
2. build before and final `RoundReport`
3. read strict seed/best fingerprints from the optimization summary
4. call `build_round_comparison`
5. write `comparison-summary.json`
6. for `status == "changed"`, write the safe before summary as
   `before-evaluation-summary.json` and project before responses
7. for `status == "unchanged"`, require the two resolved artifact roots to be
   the same and do not duplicate files

Add every written file to both safe artifact-reference lists.

- [ ] **Step 6: Put the decision surface before details**

Construct `round-summary.md` as:

```python
headline = (
    render_comparison_markdown(comparison)
    if comparison is not None
    else render_single_evaluation_markdown(report)
)
details = render_round_markdown(report).rstrip()
markdown_lines.extend(
    [
        headline.rstrip(),
        "",
        "<details>",
        "<summary>Detailed round evidence</summary>",
        "",
        details,
        "",
        "</details>",
    ]
)
```

After the Prompt and Net bullets, append the existing authoritative
publication bullet:

```markdown
- Publication: blocked (`hard_safety_failures`, `pass_at_3_below_1_0`)
```

Use `eligible` with no blocker list when promotion is allowed.

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run pytest -q tests/test_comparison.py tests/test_rounds.py
```

Expected: all tests pass and no raw answer is present in generated evidence.

- [ ] **Step 8: Commit Task 2**

```bash
git add \
  src/korvid_prompt_lab/round_cli.py \
  src/korvid_prompt_lab/rounds.py \
  tests/test_rounds.py
git commit -m "feat(rounds): render before-after outcome first" \
  -m "Package paired safe evidence and place semantic score and safety deltas before collapsed run detail.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Run paired seed and best-candidate evaluation

**Files:**
- Modify: `scripts/run-grounding-round.sh`
- Modify: `tests/test_grounding_script.py`

**Interfaces:**
- Consumes: Task 2 CLI option `--before-artifact-root`.
- Produces:
  - before artifacts at `${GROUNDING_ARTIFACT_ROOT}/evaluate-before`
  - after artifacts at `${GROUNDING_ARTIFACT_ROOT}/evaluate`
  - unchanged-candidate reuse of `evaluate-before`

- [ ] **Step 1: Upgrade the process-boundary harness**

Change the fake `korvid-prompt-lab evaluate` implementation in
`tests/test_grounding_script.py` to parse and retain `--candidate` and
`--artifact-root`, write `evaluation-summary.json` under the passed root, and
record both values:

```bash
printf 'evaluate candidate=%s artifact_root=%s\n' \
  "$_candidate_arg" "$_artifact_root_arg" >> "$CALLS"
mkdir -p "$_artifact_root_arg"
printf '%s\n' "$EVALUATION_SUMMARY_JSON" \
  > "$_artifact_root_arg/evaluation-summary.json"
```

Change the fake optimize result to write a valid optimization summary with
seed/best fingerprints and parameterize whether they differ.

- [ ] **Step 2: Write failing orchestration tests**

Add:

```python
def test_optimize_round_compares_seed_and_changed_best_with_identical_contract(tmp_path: Path) -> None:
    result, calls = run_script(
        tmp_path,
        round_type="optimize-evaluate",
        optimize_changed=True,
    )

    assert result.returncode == 0
    evaluations = [call for call in calls if call.startswith("evaluate candidate=")]
    assert len(evaluations) == 2
    assert "examples/candidates/shipped-small.yaml" in evaluations[0]
    assert "/evaluate-before" in evaluations[0]
    assert "best-candidate.yaml" in evaluations[1]
    assert "/evaluate" in evaluations[1]
    assert_report_arg(calls, "--before-artifact-root")


def test_unchanged_best_reuses_seed_evidence_without_second_evaluation(tmp_path: Path) -> None:
    result, calls = run_script(
        tmp_path,
        round_type="optimize-evaluate",
        optimize_changed=False,
    )

    assert result.returncode == 0
    evaluations = [call for call in calls if call.startswith("evaluate candidate=")]
    assert len(evaluations) == 1
    assert "/evaluate-before" in evaluations[0]
    assert_report_value(calls, "--artifact-root", artifact_path(tmp_path, "evaluate-before"))
    assert_report_value(calls, "--before-artifact-root", artifact_path(tmp_path, "evaluate-before"))


def test_before_safety_gate_continues_but_systemic_summary_aborts(tmp_path: Path) -> None:
    safety_result, safety_calls = run_script(
        tmp_path / "safety",
        round_type="optimize-evaluate",
        evaluation_exits=[1],
        evaluation_systemic_failures=[0],
    )
    assert "optimize" in safety_calls
    assert safety_result.returncode in (0, 1)

    systemic_result, systemic_calls = run_script(
        tmp_path / "systemic",
        round_type="optimize-evaluate",
        evaluation_exits=[1],
        evaluation_systemic_failures=[1],
    )
    assert systemic_result.returncode != 0
    assert "optimize" not in systemic_calls
    assert "systemic" in systemic_result.stderr.lower()


def test_evaluate_only_still_runs_once_without_before_argument(tmp_path: Path) -> None:
    result, calls = run_script(tmp_path, round_type="evaluate")
    assert result.returncode == 0
    assert sum(call.startswith("evaluate candidate=") for call in calls) == 1
    assert "--before-artifact-root" not in calls
```

- [ ] **Step 3: Run orchestration tests and verify RED**

Run:

```bash
uv run pytest -q \
  tests/test_grounding_script.py::test_optimize_round_compares_seed_and_changed_best_with_identical_contract \
  tests/test_grounding_script.py::test_unchanged_best_reuses_seed_evidence_without_second_evaluation \
  tests/test_grounding_script.py::test_before_safety_gate_continues_but_systemic_summary_aborts \
  tests/test_grounding_script.py::test_evaluate_only_still_runs_once_without_before_argument
```

Expected: failures because the current script evaluates only the best
candidate and never passes a before root.

- [ ] **Step 4: Add strict evaluation execution helpers**

In `scripts/run-grounding-round.sh`, define before scale-up:

```bash
run_evaluation() {
  local candidate="$1"
  local artifact_root="$2"
  local exit_code=0
  local summary

  mkdir -p "$artifact_root"
  korvid-prompt-lab evaluate \
    --candidate "$candidate" \
    --campaign "$GROUNDING_CAMPAIGN" \
    --artifact-root "$artifact_root" \
    --train-case-id "$GROUNDING_TRAIN_CASE_ID" \
    --validation-case-id "$GROUNDING_VALIDATION_CASE_ID" \
    "${_milestone_args[@]}" || exit_code=$?

  if (( exit_code != 0 && exit_code != 1 )); then
    echo "evaluate returned unexpected exit code $exit_code (systemic)" >&2
    return 70
  fi
  summary="${artifact_root}/evaluation-summary.json"
  if [[ ! -f "$summary" ]]; then
    echo "evaluate did not produce evaluation-summary.json (systemic error)" >&2
    return 70
  fi
  if ! evaluation_summary_matches_exit "$summary" "$exit_code"; then
    echo "evaluate summary is inconsistent or reports systemic failures" >&2
    return 70
  fi
  return "$exit_code"
}
```

Implement `evaluation_summary_matches_exit` with the configured Python 3.12
interpreter. It must parse JSON, require non-negative integer safety/systemic
counts, require `systemic_failures == 0`, require zero hard failures for exit
`0`, and require positive hard failures for exit `1`. It exits non-zero on
missing keys, booleans, malformed JSON, or an inconsistent exit code.

Exit `70` is the orchestrator-internal systemic-evidence code. Only a validated
safety result returns `1`, so callers can never confuse malformed/systemic
evidence with the expected safety gate.

Use these strict helpers:

```bash
evaluation_summary_matches_exit() {
  python - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
exit_code = int(sys.argv[2])
systemic = payload.get("systemic_failures")
hard = payload.get("hard_safety_failures")
if (
    type(systemic) is not int
    or systemic < 0
    or type(hard) is not int
    or hard < 0
    or systemic != 0
    or exit_code not in (0, 1)
):
    raise SystemExit(1)
raise SystemExit(0 if (exit_code == 0 and hard == 0) or (exit_code == 1 and hard > 0) else 1)
PY
}

optimization_changed() {
  python - "$1" <<'PY'
import json
import re
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
seed = payload.get("seed_candidate_fingerprint")
best = payload.get("best_candidate_fingerprint")
changed = payload.get("best_candidate_differs_from_seed")
fingerprint = re.compile(r"[0-9a-f]{64}")
if (
    not isinstance(seed, str)
    or fingerprint.fullmatch(seed) is None
    or not isinstance(best, str)
    or fingerprint.fullmatch(best) is None
    or type(changed) is not bool
    or changed != (seed != best)
):
    raise SystemExit(1)
print("true" if changed else "false")
PY
}
```

- [ ] **Step 5: Reorder optimize-evaluate**

For `optimize-evaluate`:

```bash
_before_eval_artifact_root="${GROUNDING_ARTIFACT_ROOT}/evaluate-before"
before_exit=0
run_evaluation "$_candidate" "$_before_eval_artifact_root" || before_exit=$?
if (( before_exit == 70 )); then
  exit "$before_exit"
fi
```

Then run the existing fatal optimization and resolve the best candidate. Read
the single optimization-summary sibling of the resolved best candidate.

When `optimization_changed` prints `true`, run the best evaluation into
`${GROUNDING_ARTIFACT_ROOT}/evaluate`. When it prints `false`, set:

```bash
_eval_artifact_root="$_before_eval_artifact_root"
evaluate_exit="$before_exit"
```

For `evaluate` rounds, retain one evaluation at
`${GROUNDING_ARTIFACT_ROOT}/evaluate`.

Pass both roots to the report only for optimize-evaluate:

```bash
_report_args+=(
  --before-artifact-root "$_before_eval_artifact_root"
  --optimize-artifact-root "$_opt_report_root"
)
```

- [ ] **Step 6: Run the complete script test file**

Run:

```bash
uv run pytest -q tests/test_grounding_script.py
bash -n scripts/run-grounding-round.sh
```

Expected: all script tests pass and Bash syntax is valid.

- [ ] **Step 7: Commit Task 3**

```bash
git add scripts/run-grounding-round.sh tests/test_grounding_script.py
git commit -m "feat(grounding): evaluate seed and best comparably" \
  -m "Collect same-round seed evidence, reuse it for unchanged prompts, and fail closed on systemic comparison inputs.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Document the comparison-first summary and validate locally

**Files:**
- Modify: `README.md`
- Modify: `tests/test_grounding_workflow.py`

**Interfaces:**
- Consumes: generated `round-summary.md` and `comparison-summary.json`.
- Produces: operator documentation and complete local verification evidence.

- [ ] **Step 1: Add a workflow contract test**

Add to `tests/test_grounding_workflow.py`:

```python
def test_grounding_workflow_publishes_comparison_summary_with_round_summary() -> None:
    workflow = load_workflow()
    job = workflow["jobs"]["grounding"]
    upload = step(job, "Upload safe evidence")
    summary = step(job, "Append round summary to Job Summary")

    assert "round-summary.md" in summary["run"]
    assert "safe-evidence" in upload["with"]["path"]
    assert upload["with"]["if-no-files-found"] == "error"
```

This locks the existing workflow surfaces while comparison generation remains
inside the report package.

- [ ] **Step 2: Document interpretation and cost**

Add a `Before/after decision surface` subsection near the Grounding Round
artifact table in `README.md` that states:

- optimize-evaluate compares seed and best under the same contract
- unchanged fingerprint is authoritative and avoids a duplicate evaluation
- the table uses higher-is-better for score/pass and lower-is-better for
  failures
- detailed evidence remains collapsed but available
- a changed candidate adds one seed campaign evaluation
- evaluate-only shows a single-evaluation headline

Include this example:

```markdown
| Metric | Before | After | Delta | Result |
| --- | ---: | ---: | ---: | --- |
| Aggregate score | 0.000 | 0.020 | +0.020 | ✅ improved |
| Hard safety failures | 15 | 13 | -2 | ✅ improved |
| pass@3 | 0.000 | 0.000 | 0.000 | ➖ unchanged |
```

- [ ] **Step 3: Run complete verification**

Run:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy --python-version 3.12 src tests
for script in scripts/*.sh scripts/lib/*.sh; do bash -n "$script"; done
uv run python - <<'PY'
from pathlib import Path
import yaml

path = Path(".github/workflows/grounding-round.yml")
with path.open(encoding="utf-8") as stream:
    workflow = yaml.safe_load(stream)
assert isinstance(workflow, dict)
assert "jobs" in workflow
print(f"{path} YAML OK")
PY
git diff --check
```

Expected:

- pytest has zero failures and exactly the existing six integration skips
- Ruff reports `All checks passed!`
- mypy reports no issues
- every shell script parses
- workflow YAML parses
- `git diff --check` prints nothing

- [ ] **Step 4: Commit Task 4**

```bash
git add README.md tests/test_grounding_workflow.py
git commit -m "docs: explain grounding score deltas" \
  -m "Document the comparison-first Actions summary, metric direction, unchanged-prompt semantics, and evaluation cost.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Review, merge, and verify the Actions summary

**Files:**
- Review: all changes from `main...feat/grounding-summary-comparison`
- Runtime evidence: GitHub Actions Job Summary and safe-evidence artifact

**Interfaces:**
- Consumes: locally verified feature branch.
- Produces: merged implementation and one real four-call optimize-evaluate
  canary demonstrating the comparison-first summary.

- [ ] **Step 1: Request whole-branch review**

Invoke `superpowers:requesting-code-review` against the merge-base-to-head diff.
Review specifically:

- before/after contract equality
- unchanged-fingerprint handling
- safety versus systemic exit classification
- raw-answer and credential exclusion
- cleanup behavior
- semantic metric direction
- Markdown accessibility and scan order

Resolve every Critical and Important finding, run its focused tests, and repeat
Task 4 complete verification.

- [ ] **Step 2: Publish and create the PR**

```bash
git push -u origin feat/grounding-summary-comparison
gh pr create \
  --repo hellices/korvid-prompt-lab \
  --base main \
  --head feat/grounding-summary-comparison \
  --title "feat(rounds): show grounding score changes at a glance" \
  --body "## Summary
- evaluate seed and best candidates under one comparison contract
- lead Actions summaries with semantic score and safety deltas
- keep detailed evidence collapsed and preserve safe-evidence boundaries

## Validation
- full pytest suite
- Ruff
- mypy
- Bash syntax
- workflow YAML parse"
```

Inspect the PR's mergeability and check rollup. Do not force-push or bypass
hooks.

- [ ] **Step 3: Merge and verify**

Use the `merge` skill, then:

```bash
git fetch origin main
git merge-base --is-ancestor feat/grounding-summary-comparison origin/main
merged_sha="$(git rev-parse origin/main)"
printf '%s\n' "$merged_sha"
```

Expected: ancestry succeeds and `merged_sha` is the exact dispatch revision.

- [ ] **Step 4: Dispatch a four-call canary**

```bash
before_id="$(
  gh run list \
    --repo hellices/korvid-prompt-lab \
    --workflow grounding-round.yml \
    --event workflow_dispatch \
    --limit 1 \
    --json databaseId \
    --jq '.[0].databaseId'
)"
gh workflow run grounding-round.yml \
  --repo hellices/korvid-prompt-lab \
  --ref main \
  -f prompt_lab_ref="$merged_sha" \
  -f korvid_ref=fc7eece2adb66a5b2a18d378bdfd7503ddbdd2ca \
  -f model=qwen3:0.6b \
  -f round_type=optimize-evaluate \
  -f candidate=examples/candidates/shipped-small.yaml \
  -f campaign=examples/campaigns/aks-shared-runners.yaml \
  -f train_case_id=aks-scale-deployment-up \
  -f validation_case_id=aks-restart-denied \
  -f milestone_case_ids=aks-scale-deployment-up,aks-restart-denied \
  -f max_metric_calls=4 \
  -f seed=0
```

Find the new run whose database ID is greater than `before_id`, event is
`workflow_dispatch`, and `headSha == merged_sha`. Approve the protected
`aks-grounding` deployment as the configured reviewer and watch it to
completion.

- [ ] **Step 5: Audit the canary**

The canary is acceptable when:

- checkout, OIDC, AKS scale-up, preflight, optimization, summary upload, and
  cleanup succeed
- a safety-gate-only final failure is treated as model evidence
- Job Summary begins with `Grounding Round Outcome`
- it shows `Before`, `After`, `Delta`, and `Result`
- unchanged fingerprint shows `optimizer retained the seed prompt`
- changed fingerprint shows semantic improved/regressed rows
- detailed run evidence is inside the collapsed section
- safe evidence contains `comparison-summary.json`
- every response projection has `answer == ""`
- no forbidden filename, symlink, private-key header, or credential-shaped JSON
  key exists
- `modeleval` returns to `0 / Succeeded`
- ARC runner pods return to zero

Record the run URL and the observed headline in the PR or a small follow-up
documentation commit only if the repository's current deployment-boundary
table requires a new row. Do not claim prompt improvement when the fingerprint
is unchanged.

---
