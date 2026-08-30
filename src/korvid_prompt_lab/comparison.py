from __future__ import annotations

import math
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
    evidence_sources: tuple[tuple[str, str, int, str, str, str], ...]


@dataclass(frozen=True, slots=True)
class EvaluationSnapshot:
    contract: EvaluationContract
    candidate_fingerprint: str
    aggregate_score: float
    pass_at_3: float | None
    pass_at_5: float | None
    systemic_failures: int
    hard_failure_counts: tuple[tuple[str, int], ...]
    run_evidence: tuple[
        tuple[
            str,
            str,
            int,
            str,
            float | None,
            float | None,
            float | None,
            tuple[str, ...],
            str,
        ],
        ...,
    ]

    @classmethod
    def from_report(cls, report: RoundReport) -> EvaluationSnapshot:
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
                evidence_sources=report.evidence_sources,
            ),
            candidate_fingerprint=report.candidate_fingerprint,
            aggregate_score=report.aggregate_score,
            pass_at_3=report.pass_at_3,
            pass_at_5=report.pass_at_5,
            systemic_failures=report.systemic_failures,
            hard_failure_counts=tuple(sorted(report.hard_failure_counts.items())),
            run_evidence=tuple(
                sorted(
                    (
                        run.case_id,
                        run.model,
                        run.repetition,
                        run.status,
                        run.completion,
                        run.verification,
                        run.efficiency,
                        run.hard_failures,
                        run.execution_mode,
                    )
                    for run in report.runs
                )
            ),
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
    not_comparable_count: int


def _require_finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} is not a finite number: {value!r}")
    return value


def _metric_result(delta: float, *, lower_is_better: bool) -> MetricResult:
    if delta == 0:
        return "unchanged"
    improved = delta < 0 if lower_is_better else delta > 0
    return "improved" if improved else "regressed"


def _make_metric(
    key: str,
    label: str,
    before: float | None,
    after: float | None,
    *,
    lower_is_better: bool = False,
    integer: bool = False,
    core: bool = True,
) -> MetricComparison:
    if before is None or after is None:
        return MetricComparison(
            key=key,
            label=label,
            before=before,
            after=after,
            delta=None,
            result="not_comparable",
            integer=integer,
            core=core,
        )
    delta: float | int = (after - before) if not integer else int(after) - int(before)
    result = _metric_result(delta, lower_is_better=lower_is_better)
    return MetricComparison(
        key=key,
        label=label,
        before=before,
        after=after,
        delta=delta,
        result=result,
        integer=integer,
        core=core,
    )


def build_round_comparison(
    before: RoundReport,
    after: RoundReport,
    *,
    seed_fingerprint: str,
    best_fingerprint: str,
) -> RoundComparison:
    snap_before = EvaluationSnapshot.from_report(before)
    snap_after = EvaluationSnapshot.from_report(after)

    # Contract identity check
    if snap_before.contract != snap_after.contract:
        raise ValueError(
            "comparison contract mismatch: before and after must share campaign, "
            "models, case repetitions, and execution modes"
        )

    contract = snap_before.contract

    # Fingerprint checks
    if snap_before.candidate_fingerprint != seed_fingerprint:
        raise ValueError(
            f"before fingerprint {snap_before.candidate_fingerprint!r} does not match seed_fingerprint"
        )
    if snap_after.candidate_fingerprint != best_fingerprint:
        raise ValueError(
            f"after fingerprint {snap_after.candidate_fingerprint!r} does not match best_fingerprint"
        )

    # Same-fingerprint invariant
    same_fingerprint = seed_fingerprint == best_fingerprint
    if same_fingerprint and snap_before != snap_after:
        raise ValueError(
            "unchanged candidate evidence: same fingerprint but snapshots differ"
        )

    # Validate finite core metrics
    _require_finite(snap_before.aggregate_score, "before aggregate_score")
    _require_finite(snap_after.aggregate_score, "after aggregate_score")

    # Build core metrics
    hard_before = sum(v for _, v in snap_before.hard_failure_counts)
    hard_after = sum(v for _, v in snap_after.hard_failure_counts)

    core_metrics: list[MetricComparison] = [
        _make_metric("aggregate_score", "Aggregate score", snap_before.aggregate_score, snap_after.aggregate_score, core=True),
        _make_metric("pass_at_3", "pass@3", snap_before.pass_at_3, snap_after.pass_at_3, core=True),
        _make_metric("pass_at_5", "pass@5", snap_before.pass_at_5, snap_after.pass_at_5, core=True),
        _make_metric("hard_safety_failures", "Hard safety failures", hard_before, hard_after, lower_is_better=True, integer=True, core=True),
        _make_metric("systemic_failures", "Systemic failures", snap_before.systemic_failures, snap_after.systemic_failures, lower_is_better=True, integer=True, core=True),
    ]

    # Build failure-category union metrics (sorted)
    all_failure_keys = sorted(
        {key for key, _ in snap_before.hard_failure_counts}
        | {key for key, _ in snap_after.hard_failure_counts}
    )
    before_failures = dict(snap_before.hard_failure_counts)
    after_failures = dict(snap_after.hard_failure_counts)
    failure_metrics: list[MetricComparison] = [
        _make_metric(
            key,
            f"`{key}`",
            before_failures.get(key, 0),
            after_failures.get(key, 0),
            lower_is_better=True,
            integer=True,
            core=False,
        )
        for key in all_failure_keys
    ]

    all_metrics = tuple(core_metrics + failure_metrics)

    # Determine outcome using only core metrics
    if same_fingerprint:
        outcome: ComparisonOutcome = "unchanged"
    else:
        core_results = [m.result for m in core_metrics if m.result != "not_comparable"]
        if any(r == "regressed" for r in core_results):
            outcome = "regressed"
        elif any(r == "improved" for r in core_results):
            outcome = "improved"
        else:
            outcome = "unchanged"

    improved_count = sum(1 for m in all_metrics if m.result == "improved")
    unchanged_count = sum(1 for m in all_metrics if m.result == "unchanged")
    regressed_count = sum(1 for m in all_metrics if m.result == "regressed")
    not_comparable_count = sum(1 for m in all_metrics if m.result == "not_comparable")

    return RoundComparison(
        status="unchanged" if same_fingerprint else "changed",
        outcome=outcome,
        seed_candidate_fingerprint=seed_fingerprint,
        best_candidate_fingerprint=best_fingerprint,
        contract=contract,
        metrics=all_metrics,
        improved_count=improved_count,
        unchanged_count=unchanged_count,
        regressed_count=regressed_count,
        not_comparable_count=not_comparable_count,
    )


def _fmt_score(value: float | None, *, integer: bool) -> str:
    if value is None:
        return "N/A"
    if integer:
        return str(int(value))
    return f"{float(value):.3f}"


def _fmt_delta(delta: float | None, *, integer: bool) -> str:
    if delta is None:
        return "N/A"
    if integer:
        di = int(delta)
        return f"+{di}" if di > 0 else str(di)
    df = float(delta)
    return f"+{df:.3f}" if df > 0 else f"{df:.3f}"


def _fmt_result(result: MetricResult) -> str:
    if result == "improved":
        return "✅ improved"
    if result == "regressed":
        return "⚠️ regressed"
    if result == "unchanged":
        return "➖ unchanged"
    return "N/A"


def _publication_line(report: RoundReport) -> str:
    if report.promotion_eligible:
        return "- Publication: eligible"
    blockers = ", ".join(f"`{b}`" for b in report.promotion_blockers)
    return f"- Publication: blocked ({blockers})"


def render_comparison_markdown(comparison: RoundComparison, report: RoundReport) -> str:
    # Branch on the comparison *status* first: only a matching fingerprint means
    # the optimizer retained the seed prompt. A changed candidate with no core
    # movement is still a changed candidate and must say so.
    if comparison.status == "unchanged":
        outcome_line = "## ➖ UNCHANGED — optimizer retained the seed prompt"
    elif comparison.outcome == "improved":
        outcome_line = "## ✅ IMPROVED — candidate changed; no core metric regressed"
    elif comparison.outcome == "regressed":
        outcome_line = "## ⚠️ REGRESSED — candidate changed; one or more core metrics regressed"
    else:
        outcome_line = "## ➖ UNCHANGED — candidate changed; no core metric moved"

    lines = [
        "# Grounding Round Outcome",
        "",
        outcome_line,
        "",
        "## Before vs after",
        "",
        "| Metric | Before | After | Delta | Result |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for m in comparison.metrics:
        label = m.label
        before_str = _fmt_score(m.before, integer=m.integer)
        after_str = _fmt_score(m.after, integer=m.integer)
        delta_str = _fmt_delta(m.delta, integer=m.integer)
        result_str = _fmt_result(m.result) if m.result != "not_comparable" else "N/A"
        lines.append(f"| {label} | {before_str} | {after_str} | {delta_str} | {result_str} |")

    # Prompt line
    if comparison.status == "unchanged":
        prompt_line = f"- Prompt: unchanged (`{comparison.seed_candidate_fingerprint}`)"
    else:
        prompt_line = (
            f"- Prompt: `{comparison.seed_candidate_fingerprint}` → `{comparison.best_candidate_fingerprint}`"
        )

    net_line = (
        f"- Net: {comparison.improved_count} improved, "
        f"{comparison.unchanged_count} unchanged, "
        f"{comparison.regressed_count} regressed"
    )
    if comparison.not_comparable_count:
        net_line += f", {comparison.not_comparable_count} not comparable"

    lines.extend(["", prompt_line, net_line, _publication_line(report)])
    return "\n".join(lines) + "\n"


def render_single_evaluation_markdown(report: RoundReport) -> str:
    snap = EvaluationSnapshot.from_report(report)
    hard_total = sum(v for _, v in snap.hard_failure_counts)

    lines = [
        "# Grounding Round Outcome",
        "",
        "## ℹ️ SINGLE EVALUATION — no before/after pair",
        "",
        "## Final scores",
        "",
        "| Metric | Final |",
        "| --- | ---: |",
        f"| Aggregate score | {snap.aggregate_score:.3f} |",
        f"| pass@3 | {_fmt_score(snap.pass_at_3, integer=False)} |",
        f"| pass@5 | {_fmt_score(snap.pass_at_5, integer=False)} |",
        f"| Hard safety failures | {hard_total} |",
        f"| Systemic failures | {snap.systemic_failures} |",
    ]

    for key, count in snap.hard_failure_counts:
        lines.append(f"| `{key}` | {count} |")

    lines.extend(["", _publication_line(report)])
    return "\n".join(lines) + "\n"


def comparison_payload(comparison: RoundComparison) -> dict[str, object]:
    evidence_sources = comparison.contract.evidence_sources
    return {
        "schema_version": 2 if evidence_sources else 1,
        "status": comparison.status,
        "outcome": comparison.outcome,
        "seed_candidate_fingerprint": comparison.seed_candidate_fingerprint,
        "best_candidate_fingerprint": comparison.best_candidate_fingerprint,
        "contract": {
            "campaign_id": comparison.contract.campaign_id,
            "models": list(comparison.contract.models),
            "case_repetitions": [list(cr) for cr in comparison.contract.case_repetitions],
            "execution_modes": list(comparison.contract.execution_modes),
            **(
                {"evidence_sources": [list(source) for source in evidence_sources]}
                if evidence_sources
                else {}
            ),
        },
        "metrics": [
            {
                "key": m.key,
                "label": m.label,
                "before": m.before,
                "after": m.after,
                "delta": m.delta,
                "result": m.result,
                "integer": m.integer,
                "core": m.core,
            }
            for m in comparison.metrics
        ],
        "improved_count": comparison.improved_count,
        "unchanged_count": comparison.unchanged_count,
        "regressed_count": comparison.regressed_count,
        "not_comparable_count": comparison.not_comparable_count,
    }
