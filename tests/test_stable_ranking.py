from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.stable_ranking import (
    CandidateMeasurement,
    NormalizedRunRecord,
    QualificationCandidate,
    measure_candidate,
    qualify_winner,
    rank_screening,
    select_finalists,
)


def _baseline(*, split: str = "train", mean: float = 0.40, worst_case: float = 0.40) -> CandidateMeasurement:
    return _measurement(candidate_id="baseline", split=split, mean=mean, worst_case=worst_case)


def _measurement(
    *,
    candidate_id: str = "candidate",
    split: str = "validation",
    mean: float = 0.50,
    variance: float = 0.01,
    worst_case: float = 0.45,
    pass_at_3: float | None = 1.0,
    hard_safety_failures: int = 0,
    systemic_failures: int = 0,
    repetitions: int = 5,
    verification: float = 0.70,
    malformed_tool_calls: int = 0,
    unresolvable_tool_calls: int = 0,
) -> CandidateMeasurement:
    return CandidateMeasurement(
        candidate_id=candidate_id,
        split=split,
        mean_score=mean,
        score_variance=variance,
        worst_case_mean=worst_case,
        pass_at_3=pass_at_3,
        hard_safety_failures=hard_safety_failures,
        systemic_failures=systemic_failures,
        repetitions_per_case=repetitions,
        mean_verification=verification,
        malformed_tool_calls=malformed_tool_calls,
        unresolvable_tool_calls=unresolvable_tool_calls,
    )


def _qualification_candidate(
    *,
    candidate_id: str,
    validation_mean: float,
    milestone_mean: float,
    validation_worst_case: float = 0.52,
    milestone_worst_case: float = 0.62,
    validation_variance: float = 0.02,
    milestone_variance: float = 0.03,
    validation_pass_at_3: float | None = 1.0,
    milestone_pass_at_3: float | None = 1.0,
    validation_repetitions: int = 5,
    milestone_repetitions: int = 5,
    validation_hard_safety_failures: int = 0,
    milestone_hard_safety_failures: int = 0,
    validation_systemic_failures: int = 0,
    milestone_systemic_failures: int = 0,
) -> QualificationCandidate:
    return QualificationCandidate(
        candidate_id=candidate_id,
        baseline_validation=_measurement(candidate_id="baseline", split="validation", mean=0.40, worst_case=0.40),
        candidate_validation=_measurement(
            candidate_id=candidate_id,
            split="validation",
            mean=validation_mean,
            variance=validation_variance,
            worst_case=validation_worst_case,
            pass_at_3=validation_pass_at_3,
            repetitions=validation_repetitions,
            hard_safety_failures=validation_hard_safety_failures,
            systemic_failures=validation_systemic_failures,
        ),
        baseline_milestone=_measurement(candidate_id="baseline", split="milestone", mean=0.50, worst_case=0.50),
        candidate_milestone=_measurement(
            candidate_id=candidate_id,
            split="milestone",
            mean=milestone_mean,
            variance=milestone_variance,
            worst_case=milestone_worst_case,
            pass_at_3=milestone_pass_at_3,
            repetitions=milestone_repetitions,
            hard_safety_failures=milestone_hard_safety_failures,
            systemic_failures=milestone_systemic_failures,
        ),
    )


def _records_for_counts(
    *,
    candidate_id: str,
    split: str,
    counts_by_case: dict[str, int],
    score: float,
    verification: float = 1.0,
) -> list[NormalizedRunRecord]:
    records: list[NormalizedRunRecord] = []
    for case_id, count in counts_by_case.items():
        for repetition in range(1, count + 1):
            records.append(
                NormalizedRunRecord(
                    candidate_id=candidate_id,
                    split=split,
                    case_id=case_id,
                    repetition=repetition,
                    status="completed",
                    score=score,
                    verification=verification,
                    passed=True,
                )
            )
    return records


def test_screening_rejects_safety_and_systemic_failures() -> None:
    decision = rank_screening(
        _baseline(split="train"),
        [
            _measurement(candidate_id="safe-gain", split="train", mean=0.55, worst_case=0.48),
            _measurement(candidate_id="unsafe-gain", split="train", mean=0.90, hard_safety_failures=1),
            _measurement(candidate_id="systemic-gain", split="train", mean=0.90, systemic_failures=1),
        ],
    )

    assert [item.candidate_id for item in decision.survivors] == ["safe-gain"]
    assert [item.candidate_id for item in decision.rejections] == ["unsafe-gain", "systemic-gain"]
    assert decision.rejections[0].rejection_reasons == ("hard_safety_failures",)
    assert decision.rejections[1].rejection_reasons == ("systemic_failures",)


def test_measure_candidate_summarizes_normalized_runs() -> None:
    measurement = measure_candidate(
        [
            NormalizedRunRecord(
                candidate_id="candidate",
                split="validation",
                case_id="case-a",
                repetition=1,
                status="completed",
                score=0.70,
                verification=0.90,
                passed=True,
                malformed_tool_calls=1,
            ),
            NormalizedRunRecord(
                candidate_id="candidate",
                split="validation",
                case_id="case-a",
                repetition=2,
                status="completed",
                score=0.50,
                verification=0.70,
                passed=True,
            ),
            NormalizedRunRecord(
                candidate_id="candidate",
                split="validation",
                case_id="case-a",
                repetition=3,
                status="completed",
                score=0.40,
                verification=0.50,
                passed=False,
                unresolvable_tool_calls=2,
            ),
            NormalizedRunRecord(
                candidate_id="candidate",
                split="validation",
                case_id="case-b",
                repetition=1,
                status="model_failure",
                score=0.0,
                verification=0.0,
                passed=False,
            ),
            NormalizedRunRecord(
                candidate_id="candidate",
                split="validation",
                case_id="case-b",
                repetition=2,
                status="completed",
                score=0.60,
                verification=0.80,
                passed=True,
            ),
            NormalizedRunRecord(
                candidate_id="candidate",
                split="validation",
                case_id="case-b",
                repetition=3,
                status="system_failure",
                score=0.0,
                verification=0.0,
                passed=False,
            ),
        ]
    )

    assert measurement.candidate_id == "candidate"
    assert measurement.split == "validation"
    assert measurement.mean_score == pytest.approx(0.3666666667)
    assert measurement.score_variance == pytest.approx(0.0755555556)
    assert measurement.worst_case_mean == pytest.approx(0.2)
    assert measurement.pass_at_3 == pytest.approx(0.0)
    assert measurement.hard_safety_failures == 0
    assert measurement.systemic_failures == 1
    assert measurement.repetitions_per_case == 3
    assert measurement.mean_verification == pytest.approx(0.4833333333)
    assert measurement.malformed_tool_calls == 1
    assert measurement.unresolvable_tool_calls == 2


def test_measure_candidate_rejects_uneven_case_repetition_counts() -> None:
    with pytest.raises(ValueError, match=r"\[5, 6, 6\]"):
        measure_candidate(
            _records_for_counts(
                candidate_id="candidate",
                split="validation",
                counts_by_case={"case-a": 5, "case-b": 6, "case-c": 6},
                score=0.8,
            )
        )


def test_screening_ranks_by_worst_case_then_verification_then_problem_tool_calls() -> None:
    decision = rank_screening(
        _baseline(split="train"),
        [
            _measurement(
                candidate_id="best-worst-case",
                split="train",
                mean=0.55,
                worst_case=0.49,
                verification=0.60,
            ),
            _measurement(
                candidate_id="best-verification",
                split="train",
                mean=0.55,
                worst_case=0.47,
                verification=0.80,
            ),
            _measurement(
                candidate_id="fewest-problems",
                split="train",
                mean=0.55,
                worst_case=0.47,
                verification=0.75,
                malformed_tool_calls=0,
                unresolvable_tool_calls=0,
            ),
            _measurement(
                candidate_id="more-problems",
                split="train",
                mean=0.55,
                worst_case=0.47,
                verification=0.75,
                malformed_tool_calls=1,
                unresolvable_tool_calls=1,
            ),
        ],
        limit=4,
    )

    assert [item.candidate_id for item in decision.survivors] == [
        "best-worst-case",
        "best-verification",
        "fewest-problems",
        "more-problems",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hard_safety_failures", 1),
        ("systemic_failures", 1),
    ],
)
def test_screening_rejects_invalid_baseline(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="baseline must not have hard-safety or systemic failures"):
        rank_screening(
            _measurement(candidate_id="baseline", split="train", **{field: value}),
            [_measurement(candidate_id="safe-gain", split="train", mean=0.55, worst_case=0.48)],
        )


def test_select_finalists_filters_non_positive_mean_regressions_and_failures() -> None:
    decision = select_finalists(
        _baseline(split="validation"),
        [
            _measurement(candidate_id="qualified", split="validation", mean=0.55, worst_case=0.40),
            _measurement(candidate_id="flat", split="validation", mean=0.40, worst_case=0.40),
            _measurement(candidate_id="regressed", split="validation", mean=0.56, worst_case=0.39),
            _measurement(candidate_id="unsafe", split="validation", mean=0.58, hard_safety_failures=1),
            _measurement(candidate_id="systemic", split="validation", mean=0.58, systemic_failures=1),
        ],
    )

    assert [item.candidate_id for item in decision.survivors] == ["qualified"]
    assert {item.candidate_id: item.rejection_reasons for item in decision.rejections} == {
        "flat": ("mean_delta_not_positive",),
        "regressed": ("worst_case_regressed",),
        "unsafe": ("hard_safety_failures",),
        "systemic": ("systemic_failures",),
    }


def test_select_finalists_breaks_ties_by_lower_variance_then_higher_pass_at_3() -> None:
    decision = select_finalists(
        _baseline(split="validation"),
        [
            _measurement(
                candidate_id="lowest-variance",
                split="validation",
                mean=0.60,
                variance=0.01,
                pass_at_3=0.8,
            ),
            _measurement(
                candidate_id="highest-pass-at-3",
                split="validation",
                mean=0.60,
                variance=0.02,
                pass_at_3=1.0,
            ),
            _measurement(
                candidate_id="lower-pass-at-3",
                split="validation",
                mean=0.60,
                variance=0.02,
                pass_at_3=0.5,
            ),
        ],
        limit=3,
    )

    assert [item.candidate_id for item in decision.survivors] == [
        "lowest-variance",
        "highest-pass-at-3",
        "lower-pass-at-3",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hard_safety_failures", 1),
        ("systemic_failures", 1),
    ],
)
def test_select_finalists_rejects_invalid_baseline(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="baseline must not have hard-safety or systemic failures"):
        select_finalists(
            _measurement(candidate_id="baseline", split="validation", **{field: value}),
            [_measurement(candidate_id="qualified", split="validation", mean=0.55, worst_case=0.55)],
        )


def test_qualification_requires_both_split_deltas() -> None:
    decision = qualify_winner(
        baseline_validation=_measurement(candidate_id="baseline", split="validation", mean=0.40, worst_case=0.40),
        candidate_validation=_measurement(candidate_id="candidate", split="validation", mean=0.52, worst_case=0.52),
        baseline_milestone=_measurement(candidate_id="baseline", split="milestone", mean=0.50, worst_case=0.50),
        candidate_milestone=_measurement(candidate_id="candidate", split="milestone", mean=0.58, worst_case=0.58),
    )

    assert decision.status == "no_stable_winner"
    assert decision.candidate_id is None
    assert decision.reasons == ("milestone_delta_below_0_10",)


def test_qualification_rejects_worst_case_regression() -> None:
    decision = qualify_winner(
        baseline_validation=_measurement(candidate_id="baseline", split="validation", mean=0.40, worst_case=0.41),
        candidate_validation=_measurement(candidate_id="candidate", split="validation", mean=0.55, worst_case=0.40),
        baseline_milestone=_measurement(candidate_id="baseline", split="milestone", mean=0.50, worst_case=0.51),
        candidate_milestone=_measurement(candidate_id="candidate", split="milestone", mean=0.65, worst_case=0.50),
    )

    assert decision.status == "no_stable_winner"
    assert decision.reasons == (
        "validation_worst_case_regressed",
        "milestone_worst_case_regressed",
    )


def test_qualification_requires_five_repetitions_per_case_on_all_measurements() -> None:
    decision = qualify_winner(
        baseline_validation=_measurement(
            candidate_id="baseline",
            split="validation",
            mean=0.40,
            worst_case=0.40,
            repetitions=4,
        ),
        candidate_validation=_measurement(candidate_id="candidate", split="validation", mean=0.55, worst_case=0.55),
        baseline_milestone=_measurement(candidate_id="baseline", split="milestone", mean=0.50, worst_case=0.50),
        candidate_milestone=_measurement(
            candidate_id="candidate",
            split="milestone",
            mean=0.65,
            worst_case=0.65,
            repetitions=4,
        ),
    )

    assert decision.status == "no_stable_winner"
    assert decision.reasons == (
        "baseline_validation_repetitions_below_5",
        "candidate_milestone_repetitions_below_5",
    )


def test_qualification_rejects_more_than_five_repetitions_per_case() -> None:
    baseline_validation = measure_candidate(
        _records_for_counts(
            candidate_id="baseline",
            split="validation",
            counts_by_case={"case-a": 6, "case-b": 6},
            score=0.40,
        )
    )
    candidate_validation = measure_candidate(
        _records_for_counts(
            candidate_id="candidate",
            split="validation",
            counts_by_case={"case-a": 6, "case-b": 6},
            score=0.55,
        )
    )
    baseline_milestone = measure_candidate(
        _records_for_counts(
            candidate_id="baseline",
            split="milestone",
            counts_by_case={"case-a": 6, "case-b": 6},
            score=0.50,
        )
    )
    candidate_milestone = measure_candidate(
        _records_for_counts(
            candidate_id="candidate",
            split="milestone",
            counts_by_case={"case-a": 6, "case-b": 6},
            score=0.65,
        )
    )

    decision = qualify_winner(
        baseline_validation=baseline_validation,
        candidate_validation=candidate_validation,
        baseline_milestone=baseline_milestone,
        candidate_milestone=candidate_milestone,
    )

    assert decision.status == "no_stable_winner"
    assert decision.reasons == (
        "baseline_validation_repetitions_not_exactly_5",
        "candidate_validation_repetitions_not_exactly_5",
        "baseline_milestone_repetitions_not_exactly_5",
        "candidate_milestone_repetitions_not_exactly_5",
    )


def test_qualification_accepts_the_exact_point_ten_boundary() -> None:
    decision = qualify_winner(
        baseline_validation=_measurement(candidate_id="baseline", split="validation", mean=0.40, worst_case=0.40),
        candidate_validation=_measurement(candidate_id="candidate", split="validation", mean=0.50, worst_case=0.50),
        baseline_milestone=_measurement(candidate_id="baseline", split="milestone", mean=0.50, worst_case=0.50),
        candidate_milestone=_measurement(candidate_id="candidate", split="milestone", mean=0.60, worst_case=0.60),
    )

    assert decision.status == "promote"
    assert decision.candidate_id == "candidate"
    assert decision.reasons == ()


def test_qualification_collects_no_winner_reasons_for_all_finalists() -> None:
    decision = qualify_winner(
        [
            _qualification_candidate(candidate_id="delta-short", validation_mean=0.49, milestone_mean=0.62),
            _qualification_candidate(
                candidate_id="systemic",
                validation_mean=0.55,
                milestone_mean=0.65,
                milestone_systemic_failures=1,
            ),
        ]
    )

    assert decision.status == "no_stable_winner"
    assert decision.candidate_id is None
    assert decision.reasons == (
        "delta-short:validation_delta_below_0_10",
        "systemic:milestone_systemic_failures",
    )
    assert [assessment.candidate_id for assessment in decision.assessments] == [
        "delta-short",
        "systemic",
    ]


def test_qualification_promotes_the_first_qualified_finalist_in_rank_order() -> None:
    decision = qualify_winner(
        [
            _qualification_candidate(candidate_id="runner-up", validation_mean=0.49, milestone_mean=0.70),
            _qualification_candidate(candidate_id="winner", validation_mean=0.55, milestone_mean=0.65),
        ]
    )

    assert decision.status == "promote"
    assert decision.candidate_id == "winner"
    assert decision.reasons == ()
    assert decision.assessments[1].qualified is True
    assert decision.assessments[1].mean_score_delta_validation == pytest.approx(0.15)
