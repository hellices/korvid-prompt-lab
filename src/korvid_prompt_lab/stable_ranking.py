from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import pvariance
from typing import Literal

__all__ = [
    "CandidateMeasurement",
    "NormalizedRunRecord",
    "QualificationAssessment",
    "QualificationCandidate",
    "QualificationDecision",
    "RankedCandidate",
    "StageDecision",
    "measure_candidate",
    "qualify_winner",
    "rank_screening",
    "select_finalists",
]

_SCOREABLE_STATUSES = {"completed", "model_failure"}
_FLOAT_TOLERANCE = 1e-12


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_finite_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def _require_probability(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    probability = _require_finite_float(value, field_name)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return probability


def _require_limit(value: int, field_name: str) -> int:
    limit = _require_non_negative_int(value, field_name)
    if limit == 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return limit


def _meets_or_exceeds(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(value, threshold, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE)


@dataclass(frozen=True, slots=True)
class NormalizedRunRecord:
    candidate_id: str
    split: str
    case_id: str
    repetition: int
    status: str
    score: float
    verification: float
    passed: bool
    hard_safety_failures: int = 0
    malformed_tool_calls: int = 0
    unresolvable_tool_calls: int = 0

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.split, "split")
        _require_text(self.case_id, "case_id")
        _require_limit(self.repetition, "repetition")
        _require_text(self.status, "status")
        _require_finite_float(self.score, "score")
        verification = _require_probability(self.verification, "verification")
        if verification is None:  # pragma: no cover - verification is never optional
            raise ValueError("verification must be present")
        _require_non_negative_int(self.hard_safety_failures, "hard_safety_failures")
        _require_non_negative_int(self.malformed_tool_calls, "malformed_tool_calls")
        _require_non_negative_int(self.unresolvable_tool_calls, "unresolvable_tool_calls")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")  # noqa: TRY004 - preserve validation API


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
    mean_verification: float = 0.0
    malformed_tool_calls: int = 0
    unresolvable_tool_calls: int = 0

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.split, "split")
        _require_finite_float(self.mean_score, "mean_score")
        variance = _require_finite_float(self.score_variance, "score_variance")
        if variance < 0.0:
            raise ValueError("score_variance must be non-negative")
        _require_finite_float(self.worst_case_mean, "worst_case_mean")
        _require_probability(self.pass_at_3, "pass_at_3")
        _require_non_negative_int(self.hard_safety_failures, "hard_safety_failures")
        _require_non_negative_int(self.systemic_failures, "systemic_failures")
        _require_limit(self.repetitions_per_case, "repetitions_per_case")
        verification = _require_probability(self.mean_verification, "mean_verification")
        if verification is None:  # pragma: no cover - mean_verification is never optional
            raise ValueError("mean_verification must be present")
        _require_non_negative_int(self.malformed_tool_calls, "malformed_tool_calls")
        _require_non_negative_int(self.unresolvable_tool_calls, "unresolvable_tool_calls")

    @property
    def problem_tool_calls(self) -> int:
        return self.malformed_tool_calls + self.unresolvable_tool_calls


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: CandidateMeasurement
    baseline: CandidateMeasurement
    mean_score_delta: float
    worst_case_delta: float
    verification_delta: float
    problem_tool_calls: int
    rejection_reasons: tuple[str, ...] = ()

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def split(self) -> str:
        return self.candidate.split


@dataclass(frozen=True, slots=True)
class StageDecision:
    stage: str
    baseline: CandidateMeasurement
    rankings: tuple[RankedCandidate, ...]
    survivors: tuple[RankedCandidate, ...]
    rejections: tuple[RankedCandidate, ...]
    limit: int


@dataclass(frozen=True, slots=True)
class QualificationCandidate:
    candidate_id: str
    baseline_validation: CandidateMeasurement
    candidate_validation: CandidateMeasurement
    baseline_milestone: CandidateMeasurement
    candidate_milestone: CandidateMeasurement

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        if self.candidate_validation.candidate_id != self.candidate_id:
            raise ValueError("candidate_validation candidate_id must match candidate_id")
        if self.candidate_milestone.candidate_id != self.candidate_id:
            raise ValueError("candidate_milestone candidate_id must match candidate_id")
        if self.baseline_validation.split != "validation":
            raise ValueError("baseline_validation split must be 'validation'")
        if self.candidate_validation.split != "validation":
            raise ValueError("candidate_validation split must be 'validation'")
        if self.baseline_milestone.split != "milestone":
            raise ValueError("baseline_milestone split must be 'milestone'")
        if self.candidate_milestone.split != "milestone":
            raise ValueError("candidate_milestone split must be 'milestone'")


@dataclass(frozen=True, slots=True)
class QualificationAssessment:
    candidate_id: str
    mean_score_delta_validation: float
    mean_score_delta_milestone: float
    worst_case_delta_validation: float
    worst_case_delta_milestone: float
    combined_variance: float
    combined_pass_at_3: float | None
    qualified: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualificationDecision:
    status: Literal["promote", "no_stable_winner"]
    candidate_id: str | None
    reasons: tuple[str, ...]
    assessments: tuple[QualificationAssessment, ...] = ()


def measure_candidate(records: Iterable[NormalizedRunRecord]) -> CandidateMeasurement:
    normalized_records = tuple(records)
    if not normalized_records:
        raise ValueError("records must not be empty")

    first = normalized_records[0]
    candidate_id = first.candidate_id
    split = first.split
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_passes: dict[str, dict[int, bool]] = defaultdict(dict)
    scores: list[float] = []
    verifications: list[float] = []
    hard_safety_failures = 0
    systemic_failures = 0
    malformed_tool_calls = 0
    unresolvable_tool_calls = 0

    for record in normalized_records:
        if record.candidate_id != candidate_id:
            raise ValueError("records must describe exactly one candidate_id")
        if record.split != split:
            raise ValueError("records must describe exactly one split")
        if record.repetition in grouped_passes[record.case_id]:
            raise ValueError(
                f"duplicate repetition {record.repetition} for case {record.case_id}"
            )

        grouped_scores[record.case_id].append(record.score)
        grouped_passes[record.case_id][record.repetition] = record.passed
        scores.append(record.score)
        verifications.append(record.verification)
        hard_safety_failures += record.hard_safety_failures
        malformed_tool_calls += record.malformed_tool_calls
        unresolvable_tool_calls += record.unresolvable_tool_calls
        if record.status not in _SCOREABLE_STATUSES:
            systemic_failures += 1

    if not grouped_scores:
        raise ValueError("records must include at least one case")

    per_case_means = [sum(case_scores) / len(case_scores) for case_scores in grouped_scores.values()]
    repetitions_per_case = min(len(repetitions) for repetitions in grouped_passes.values())
    pass_at_3 = _pass_at_k(grouped_passes, 3)

    return CandidateMeasurement(
        candidate_id=candidate_id,
        split=split,
        mean_score=sum(scores) / len(scores),
        score_variance=0.0 if len(scores) == 1 else pvariance(scores),
        worst_case_mean=min(per_case_means),
        pass_at_3=pass_at_3,
        hard_safety_failures=hard_safety_failures,
        systemic_failures=systemic_failures,
        repetitions_per_case=repetitions_per_case,
        mean_verification=sum(verifications) / len(verifications),
        malformed_tool_calls=malformed_tool_calls,
        unresolvable_tool_calls=unresolvable_tool_calls,
    )


def rank_screening(
    baseline: CandidateMeasurement,
    candidates: Sequence[CandidateMeasurement],
    *,
    limit: int = 3,
    stage: str = "screening",
) -> StageDecision:
    if baseline.split == "milestone":
        raise ValueError("screening baseline must not use the milestone split")
    _require_limit(limit, "limit")

    eligible: list[RankedCandidate] = []
    rejected: list[RankedCandidate] = []
    for candidate in candidates:
        ranked = _paired_candidate(baseline, candidate)
        if ranked.rejection_reasons:
            rejected.append(ranked)
        else:
            eligible.append(ranked)

    ranked_eligible = sorted(
        eligible,
        key=lambda item: (
            -item.mean_score_delta,
            -item.worst_case_delta,
            -item.verification_delta,
            item.problem_tool_calls,
            item.candidate_id,
        ),
    )
    survivors = tuple(ranked_eligible[:limit])
    cutoff_rejections = tuple(
        _replace_rejection(item, "screening_rank_below_cutoff")
        for item in ranked_eligible[limit:]
    )
    all_rejections = tuple(rejected) + cutoff_rejections
    return StageDecision(
        stage=stage,
        baseline=baseline,
        rankings=tuple(survivors) + cutoff_rejections + tuple(rejected),
        survivors=survivors,
        rejections=all_rejections,
        limit=limit,
    )


def select_finalists(
    baseline: CandidateMeasurement,
    candidates: Sequence[CandidateMeasurement],
    *,
    limit: int = 2,
    stage: str = "validation",
) -> StageDecision:
    _require_limit(limit, "limit")

    eligible: list[RankedCandidate] = []
    rejected: list[RankedCandidate] = []
    for candidate in candidates:
        ranked = _paired_candidate(baseline, candidate)
        reasons = list(ranked.rejection_reasons)
        if ranked.mean_score_delta <= 0.0:
            reasons.append("mean_delta_not_positive")
        if ranked.worst_case_delta < 0.0:
            reasons.append("worst_case_regressed")
        if reasons:
            rejected.append(_replace_rejection(ranked, *reasons))
        else:
            eligible.append(ranked)

    ranked_eligible = sorted(
        eligible,
        key=lambda item: (
            -item.mean_score_delta,
            item.candidate.score_variance,
            -(item.candidate.pass_at_3 if item.candidate.pass_at_3 is not None else -1.0),
            item.candidate_id,
        ),
    )
    survivors = tuple(ranked_eligible[:limit])
    cutoff_rejections = tuple(
        _replace_rejection(item, "finalist_rank_below_cutoff")
        for item in ranked_eligible[limit:]
    )
    all_rejections = tuple(rejected) + cutoff_rejections
    return StageDecision(
        stage=stage,
        baseline=baseline,
        rankings=tuple(survivors) + cutoff_rejections + tuple(rejected),
        survivors=survivors,
        rejections=all_rejections,
        limit=limit,
    )


def qualify_winner(
    qualification: QualificationCandidate | Sequence[QualificationCandidate] | None = None,
    *,
    baseline_validation: CandidateMeasurement | None = None,
    candidate_validation: CandidateMeasurement | None = None,
    baseline_milestone: CandidateMeasurement | None = None,
    candidate_milestone: CandidateMeasurement | None = None,
    minimum_mean_delta: float = 0.10,
    required_repetitions: int = 5,
) -> QualificationDecision:
    threshold = _require_finite_float(minimum_mean_delta, "minimum_mean_delta")
    if threshold < 0.0:
        raise ValueError("minimum_mean_delta must be non-negative")
    repetition_floor = _require_limit(required_repetitions, "required_repetitions")

    candidates = _coerce_qualification_candidates(
        qualification=qualification,
        baseline_validation=baseline_validation,
        candidate_validation=candidate_validation,
        baseline_milestone=baseline_milestone,
        candidate_milestone=candidate_milestone,
    )
    if not candidates:
        return QualificationDecision(
            status="no_stable_winner",
            candidate_id=None,
            reasons=("no_finalists",),
            assessments=(),
        )

    assessments: list[QualificationAssessment] = []
    for finalist in candidates:
        assessment = _assess_candidate(
            finalist,
            minimum_mean_delta=threshold,
            required_repetitions=repetition_floor,
        )
        assessments.append(assessment)
        if assessment.qualified:
            return QualificationDecision(
                status="promote",
                candidate_id=assessment.candidate_id,
                reasons=(),
                assessments=tuple(assessments),
            )

    if len(assessments) == 1:
        reasons = assessments[0].reasons
    else:
        reasons = tuple(
            f"{assessment.candidate_id}:{reason}"
            for assessment in assessments
            for reason in assessment.reasons
        )
    return QualificationDecision(
        status="no_stable_winner",
        candidate_id=None,
        reasons=reasons,
        assessments=tuple(assessments),
    )


def _paired_candidate(baseline: CandidateMeasurement, candidate: CandidateMeasurement) -> RankedCandidate:
    if baseline.split != candidate.split:
        raise ValueError("baseline and candidate must describe the same split")
    if candidate.candidate_id == baseline.candidate_id:
        raise ValueError("candidate_id must differ from the baseline candidate_id")

    reasons: list[str] = []
    if candidate.hard_safety_failures > 0:
        reasons.append("hard_safety_failures")
    if candidate.systemic_failures > 0:
        reasons.append("systemic_failures")

    return RankedCandidate(
        candidate=candidate,
        baseline=baseline,
        mean_score_delta=candidate.mean_score - baseline.mean_score,
        worst_case_delta=candidate.worst_case_mean - baseline.worst_case_mean,
        verification_delta=candidate.mean_verification - baseline.mean_verification,
        problem_tool_calls=candidate.problem_tool_calls,
        rejection_reasons=tuple(reasons),
    )


def _replace_rejection(candidate: RankedCandidate, *reasons: str) -> RankedCandidate:
    unique_reasons = tuple(dict.fromkeys(candidate.rejection_reasons + tuple(reasons)))
    return RankedCandidate(
        candidate=candidate.candidate,
        baseline=candidate.baseline,
        mean_score_delta=candidate.mean_score_delta,
        worst_case_delta=candidate.worst_case_delta,
        verification_delta=candidate.verification_delta,
        problem_tool_calls=candidate.problem_tool_calls,
        rejection_reasons=unique_reasons,
    )


def _pass_at_k(grouped_passes: dict[str, dict[int, bool]], k: int) -> float | None:
    passed_groups = 0
    total_groups = 0
    for repetitions in grouped_passes.values():
        ordered = sorted(repetitions)
        if len(ordered) < k:
            return None
        total_groups += 1
        if all(repetitions[index] for index in ordered[:k]):
            passed_groups += 1
    if total_groups == 0:
        return None
    return passed_groups / total_groups


def _coerce_qualification_candidates(
    *,
    qualification: QualificationCandidate | Sequence[QualificationCandidate] | None,
    baseline_validation: CandidateMeasurement | None,
    candidate_validation: CandidateMeasurement | None,
    baseline_milestone: CandidateMeasurement | None,
    candidate_milestone: CandidateMeasurement | None,
) -> tuple[QualificationCandidate, ...]:
    if qualification is not None:
        if any(
            value is not None
            for value in (
                baseline_validation,
                candidate_validation,
                baseline_milestone,
                candidate_milestone,
            )
        ):
            raise ValueError("qualification positional input cannot be mixed with explicit measurements")
        if isinstance(qualification, QualificationCandidate):
            return (qualification,)
        return tuple(qualification)

    required = {
        "baseline_validation": baseline_validation,
        "candidate_validation": candidate_validation,
        "baseline_milestone": baseline_milestone,
        "candidate_milestone": candidate_milestone,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"missing qualification measurement(s): {', '.join(missing)}")

    assert baseline_validation is not None
    assert candidate_validation is not None
    assert baseline_milestone is not None
    assert candidate_milestone is not None
    return (
        QualificationCandidate(
            candidate_id=candidate_validation.candidate_id,
            baseline_validation=baseline_validation,
            candidate_validation=candidate_validation,
            baseline_milestone=baseline_milestone,
            candidate_milestone=candidate_milestone,
        ),
    )


def _assess_candidate(
    finalist: QualificationCandidate,
    *,
    minimum_mean_delta: float,
    required_repetitions: int,
) -> QualificationAssessment:
    validation_delta = finalist.candidate_validation.mean_score - finalist.baseline_validation.mean_score
    milestone_delta = finalist.candidate_milestone.mean_score - finalist.baseline_milestone.mean_score
    validation_worst_case_delta = (
        finalist.candidate_validation.worst_case_mean - finalist.baseline_validation.worst_case_mean
    )
    milestone_worst_case_delta = (
        finalist.candidate_milestone.worst_case_mean - finalist.baseline_milestone.worst_case_mean
    )
    reasons: list[str] = []

    for label, measurement in (
        ("baseline_validation", finalist.baseline_validation),
        ("candidate_validation", finalist.candidate_validation),
        ("baseline_milestone", finalist.baseline_milestone),
        ("candidate_milestone", finalist.candidate_milestone),
    ):
        if measurement.repetitions_per_case < required_repetitions:
            reasons.append(f"{label}_repetitions_below_{required_repetitions}")

    for label, measurement in (
        ("baseline_validation", finalist.baseline_validation),
        ("validation", finalist.candidate_validation),
        ("baseline_milestone", finalist.baseline_milestone),
        ("milestone", finalist.candidate_milestone),
    ):
        if measurement.hard_safety_failures > 0:
            reasons.append(f"{label}_hard_safety_failures")
        if measurement.systemic_failures > 0:
            reasons.append(f"{label}_systemic_failures")

    if not _meets_or_exceeds(validation_delta, minimum_mean_delta):
        reasons.append(f"validation_delta_below_{minimum_mean_delta:.2f}".replace(".", "_", 1))
    if not _meets_or_exceeds(milestone_delta, minimum_mean_delta):
        reasons.append(f"milestone_delta_below_{minimum_mean_delta:.2f}".replace(".", "_", 1))
    if validation_worst_case_delta < 0.0:
        reasons.append("validation_worst_case_regressed")
    if milestone_worst_case_delta < 0.0:
        reasons.append("milestone_worst_case_regressed")

    pass_values = [
        value
        for value in (
            finalist.candidate_validation.pass_at_3,
            finalist.candidate_milestone.pass_at_3,
        )
        if value is not None
    ]
    return QualificationAssessment(
        candidate_id=finalist.candidate_id,
        mean_score_delta_validation=validation_delta,
        mean_score_delta_milestone=milestone_delta,
        worst_case_delta_validation=validation_worst_case_delta,
        worst_case_delta_milestone=milestone_worst_case_delta,
        combined_variance=(
            finalist.candidate_validation.score_variance + finalist.candidate_milestone.score_variance
        ),
        combined_pass_at_3=None if len(pass_values) != 2 else sum(pass_values) / len(pass_values),
        qualified=not reasons,
        reasons=tuple(reasons),
    )
