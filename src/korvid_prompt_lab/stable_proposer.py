from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import dspy  # type: ignore[import-untyped]
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.exceptions import (
    Timeout as LiteLLMTimeout,
)

from .reflection import canonicalize_proposal_text
from .stable_candidates import CandidateAxis
from .stable_ranking import CandidateMeasurement, QualificationCandidate

__all__ = [
    "BoundedAggregateFeedback",
    "BoundedAppendProposalRequest",
    "BoundedAppendProposalSignature",
    "BoundedAppendProposer",
    "build_proposal_request",
]

_MAX_BOUNDED_APPEND_LENGTH = 480
_SAFE_PROPOSE_FAILURES = (
    ValueError,
    TimeoutError,
    ConnectionError,
    subprocess.TimeoutExpired,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    LiteLLMTimeout,
)
_AGGREGATE_FIELD_NAMES = (
    "mean_score",
    "score_variance",
    "worst_case_mean",
    "pass_at_3",
    "hard_safety_failures",
    "systemic_failures",
    "repetitions_per_case",
    "mean_verification",
    "malformed_tool_calls",
    "unresolvable_tool_calls",
)


@dataclass(frozen=True, slots=True)
class BoundedAggregateFeedback:
    mean_score: float
    score_variance: float
    worst_case_mean: float
    pass_at_3: float | None
    hard_safety_failures: int
    systemic_failures: int
    repetitions_per_case: int
    mean_verification: float
    malformed_tool_calls: int
    unresolvable_tool_calls: int

    def __post_init__(self) -> None:
        _validate_bounded_feedback(self)


@dataclass(frozen=True, slots=True)
class BoundedAppendProposalRequest:
    finalist_append: str
    failure_axis: CandidateAxis
    bounded_feedback: BoundedAggregateFeedback

    def __post_init__(self) -> None:
        _validate_bounded_append_request(self)


class BoundedAppendProposalSignature(dspy.Signature):
    current_append: str = dspy.InputField(desc="Canonical finalist append text.")
    failure_axis: str = dspy.InputField(desc="One known failure axis from the structured signal.")
    bounded_feedback_json: str = dspy.InputField(
        desc="Compact bounded aggregate feedback encoded as JSON."
    )
    revised_append: str = dspy.OutputField(
        desc="Canonical prompt append, at most 480 characters."
    )


class BoundedAppendProposer:
    def __init__(self, reflection_lm: object) -> None:
        self.reflection_lm = reflection_lm
        self._predictor: dspy.Predict | None = None

    def propose(self, request: BoundedAppendProposalRequest) -> str:
        _validate_bounded_append_request(request)
        prediction = self._predictor_or_raise()(
            current_append=request.finalist_append,
            failure_axis=request.failure_axis.value,
            bounded_feedback_json=json.dumps(asdict(request.bounded_feedback), ensure_ascii=False, sort_keys=True),
            lm=self.reflection_lm,
        )
        revised = getattr(prediction, "revised_append", None)
        normalized = canonicalize_proposal_text(revised, context="revised_append")
        if len(normalized) > _MAX_BOUNDED_APPEND_LENGTH:
            raise ValueError(f"revised_append must be at most {_MAX_BOUNDED_APPEND_LENGTH} characters")
        return normalized

    def safe_propose(self, request_or_context: Any, **kwargs: Any) -> str | None:
        try:
            request = (
                request_or_context
                if isinstance(request_or_context, BoundedAppendProposalRequest)
                else build_proposal_request(request_or_context, **kwargs)
            )
            return self.propose(request)
        except _SAFE_PROPOSE_FAILURES:
            return None

    def _predictor_or_raise(self) -> dspy.Predict:
        if self._predictor is None:
            self._predictor = dspy.Predict(BoundedAppendProposalSignature)
        return self._predictor


def build_proposal_request(
    context: Any,
    *,
    finalist_append: str | None = None,
    failure_axis: CandidateAxis | str | None = None,
    bounded_feedback: Any | None = None,
) -> BoundedAppendProposalRequest:
    append_text = finalist_append if finalist_append is not None else _lookup_text(
        context, ("finalist_append", "append")
    )
    axis_value = failure_axis if failure_axis is not None else _lookup_axis(context)
    feedback_source = bounded_feedback if bounded_feedback is not None else context
    return BoundedAppendProposalRequest(
        finalist_append=_require_canonical_text(append_text, "finalist_append"),
        failure_axis=_coerce_axis(axis_value),
        bounded_feedback=_coerce_bounded_feedback(feedback_source),
    )


def _coerce_axis(value: CandidateAxis | str | Any) -> CandidateAxis:
    if isinstance(value, CandidateAxis):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("failure_axis must be a known failure axis")
    try:
        return CandidateAxis(value)
    except ValueError as exc:
        raise ValueError(f"unknown failure_axis: {value}") from exc


def _lookup_axis(context: Any) -> CandidateAxis | str:
    for field_name in ("failure_axis", "axis", "known_failure_axis"):
        value = _lookup_optional(context, field_name)
        if value is not _MISSING:
            return value
    raise ValueError("failure_axis must be provided")


def _lookup_text(context: Any, field_names: tuple[str, ...]) -> str:
    for field_name in field_names:
        value = _lookup_optional(context, field_name)
        if value is not _MISSING:
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")
            return value
    raise ValueError(f"missing required field: {field_names[0]}")


def _lookup_optional(context: Any, field_name: str) -> Any:
    if isinstance(context, Mapping):
        return context.get(field_name, _MISSING)
    return getattr(context, field_name, _MISSING)


def _bounded_feedback_payload(source: Any) -> BoundedAggregateFeedback:
    if isinstance(source, BoundedAggregateFeedback):
        return source
    if isinstance(source, BoundedAppendProposalRequest):
        return _bounded_feedback_payload(source.bounded_feedback)
    if isinstance(source, CandidateMeasurement):
        return _measurement_payload(source)
    if isinstance(source, QualificationCandidate):
        return _bounded_feedback_payload(source.candidate_validation)
    if isinstance(source, Mapping):
        if "candidate_validation" in source:
            return _bounded_feedback_payload(source["candidate_validation"])
        if any(key in source for key in _AGGREGATE_FIELD_NAMES):
            return _filter_measurement_mapping(source)
        if "bounded_feedback" in source:
            return _bounded_feedback_payload(source["bounded_feedback"])
    if hasattr(source, "candidate_validation"):
        return _bounded_feedback_payload(source.candidate_validation)
    if any(hasattr(source, field_name) for field_name in _AGGREGATE_FIELD_NAMES):
        return _filter_measurement_object(source)
    if hasattr(source, "bounded_feedback"):
        return _bounded_feedback_payload(source.bounded_feedback)
    raise ValueError("bounded feedback must be a candidate measurement or qualification candidate")


def _coerce_bounded_feedback(source: Any) -> BoundedAggregateFeedback:
    try:
        return _bounded_feedback_payload(source)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounded_feedback must be a valid bounded aggregate feedback") from exc


def _measurement_payload(measurement: CandidateMeasurement) -> BoundedAggregateFeedback:
    return BoundedAggregateFeedback(
        mean_score=measurement.mean_score,
        score_variance=measurement.score_variance,
        worst_case_mean=measurement.worst_case_mean,
        pass_at_3=measurement.pass_at_3,
        hard_safety_failures=measurement.hard_safety_failures,
        systemic_failures=measurement.systemic_failures,
        repetitions_per_case=measurement.repetitions_per_case,
        mean_verification=measurement.mean_verification,
        malformed_tool_calls=measurement.malformed_tool_calls,
        unresolvable_tool_calls=measurement.unresolvable_tool_calls,
    )


def _filter_measurement_mapping(mapping: Mapping[str, Any]) -> BoundedAggregateFeedback:
    payload = {field_name: mapping[field_name] for field_name in _AGGREGATE_FIELD_NAMES if field_name in mapping}
    if len(payload) != len(_AGGREGATE_FIELD_NAMES):
        raise ValueError("bounded feedback must include a known aggregate measurement")
    return BoundedAggregateFeedback(**payload)


def _filter_measurement_object(source: Any) -> BoundedAggregateFeedback:
    payload: dict[str, Any] = {}
    for field_name in _AGGREGATE_FIELD_NAMES:
        value = getattr(source, field_name, _MISSING)
        if value is not _MISSING:
            payload[field_name] = value
    if len(payload) != len(_AGGREGATE_FIELD_NAMES):
        raise ValueError("bounded feedback must include a known aggregate measurement")
    return BoundedAggregateFeedback(**payload)


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_canonical_text(value: Any, context: str) -> str:
    text = _require_text(value, context)
    if text != text.strip():
        raise ValueError(f"{context} must use canonical outer whitespace")
    return text


def _require_finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{context} must be a finite number")
    return normalized


def _require_non_negative_float(value: Any, context: str) -> float:
    normalized = _require_finite_float(value, context)
    if normalized < 0.0:
        raise ValueError(f"{context} must be a non-negative number")
    return normalized


def _require_probability(value: Any, context: str) -> float | None:
    if value is None:
        return None
    normalized = _require_finite_float(value, context)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{context} must be between 0.0 and 1.0")
    return normalized


def _require_non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, context: str) -> int:
    normalized = _require_non_negative_int(value, context)
    if normalized == 0:
        raise ValueError(f"{context} must be a positive integer")
    return normalized


def _validate_bounded_feedback(feedback: BoundedAggregateFeedback) -> BoundedAggregateFeedback:
    if type(feedback) is not BoundedAggregateFeedback:
        raise TypeError("bounded_feedback must be a BoundedAggregateFeedback instance")
    _require_finite_float(feedback.mean_score, "mean_score")
    _require_non_negative_float(feedback.score_variance, "score_variance")
    _require_finite_float(feedback.worst_case_mean, "worst_case_mean")
    _require_probability(feedback.pass_at_3, "pass_at_3")
    _require_non_negative_int(feedback.hard_safety_failures, "hard_safety_failures")
    _require_non_negative_int(feedback.systemic_failures, "systemic_failures")
    _require_positive_int(feedback.repetitions_per_case, "repetitions_per_case")
    if _require_probability(feedback.mean_verification, "mean_verification") is None:
        raise TypeError("mean_verification must be a probability")
    _require_non_negative_int(feedback.malformed_tool_calls, "malformed_tool_calls")
    _require_non_negative_int(feedback.unresolvable_tool_calls, "unresolvable_tool_calls")
    return feedback


def _validate_bounded_append_request(request: BoundedAppendProposalRequest) -> BoundedAggregateFeedback:
    if type(request) is not BoundedAppendProposalRequest:
        raise ValueError("request must be a BoundedAppendProposalRequest instance")
    _require_canonical_text(request.finalist_append, "finalist_append")
    if not isinstance(request.failure_axis, CandidateAxis) or type(request.failure_axis) is not CandidateAxis:
        raise ValueError("failure_axis must be a known failure axis")
    if type(request.bounded_feedback) is not BoundedAggregateFeedback:
        raise ValueError("bounded_feedback must be a BoundedAggregateFeedback instance")
    return _validate_bounded_feedback(request.bounded_feedback)


_MISSING = object()
