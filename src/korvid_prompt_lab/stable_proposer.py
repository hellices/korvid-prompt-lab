from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import dspy  # type: ignore[import-untyped]

from .reflection import canonicalize_proposal_text
from .stable_candidates import CandidateAxis
from .stable_ranking import CandidateMeasurement, QualificationCandidate

__all__ = [
    "BoundedAppendProposalRequest",
    "BoundedAppendProposalSignature",
    "BoundedAppendProposer",
    "build_proposal_request",
]

_MEASUREMENT_FIELD_NAMES = (
    "candidate_id",
    "split",
    "mean_score",
    "score_variance",
    "worst_case_mean",
    "pass_at_3",
    "hard_safety_failures",
    "systemic_failures",
    "repetitions_per_case",
    "per_case_repetition_counts",
    "mean_verification",
    "malformed_tool_calls",
    "unresolvable_tool_calls",
)


@dataclass(frozen=True, slots=True)
class BoundedAppendProposalRequest:
    finalist_append: str
    failure_axis: CandidateAxis
    bounded_feedback: dict[str, Any]


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
        prediction = self._predictor_or_raise()(
            current_append=request.finalist_append,
            failure_axis=request.failure_axis.value,
            bounded_feedback_json=json.dumps(
                request.bounded_feedback, ensure_ascii=False, sort_keys=True
            ),
            lm=self.reflection_lm,
        )
        revised = getattr(prediction, "revised_append", None)
        return canonicalize_proposal_text(revised, context="revised_append")

    def safe_propose(self, request_or_context: Any, **kwargs: Any) -> str | None:
        try:
            request = (
                request_or_context
                if isinstance(request_or_context, BoundedAppendProposalRequest)
                else build_proposal_request(request_or_context, **kwargs)
            )
            return self.propose(request)
        except (ValueError, TimeoutError, subprocess.TimeoutExpired):
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
        finalist_append=canonicalize_proposal_text(append_text, context="finalist_append"),
        failure_axis=_coerce_axis(axis_value),
        bounded_feedback=_bounded_feedback_payload(feedback_source),
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


def _bounded_feedback_payload(source: Any) -> dict[str, Any]:
    if isinstance(source, BoundedAppendProposalRequest):
        return _bounded_feedback_payload(source.bounded_feedback)
    if isinstance(source, CandidateMeasurement):
        return _measurement_payload(source)
    if isinstance(source, QualificationCandidate):
        return _qualification_payload(source)
    if isinstance(source, Mapping):
        if "candidate_validation" in source:
            payload_mapping: dict[str, Any] = {
                "candidate_validation": _bounded_feedback_payload(source["candidate_validation"])
            }
            if "baseline_validation" in source:
                payload_mapping["baseline_validation"] = _bounded_feedback_payload(source["baseline_validation"])
            if "baseline_milestone" in source:
                payload_mapping["baseline_milestone"] = _bounded_feedback_payload(source["baseline_milestone"])
            if "candidate_milestone" in source:
                payload_mapping["candidate_milestone"] = _bounded_feedback_payload(source["candidate_milestone"])
            if "candidate_id" in source:
                payload_mapping["candidate_id"] = _require_text(source["candidate_id"], "candidate_id")
            return payload_mapping
        if any(key in source for key in _MEASUREMENT_FIELD_NAMES):
            return _filter_measurement_mapping(source)
        if "bounded_feedback" in source:
            return _bounded_feedback_payload(source["bounded_feedback"])
    if hasattr(source, "candidate_validation"):
        payload_object: dict[str, Any] = {
            "candidate_validation": _bounded_feedback_payload(source.candidate_validation)
        }
        for field_name in ("baseline_validation", "baseline_milestone", "candidate_milestone", "candidate_id"):
            value = getattr(source, field_name, _MISSING)
            if value is _MISSING:
                continue
            if field_name == "candidate_id":
                payload_object[field_name] = _require_text(value, field_name)
            else:
                payload_object[field_name] = _bounded_feedback_payload(value)
        return payload_object
    if any(hasattr(source, field_name) for field_name in _MEASUREMENT_FIELD_NAMES):
        return _filter_measurement_object(source)
    if hasattr(source, "bounded_feedback"):
        return _bounded_feedback_payload(source.bounded_feedback)
    raise ValueError("bounded feedback must be a candidate measurement or qualification candidate")


def _qualification_payload(candidate: QualificationCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "baseline_validation": _measurement_payload(candidate.baseline_validation),
        "candidate_validation": _measurement_payload(candidate.candidate_validation),
        "baseline_milestone": _measurement_payload(candidate.baseline_milestone),
        "candidate_milestone": _measurement_payload(candidate.candidate_milestone),
    }


def _measurement_payload(measurement: CandidateMeasurement) -> dict[str, Any]:
    return {field_name: getattr(measurement, field_name) for field_name in _MEASUREMENT_FIELD_NAMES}


def _filter_measurement_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in _MEASUREMENT_FIELD_NAMES:
        if field_name in mapping:
            payload[field_name] = mapping[field_name]
    if not payload:
        raise ValueError("bounded feedback must include a known aggregate measurement")
    return payload


def _filter_measurement_object(source: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in _MEASUREMENT_FIELD_NAMES:
        value = getattr(source, field_name, _MISSING)
        if value is not _MISSING:
            payload[field_name] = value
    if not payload:
        raise ValueError("bounded feedback must include a known aggregate measurement")
    return payload


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


_MISSING = object()
