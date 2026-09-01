from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .contracts import Candidate

__all__ = ["CandidateAxis", "StructuredCandidate", "build_structured_candidates"]


class CandidateAxis(StrEnum):
    EVIDENCE_FIRST = "evidence-first"
    ONE_TOOL_AT_A_TIME = "one-tool-at-a-time"
    CITE_BEFORE_CONCLUSION = "cite-before-conclusion"
    STOP_WITH_UNCERTAINTY = "stop-with-uncertainty"


@dataclass(frozen=True, slots=True)
class StructuredCandidate:
    axes: tuple[CandidateAxis, ...]
    candidate: Candidate


_APPEND_TEXT_BY_AXIS: dict[CandidateAxis, str] = {
    CandidateAxis.EVIDENCE_FIRST: "inspect runtime evidence before stating a diagnosis.",
    CandidateAxis.ONE_TOOL_AT_A_TIME: (
        "choose the single highest-value read tool, inspect its result, then decide the next step."
    ),
    CandidateAxis.CITE_BEFORE_CONCLUSION: (
        "name the observed evidence and its source before the final conclusion."
    ),
    CandidateAxis.STOP_WITH_UNCERTAINTY: (
        "when evidence is insufficient, state what is missing and stop instead of guessing."
    ),
}

_MATRIX: tuple[tuple[CandidateAxis, ...], ...] = (
    (CandidateAxis.EVIDENCE_FIRST,),
    (CandidateAxis.ONE_TOOL_AT_A_TIME,),
    (CandidateAxis.CITE_BEFORE_CONCLUSION,),
    (CandidateAxis.STOP_WITH_UNCERTAINTY,),
    (CandidateAxis.EVIDENCE_FIRST, CandidateAxis.ONE_TOOL_AT_A_TIME),
    (CandidateAxis.EVIDENCE_FIRST, CandidateAxis.CITE_BEFORE_CONCLUSION),
    (CandidateAxis.CITE_BEFORE_CONCLUSION, CandidateAxis.STOP_WITH_UNCERTAINTY),
    tuple(CandidateAxis),
)


def _candidate_id(axes: tuple[CandidateAxis, ...]) -> str:
    return "+".join(axis.value for axis in axes)


def _render_append(axes: tuple[CandidateAxis, ...]) -> str:
    append = "\n".join(_APPEND_TEXT_BY_AXIS[axis] for axis in axes)
    if append != append.strip():
        raise ValueError("append text must use canonical outer whitespace")
    if not append:
        raise ValueError("append text must not be empty")
    if len(append) > 480:
        raise ValueError("append text must be at most 480 characters")
    return append


def build_structured_candidates(baseline: Candidate) -> tuple[StructuredCandidate, ...]:
    if set(baseline.components) != {"system"}:
        raise ValueError("baseline components must be exactly {'system'}")
    system = baseline.components["system"]

    candidates: list[StructuredCandidate] = []
    for axes in _MATRIX:
        candidate = Candidate.from_mapping(
            {
                "schema_version": 1,
                "candidate_id": _candidate_id(axes),
                "components": {
                    "system": system,
                    "append": _render_append(axes),
                },
                "metadata": baseline.metadata,
            }
        )
        candidates.append(StructuredCandidate(axes=axes, candidate=candidate))
    return tuple(candidates)
