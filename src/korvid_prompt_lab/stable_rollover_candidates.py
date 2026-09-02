from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from .contracts import Candidate
from .stable_candidates import CandidateAxis, StructuredCandidate
from .stable_rollover import PriorCampaignEvidence

__all__ = ["RolloverCandidateAxis", "StructuredCandidate", "build_rollover_candidates"]


class RolloverCandidateAxis(StrEnum):
    DECISIVE_READ_FIRST = "decisive-read-first"
    CONTINUE_BEFORE_UNCERTAINTY = "continue-before-uncertainty"
    BOUNDED_UNCERTAINTY = "bounded-uncertainty"
    EVIDENCE_LINKED_CONCLUSION = "evidence-linked-conclusion"


@dataclass(frozen=True, slots=True)
class _RolloverCandidateSpec:
    axes: tuple[RolloverCandidateAxis, ...]


_AXIS_APPEND_TEXT: dict[RolloverCandidateAxis, str] = {
    RolloverCandidateAxis.DECISIVE_READ_FIRST: (
        "gather the smallest relevant read-only evidence needed to distinguish likely causes before concluding."
    ),
    RolloverCandidateAxis.CONTINUE_BEFORE_UNCERTAINTY: (
        "when initial evidence is insufficient, inspect the next highest-value source before stopping."
    ),
    RolloverCandidateAxis.BOUNDED_UNCERTAINTY: (
        "after relevant read-only evidence is exhausted, state exactly what remains unknown and stop instead of guessing."
    ),
    RolloverCandidateAxis.EVIDENCE_LINKED_CONCLUSION: (
        "tie each conclusion to observed evidence and avoid unsupported remediation."
    ),
}

_ROLLOVER_SEED_LINE = "name the observed evidence and its source before the final conclusion."

_MATRIX: tuple[_RolloverCandidateSpec, ...] = (
    _RolloverCandidateSpec((RolloverCandidateAxis.DECISIVE_READ_FIRST,)),
    _RolloverCandidateSpec((RolloverCandidateAxis.CONTINUE_BEFORE_UNCERTAINTY,)),
    _RolloverCandidateSpec((RolloverCandidateAxis.BOUNDED_UNCERTAINTY,)),
    _RolloverCandidateSpec((RolloverCandidateAxis.EVIDENCE_LINKED_CONCLUSION,)),
    _RolloverCandidateSpec(
        (
            RolloverCandidateAxis.DECISIVE_READ_FIRST,
            RolloverCandidateAxis.CONTINUE_BEFORE_UNCERTAINTY,
        )
    ),
    _RolloverCandidateSpec(
        (
            RolloverCandidateAxis.DECISIVE_READ_FIRST,
            RolloverCandidateAxis.BOUNDED_UNCERTAINTY,
        )
    ),
    _RolloverCandidateSpec(
        (
            RolloverCandidateAxis.BOUNDED_UNCERTAINTY,
            RolloverCandidateAxis.EVIDENCE_LINKED_CONCLUSION,
        )
    ),
    _RolloverCandidateSpec(tuple(RolloverCandidateAxis)),
)


def _candidate_id(axes: tuple[RolloverCandidateAxis, ...]) -> str:
    return "+".join(axis.value for axis in axes)


def _extract_rollover_seed(prior_append: str) -> str:
    if prior_append != prior_append.strip():
        raise ValueError("prior finalist append must use canonical outer whitespace")

    lines = tuple(line.strip() for line in prior_append.splitlines() if line.strip())
    if _ROLLOVER_SEED_LINE in lines:
        return _ROLLOVER_SEED_LINE
    if len(lines) == 1:
        return lines[0]
    raise ValueError("prior finalist append must contain the rollover seed line")


def _render_append(prior_seed: str, axes: tuple[RolloverCandidateAxis, ...]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in (prior_seed, *(_AXIS_APPEND_TEXT[axis] for axis in axes)):
        line = raw_line.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    append = "\n".join(lines)
    if append != append.strip():
        raise ValueError("append text must use canonical outer whitespace")
    if len(append) > 480:
        raise ValueError("append text must be at most 480 characters")
    return append


def build_rollover_candidates(
    baseline: Candidate,
    prior: PriorCampaignEvidence,
) -> tuple[StructuredCandidate, ...]:
    if set(baseline.components) != {"system"}:
        raise ValueError("baseline components must be exactly {'system'}")

    system = baseline.components["system"]
    prior_seed = _extract_rollover_seed(prior.finalist.append)
    metadata = {
        **baseline.metadata,
        "rollover_from": prior.summary_sha256,
        "prior_finalist_fingerprint": prior.finalist.candidate_fingerprint,
    }

    candidates: list[StructuredCandidate] = []
    for spec in _MATRIX:
        candidate = Candidate.from_mapping(
            {
                "schema_version": 1,
                "candidate_id": _candidate_id(spec.axes),
                "components": {
                    "system": system,
                    "append": _render_append(prior_seed, spec.axes),
                },
                "metadata": metadata,
            }
        )
        candidates.append(StructuredCandidate(axes=cast(tuple[CandidateAxis, ...], spec.axes), candidate=candidate))
    return tuple(candidates)
