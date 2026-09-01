from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.contracts import Candidate
from korvid_prompt_lab.stable_candidates import (
    CandidateAxis,
    StructuredCandidate,
    build_structured_candidates,
)


def _baseline(*, system: str = "You are korvid's bounded Kubernetes operator.") -> Candidate:
    return Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "korvid-baseline-small",
            "components": {
                "system": system,
                "append": "Verify the postcondition before reporting completion.",
            },
        }
    )


def test_structured_matrix_has_eight_unique_append_candidates() -> None:
    candidates = build_structured_candidates(_baseline())

    assert len(candidates) == 8
    assert all(isinstance(item, StructuredCandidate) for item in candidates)
    assert [item.axes for item in candidates] == [
        (CandidateAxis.EVIDENCE_FIRST,),
        (CandidateAxis.ONE_TOOL_AT_A_TIME,),
        (CandidateAxis.CITE_BEFORE_CONCLUSION,),
        (CandidateAxis.STOP_WITH_UNCERTAINTY,),
        (CandidateAxis.EVIDENCE_FIRST, CandidateAxis.ONE_TOOL_AT_A_TIME),
        (CandidateAxis.EVIDENCE_FIRST, CandidateAxis.CITE_BEFORE_CONCLUSION),
        (CandidateAxis.CITE_BEFORE_CONCLUSION, CandidateAxis.STOP_WITH_UNCERTAINTY),
        tuple(CandidateAxis),
    ]
    assert len({item.candidate.fingerprint for item in candidates}) == 8
    assert all(set(item.candidate.components) == {"system", "append"} for item in candidates)
    assert all(len(item.candidate.components["append"]) <= 480 for item in candidates)
    assert [item.candidate.candidate_id for item in candidates] == [
        "evidence-first",
        "one-tool-at-a-time",
        "cite-before-conclusion",
        "stop-with-uncertainty",
        "evidence-first+one-tool-at-a-time",
        "evidence-first+cite-before-conclusion",
        "cite-before-conclusion+stop-with-uncertainty",
        "evidence-first+one-tool-at-a-time+cite-before-conclusion+stop-with-uncertainty",
    ]
    assert [item.candidate.components["append"] for item in candidates] == [
        "inspect runtime evidence before stating a diagnosis.",
        "choose the single highest-value read tool, inspect its result, then decide the next step.",
        "name the observed evidence and its source before the final conclusion.",
        "when evidence is insufficient, state what is missing and stop instead of guessing.",
        (
            "inspect runtime evidence before stating a diagnosis.\n"
            "choose the single highest-value read tool, inspect its result, then decide the next step."
        ),
        (
            "inspect runtime evidence before stating a diagnosis.\n"
            "name the observed evidence and its source before the final conclusion."
        ),
        (
            "name the observed evidence and its source before the final conclusion.\n"
            "when evidence is insufficient, state what is missing and stop instead of guessing."
        ),
        (
            "inspect runtime evidence before stating a diagnosis.\n"
            "choose the single highest-value read tool, inspect its result, then decide the next step.\n"
            "name the observed evidence and its source before the final conclusion.\n"
            "when evidence is insufficient, state what is missing and stop instead of guessing."
        ),
    ]


def test_matrix_preserves_exact_baseline_system_prompt() -> None:
    baseline = _baseline(system="  exact installed prompt  ")

    assert {
        item.candidate.components["system"] for item in build_structured_candidates(baseline)
    } == {"  exact installed prompt  "}
