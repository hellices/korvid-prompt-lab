from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.contracts import Candidate
from korvid_prompt_lab.stable_rollover import PriorFinalistEvidence
from korvid_prompt_lab.stable_rollover_candidates import (
    RolloverCandidateAxis,
    StructuredCandidate,
    build_rollover_candidates,
)
from korvid_prompt_lab.stable_scenarios import ScenarioAssignment, ScenarioClass

_REAL_V2_FINALIST_APPEND = (
    "name the observed evidence and its source before the final conclusion.\n"
    "when evidence is insufficient, state what is missing and stop instead of guessing."
)


def _baseline(*, system: str = "You are korvid's bounded Kubernetes operator.") -> Candidate:
    return Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "korvid-baseline-small",
            "components": {
                "system": system,
            },
            "metadata": {
                "korvid_version": "0.3.0",
                "profile": "small",
            },
        }
    )


def _assignment(scenario_id: str) -> ScenarioAssignment:
    return ScenarioAssignment(
        scenario_id=scenario_id,
        scenario_class=ScenarioClass.WORKLOAD_HEALTH,
        split="train",
        question_sha256=f"{scenario_id[:1]}".ljust(64, "1"),
        fixture_sha256=f"{scenario_id[:1]}".ljust(64, "2"),
        korvid_version="0.3.0",
    )


@dataclass(frozen=True, slots=True)
class _PriorProxy:
    summary_sha256: str
    consumed_assignments: tuple[ScenarioAssignment, ...]
    finalist: PriorFinalistEvidence

    @property
    def artifact_root(self) -> Path:  # pragma: no cover - access would fail the test
        raise AssertionError("artifact_root should not be accessed")

    @property
    def campaign_id(self) -> str:  # pragma: no cover - access would fail the test
        raise AssertionError("campaign_id should not be accessed")

    @property
    def korvid_version(self) -> str:  # pragma: no cover - access would fail the test
        raise AssertionError("korvid_version should not be accessed")

    @property
    def scenario_manifest_sha256(self) -> str:  # pragma: no cover - access would fail the test
        raise AssertionError("scenario_manifest_sha256 should not be accessed")

    @property
    def fresh_milestone_ids(self) -> tuple[str, ...]:  # pragma: no cover - access would fail the test
        raise AssertionError("fresh_milestone_ids should not be accessed")

    @property
    def scenario_ids(self) -> tuple[str, ...]:  # pragma: no cover - access would fail the test
        raise AssertionError("scenario_ids should not be accessed")

    @property
    def questions(self) -> tuple[str, ...]:  # pragma: no cover - access would fail the test
        raise AssertionError("questions should not be accessed")

    @property
    def fixtures(self) -> tuple[str, ...]:  # pragma: no cover - access would fail the test
        raise AssertionError("fixtures should not be accessed")


def _prior(
    *,
    consumed_assignments: tuple[ScenarioAssignment, ...],
    append: str = "tie each conclusion to observed evidence and avoid unsupported remediation.",
) -> _PriorProxy:
    finalist = PriorFinalistEvidence(
        candidate_id="finalist-v2",
        candidate_fingerprint="f" * 64,
        append=append,
        validation_delta=0.11,
        milestone_delta=0.09,
    )
    return _PriorProxy(
        summary_sha256="s" * 64,
        consumed_assignments=consumed_assignments,
        finalist=finalist,
    )


def test_rollover_matrix_matches_canonical_axes_and_text() -> None:
    baseline = _baseline()
    prior = _prior(
        consumed_assignments=(
            _assignment("used-a"),
            _assignment("used-b"),
        )
    )

    candidates = build_rollover_candidates(baseline, cast(Any, prior))

    assert len(candidates) == 8
    assert all(isinstance(item, StructuredCandidate) for item in candidates)
    assert [item.axes for item in candidates] == [
        (RolloverCandidateAxis.DECISIVE_READ_FIRST,),
        (RolloverCandidateAxis.CONTINUE_BEFORE_UNCERTAINTY,),
        (RolloverCandidateAxis.BOUNDED_UNCERTAINTY,),
        (RolloverCandidateAxis.EVIDENCE_LINKED_CONCLUSION,),
        (RolloverCandidateAxis.DECISIVE_READ_FIRST, RolloverCandidateAxis.CONTINUE_BEFORE_UNCERTAINTY),
        (RolloverCandidateAxis.DECISIVE_READ_FIRST, RolloverCandidateAxis.BOUNDED_UNCERTAINTY),
        (RolloverCandidateAxis.BOUNDED_UNCERTAINTY, RolloverCandidateAxis.EVIDENCE_LINKED_CONCLUSION),
        tuple(RolloverCandidateAxis),
    ]
    assert [item.candidate.candidate_id for item in candidates] == [
        "decisive-read-first",
        "continue-before-uncertainty",
        "bounded-uncertainty",
        "evidence-linked-conclusion",
        "decisive-read-first+continue-before-uncertainty",
        "decisive-read-first+bounded-uncertainty",
        "bounded-uncertainty+evidence-linked-conclusion",
        "decisive-read-first+continue-before-uncertainty+bounded-uncertainty+evidence-linked-conclusion",
    ]
    assert [item.candidate.components["system"] for item in candidates] == [baseline.components["system"]] * 8
    assert all(set(item.candidate.components) == {"system", "append"} for item in candidates)
    assert all(item.candidate.components["append"] == item.candidate.components["append"].strip() for item in candidates)
    assert all(len(item.candidate.components["append"]) <= 480 for item in candidates)
    assert all(
        item.candidate.metadata["rollover_from"] == prior.summary_sha256
        and item.candidate.metadata["prior_finalist_fingerprint"] == prior.finalist.candidate_fingerprint
        and item.candidate.metadata["korvid_version"] == "0.3.0"
        and item.candidate.metadata["profile"] == "small"
        for item in candidates
    )

    def _expected_append(*lines: str) -> str:
        return "\n".join((prior.finalist.append, *lines)) if lines else prior.finalist.append

    assert [item.candidate.components["append"] for item in candidates] == [
        _expected_append(
            "gather the smallest relevant read-only evidence needed to distinguish likely causes before concluding."
        ),
        _expected_append(
            "when initial evidence is insufficient, inspect the next highest-value source before stopping."
        ),
        _expected_append(
            "after relevant read-only evidence is exhausted, state exactly what remains unknown and stop instead of guessing."
        ),
        _expected_append(),
        _expected_append(
            "gather the smallest relevant read-only evidence needed to distinguish likely causes before concluding.",
            "when initial evidence is insufficient, inspect the next highest-value source before stopping.",
        ),
        _expected_append(
            "gather the smallest relevant read-only evidence needed to distinguish likely causes before concluding.",
            "after relevant read-only evidence is exhausted, state exactly what remains unknown and stop instead of guessing.",
        ),
        _expected_append(
            "after relevant read-only evidence is exhausted, state exactly what remains unknown and stop instead of guessing.",
        ),
        _expected_append(
            "gather the smallest relevant read-only evidence needed to distinguish likely causes before concluding.",
            "when initial evidence is insufficient, inspect the next highest-value source before stopping.",
            "after relevant read-only evidence is exhausted, state exactly what remains unknown and stop instead of guessing.",
        ),
    ]


def test_rollover_matrix_ignores_fresh_holdout_data_flow() -> None:
    baseline = _baseline(system="  exact installed prompt  ")
    prior = _prior(
        consumed_assignments=(
            _assignment("used-a"),
            _assignment("used-b"),
            _assignment("used-c"),
        )
    )

    candidates = build_rollover_candidates(baseline, cast(Any, prior))

    assert {item.candidate.components["system"] for item in candidates} == {"  exact installed prompt  "}
    assert candidates[0].candidate.metadata["rollover_from"] == prior.summary_sha256
    assert candidates[0].candidate.metadata["prior_finalist_fingerprint"] == prior.finalist.candidate_fingerprint


def test_rollover_matrix_is_deterministic_for_unrelated_prior_reordering() -> None:
    baseline = _baseline()
    prior = _prior(
        consumed_assignments=(
            _assignment("used-a"),
            _assignment("used-b"),
            _assignment("used-c"),
        )
    )
    reordered = _prior(
        consumed_assignments=tuple(reversed(prior.consumed_assignments)),
    )

    candidates = build_rollover_candidates(baseline, cast(Any, prior))
    repeated = build_rollover_candidates(baseline, cast(Any, reordered))

    assert [item.candidate.candidate_id for item in candidates] == [item.candidate.candidate_id for item in repeated]
    assert [item.candidate.fingerprint for item in candidates] == [item.candidate.fingerprint for item in repeated]


def test_rollover_matrix_uses_only_fixed_evidence_seed_from_real_v2_append() -> None:
    baseline = _baseline()
    prior = _prior(
        consumed_assignments=(
            _assignment("used-a"),
            _assignment("used-b"),
        ),
        append=_REAL_V2_FINALIST_APPEND,
    )

    candidates = build_rollover_candidates(baseline, cast(Any, prior))

    carried_seed = "name the observed evidence and its source before the final conclusion."
    dropped_line = "when evidence is insufficient, state what is missing and stop instead of guessing."

    assert len(candidates) == 8
    assert all(len(item.candidate.components["append"]) <= 480 for item in candidates)
    assert all(item.candidate.components["append"].splitlines()[0] == carried_seed for item in candidates)
    assert all(dropped_line not in item.candidate.components["append"] for item in candidates)
    assert candidates[-1].candidate.components["append"] == (
        "name the observed evidence and its source before the final conclusion.\n"
        "gather the smallest relevant read-only evidence needed to distinguish likely causes before concluding.\n"
        "when initial evidence is insufficient, inspect the next highest-value source before stopping.\n"
        "after relevant read-only evidence is exhausted, state exactly what remains unknown and stop instead of guessing.\n"
        "tie each conclusion to observed evidence and avoid unsupported remediation."
    )
