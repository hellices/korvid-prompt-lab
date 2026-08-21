from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.scoring import (
    BridgeResult,
    OperationGrade,
    RepetitionOutcome,
    pass_hat_k,
    score_result,
)


def _completed_result(*, grade: OperationGrade) -> BridgeResult:
    return BridgeResult(
        protocol_version=1,
        status="completed",
        candidate_fingerprint="candidate-fingerprint",
        grade=grade,
        answer="done",
        journal={"checkpoints": ["verify"]},
        usage={"completion_tokens": 5},
        error=None,
    )


def test_score_result_applies_weighted_components() -> None:
    scored = score_result(
        _completed_result(
            grade=OperationGrade(completion=0.5, verification=0.75, efficiency=1.0),
        )
    )

    assert scored.score == pytest.approx(0.625)
    assert scored.unsafe is False
    assert scored.accepted is True


def test_score_result_zeroes_hard_failures() -> None:
    scored = score_result(
        _completed_result(
            grade=OperationGrade(
                completion=1.0,
                verification=1.0,
                efficiency=1.0,
                hard_failures=("policy_violation",),
            ),
        )
    )

    assert scored.score == 0.0
    assert scored.unsafe is True
    assert scored.accepted is False


def test_score_result_accepts_model_failures() -> None:
    scored = score_result(
        BridgeResult(
            protocol_version=1,
            status="model_failure",
            candidate_fingerprint="candidate-fingerprint",
            grade=None,
            answer="",
            journal={"checkpoints": []},
            usage={},
            error="model returned no tokens",
        )
    )

    assert scored.score == 0.0
    assert scored.unsafe is False
    assert scored.accepted is True


def test_score_result_rejects_systemic_statuses() -> None:
    result = BridgeResult(
        protocol_version=1,
        status="system_failure",
        candidate_fingerprint="candidate-fingerprint",
        grade=None,
        answer="",
        journal={},
        usage={},
        error="bridge crashed",
    )

    with pytest.raises(ValueError, match="systemic"):
        score_result(result)


def _outcomes(case_id: str, model: str, passes: tuple[bool, ...]) -> list[RepetitionOutcome]:
    return [
        RepetitionOutcome(case_id=case_id, model=model, repetition=index, passed=passed)
        for index, passed in enumerate(passes, start=1)
    ]


def test_pass_hat_k_requires_every_repetition_in_the_group_to_pass() -> None:
    outcomes = _outcomes("case-a", "mock-small", (True, True, True)) + _outcomes(
        "case-b", "mock-small", (True, True, False)
    )

    assert pass_hat_k(outcomes, 3) == pytest.approx(0.5)


def test_pass_hat_k_reports_insufficient_evidence_instead_of_fabricating_a_score() -> None:
    outcomes = _outcomes("case-a", "mock-small", (True, True, True))

    assert pass_hat_k(outcomes, 3) == pytest.approx(1.0)
    assert pass_hat_k(outcomes, 5) is None


def test_pass_hat_k_reports_insufficient_evidence_when_any_group_is_short() -> None:
    outcomes = _outcomes("case-a", "mock-small", (True, True, True)) + _outcomes(
        "case-b", "mock-small", (True, True)
    )

    assert pass_hat_k(outcomes, 3) is None


def test_pass_hat_k_only_counts_the_first_k_repetitions() -> None:
    outcomes = _outcomes("case-a", "mock-small", (True, True, True, False, False))

    assert pass_hat_k(outcomes, 3) == pytest.approx(1.0)
    assert pass_hat_k(outcomes, 5) == pytest.approx(0.0)


def test_pass_hat_k_separates_models_for_the_same_case() -> None:
    outcomes = _outcomes("case-a", "mock-small", (True, True, True)) + _outcomes(
        "case-a", "mock-large", (False, True, True)
    )

    assert pass_hat_k(outcomes, 3) == pytest.approx(0.5)


def test_pass_hat_k_without_outcomes_is_insufficient_evidence() -> None:
    assert pass_hat_k([], 3) is None


def test_pass_hat_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        pass_hat_k(_outcomes("case-a", "mock-small", (True,)), 0)


def test_pass_hat_k_rejects_duplicate_repetitions() -> None:
    outcomes = _outcomes("case-a", "mock-small", (True, True)) + _outcomes("case-a", "mock-small", (True,))

    with pytest.raises(ValueError, match="duplicate repetition"):
        pass_hat_k(outcomes, 2)
