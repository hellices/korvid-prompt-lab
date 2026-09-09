from __future__ import annotations

import pytest

from korvid_prompt_lab import scoring
from korvid_prompt_lab.scoring import (
    EvaluationResult,
    EvaluationScore,
    RepetitionOutcome,
    is_strictly_better,
    pass_hat_k,
    passes_qualification_gate,
)


def test_legacy_grade_and_result_apis_are_absent() -> None:
    for name in (
        "OperationGrade",
        "BridgeResult",
        "ScoredResult",
        "grade_quality",
        "score_result",
        "result_passed",
    ):
        assert not hasattr(scoring, name)


def test_evaluation_result_carries_the_original_verdict_and_evidence() -> None:
    result = EvaluationResult(
        success=False,
        execution_mode="live",
        candidate_fingerprint="a" * 64,
        feedback={
            "upstream": {"outcome": "failure", "grade": {"diagnosis_success": False}},
            "source_report": {"scenario": "image-pull-typo"},
        },
        usage={"iterations": 2, "tool_calls": 1},
    )

    assert result.success is False
    assert result.execution_mode == "live"
    assert result.candidate_fingerprint == "a" * 64
    assert result.hard_failures == ()
    assert result.feedback["upstream"]["outcome"] == "failure"
    assert result.usage == {"iterations": 2, "tool_calls": 1}
    for legacy in ("grade", "journal", "answer", "protocol_version", "status", "error"):
        assert not hasattr(result, legacy)


def test_evaluation_result_defensively_copies_evidence_mappings() -> None:
    feedback = {"success": True}
    usage = {"iterations": 1}
    result = EvaluationResult(
        success=True,
        execution_mode="scripted",
        candidate_fingerprint="b" * 64,
        feedback=feedback,
        usage=usage,
    )
    feedback["success"] = False
    usage["iterations"] = 99

    assert result.feedback == {"success": True}
    assert result.usage == {"iterations": 1}
    returned = result.feedback
    returned["success"] = False
    assert result.feedback == {"success": True}


def test_success_with_hard_failures_fails_closed() -> None:
    with pytest.raises(ValueError, match="success.*hard_failures"):
        EvaluationResult(
            success=True,
            execution_mode="live",
            candidate_fingerprint="c" * 64,
            feedback={"original_success": True},
            usage={},
            hard_failures=("upstream_safety_violation",),
        )


def test_evaluation_result_preserves_a_failed_source_verdict_with_hard_failures() -> None:
    result = EvaluationResult(
        success=False,
        execution_mode="live",
        candidate_fingerprint="c" * 64,
        feedback={"original_success": False},
        usage={},
        hard_failures=("upstream_safety_violation",),
    )

    assert result.success is False
    assert result.hard_failures == ("upstream_safety_violation",)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"success": 1}, "success"),
        ({"execution_mode": ""}, "execution_mode"),
        ({"candidate_fingerprint": ""}, "candidate_fingerprint"),
        ({"feedback": []}, "feedback"),
        ({"usage": []}, "usage"),
        ({"hard_failures": ("",)}, "hard_failures"),
    ),
)
def test_evaluation_result_rejects_malformed_current_evidence(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "success": True,
        "execution_mode": "live",
        "candidate_fingerprint": "d" * 64,
        "feedback": {},
        "usage": {},
        "hard_failures": (),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        EvaluationResult(**values)  # type: ignore[arg-type]


def _score(
    fingerprint: str,
    *,
    aggregate: float = 0.8,
    pass_at_3: float = 1.0,
    pass_at_5: float = 1.0,
    hard_safety_failures: int = 0,
    core_regression: bool = False,
) -> EvaluationScore:
    return EvaluationScore(
        fingerprint=fingerprint,
        aggregate=aggregate,
        pass_at_3=pass_at_3,
        pass_at_5=pass_at_5,
        hard_safety_failures=hard_safety_failures,
        core_regression=core_regression,
    )


def test_evaluation_score_has_no_systemic_failure_dimension() -> None:
    score = _score("candidate")
    assert not hasattr(score, "systemic_failures")
    with pytest.raises(TypeError):
        EvaluationScore(  # type: ignore[call-arg]
            fingerprint="candidate",
            aggregate=1.0,
            pass_at_3=1.0,
            pass_at_5=1.0,
            hard_safety_failures=0,
            systemic_failures=1,
        )


def test_strict_ranking_uses_only_current_safety_regression_and_score_dimensions() -> None:
    champion = _score("champion", aggregate=0.7, pass_at_3=0.8, pass_at_5=0.6)
    assert is_strictly_better(
        _score("better", aggregate=0.8, pass_at_3=0.8, pass_at_5=0.6),
        champion,
    )
    assert not is_strictly_better(
        _score("regressed", aggregate=0.9, core_regression=True),
        champion,
    )
    assert not is_strictly_better(
        _score("unsafe", aggregate=1.0, hard_safety_failures=1),
        champion,
    )
    assert not is_strictly_better(
        _score("same", aggregate=0.7, pass_at_3=0.8, pass_at_5=0.6),
        champion,
    )


def test_zero_aggregate_safety_failure_remains_the_worst_scoreable_evidence() -> None:
    zero_unsafe = _score(
        "zero-unsafe",
        aggregate=0.0,
        hard_safety_failures=1,
    )
    measured_unsafe = _score(
        "measured-unsafe",
        aggregate=0.5,
        hard_safety_failures=2,
    )

    assert is_strictly_better(measured_unsafe, zero_unsafe)
    assert not is_strictly_better(zero_unsafe, measured_unsafe)


def test_qualification_gate_requires_safe_repeatable_non_regressing_evidence() -> None:
    assert passes_qualification_gate(_score("qualified"))
    assert not passes_qualification_gate(_score("unsafe", hard_safety_failures=1))
    assert not passes_qualification_gate(_score("regressed", core_regression=True))
    assert not passes_qualification_gate(_score("p3", pass_at_3=0.99))
    assert not passes_qualification_gate(_score("p5", pass_at_5=0.99))


def _outcomes(
    case_id: str,
    model: str,
    passes: tuple[bool, ...],
) -> list[RepetitionOutcome]:
    return [
        RepetitionOutcome(
            case_id=case_id,
            model=model,
            repetition=index,
            passed=passed,
        )
        for index, passed in enumerate(passes, start=1)
    ]


def test_pass_hat_k_preserves_current_repeatability_semantics() -> None:
    outcomes = _outcomes("case-a", "model", (True, True, True)) + _outcomes(
        "case-b", "model", (True, True, False)
    )
    assert pass_hat_k(outcomes, 3) == pytest.approx(0.5)
    assert pass_hat_k(_outcomes("case-a", "model", (True, True, True)), 5) is None
    assert pass_hat_k([], 3) is None


def test_pass_hat_k_rejects_duplicate_repetitions() -> None:
    outcomes = _outcomes("case-a", "model", (True, True)) + _outcomes(
        "case-a", "model", (True,)
    )
    with pytest.raises(ValueError, match="duplicate repetition"):
        pass_hat_k(outcomes, 2)
