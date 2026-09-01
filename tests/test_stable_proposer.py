from __future__ import annotations

import inspect
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.exceptions import (
    Timeout as LiteLLMTimeout,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.stable_candidates import CandidateAxis
from korvid_prompt_lab.stable_proposer import (
    BoundedAggregateFeedback,
    BoundedAppendProposalRequest,
    BoundedAppendProposer,
    build_proposal_request,
)
from korvid_prompt_lab.stable_ranking import CandidateMeasurement


def _measurement() -> CandidateMeasurement:
    return CandidateMeasurement(
        candidate_id="candidate-1",
        split="validation",
        mean_score=0.61,
        score_variance=0.02,
        worst_case_mean=0.54,
        pass_at_3=1.0,
        hard_safety_failures=0,
        systemic_failures=0,
        repetitions_per_case=5,
        per_case_repetition_counts=(("case-1", 3), ("case-2", 2)),
        mean_verification=0.82,
        malformed_tool_calls=0,
        unresolvable_tool_calls=1,
    )


def test_build_proposal_request_keeps_only_bounded_aggregate_feedback() -> None:
    measurement = _measurement()

    request = build_proposal_request(
        object(),
        finalist_append="inspect runtime evidence before stating a diagnosis.",
        failure_axis=CandidateAxis.EVIDENCE_FIRST.value,
        bounded_feedback=measurement,
    )
    encoded = json.dumps(asdict(request), ensure_ascii=False, sort_keys=True)

    assert isinstance(request, BoundedAppendProposalRequest)
    assert request.finalist_append == "inspect runtime evidence before stating a diagnosis."
    assert request.failure_axis is CandidateAxis.EVIDENCE_FIRST
    assert "candidate-1" not in encoded
    assert "case-1" not in encoded
    assert "case-2" not in encoded
    assert "per_case_repetition_counts" not in encoded
    assert asdict(request.bounded_feedback) == {
        "mean_score": 0.61,
        "score_variance": 0.02,
        "worst_case_mean": 0.54,
        "pass_at_3": 1.0,
        "hard_safety_failures": 0,
        "systemic_failures": 0,
        "repetitions_per_case": 5,
        "mean_verification": 0.82,
        "malformed_tool_calls": 0,
        "unresolvable_tool_calls": 1,
    }


def test_bounded_append_proposal_request_rejects_non_bounded_feedback_instances() -> None:
    measurement = _measurement()
    payload = asdict(measurement)

    with pytest.raises(TypeError, match="bounded_feedback"):
        BoundedAppendProposalRequest(
            finalist_append="inspect runtime evidence before stating a diagnosis.",
            failure_axis=CandidateAxis.EVIDENCE_FIRST,
            bounded_feedback=measurement,
        )
    with pytest.raises(TypeError, match="bounded_feedback"):
        BoundedAppendProposalRequest(
            finalist_append="inspect runtime evidence before stating a diagnosis.",
            failure_axis=CandidateAxis.EVIDENCE_FIRST,
            bounded_feedback=payload,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mean_score", float("nan"), "finite"),
        ("score_variance", -0.1, "non-negative"),
        ("worst_case_mean", float("inf"), "finite"),
        ("pass_at_3", 1.1, "between 0.0 and 1.0"),
        ("hard_safety_failures", -1, "non-negative integer"),
        ("repetitions_per_case", -1, "non-negative integer"),
        ("mean_verification", float("-inf"), "finite"),
        ("malformed_tool_calls", -1, "non-negative integer"),
        ("unresolvable_tool_calls", -1, "non-negative integer"),
    ],
)
def test_bounded_aggregate_feedback_validates_numeric_bounds(
    field: str, value: object, message: str
) -> None:
    kwargs = {
        "mean_score": 0.61,
        "score_variance": 0.02,
        "worst_case_mean": 0.54,
        "pass_at_3": 1.0,
        "hard_safety_failures": 0,
        "systemic_failures": 0,
        "repetitions_per_case": 5,
        "mean_verification": 0.82,
        "malformed_tool_calls": 0,
        "unresolvable_tool_calls": 1,
    }
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        BoundedAggregateFeedback(**kwargs)


def test_build_proposal_request_rejects_noncanonical_finalist_append_outer_whitespace() -> None:
    with pytest.raises(ValueError, match="canonical outer whitespace"):
        build_proposal_request(
            object(),
            finalist_append=" inspect runtime evidence before stating a diagnosis. ",
            failure_axis=CandidateAxis.EVIDENCE_FIRST.value,
            bounded_feedback=_measurement(),
        )


def test_build_proposal_request_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError, match="unknown failure_axis"):
        build_proposal_request(
            object(),
            finalist_append="inspect runtime evidence before stating a diagnosis.",
            failure_axis="not-an-axis",
            bounded_feedback=_measurement(),
        )


def test_bounded_append_proposer_is_lazy_and_canonicalizes_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor_inits: list[object] = []
    predictor_calls: list[dict[str, object]] = []

    class FakePredict:
        def __init__(self, signature: object) -> None:
            predictor_inits.append(signature)

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            predictor_calls.append(kwargs)
            return SimpleNamespace(revised_append="  Tighten verification before concluding.  ")

    monkeypatch.setattr("korvid_prompt_lab.stable_proposer.dspy.Predict", FakePredict)

    proposer = BoundedAppendProposer(reflection_lm={"model": "reflection-lm"})
    request = build_proposal_request(
        object(),
        finalist_append="inspect runtime evidence before stating a diagnosis.",
        failure_axis=CandidateAxis.EVIDENCE_FIRST.value,
        bounded_feedback=_measurement(),
    )

    assert predictor_inits == []

    revised = proposer.propose(request)

    assert revised == "Tighten verification before concluding."
    assert len(predictor_inits) == 1
    assert predictor_calls == [
        {
            "current_append": request.finalist_append,
            "failure_axis": request.failure_axis.value,
            "bounded_feedback_json": json.dumps(
                asdict(request.bounded_feedback), ensure_ascii=False, sort_keys=True
            ),
            "lm": {"model": "reflection-lm"},
        }
    ]


@pytest.mark.parametrize(
    "revised_append",
    [
        "   ",
        "x" * 481,
    ],
)
def test_bounded_append_proposer_rejects_blank_or_overlong_output(
    monkeypatch: pytest.MonkeyPatch, revised_append: str
) -> None:
    class FakePredict:
        def __init__(self, signature: object) -> None:
            self.signature = signature

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(revised_append=revised_append)

    monkeypatch.setattr("korvid_prompt_lab.stable_proposer.dspy.Predict", FakePredict)

    proposer = BoundedAppendProposer(reflection_lm=object())
    request = build_proposal_request(
        object(),
        finalist_append="inspect runtime evidence before stating a diagnosis.",
        failure_axis=CandidateAxis.EVIDENCE_FIRST.value,
        bounded_feedback=_measurement(),
    )

    with pytest.raises(ValueError, match="blank|480"):
        proposer.propose(request)


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("slow"),
        subprocess.TimeoutExpired("dspy", 1.0),
        ConnectionError("transport broke"),
        APIConnectionError("api connection failed", "openai", "model"),
        RateLimitError("rate limited", "openai", "model"),
        ServiceUnavailableError("service unavailable", "openai", "model"),
        InternalServerError("server exploded", "openai", "model"),
        LiteLLMTimeout("request timed out", "model", "openai"),
    ],
)
def test_bounded_append_proposer_safe_propose_returns_none_on_transient_failures(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    assert inspect.signature(APIConnectionError).parameters["llm_provider"].name == "llm_provider"
    assert inspect.signature(RateLimitError).parameters["llm_provider"].name == "llm_provider"

    class FakePredict:
        def __init__(self, signature: object) -> None:
            self.signature = signature

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            raise error

    monkeypatch.setattr("korvid_prompt_lab.stable_proposer.dspy.Predict", FakePredict)

    proposer = BoundedAppendProposer(reflection_lm=object())
    assert (
        proposer.safe_propose(
            object(),
            finalist_append="inspect runtime evidence before stating a diagnosis.",
            failure_axis=CandidateAxis.EVIDENCE_FIRST.value,
            bounded_feedback=_measurement(),
        )
        is None
    )


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError("auth failed", "openai", "model"),
        BadRequestError("bad request", "model", "openai"),
    ],
)
def test_bounded_append_proposer_safe_propose_propagates_auth_and_bad_request_errors(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    assert inspect.signature(AuthenticationError).parameters["llm_provider"].name == "llm_provider"
    assert inspect.signature(BadRequestError).parameters["llm_provider"].name == "llm_provider"

    class FakePredict:
        def __init__(self, signature: object) -> None:
            self.signature = signature

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            raise error

    monkeypatch.setattr("korvid_prompt_lab.stable_proposer.dspy.Predict", FakePredict)

    proposer = BoundedAppendProposer(reflection_lm=object())

    with pytest.raises(type(error), match=str(error)):
        proposer.safe_propose(
            object(),
            finalist_append="inspect runtime evidence before stating a diagnosis.",
            failure_axis=CandidateAxis.EVIDENCE_FIRST.value,
            bounded_feedback=_measurement(),
        )


@pytest.mark.parametrize(
    "error",
    [RuntimeError("boom"), PermissionError("denied"), TypeError("bug")],
)
def test_bounded_append_proposer_safe_propose_does_not_swallow_programming_errors(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    class FakePredict:
        def __init__(self, signature: object) -> None:
            self.signature = signature

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            raise error

    monkeypatch.setattr("korvid_prompt_lab.stable_proposer.dspy.Predict", FakePredict)

    proposer = BoundedAppendProposer(reflection_lm=object())

    with pytest.raises(type(error), match=str(error)):
        proposer.safe_propose(
            object(),
            finalist_append="inspect runtime evidence before stating a diagnosis.",
            failure_axis=CandidateAxis.EVIDENCE_FIRST.value,
            bounded_feedback=_measurement(),
        )
