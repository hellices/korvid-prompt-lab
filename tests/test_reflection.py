from __future__ import annotations

import json
from types import SimpleNamespace

import dspy  # type: ignore[import-untyped]
import litellm  # type: ignore[import-untyped]
import pytest

from korvid_prompt_lab.experiment_budget import BudgetExhausted, ExperimentBudget
from korvid_prompt_lab.reflection import (
    DSPyInstructionProposer,
    ProposalProviderError,
    ProposalRejected,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_dspy_proposer_is_lazy_and_serializes_only_tier_pack_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor_inits: list[object] = []
    predictor_calls: list[dict[str, object]] = []

    class FakePredict:
        def __init__(self, signature: object) -> None:
            predictor_inits.append(signature)

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            predictor_calls.append(kwargs)
            return SimpleNamespace(
                revised_component_text="Use decisive source evidence."
            )

    monkeypatch.setattr("korvid_prompt_lab.reflection.dspy.Predict", FakePredict)
    budget = ExperimentBudget(0, 1, 10.0)
    lm = object()
    proposer = DSPyInstructionProposer(lm, budget=budget)
    records = {
        "tier_pack": [
            {
                "Inputs": {"case_id": "image-pull-typo"},
                "Generated Outputs": {"turns": []},
                "Feedback": {"success": False},
            }
        ]
    }

    assert predictor_inits == []
    assert proposer(
        {"tier_pack": "Original operating prompt."},
        records,
        ["tier_pack"],
    ) == {"tier_pack": "Use decisive source evidence."}
    assert len(predictor_inits) == 1
    assert predictor_calls == [
        {
            "current_component_text": "Original operating prompt.",
            "reflection_records_json": json.dumps(
                records["tier_pack"],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "lm": lm,
        }
    ]


def test_dspy_proposer_requires_a_budget() -> None:
    with pytest.raises(TypeError):
        DSPyInstructionProposer(object())  # type: ignore[call-arg]


def test_dspy_proposer_rejects_unknown_or_blank_tier_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer = DSPyInstructionProposer(
        object(),
        budget=ExperimentBudget(0, 1, 10.0),
    )
    with pytest.raises(ValueError, match="missing"):
        proposer({"tier_pack": "Original."}, {}, ["missing"])

    class BlankPredict:
        def __init__(self, signature: object) -> None:
            self.signature = signature

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(revised_component_text="   ")

    monkeypatch.setattr("korvid_prompt_lab.reflection.dspy.Predict", BlankPredict)
    with pytest.raises(ProposalRejected) as error:
        proposer({"tier_pack": "Original."}, {"tier_pack": []}, ["tier_pack"])
    assert error.value.error_label == "blank_proposal"


def test_dspy_proposer_clips_teacher_timeout_to_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakePredict:
        def __init__(self, signature: object) -> None:
            self.signature = signature

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(revised_component_text="Revised.")

    monkeypatch.setattr("korvid_prompt_lab.reflection.dspy.Predict", FakePredict)
    clock = FakeClock()
    lm = dspy.LM("openai/test", api_key="unused", timeout=180.0)
    proposer = DSPyInstructionProposer(
        lm,
        budget=ExperimentBudget(0, 1, 7.0, clock=clock),
    )

    proposer({"tier_pack": "Original."}, {"tier_pack": []}, ["tier_pack"])

    clipped = calls[0]["lm"]
    assert isinstance(clipped, dspy.LM)
    assert clipped is not lm
    assert clipped.kwargs["timeout"] == 7.0
    assert clipped.num_retries == 0
    assert lm.kwargs["timeout"] == 180.0


def test_dspy_proposer_preserves_non_provider_lm_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakePredict:
        def __init__(self, signature: object) -> None:
            self.signature = signature

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(revised_component_text="Revised.")

    monkeypatch.setattr("korvid_prompt_lab.reflection.dspy.Predict", FakePredict)
    lm = object()
    proposer = DSPyInstructionProposer(
        lm,
        budget=ExperimentBudget(0, 1, 10.0),
    )

    proposer({"tier_pack": "Original."}, {"tier_pack": []}, ["tier_pack"])

    assert calls[0]["lm"] is lm


def test_teacher_result_is_discarded_when_the_deadline_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()

    class SlowPredict:
        def __init__(self, signature: object) -> None:
            self.signature = signature

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            clock.now = 6.0
            return SimpleNamespace(revised_component_text="Too late.")

    monkeypatch.setattr("korvid_prompt_lab.reflection.dspy.Predict", SlowPredict)
    proposer = DSPyInstructionProposer(
        object(),
        budget=ExperimentBudget(0, 1, 5.0, clock=clock),
    )

    with pytest.raises(BudgetExhausted) as error:
        proposer({"tier_pack": "Original."}, {"tier_pack": []}, ["tier_pack"])

    assert error.value.reason == "wall_clock"


def test_teacher_parse_failure_does_not_retry_with_a_json_fallback() -> None:
    lm = dspy.utils.DummyLM(
        [
            {"wrong_field": "malformed"},
            {"revised_component_text": "fallback must not run"},
        ]
    )
    proposer = DSPyInstructionProposer(
        lm,
        budget=ExperimentBudget(0, 1, 10.0),
    )

    with pytest.raises(ProposalRejected) as error:
        proposer({"tier_pack": "Original."}, {"tier_pack": []}, ["tier_pack"])

    assert error.value.error_label == "unparseable_teacher_response"
    assert len(lm.history) == 1


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan")])
def test_dspy_proposer_rejects_invalid_teacher_timeout(timeout: object) -> None:
    lm = dspy.LM("openai/test", api_key="unused", timeout=timeout)
    proposer = DSPyInstructionProposer(
        lm,
        budget=ExperimentBudget(0, 1, 10.0),
    )

    with pytest.raises(ValueError, match="timeout must be a finite positive number"):
        proposer({"tier_pack": "Original."}, {"tier_pack": []}, ["tier_pack"])


@pytest.mark.parametrize("deadline_expired", [False, True])
def test_teacher_timeout_is_classified_by_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
    deadline_expired: bool,
) -> None:
    clock = FakeClock()

    class TimeoutPredict:
        def __init__(self, signature: object) -> None:
            self.signature = signature

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            clock.now = 5.01 if deadline_expired else 1.0
            raise litellm.exceptions.Timeout(
                "teacher timed out",
                model="test",
                llm_provider="openai",
            )

    monkeypatch.setattr("korvid_prompt_lab.reflection.dspy.Predict", TimeoutPredict)
    proposer = DSPyInstructionProposer(
        dspy.LM("openai/test", api_key="unused", timeout=5.0),
        budget=ExperimentBudget(0, 1, 5.0, clock=clock),
    )

    if deadline_expired:
        with pytest.raises(BudgetExhausted) as budget_error:
            proposer({"tier_pack": "Original."}, {"tier_pack": []}, ["tier_pack"])
        assert budget_error.value.reason == "wall_clock"
    else:
        with pytest.raises(ProposalProviderError) as provider_error:
            proposer({"tier_pack": "Original."}, {"tier_pack": []}, ["tier_pack"])
        assert provider_error.value.reason == "provider_timeout"
