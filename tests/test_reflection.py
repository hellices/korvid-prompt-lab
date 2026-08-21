from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.reflection import DSPyInstructionProposer


def test_dspy_instruction_proposer_is_lazy_and_serializes_reflection_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor_inits: list[object] = []
    predictor_calls: list[dict[str, object]] = []

    class FakePredict:
        def __init__(self, signature: object) -> None:
            predictor_inits.append(signature)

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            predictor_calls.append(kwargs)
            return SimpleNamespace(revised_component_text="Tighten postcondition verification.")

    monkeypatch.setattr("korvid_prompt_lab.reflection.dspy.Predict", FakePredict)

    lm = object()
    proposer = DSPyInstructionProposer(reflection_lm=lm)

    assert predictor_inits == []

    candidate = {
        "system": "Stay safe.",
        "append": "Verify before reporting completion.",
    }
    reflective_dataset: dict[str, list[dict[str, Any]]] = {
        "system": [{"Feedback": "No changes"}],
        "append": [
            {
                "Inputs": {"case_id": "case-1", "model": "mock-small"},
                "Generated Outputs": {"outcome": "unsafe"},
                "Feedback": "Require explicit verification before success.",
            }
        ],
    }

    proposals = proposer(candidate, reflective_dataset, ["append"])

    assert proposals == {"append": "Tighten postcondition verification."}
    assert len(predictor_inits) == 1
    assert predictor_calls == [
        {
            "current_component_text": "Verify before reporting completion.",
            "reflection_records_json": json.dumps(reflective_dataset["append"], ensure_ascii=False, sort_keys=True),
            "lm": lm,
        }
    ]


def test_dspy_instruction_proposer_rejects_unknown_component_requests() -> None:
    proposer = DSPyInstructionProposer(reflection_lm=object())

    with pytest.raises(ValueError, match="missing"):
        proposer({"system": "Stay safe."}, {"system": []}, ["missing"])


def test_dspy_instruction_proposer_rejects_blank_proposals(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePredict:
        def __init__(self, signature: object) -> None:
            self.signature = signature

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(revised_component_text="   ")

    monkeypatch.setattr("korvid_prompt_lab.reflection.dspy.Predict", FakePredict)

    proposer = DSPyInstructionProposer(reflection_lm=object())

    with pytest.raises(ValueError, match="blank"):
        proposer({"system": "Stay safe."}, {"system": []}, ["system"])
