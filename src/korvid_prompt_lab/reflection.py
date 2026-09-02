from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import dspy  # type: ignore[import-untyped]


def canonicalize_proposal_text(text: Any, *, context: str) -> str:
    if not isinstance(text, str):
        raise ValueError(f"{context} must be a string")  # noqa: TRY004 - preserve validation API
    normalized = text.strip()
    if not normalized:
        raise ValueError(f"{context} must not be blank")
    return normalized


class ReflectionProposalSignature(dspy.Signature):
    current_component_text: str = dspy.InputField(desc="Current prompt component text.")
    reflection_records_json: str = dspy.InputField(desc="Compact reflection records encoded as JSON.")
    revised_component_text: str = dspy.OutputField(desc="Improved prompt component text.")


class DSPyInstructionProposer:
    def __init__(self, reflection_lm: object) -> None:
        self.reflection_lm = reflection_lm
        self._predictor: dspy.Predict | None = None

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        proposals: dict[str, str] = {}
        for component_name in components_to_update:
            if component_name not in candidate:
                raise ValueError(f"missing candidate component: {component_name}")
            records = list(reflective_dataset.get(component_name, ()))
            prediction = self._get_predictor()(
                current_component_text=candidate[component_name],
                reflection_records_json=json.dumps(records, ensure_ascii=False, sort_keys=True),
                lm=self.reflection_lm,
            )
            revised = getattr(prediction, "revised_component_text", None)
            proposals[component_name] = canonicalize_proposal_text(
                revised, context=f"blank proposal for component: {component_name}"
            )
        return proposals

    def _get_predictor(self) -> dspy.Predict:
        if self._predictor is None:
            self._predictor = dspy.Predict(ReflectionProposalSignature)
        return self._predictor
