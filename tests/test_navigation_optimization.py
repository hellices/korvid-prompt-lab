from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from korvid.agent.provider import LLMProvider

from korvid_prompt_lab.adapter import KorvidGEPAAdapter
from korvid_prompt_lab.config import load_campaign, load_candidate
from korvid_prompt_lab.navigation import KorvidNavigationRunner, initialize_navigation
from korvid_prompt_lab.optimize import optimize_campaign


class PromptAwareProvider(LLMProvider):
    """Deterministic contract probe, not evidence of a model improvement."""

    @property
    def name(self) -> str:
        return "scripted"

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
        stream: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        if messages[-1]["role"] == "tool" or "tool-first-test" not in messages[0]["content"]:
            yield {"type": "text_delta", "text": "Done."}
            return
        prompt = messages[-1]["content"]
        namespace = "shop" if "shop" in prompt else "monitoring"
        yield {
            "type": "tool_call", "id": "nav", "name": "navigate",
            "arguments": json.dumps({"view": "pods", "namespace": namespace}),
        }


def test_gepa_navigation_proposals_get_action_and_state_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NAVIGATION_MODEL_URL", "http://127.0.0.1:11434/v1")
    candidate_path, campaign_path = initialize_navigation(tmp_path / "nav", model="qwen3:0.6b")
    candidate = load_candidate(candidate_path)
    campaign = load_campaign(campaign_path)
    train = [case for case in campaign.cases if case.case_id == "train-pods"]
    validation = [case for case in campaign.cases if case.case_id == "validation-pods"]
    runner = KorvidNavigationRunner(campaign, provider_factory=PromptAwareProvider)
    adapter = KorvidGEPAAdapter(runner, tmp_path / "trace")
    batch = adapter.evaluate(train, candidate.components, capture_traces=True)
    records = adapter.make_reflective_dataset(candidate.components, batch, ["system"])["system"]
    assert records[0]["Inputs"]["request"] == train[0].prompt
    assert records[0]["Inputs"]["available_mcp_tools"]
    assert records[0]["Feedback"]["missing_postconditions"]
    assert "holdout" not in json.dumps(records)

    proposals: list[object] = []

    def propose(
        candidate: dict[str, str], reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        proposals.append(reflective_dataset)
        return {name: candidate[name] + " tool-first-test" for name in components_to_update}

    artifacts = optimize_campaign(
        runner=runner, seed_candidate=candidate, train_cases=train,
        validation_cases=validation, artifact_root=tmp_path / "optimize",
        max_metric_calls=6, candidate_proposer=propose,
    )
    assert proposals
    assert artifacts.best_candidate.fingerprint != candidate.fingerprint
    assert json.loads(artifacts.summary_path.read_text())["execution_modes"] == ["scripted"]


def test_library_search_rejects_holdout_before_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NAVIGATION_MODEL_URL", "http://127.0.0.1:11434/v1")
    candidate_path, campaign_path = initialize_navigation(tmp_path / "nav", model="qwen3:0.6b")
    campaign = load_campaign(campaign_path)
    runner = KorvidNavigationRunner(campaign, provider_factory=PromptAwareProvider)
    with pytest.raises(ValueError, match="holdout"):
        optimize_campaign(
            runner=runner, seed_candidate=load_candidate(candidate_path),
            train_cases=[case for case in campaign.cases if case.case_id == "train-pods"],
            validation_cases=[case for case in campaign.cases if case.case_id == "holdout-pods"],
            artifact_root=tmp_path / "optimize", max_metric_calls=2,
        )
    assert not (tmp_path / "optimize").exists()
