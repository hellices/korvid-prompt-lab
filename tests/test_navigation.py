from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]
from korvid.evals.scripted import ScriptedProvider

from korvid_prompt_lab.config import load_campaign, load_candidate
from korvid_prompt_lab.contracts import KorvidNavigationServing
from korvid_prompt_lab.navigation import (
    KorvidNavigationRunner,
    initialize_navigation,
    local_model_provider,
)
from korvid_prompt_lab.runner import BridgeInvocationError
from korvid_prompt_lab.scoring import result_passed, score_result


def test_initialize_creates_small_prompt_and_full_navigation_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NAVIGATION_MODEL_URL", "http://127.0.0.1:11434/v1")
    candidate_path, campaign_path = initialize_navigation(tmp_path / "nav", model="qwen3:0.6b")
    candidate = load_candidate(candidate_path)
    campaign = load_campaign(campaign_path)
    assert isinstance(campaign.serving, KorvidNavigationServing)
    assert campaign.repetitions == 3
    assert len(campaign.cases) == 24
    assert len(candidate.components["system"]) < 2500
    assert "helmreleases" in candidate.components["system"]
    with pytest.raises(FileExistsError):
        initialize_navigation(tmp_path / "nav", model="qwen3:0.6b")


def test_navigation_runner_scores_actual_state_and_records_action_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NAVIGATION_MODEL_URL", "http://127.0.0.1:11434/v1")
    candidate_path, campaign_path = initialize_navigation(tmp_path / "nav", model="qwen3:0.6b")
    campaign = load_campaign(campaign_path)
    case = next(case for case in campaign.cases if case.case_id == "train-helm")
    candidate = load_candidate(candidate_path)
    provider = ScriptedProvider([
        [{"type": "tool_call", "id": "a", "name": "navigate",
          "arguments": json.dumps({"view": "helm", "namespace": "shop"})}],
        [{"type": "text_delta", "text": "Opened."}],
    ])
    runner = KorvidNavigationRunner(campaign, provider_factory=lambda: provider)
    result = runner.run(candidate, case, tmp_path / "run")
    assert result.execution_mode == "scripted"
    assert result_passed(score_result(result))
    feedback = result.journal["navigation_feedback"]
    assert feedback["prompt"] == case.prompt
    assert feedback["observed"]["kind"] == "helmreleases"
    assert feedback["calls"][0]["arguments"] == {"view": "helm", "namespace": "shop"}
    assert feedback["missing_postconditions"] == []
    persisted = json.loads((tmp_path / "run" / "response.json").read_text())
    assert persisted["answer"] == ""
    assert persisted["evidence_source"]["kind"] == "korvid_navigation"
    assert persisted["usage"]["tool_calls"] == 1
    assert persisted["usage"]["iterations"] == 2


def test_text_claim_alone_never_passes_navigation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KORVID_NAVIGATION_MODEL_URL", "http://127.0.0.1:11434/v1")
    candidate_path, campaign_path = initialize_navigation(tmp_path / "nav", model="qwen3:0.6b")
    campaign = load_campaign(campaign_path)
    case = next(case for case in campaign.cases if case.case_id == "train-helm")
    provider = ScriptedProvider([[{"type": "text_delta", "text": "I opened the Helm screen."}]])
    result = KorvidNavigationRunner(campaign, provider_factory=lambda: provider).run(
        load_candidate(candidate_path), case, tmp_path / "run",
    )
    assert not result_passed(score_result(result))
    assert result.grade is not None and result.grade.completion == 0
    assert result.journal["navigation_feedback"]["missing_postconditions"]
    assert result.usage["iterations"] == 1


def test_local_model_client_never_uses_environment_proxy() -> None:
    async def check() -> None:
        serving = KorvidNavigationServing(
            backend="korvid_navigation", base_url="http://127.0.0.1:11434/v1",
            timeout_seconds=120, max_iterations=4,
        )
        async with local_model_provider(serving, "qwen3:0.6b") as provider:
            assert not provider._get_client().trust_env
            assert not provider._get_client().follow_redirects

    asyncio.run(check())


def test_model_runtime_failure_is_not_prompt_quality_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NAVIGATION_MODEL_URL", "http://127.0.0.1:11434/v1")
    candidate_path, campaign_path = initialize_navigation(tmp_path / "nav", model="qwen3:0.6b")
    campaign = load_campaign(campaign_path)
    case = next(case for case in campaign.cases if case.case_id == "train-helm")
    runner = KorvidNavigationRunner(campaign, provider_factory=lambda: ScriptedProvider([]))
    with pytest.raises(BridgeInvocationError, match="runtime"):
        runner.run(load_candidate(candidate_path), case, tmp_path / "run")
    assert not (tmp_path / "run" / "response.json").exists()


def test_truncated_navigation_pack_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NAVIGATION_MODEL_URL", "http://127.0.0.1:11434/v1")
    _, campaign_path = initialize_navigation(tmp_path / "nav", model="qwen3:0.6b")
    payload = yaml.safe_load(campaign_path.read_text())
    payload["cases"] = [
        case for case in payload["cases"] if case["case_id"] in {"train-pods", "validation-pods"}
    ]
    campaign_path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="complete.*24"):
        load_campaign(campaign_path)
