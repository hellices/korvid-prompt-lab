from __future__ import annotations

import json
from pathlib import Path

import pytest

from korvid_prompt_lab.contracts import (
    Campaign,
    Candidate,
    EvalCase,
    KorvidUpstreamServing,
)
from korvid_prompt_lab.scoring import EvaluationResult
from korvid_prompt_lab.upstream_contract import KORVID_REVISION, prompt_candidate


class GradedRunner:
    def __init__(self, campaign: Campaign, *, success: bool = True, seconds: float = 2.5) -> None:
        self.campaign = campaign
        self.success = success
        self.seconds = seconds
        self.seen: list[tuple[str, int, int]] = []

    def run(
        self, candidate: Candidate, case: EvalCase, run_dir: Path | str, *,
        repetition: int = 1, seed: int = 0,
    ) -> EvaluationResult:
        self.seen.append((case.case_id, repetition, seed))
        return EvaluationResult(
            success=self.success, execution_mode="scripted",
            candidate_fingerprint=candidate.fingerprint,
            feedback={
                "success": self.success, "reference": f"scenarios/{case.case_id}",
                "source_sha256": "1" * 64, "questions": [case.prompt],
                "turns": [{"grade": {"diagnosis_success": self.success}}],
                "calls": [{
                    "name": "diagnose_pod", "arguments": {"namespace": "monitoring"},
                    "ok": True, "result": "raw synthetic result",
                }],
            }, usage={"tool_calls": 1, "wall_time_seconds": self.seconds, "diagnostic_calls": 1},
        )


@pytest.fixture
def campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Campaign:
    return Campaign(
        campaign_id="assessment", repetitions=5, models=("ollama/qwen3:0.6b",),
        cases=(EvalCase("healthy-deployment", "korvid-scenario", "Test display only", ("ollama/qwen3:0.6b",)),),
        serving=KorvidUpstreamServing("korvid_upstream", str(tmp_path), "http://127.0.0.1:11434", KORVID_REVISION, 240),
    )


def test_assessment_uses_requested_split_and_paired_seeds(campaign: Campaign, tmp_path: Path) -> None:
    from korvid_prompt_lab.experiment_evaluation import assess

    runner = GradedRunner(campaign)
    result = assess(runner, prompt_candidate("Original prompt"), campaign.cases, tmp_path / "assessment", seed=12)
    assert len(runner.seen) == 5
    assert {item[0] for item in runner.seen} == {"healthy-deployment"}
    assert {item[2] for item in runner.seen} == set(range(12, 17))
    assert result.success_rate == 1.0
    assert result.score.pass_at_3 == result.score.pass_at_5 == 1.0
    assert result.execution_modes == ("scripted",)
    stored = json.loads((tmp_path / "assessment" / "assessment.json").read_text())
    assert stored["seeds"] == list(range(12, 17))
    assert stored["success_rate"] == 1.0


def test_latency_gain_does_not_change_the_original_verdict(campaign: Campaign, tmp_path: Path) -> None:
    from korvid_prompt_lab.experiment_evaluation import assess

    before = assess(
        GradedRunner(campaign, seconds=5.0), prompt_candidate("Original"),
        campaign.cases, tmp_path / "before", seed=0,
    )
    after = assess(
        GradedRunner(campaign), prompt_candidate("Changed"),
        campaign.cases, tmp_path / "after", seed=0,
    )
    assert not after.improves(before)
    assert not after.improves_success(before)
    assert after.wall_time_seconds < before.wall_time_seconds


def test_assessment_rejects_unpaired_comparison(campaign: Campaign, tmp_path: Path) -> None:
    from korvid_prompt_lab.experiment_evaluation import assess

    before = assess(GradedRunner(campaign), prompt_candidate("Original"), campaign.cases, tmp_path / "before", seed=0)
    after = assess(GradedRunner(campaign), prompt_candidate("Changed"), campaign.cases, tmp_path / "after", seed=5)
    with pytest.raises(ValueError, match="same cases and seeds"):
        after.improves(before)


def test_assessment_does_not_grade_system_errors(campaign: Campaign, tmp_path: Path) -> None:
    from korvid_prompt_lab.experiment_evaluation import assess
    from korvid_prompt_lab.runner import BridgeInvocationError

    class BrokenRunner(GradedRunner):
        def run(
            self, candidate: Candidate, case: EvalCase, run_dir: Path | str, *,
            repetition: int = 1, seed: int = 0,
        ) -> EvaluationResult:
            raise BridgeInvocationError("provider unavailable")

    with pytest.raises(BridgeInvocationError):
        assess(BrokenRunner(campaign), prompt_candidate("Original"), campaign.cases, tmp_path / "broken", seed=0)
    assert not (tmp_path / "broken" / "assessment.json").exists()


def test_assessment_preserves_failure_context_and_separate_efficiency_metrics(campaign: Campaign, tmp_path: Path) -> None:
    from korvid_prompt_lab.experiment_evaluation import assess

    result = assess(GradedRunner(campaign), prompt_candidate("Original"), campaign.cases, tmp_path / "assessment", seed=0)
    stored = result.to_mapping()
    assert stored["diagnostic_calls"] == 5
    assert stored["wall_time_seconds"] == 12.5
    assert stored["runs"][0]["feedback"]["success"] is True
    assert stored["runs"][0]["feedback"]["calls"][0]["name"] == "diagnose_pod"
    assert "raw synthetic result" not in json.dumps(stored)
