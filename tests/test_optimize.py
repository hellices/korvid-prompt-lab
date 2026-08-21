from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from gepa import GEPAResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.config import load_candidate
from korvid_prompt_lab.contracts import Campaign, Candidate, EvalCase, ProcessServing
from korvid_prompt_lab.optimize import OptimizationArtifacts, optimize_campaign
from korvid_prompt_lab.runner import KorvidProcessRunner

ROOT = Path(__file__).resolve().parents[1]


def _seed_candidate() -> Candidate:
    return Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "candidate-1",
            "components": {
                "system": "Stay safe.",
                "append": "Verify the postcondition before reporting completion.",
            },
            "metadata": {"source": "seed"},
        }
    )


def _case(case_id: str) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        template_id="template-1",
        prompt="Confirm the postcondition.",
        models=("mock-small",),
    )


def _runner(cases: list[EvalCase]) -> KorvidProcessRunner:
    command = (
        sys.executable,
        str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
        "--request",
        "{request}",
        "--response",
        "{response}",
    )
    campaign = Campaign(
        schema_version=1,
        campaign_id="campaign-1",
        repetitions=1,
        models=("mock-small",),
        cases=tuple(cases),
        serving=ProcessServing(backend="process", command=command),
    )
    return KorvidProcessRunner(campaign, timeout_seconds=1.0)


def _fake_gepa_result(run_dir: str) -> GEPAResult:
    return GEPAResult(
        candidates=[
            {
                "system": "Stay safe.",
                "append": "Verify the postcondition before reporting completion.",
            },
            {
                "system": "Stay safe and verify approvals.",
                "append": "Verify the postcondition before reporting completion.",
            },
        ],
        parents=[[None], [0]],
        val_aggregate_scores=[0.6, 0.9],
        val_subscores=[{"val-1": 0.6}, {"val-1": 0.9}],
        per_val_instance_best_candidates={"val-1": {1}},
        discovery_eval_counts=[1, 2],
        total_metric_calls=7,
        num_full_val_evals=2,
        run_dir=run_dir,
        seed=0,
    )


def test_optimize_campaign_calls_gepa_and_persists_best_candidate_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_case = _case("train-1")
    validation_case = _case("val-1")
    runner = _runner([train_case, validation_case])
    seed_candidate = _seed_candidate()
    captured: dict[str, Any] = {}

    def fake_optimize(**kwargs: object) -> GEPAResult:
        captured.update(kwargs)
        return _fake_gepa_result(cast(str, kwargs["run_dir"]))

    monkeypatch.setattr("korvid_prompt_lab.optimize.gepa.optimize", fake_optimize)

    result = optimize_campaign(
        runner=runner,
        seed_candidate=seed_candidate,
        train_cases=[train_case],
        validation_cases=[validation_case],
        artifact_root=tmp_path / "artifacts",
        max_metric_calls=7,
    )

    assert isinstance(result, OptimizationArtifacts)
    assert captured["seed_candidate"] == seed_candidate.components
    assert captured["trainset"] == [train_case]
    assert captured["valset"] == [validation_case]
    assert captured["max_metric_calls"] == 7
    assert captured["custom_candidate_proposer"] is None

    persisted_candidate = load_candidate(result.best_candidate_path)
    assert persisted_candidate.candidate_id == seed_candidate.candidate_id
    assert persisted_candidate.metadata == seed_candidate.metadata
    assert persisted_candidate.components == _fake_gepa_result(str(tmp_path / "artifacts" / "gepa")).best_candidate

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["best_idx"] == 1
    assert summary["best_validation_score"] == 0.9
    assert summary["total_metric_calls"] == 7
    assert summary["best_candidate_fingerprint"] == persisted_candidate.fingerprint


def test_optimize_campaign_uses_optional_dspy_proposer_only_when_reflection_lm_is_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_case = _case("train-1")
    validation_case = _case("val-1")
    runner = _runner([train_case, validation_case])
    seed_candidate = _seed_candidate()
    created_with: list[object] = []
    captured_proposers: list[object] = []
    proposer = object()

    def fake_proposer_factory(reflection_lm: object) -> object:
        created_with.append(reflection_lm)
        return proposer

    def fake_optimize(**kwargs: object) -> GEPAResult:
        captured_proposers.append(kwargs["custom_candidate_proposer"])
        return _fake_gepa_result(cast(str, kwargs["run_dir"]))

    monkeypatch.setattr("korvid_prompt_lab.optimize.DSPyInstructionProposer", fake_proposer_factory)
    monkeypatch.setattr("korvid_prompt_lab.optimize.gepa.optimize", fake_optimize)

    optimize_campaign(
        runner=runner,
        seed_candidate=seed_candidate,
        train_cases=[train_case],
        validation_cases=[validation_case],
        artifact_root=tmp_path / "without-lm",
        max_metric_calls=3,
    )
    optimize_campaign(
        runner=runner,
        seed_candidate=seed_candidate,
        train_cases=[train_case],
        validation_cases=[validation_case],
        artifact_root=tmp_path / "with-lm",
        max_metric_calls=3,
        reflection_lm="reflection-lm",
    )

    assert created_with == ["reflection-lm"]
    assert captured_proposers == [None, proposer]
