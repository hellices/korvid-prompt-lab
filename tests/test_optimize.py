from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
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
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from fake_korvid_bridge import TUNED_MARKER


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


def _recording_proposer(proposals: list[list[str]]) -> Any:
    def propose(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        proposals.append(list(components_to_update))
        return {name: f"{candidate[name]} {TUNED_MARKER}" for name in components_to_update}

    return propose


def test_optimize_campaign_runs_real_gepa_and_persists_a_candidate_that_beats_the_seed(tmp_path: Path) -> None:
    train_cases = [_case("train-1"), _case("train-2"), _case("train-3")]
    validation_cases = [_case("val-1"), _case("val-2")]
    runner = _runner(train_cases + validation_cases)
    seed_candidate = _seed_candidate()
    proposals: list[list[str]] = []

    artifacts = optimize_campaign(
        runner=runner,
        seed_candidate=seed_candidate,
        train_cases=train_cases,
        validation_cases=validation_cases,
        artifact_root=tmp_path / "artifacts",
        max_metric_calls=16,
        candidate_proposer=_recording_proposer(proposals),
    )

    assert proposals, "real GEPA must invoke the injected proposal contract"
    assert artifacts.best_candidate.components != seed_candidate.components
    assert TUNED_MARKER in "".join(artifacts.best_candidate.components.values())

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["seed_candidate_fingerprint"] == seed_candidate.fingerprint
    assert summary["best_candidate_differs_from_seed"] is True
    assert summary["train_case_ids"] == ["train-1", "train-2", "train-3"]
    assert summary["validation_case_ids"] == ["val-1", "val-2"]

    persisted = load_candidate(artifacts.best_candidate_path)
    assert persisted.components == artifacts.best_candidate.components


def test_optimize_campaign_never_resumes_a_stale_run_when_the_seed_changes(tmp_path: Path) -> None:
    train_cases = [_case("train-1"), _case("train-2"), _case("train-3")]
    validation_cases = [_case("val-1"), _case("val-2")]
    runner = _runner(train_cases + validation_cases)
    seed_candidate = _seed_candidate()
    artifact_root = tmp_path / "artifacts"
    first_proposals: list[list[str]] = []
    second_proposals: list[list[str]] = []

    first = optimize_campaign(
        runner=runner,
        seed_candidate=seed_candidate,
        train_cases=train_cases,
        validation_cases=validation_cases,
        artifact_root=artifact_root,
        max_metric_calls=16,
        seed=1,
        candidate_proposer=_recording_proposer(first_proposals),
    )
    first_summary = json.loads(first.summary_path.read_text(encoding="utf-8"))

    second = optimize_campaign(
        runner=runner,
        seed_candidate=seed_candidate,
        train_cases=train_cases,
        validation_cases=validation_cases,
        artifact_root=artifact_root,
        max_metric_calls=16,
        seed=2,
        candidate_proposer=_recording_proposer(second_proposals),
    )
    second_summary = json.loads(second.summary_path.read_text(encoding="utf-8"))

    assert first_proposals, "the first run must actually search"
    assert second_proposals, "a changed seed must start a fresh search instead of resuming stale state"

    assert first_summary["seed"] == 1
    assert second_summary["seed"] == 2
    assert first_summary["run_id"] != second_summary["run_id"]
    assert first_summary["run_dir"] != second_summary["run_dir"]
    assert second_summary["total_metric_calls"] == first_summary["total_metric_calls"]
    assert second_summary["num_candidates"] == first_summary["num_candidates"]

    assert first.summary_path != second.summary_path
    assert first.best_candidate_path != second.best_candidate_path
    assert json.loads(first.summary_path.read_text(encoding="utf-8")) == first_summary
    assert load_candidate(first.best_candidate_path).components == first.best_candidate.components
    assert not (artifact_root / "gepa" / "gepa_state.bin").exists()


def test_optimize_campaign_refuses_to_reuse_an_existing_invocation_directory(tmp_path: Path) -> None:
    train_cases = [_case("train-1"), _case("train-2")]
    validation_cases = [_case("val-1")]
    runner = _runner(train_cases + validation_cases)
    seed_candidate = _seed_candidate()
    artifact_root = tmp_path / "artifacts"

    def run() -> OptimizationArtifacts:
        return optimize_campaign(
            runner=runner,
            seed_candidate=seed_candidate,
            train_cases=train_cases,
            validation_cases=validation_cases,
            artifact_root=artifact_root,
            max_metric_calls=8,
            seed=3,
            candidate_proposer=_recording_proposer([]),
        )

    first = run()

    with pytest.raises(ValueError, match="already exists"):
        run()

    assert json.loads(first.summary_path.read_text(encoding="utf-8"))["run_id"] == first.run_id


def test_optimize_campaign_records_the_run_identity_next_to_the_artifacts(tmp_path: Path) -> None:
    train_cases = [_case("train-1"), _case("train-2")]
    validation_cases = [_case("val-1")]
    runner = _runner(train_cases + validation_cases)
    seed_candidate = _seed_candidate()

    artifacts = optimize_campaign(
        runner=runner,
        seed_candidate=seed_candidate,
        train_cases=train_cases,
        validation_cases=validation_cases,
        artifact_root=tmp_path / "artifacts",
        max_metric_calls=8,
        seed=7,
        candidate_proposer=_recording_proposer([]),
    )

    identity = json.loads((artifacts.invocation_dir / "run-identity.json").read_text(encoding="utf-8"))
    assert identity["seed"] == 7
    assert identity["seed_candidate_fingerprint"] == seed_candidate.fingerprint
    assert identity["train_case_ids"] == ["train-1", "train-2"]
    assert identity["validation_case_ids"] == ["val-1"]
    assert identity["max_metric_calls"] == 8
    assert identity["proposal_source"] == "candidate_proposer"
    assert identity["campaign_id"] == "campaign-1"
    assert artifacts.summary_path.parent == artifacts.invocation_dir
    assert artifacts.best_candidate_path.parent == artifacts.invocation_dir


def test_optimize_campaign_passes_the_seed_to_gepa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_case = _case("train-1")
    validation_case = _case("val-1")
    runner = _runner([train_case, validation_case])
    captured: dict[str, Any] = {}

    def fake_optimize(**kwargs: object) -> GEPAResult:
        captured.update(kwargs)
        return _fake_gepa_result(cast(str, kwargs["run_dir"]))

    monkeypatch.setattr("korvid_prompt_lab.optimize.gepa.optimize", fake_optimize)

    optimize_campaign(
        runner=runner,
        seed_candidate=_seed_candidate(),
        train_cases=[train_case],
        validation_cases=[validation_case],
        artifact_root=tmp_path / "artifacts",
        max_metric_calls=5,
        seed=11,
    )

    assert captured["seed"] == 11


@pytest.mark.parametrize("seed", [-1, True, 1.5, "1"])
def test_optimize_campaign_rejects_invalid_seeds(tmp_path: Path, seed: object) -> None:
    train_case = _case("train-1")
    validation_case = _case("val-1")
    runner = _runner([train_case, validation_case])

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        optimize_campaign(
            runner=runner,
            seed_candidate=_seed_candidate(),
            train_cases=[train_case],
            validation_cases=[validation_case],
            artifact_root=tmp_path / "artifacts",
            max_metric_calls=4,
            seed=cast(int, seed),
        )


def test_optimize_campaign_rejects_combining_reflection_lm_and_candidate_proposer(tmp_path: Path) -> None:
    train_case = _case("train-1")
    validation_case = _case("val-1")
    runner = _runner([train_case, validation_case])

    with pytest.raises(ValueError, match="reflection_lm"):
        optimize_campaign(
            runner=runner,
            seed_candidate=_seed_candidate(),
            train_cases=[train_case],
            validation_cases=[validation_case],
            artifact_root=tmp_path / "artifacts",
            max_metric_calls=4,
            reflection_lm="reflection-lm",
            candidate_proposer=_recording_proposer([]),
        )


@pytest.mark.parametrize(
    ("train_case_ids", "validation_case_ids", "message"),
    [
        ((), ("val-1",), "train_cases must not be empty"),
        (("train-1",), (), "validation_cases must not be empty"),
        (("train-1", "val-1"), ("val-1",), "train and validation case sets must be disjoint"),
        (("val-1",), ("val-1",), "train and validation case sets must be disjoint"),
    ],
)
def test_optimize_campaign_requires_explicit_disjoint_case_splits(
    tmp_path: Path,
    train_case_ids: tuple[str, ...],
    validation_case_ids: tuple[str, ...],
    message: str,
) -> None:
    all_cases = [_case("train-1"), _case("val-1")]
    runner = _runner(all_cases)

    with pytest.raises(ValueError, match=message):
        optimize_campaign(
            runner=runner,
            seed_candidate=_seed_candidate(),
            train_cases=[_case(case_id) for case_id in train_case_ids],
            validation_cases=[_case(case_id) for case_id in validation_case_ids],
            artifact_root=tmp_path / "artifacts",
            max_metric_calls=4,
        )
