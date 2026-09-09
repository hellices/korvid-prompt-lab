"""Control-plane tests: synthetic grades and teacher, not model-quality evidence."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import dspy  # type: ignore[import-untyped]
import pytest
import yaml  # type: ignore[import-untyped]

from korvid_prompt_lab.contracts import Campaign, Candidate, EvalCase
from korvid_prompt_lab.scoring import EvaluationResult
from korvid_prompt_lab.upstream_contract import KORVID_REVISION


def experiment_mapping(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 2, "campaign_id": "native-test",
        "runtime": {
            "backend": "korvid_upstream", "source_root": "env:KORVID_NATIVE_SOURCE_ROOT",
            "korvid_revision": KORVID_REVISION, "timeout_seconds": 240,
        },
        "serving": {"backend": "loopback", "base_url": "env:KORVID_NATIVE_MODEL_URL"},
        "model": {
            "reference": "ollama/qwen3:0.6b", "digest": "sha256:" + "1" * 64,
            "options": {"native_thinking": True, "think": False, "num_ctx": 16384, "temperature": 0.0},
        },
        "reflection": {
            "reference": "ollama_chat/qwen3:14b", "digest": "sha256:" + "2" * 64,
            "timeout_seconds": 180,
            "options": {"max_tokens": 512, "num_ctx": 4096, "temperature": 0.2, "reasoning_effort": "disable"},
        },
        "evaluation": {
            "repetitions": 5, "seed": 0,
            "train": ["scenarios/healthy-deployment", "scenarios/image-pull-typo", "scenarios/oom-killed"],
            "validation": ["journeys/tui-follow", "scenarios/healthy-service-endpoints"],
            "holdout": ["journeys/compare-namespaces", "scenarios/healthy-restart-history"],
        },
        "search": {
            "stages": [{"name": "explore", "metric_calls": 16, "seeds": [0]}],
            "total_metric_calls": 32, "max_evaluations": 256, "max_proposals": 4,
            "wall_clock_seconds": 3600, "stagnation_attempt_limit": 3,
        },
    }


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from korvid_prompt_lab import experiment
    from korvid_prompt_lab.experiment_config import load_experiment
    from korvid_prompt_lab.upstream_contract import tier_pack_from_candidate

    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "korvid"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment_mapping(tmp_path)), encoding="utf-8")
    spec = load_experiment(path)
    seen: list[dict[str, Any]] = []
    lifecycle: list[str] = []
    state: dict[str, Any] = {"seen": seen, "lifecycle": lifecycle, "spec": spec, "fail": False, "gain": True}
    original = "Original shipped prompt.\n"
    cases = tuple(EvalCase(
        case_id=ref.split("/")[1],
        template_id="korvid-journey" if ref.startswith("journeys/") else "korvid-scenario",
        prompt=f"Unit-test display for {ref}", models=(spec.model.reference,),
    ) for ref in spec.references)
    def campaign(self, endpoint: str) -> Campaign:
        return Campaign(
            campaign_id=spec.campaign_id, repetitions=self.repetitions,
            models=(spec.model.reference,), cases=cases, serving=replace(self.runtime, base_url=endpoint),
            evaluation_splits=tuple(
                (split, tuple(ref.split("/")[1] for ref in self.case_splits[split]))
                for split in ("train", "validation", "holdout")
            ),
        )
    monkeypatch.setattr(type(spec), "campaign", campaign)

    class Session:
        base_url = "http://127.0.0.1:11434"
        evidence: ClassVar[dict[str, Any]] = {"test_double": True}

        def verify(self) -> None:
            lifecycle.append("verify")

    @contextmanager
    def session(*args: Any, **kwargs: Any) -> Iterator[Session]:
        lifecycle.append("open")
        try:
            yield Session()
        finally:
            lifecycle.append("closed")

    class Runner:
        def __init__(self, campaign: Campaign, *, prompt_path: Path | None = None) -> None:
            self.campaign = campaign
            self.prompt_path = prompt_path

        def run(
            self, candidate: Candidate, case: EvalCase, run_dir: Path | str, *,
            repetition: int = 1, seed: int = 0,
        ) -> EvaluationResult:
            if state["fail"]:
                raise OSError("synthetic failure")
            text = tier_pack_from_candidate(candidate)
            seen.append({
                "case": case.case_id, "seed": seed, "text": text,
                "exported": self.prompt_path is not None,
            })
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            (Path(run_dir) / "upstream-summary.json").write_text(json.dumps({
                "runtime": {"fingerprint": "3" * 64},
                "evaluation_override_verified": self.prompt_path is not None,
            }))
            complete = text != original and state["gain"]
            if state.get("partial_first") and text.startswith("Use the requested resource kind"):
                complete = case.case_id.startswith("healthy-")
            if self.prompt_path is not None and state.get("export_failure"):
                complete = False
            return EvaluationResult(
                success=bool(complete), execution_mode="live",
                candidate_fingerprint=candidate.fingerprint,
                feedback={
                    "success": bool(complete), "reference": case.case_id, "source_sha256": "4" * 64,
                    "questions": [case.prompt], "turns": [{
                        "outcome": "success" if complete else "failure",
                        "grade": {"diagnosis_success": bool(complete), "evidence_fetched": bool(complete)},
                    }], "calls": [],
                }, usage={"tool_calls": 1, "wall_time_seconds": 1.0, "iterations": 1},
            )

    def export(snapshot, candidate: Candidate, campaign: Campaign, directory: Path, *, seed: int = 0) -> Path:
        directory.mkdir(parents=True)
        path = directory / "optimized-prompt.txt"
        path.write_text(tier_pack_from_candidate(candidate))
        return path

    teacher = dspy.utils.DummyLM([
        {"revised_component_text": "Use the requested resource kind and namespace."},
        {"revised_component_text": "Use exact tool arguments from the user request."},
    ])
    monkeypatch.setattr(experiment, "model_session", session)
    monkeypatch.setattr(experiment, "KorvidUpstreamRunner", Runner)
    monkeypatch.setattr(experiment, "validate_source", lambda root: root)
    monkeypatch.setattr(experiment, "inspect_upstream", lambda *args, **kwargs: {
        "runtime": {"fingerprint": "3" * 64},
        "prompt": {"text": original, "pack_id": "low-korvid-operator", "baseline_equivalent": True, "source_sha256": "5" * 64},
        "cases": [{"reference": ref, "case_id": ref.split("/")[1], "source_sha256": "6" * 64} for ref in spec.references],
    })
    monkeypatch.setattr(experiment, "lab_identity", lambda: {"test_double": True})
    monkeypatch.setattr(experiment, "export_upstream_prompt", export)
    monkeypatch.setattr(experiment, "build_reflection_lm", lambda *args, **kwargs: teacher)
    monkeypatch.setattr(
        "korvid_prompt_lab.reflection.validate_upstream_candidate", lambda *_args: None,
    )
    return state


def test_canonical_run_executes_real_gepa_dspy_and_freezes_holdout(
    tmp_path: Path, harness: dict[str, Any],
) -> None:
    from korvid_prompt_lab.experiment import run_experiment

    root = tmp_path / "run"
    result = run_experiment(harness["spec"], root)
    assert result["pipeline_completed"] is True
    assert result["prompt_improved"] is True
    assert result["qualified"] is True
    assert result["proposal_attempts"] >= 1
    assert result["distinct_proposals"] >= 1
    assert result["prompt_override_verified"] is True
    assert result["product_application_verified"] is False
    assert result["evaluations"] == len(harness["seen"])
    assert harness["lifecycle"][-1] == "closed"
    assert any(run["exported"] for run in harness["seen"])
    assert (root / "candidate-for-review" / "optimized-prompt.txt").is_file()
    receipt = json.loads((root / "candidate-for-review" / "application-verification.json").read_text())
    assert receipt["prompt_override_verified"] is True
    assert receipt["product_application_verified"] is False
    assert receipt["candidate_fingerprint"] == result["candidate_fingerprint"]
    identities = list(root.glob("search/*/invocations/*/run-identity.json"))
    assert identities
    for path in identities:
        identity = json.loads(path.read_text())
        assert identity["run_context"]["manifest_fingerprint"] == harness["spec"].fingerprint
        holdout = {ref.split("/")[1] for ref in harness["spec"].case_splits["holdout"]}
        assert not holdout.intersection(identity["train_case_ids"] + identity["validation_case_ids"])
        assert not any(case_id in json.dumps(identity) for case_id in holdout)
    holdout_index = next(i for i, run in enumerate(harness["seen"]) if run["case"] in holdout)
    assert all(run["case"] in holdout or run["exported"] for run in harness["seen"][holdout_index:])
    assert json.loads((root / "experiment-summary.json").read_text())["qualified"] is True


def test_no_gain_does_not_spend_holdout_or_claim_improvement(tmp_path: Path, harness: dict[str, Any]) -> None:
    from korvid_prompt_lab.experiment import run_experiment

    harness["gain"] = False
    result = run_experiment(harness["spec"], tmp_path / "run")
    assert result["status"] == "NOT_CONVERGED"
    assert result["pipeline_completed"] is True
    assert result["prompt_improved"] is False
    assert result["holdout_used"] is False
    assert not any(run["case"] in {"compare-namespaces", "healthy-restart-history"} for run in harness["seen"])


def test_failure_cleans_session_and_persists_non_success(tmp_path: Path, harness: dict[str, Any]) -> None:
    from korvid_prompt_lab.experiment import run_experiment

    harness["fail"] = True
    root = tmp_path / "run"
    with pytest.raises(OSError, match="synthetic"):
        run_experiment(harness["spec"], root)
    assert harness["lifecycle"][-1] == "closed"
    report = json.loads((root / "experiment-summary.json").read_text())
    assert report["status"] == "SYSTEM_ERROR"
    assert report["pipeline_completed"] is False
    assert report["qualified"] is False


def test_run_refuses_existing_artifact_root(tmp_path: Path, harness: dict[str, Any]) -> None:
    from korvid_prompt_lab.experiment import run_experiment

    with pytest.raises(FileExistsError):
        run_experiment(harness["spec"], tmp_path)
    assert not harness["lifecycle"]


def test_run_command_is_the_primary_composable_entry_point() -> None:
    from korvid_prompt_lab.cli import build_parser

    parsed = build_parser().parse_args([
        "run", "--experiment", "native.yaml", "--artifact-root", "artifacts/run", "--check-only",
    ])
    assert parsed.check_only is True
    assert parsed.allow_capacity_changes is False


def test_multiple_search_seeds_produce_distinct_candidates(tmp_path: Path, harness: dict[str, Any]) -> None:
    from korvid_prompt_lab.contracts import SearchStage
    from korvid_prompt_lab.experiment import run_experiment

    harness["partial_first"] = True
    spec = replace(
        harness["spec"], stages=(SearchStage("explore", 16, (0, 1)),), total_metric_calls=64,
    )
    report = run_experiment(spec, tmp_path / "run")
    assert report["distinct_proposals"] == 2
    assert report["proposal_attempts"] == 2
    assert report["evaluations"] == len(harness["seen"])
    assert report["qualified"] is True


def test_budget_exhaustion_is_not_a_completed_qualification(tmp_path: Path, harness: dict[str, Any]) -> None:
    from korvid_prompt_lab.experiment import run_experiment

    report = run_experiment(replace(harness["spec"], max_evaluations=2), tmp_path / "run")
    assert report["status"] == "NOT_CONVERGED"
    assert report["stop_reason"] == "max_evaluations"
    assert report["pipeline_completed"] is False
    assert report["qualified"] is False
    assert report["evaluations"] == len(harness["seen"]) == 2
    assert harness["lifecycle"][-1] == "closed"


def test_nonzero_base_seed_is_used_by_search_and_export(tmp_path: Path, harness: dict[str, Any]) -> None:
    from korvid_prompt_lab.experiment import run_experiment

    report = run_experiment(replace(harness["spec"], evaluation_seed=12), tmp_path / "run")
    assert report["qualified"]
    assert all(run["seed"] >= 12 for run in harness["seen"])
    assert {run["seed"] for run in harness["seen"] if run["exported"]} == {12}


def test_proposal_cap_preserves_an_already_found_candidate(tmp_path: Path, harness: dict[str, Any]) -> None:
    from korvid_prompt_lab.contracts import SearchStage
    from korvid_prompt_lab.experiment import run_experiment

    harness["partial_first"] = True
    spec = replace(
        harness["spec"], stages=(SearchStage("explore", 40, (0,)),),
        total_metric_calls=64, max_proposals=1,
    )
    report = run_experiment(spec, tmp_path / "run")
    assert report["proposal_attempts"] == 1
    assert report["candidate_fingerprint"] != report["baseline_fingerprint"]
    assert report["selected_validation"]["success_rate"] > report["baseline_validation"]["success_rate"]
    assert report["holdout_used"]


def test_hard_search_cap_preserves_the_last_fully_evaluated_candidate(tmp_path: Path, harness: dict[str, Any]) -> None:
    from korvid_prompt_lab.contracts import SearchStage
    from korvid_prompt_lab.experiment import run_experiment

    harness["partial_first"] = True
    spec = replace(
        harness["spec"], stages=(SearchStage("explore", 40, (0,)),), total_metric_calls=40,
    )
    report = run_experiment(spec, tmp_path / "run")
    assert report["search_metric_calls"] <= 40
    assert report["candidate_fingerprint"] != report["baseline_fingerprint"]
    assert report["holdout_used"]
    assert report["evaluations"] == len(harness["seen"])


def test_stage_labels_never_become_output_paths(tmp_path: Path, harness: dict[str, Any]) -> None:
    from korvid_prompt_lab.contracts import SearchStage
    from korvid_prompt_lab.experiment import run_experiment

    spec = replace(harness["spec"], stages=(SearchStage(str(tmp_path / "outside"), 16, (0,)),))
    root = tmp_path / "run"
    report = run_experiment(spec, root)
    assert not (tmp_path / "outside-0").exists()
    assert list(root.glob("search/*/attempt.json"))
    assert report["distinct_proposals"] == 1


def test_prompt_reload_fidelity_is_not_the_export_case_verdict(tmp_path: Path, harness: dict[str, Any]) -> None:
    from korvid_prompt_lab.experiment import run_experiment

    harness["export_failure"] = True
    report = run_experiment(harness["spec"], tmp_path / "run")
    assert report["prompt_override_verified"] is True
    assert report["export_check_success"] is False
    assert report["product_application_verified"] is False
    assert report["qualified"] is False
