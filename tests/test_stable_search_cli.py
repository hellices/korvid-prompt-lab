from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import yaml  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import korvid_prompt_lab.cli as stable_search_cli
from korvid_prompt_lab import stable_rollover
from korvid_prompt_lab.cli import main
from korvid_prompt_lab.contracts import Candidate
from korvid_prompt_lab.runner import BridgeProcessExitError, BridgeSystemError
from korvid_prompt_lab.stable_candidates import StructuredCandidate
from korvid_prompt_lab.stable_rollover import (
    PriorCampaignEvidence,
    PriorFinalistEvidence,
)
from korvid_prompt_lab.stable_scenarios import (
    FreshHoldoutExhaustedError,
    RolloverScenarioManifest,
    ScenarioAssignment,
    ScenarioClass,
    ScenarioManifest,
    ScenarioSplitSummary,
)
from korvid_prompt_lab.stable_search import StableSearchConfig

ROOT = Path(__file__).resolve().parents[1]
FAKE_KORVID_EVALS = ROOT / "tests" / "fixtures" / "fake_korvid_evals.py"


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            exit_code = main(args)
        except SystemExit as exc:
            exit_code = int(exc.code) if isinstance(exc.code, int) else 1
    return exit_code, stdout.getvalue(), stderr.getvalue()


def _baseline() -> Candidate:
    return Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "korvid-baseline-small",
            "components": {"system": "Stay safe."},
            "metadata": {"korvid_version": "0.3.0", "profile": "small"},
        }
    )


def _structured_candidate() -> StructuredCandidate:
    return StructuredCandidate(
        axes=(),
        candidate=Candidate.from_mapping(
            {
                "schema_version": 1,
                "candidate_id": "evidence-first",
                "components": {
                    "system": "Stay safe.",
                    "append": "inspect runtime evidence before stating a diagnosis.",
                },
                "metadata": {"source": "test"},
            }
        ),
    )


def _rollover_structured_candidates() -> tuple[StructuredCandidate, ...]:
    system = _baseline().components["system"]
    candidates: list[StructuredCandidate] = []
    for index, candidate_id in enumerate(
        (
            "decisive-read-first",
            "continue-before-uncertainty",
            "bounded-uncertainty",
            "evidence-linked-conclusion",
            "decisive-read-first+continue-before-uncertainty",
            "decisive-read-first+bounded-uncertainty",
            "bounded-uncertainty+evidence-linked-conclusion",
            "decisive-read-first+continue-before-uncertainty+bounded-uncertainty+evidence-linked-conclusion",
        ),
        start=1,
    ):
        candidates.append(
            StructuredCandidate(
                axes=(),
                candidate=Candidate.from_mapping(
                    {
                        "schema_version": 1,
                        "candidate_id": candidate_id,
                        "components": {
                            "system": system,
                            "append": f"rollover append {index}",
                        },
                        "metadata": {
                            "rollover_from": "c" * 64,
                            "prior_finalist_fingerprint": "e" * 64,
                        },
                    }
                ),
            )
        )
    return tuple(candidates)


def _manifest() -> ScenarioManifest:
    cases = {
        "oom-killed": ("train", ScenarioClass.WORKLOAD_HEALTH),
        "image-pull-typo": ("validation", ScenarioClass.IMAGE_CONFIG),
        "healthy-deployment": ("milestone", ScenarioClass.HEALTHY_CONTROL),
    }
    assignments = tuple(
        ScenarioAssignment(
            scenario_id=scenario_id,
            scenario_class=scenario_class,
            split=cast(Literal["train", "validation", "milestone"], split),
            question_sha256=f"question-{scenario_id}",
            fixture_sha256=f"fixture-{scenario_id}",
            korvid_version="0.3.0",
        )
        for scenario_id, (split, scenario_class) in cases.items()
    )
    return ScenarioManifest(
        korvid_version="0.3.0",
        assignments=assignments,
        train=("oom-killed",),
        validation=("image-pull-typo",),
        milestone=("healthy-deployment",),
        split_summaries=(
            ScenarioSplitSummary(
                split_name="train",
                classes=(ScenarioClass.WORKLOAD_HEALTH,),
                scenario_ids=("oom-killed",),
            ),
            ScenarioSplitSummary(
                split_name="validation",
                classes=(ScenarioClass.IMAGE_CONFIG,),
                scenario_ids=("image-pull-typo",),
            ),
            ScenarioSplitSummary(
                split_name="milestone",
                classes=(ScenarioClass.HEALTHY_CONTROL,),
                scenario_ids=("healthy-deployment",),
            ),
        ),
    )


def _prior_evidence(prior_root: Path) -> PriorCampaignEvidence:
    assignments = (
        ScenarioAssignment(
            scenario_id="used-validation",
            scenario_class=ScenarioClass.NETWORKING,
            split="validation",
            question_sha256="2" * 64,
            fixture_sha256="f" * 64,
            korvid_version="0.3.0",
        ),
        ScenarioAssignment(
            scenario_id="used-train",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            split="train",
            question_sha256="1" * 64,
            fixture_sha256="a" * 64,
            korvid_version="0.3.0",
        ),
    )
    return PriorCampaignEvidence(
        artifact_root=prior_root.resolve(),
        campaign_id="stable-search-korvid-small",
        korvid_version="0.3.0",
        summary_sha256="c" * 64,
        scenario_manifest_sha256="d" * 64,
        consumed_assignments=assignments,
        finalist=PriorFinalistEvidence(
            candidate_id="cite-before-conclusion+stop-with-uncertainty",
            candidate_fingerprint="e" * 64,
            append="name the observed evidence and its source before the final conclusion.",
            validation_delta=0.12,
            milestone_delta=-0.08,
        ),
    )


def _rollover_manifest() -> RolloverScenarioManifest:
    assignments = (
        ScenarioAssignment(
            scenario_id="missing-configmap-mount",
            scenario_class=ScenarioClass.STORAGE,
            split="train",
            question_sha256="9" * 64,
            fixture_sha256="9" * 64,
            korvid_version="0.3.0",
        ),
        ScenarioAssignment(
            scenario_id="oom-killed",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            split="train",
            question_sha256="8" * 64,
            fixture_sha256="8" * 64,
            korvid_version="0.3.0",
        ),
        ScenarioAssignment(
            scenario_id="service-selector-mismatch",
            scenario_class=ScenarioClass.NETWORKING,
            split="validation",
            question_sha256="7" * 64,
            fixture_sha256="7" * 64,
            korvid_version="0.3.0",
        ),
        ScenarioAssignment(
            scenario_id="pvc-pending-no-storageclass",
            scenario_class=ScenarioClass.STORAGE,
            split="milestone",
            question_sha256="6" * 64,
            fixture_sha256="e" * 64,
            korvid_version="0.3.0",
        ),
        ScenarioAssignment(
            scenario_id="pending-insufficient-cpu",
            scenario_class=ScenarioClass.SCHEDULING_RESOURCES,
            split="milestone",
            question_sha256="5" * 64,
            fixture_sha256="b" * 64,
            korvid_version="0.3.0",
        ),
    )
    manifest = ScenarioManifest(
        korvid_version="0.3.0",
        assignments=assignments,
        train=("missing-configmap-mount", "oom-killed"),
        validation=("service-selector-mismatch",),
        milestone=("pvc-pending-no-storageclass", "pending-insufficient-cpu"),
        split_summaries=(
            ScenarioSplitSummary(
                split_name="train",
                classes=(ScenarioClass.STORAGE, ScenarioClass.WORKLOAD_HEALTH),
                scenario_ids=("missing-configmap-mount", "oom-killed"),
            ),
            ScenarioSplitSummary(
                split_name="validation",
                classes=(ScenarioClass.NETWORKING,),
                scenario_ids=("service-selector-mismatch",),
            ),
            ScenarioSplitSummary(
                split_name="milestone",
                classes=(ScenarioClass.STORAGE, ScenarioClass.SCHEDULING_RESOURCES),
                scenario_ids=("pvc-pending-no-storageclass", "pending-insufficient-cpu"),
            ),
        ),
    )
    return RolloverScenarioManifest(
        manifest=manifest,
        consumed_ids=("used-train", "used-validation"),
        fresh_milestone_ids=("fresh-milestone-b", "fresh-milestone-a"),
        audit_reserve_ids=("fresh-audit",),
    )


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def test_stable_search_cli_runs_the_readonly_campaign_and_writes_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "korvid_prompt_lab.korvid_readonly._KORVID_EVALS_COMMAND",
        (sys.executable, str(FAKE_KORVID_EVALS)),
    )
    monkeypatch.setattr("korvid_prompt_lab.cli.build_scenario_manifest", lambda target_per_split=6: _manifest())
    monkeypatch.setenv("KORVID_READONLY_BASE_URL", "http://127.0.0.1:41001")
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "completed")
    record_path = tmp_path / "korvid-evals-record.json"
    monkeypatch.setenv("FAKE_KORVID_EVALS_RECORD", str(record_path))

    artifact_root = tmp_path / "stable-search"
    exit_code, stdout, stderr = _run_cli(
        [
            "stable-search",
            "--artifact-root",
            str(artifact_root),
            "--json",
        ]
    )

    assert exit_code == 0, stderr
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["decision"]["status"] == "no_stable_winner"
    assert payload["campaign_id"] == "stable-search-korvid-small"
    assert (artifact_root / "stable-search-summary.json").exists()
    recorded = json.loads(record_path.read_text(encoding="utf-8"))
    assert recorded["env"]["KORVID_EVAL_BASE_URL"] == "http://127.0.0.1:41001/v1"
    assert recorded["env"]["KORVID_EVAL_MODEL"] == "qwen3:0.6b"


def test_stable_search_cli_rejects_existing_artifact_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "existing"
    artifact_root.mkdir()

    exit_code, stdout, stderr = _run_cli(
        [
            "stable-search",
            "--artifact-root",
            str(artifact_root),
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "already exists" in stderr


def test_stable_search_cli_requires_reflection_model_when_proposer_is_enabled(
    tmp_path: Path,
) -> None:
    exit_code, stdout, stderr = _run_cli(
        [
            "stable-search",
            "--artifact-root",
            str(tmp_path / "stable-search"),
            "--enable-bounded-proposer",
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "--reflection-model" in stderr


def test_stable_search_cli_requires_enable_flag_when_reflection_model_is_set(
    tmp_path: Path,
) -> None:
    exit_code, stdout, stderr = _run_cli(
        [
            "stable-search",
            "--artifact-root",
            str(tmp_path / "stable-search"),
            "--reflection-model",
            "ollama_chat/qwen3:4b",
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "--enable-bounded-proposer" in stderr


@dataclass(frozen=True, slots=True)
class _FakeArtifacts:
    summary_path: Path


def test_stable_search_cli_builds_the_bounded_proposer_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    summary_path = tmp_path / "stable-search-summary.json"
    summary_path.write_text(
        json.dumps({"decision": {"status": "no_stable_winner", "candidate_id": None, "reasons": ["no_finalists"]}}),
        encoding="utf-8",
    )

    class FakeRunner:
        def __init__(self, campaign: object) -> None:
            captured["campaign"] = campaign
            self.campaign = campaign

    def fake_build_reflection_lm(model_name: str) -> object:
        captured["reflection_model"] = model_name
        return {"model": model_name}

    def fake_run_stable_search(**kwargs: Any) -> _FakeArtifacts:
        captured["search_kwargs"] = kwargs
        return _FakeArtifacts(summary_path=summary_path)

    monkeypatch.setattr("korvid_prompt_lab.cli.build_baseline_candidate", lambda profile: _baseline())
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_structured_candidates",
        lambda baseline: (_structured_candidate(),),
    )
    monkeypatch.setattr("korvid_prompt_lab.cli.build_scenario_manifest", lambda target_per_split=6: _manifest())
    monkeypatch.setattr("korvid_prompt_lab.cli.KorvidReadonlyRunner", FakeRunner)
    monkeypatch.setattr("korvid_prompt_lab.cli._build_reflection_lm", fake_build_reflection_lm)
    monkeypatch.setattr("korvid_prompt_lab.cli.run_stable_search", fake_run_stable_search)
    monkeypatch.setenv("KORVID_READONLY_BASE_URL", "http://127.0.0.1:41001")

    exit_code, stdout, stderr = _run_cli(
        [
            "stable-search",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--enable-bounded-proposer",
            "--reflection-model",
            "ollama_chat/qwen3:4b",
            "--json",
        ]
    )

    assert exit_code == 0, stderr
    assert stderr == ""
    assert json.loads(stdout)["decision"]["status"] == "no_stable_winner"
    assert captured["reflection_model"] == "ollama_chat/qwen3:4b"
    assert captured["campaign"].models == ("qwen3:0.6b",)
    assert captured["campaign"].repetitions == 5
    assert captured["campaign"].serving.base_url == "http://127.0.0.1:41001"
    assert (
        captured["search_kwargs"]["extension"].bounded_append_proposer.reflection_lm
        == {"model": "ollama_chat/qwen3:4b"}
    )


def test_stable_search_cli_passes_non_default_target_per_split_to_manifest_and_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    summary_path = tmp_path / "stable-search-summary.json"
    summary_path.write_text(
        json.dumps({"decision": {"status": "no_stable_winner", "candidate_id": None, "reasons": ["no_finalists"]}}),
        encoding="utf-8",
    )

    class FakeRunner:
        def __init__(self, campaign: object) -> None:
            self.campaign = campaign

    def fake_build_scenario_manifest(*, target_per_split: int = 6) -> ScenarioManifest:
        captured["target_per_split"] = target_per_split
        return _manifest()

    def fake_run_stable_search(**kwargs: Any) -> _FakeArtifacts:
        captured["manifest"] = kwargs["manifest"]
        return _FakeArtifacts(summary_path=summary_path)

    monkeypatch.setattr("korvid_prompt_lab.cli.build_baseline_candidate", lambda profile: _baseline())
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_structured_candidates",
        lambda baseline: (_structured_candidate(),),
    )
    monkeypatch.setattr("korvid_prompt_lab.cli.build_scenario_manifest", fake_build_scenario_manifest)
    monkeypatch.setattr("korvid_prompt_lab.cli.KorvidReadonlyRunner", FakeRunner)
    monkeypatch.setattr("korvid_prompt_lab.cli.run_stable_search", fake_run_stable_search)
    monkeypatch.setenv("KORVID_READONLY_BASE_URL", "http://127.0.0.1:41001")

    exit_code, stdout, stderr = _run_cli(
        [
            "stable-search",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--target-per-split",
            "4",
            "--json",
        ]
    )

    assert exit_code == 0, stderr
    assert stderr == ""
    assert json.loads(stdout)["decision"]["status"] == "no_stable_winner"
    assert captured["target_per_split"] == 4
    assert captured["manifest"] == _manifest()


@pytest.mark.parametrize("json_output", [False, True])
def test_stable_search_cli_redacts_systemic_bridge_error_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    json_output: bool,
) -> None:
    class FakeRunner:
        def __init__(self, campaign: object) -> None:
            self.campaign = campaign

    def fake_run_stable_search(**kwargs: Any) -> _FakeArtifacts:
        raise BridgeProcessExitError("TOKEN=TOP_SECRET raw-answer=LEAK")

    monkeypatch.setattr("korvid_prompt_lab.cli.build_baseline_candidate", lambda profile: _baseline())
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_structured_candidates",
        lambda baseline: (_structured_candidate(),),
    )
    monkeypatch.setattr("korvid_prompt_lab.cli.build_scenario_manifest", lambda target_per_split=6: _manifest())
    monkeypatch.setattr("korvid_prompt_lab.cli.KorvidReadonlyRunner", FakeRunner)
    monkeypatch.setattr("korvid_prompt_lab.cli.run_stable_search", fake_run_stable_search)
    monkeypatch.setenv("KORVID_READONLY_BASE_URL", "http://127.0.0.1:41001")

    args = [
        "stable-search",
        "--artifact-root",
        str(tmp_path / "artifacts"),
    ]
    if json_output:
        args.append("--json")

    exit_code, stdout, stderr = _run_cli(args)

    assert exit_code == 1
    assert "TOP_SECRET" not in stdout + stderr
    assert "raw-answer" not in stdout + stderr
    if json_output:
        assert stderr == ""
        assert json.loads(stdout) == {
            "error_label": "bridge_process_exit_error",
            "status": "system_error",
        }
    else:
        assert stdout == ""
        assert stderr == "stable-search failed: systemic bridge error: bridge_process_exit_error\n"


def test_stable_search_rollover_cli_reuses_orchestrator_and_writes_bounded_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {"order": []}
    prior_root = tmp_path / "prior-root"
    prior_root.mkdir()
    artifact_root = tmp_path / "rollover-artifacts"
    prior = _prior_evidence(prior_root)
    baseline = _baseline()
    candidates = _rollover_structured_candidates()
    rollover = _rollover_manifest()
    summary = {
        "campaign_id": "stable-search-korvid-small",
        "decision": {"status": "no_stable_winner", "candidate_id": None, "reasons": ["no_finalists"]},
    }
    summary_path = _write_json(tmp_path / "stable-search-summary.json", summary)

    class FakeRunner:
        def __init__(self, campaign: object) -> None:
            captured["order"].append("runner")
            captured["campaign"] = campaign
            self.campaign = campaign

    def fake_load_prior_campaign_evidence(root: Path) -> PriorCampaignEvidence:
        captured["order"].append("prior")
        captured["prior_root"] = root
        return prior

    def fake_build_baseline_candidate(profile: str) -> Candidate:
        captured["order"].append("baseline")
        captured["profile"] = profile
        return baseline

    def fake_build_rollover_candidates(
        baseline_candidate: Candidate,
        prior_evidence: PriorCampaignEvidence,
    ) -> tuple[StructuredCandidate, ...]:
        captured["order"].append("candidates")
        captured["candidate_inputs"] = (baseline_candidate, prior_evidence)
        return candidates

    def fake_build_rollover_scenario_manifest(
        consumed: tuple[ScenarioAssignment, ...],
        *,
        target_per_split: int = 6,
    ) -> RolloverScenarioManifest:
        captured["order"].append("manifest")
        captured["manifest_inputs"] = {
            "consumed": consumed,
            "target_per_split": target_per_split,
        }
        return rollover

    def fake_run_stable_search(**kwargs: Any) -> _FakeArtifacts:
        captured["order"].append("run")
        captured["run_kwargs"] = kwargs
        return _FakeArtifacts(summary_path=summary_path)

    monkeypatch.setattr(
        "korvid_prompt_lab.cli.load_prior_campaign_evidence",
        fake_load_prior_campaign_evidence,
        raising=False,
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_rollover_candidates",
        fake_build_rollover_candidates,
        raising=False,
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_rollover_scenario_manifest",
        fake_build_rollover_scenario_manifest,
        raising=False,
    )
    monkeypatch.setattr("korvid_prompt_lab.cli.build_baseline_candidate", fake_build_baseline_candidate)
    monkeypatch.setattr("korvid_prompt_lab.cli.KorvidReadonlyRunner", FakeRunner)
    monkeypatch.setattr("korvid_prompt_lab.cli.run_stable_search", fake_run_stable_search)
    monkeypatch.setenv("KORVID_READONLY_BASE_URL", "http://127.0.0.1:41001")

    exit_code, stdout, stderr = _run_cli(
        [
            "stable-search-rollover",
            "--prior-artifact-root",
            str(prior_root),
            "--artifact-root",
            str(artifact_root),
            "--json",
        ]
    )

    assert exit_code == 0, stderr
    assert stderr == ""
    assert captured["prior_root"] == prior_root
    assert captured["profile"] == "small"
    assert captured["candidate_inputs"] == (baseline, prior)
    assert captured["manifest_inputs"] == {
        "consumed": prior.consumed_assignments,
        "target_per_split": 6,
    }
    assert captured["order"].index("candidates") < captured["order"].index("manifest")
    assert captured["run_kwargs"]["runner"].campaign == captured["campaign"]
    assert captured["run_kwargs"]["baseline"] == baseline
    assert captured["run_kwargs"]["candidates"] == candidates
    assert len(captured["run_kwargs"]["candidates"]) == 8
    assert captured["run_kwargs"]["manifest"] == rollover.manifest
    assert captured["run_kwargs"]["artifact_root"] == artifact_root
    assert captured["run_kwargs"]["config"] == StableSearchConfig()
    assert captured["campaign"].repetitions == 5
    assert json.loads(stdout) == summary

    lineage_path = artifact_root / "rollover-lineage.json"
    payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "prior": {
            "campaign_id": "stable-search-korvid-small",
            "decision": "no_stable_winner",
            "stable_search_summary_sha256": "c" * 64,
            "scenario_manifest_sha256": "d" * 64,
            "finalist_id": "cite-before-conclusion+stop-with-uncertainty",
            "finalist_fingerprint": "e" * 64,
        },
        "scenario_consumption": {
            "korvid_version": "0.3.0",
            "consumed": ["a" * 64, "f" * 64],
            "fresh_milestone": ["b" * 64, "e" * 64],
            "counts": {
                "train": 2,
                "validation": 1,
                "milestone": 2,
                "audit_reserve": 1,
            },
        },
        "candidate_matrix_version": "rollover-v1",
        "max_target_calls": 306,
        "terminal_reason": "no_stable_winner",
    }
    text = lineage_path.read_text(encoding="utf-8")
    assert str(prior_root.resolve()) not in text
    for forbidden in ("question", "fixture_state", "endpoint", "raw_answer", "raw_error", "http://127.0.0.1:41001"):
        assert forbidden not in text


def test_stable_search_rollover_cli_rejects_existing_artifact_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "existing"
    artifact_root.mkdir()

    exit_code, stdout, stderr = _run_cli(
        [
            "stable-search-rollover",
            "--prior-artifact-root",
            str(tmp_path / "prior-root"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "already exists" in stderr


def test_stable_search_rollover_cli_reports_fresh_holdout_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prior_root = tmp_path / "prior-root"
    prior_root.mkdir()
    artifact_root = tmp_path / "rollover-artifacts"
    prior = _prior_evidence(prior_root)

    monkeypatch.setattr(
        "korvid_prompt_lab.cli.load_prior_campaign_evidence",
        lambda root: prior,
        raising=False,
    )
    monkeypatch.setattr("korvid_prompt_lab.cli.build_baseline_candidate", lambda profile: _baseline())
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_rollover_candidates",
        lambda baseline, prior_evidence: _rollover_structured_candidates(),
        raising=False,
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_rollover_scenario_manifest",
        lambda consumed, target_per_split=6: (_ for _ in ()).throw(FreshHoldoutExhaustedError("fresh holdout exhausted")),
        raising=False,
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "stable-search-rollover",
            "--prior-artifact-root",
            str(prior_root),
            "--artifact-root",
            str(artifact_root),
            "--json",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout) == {
        "status": "no_stable_winner",
        "terminal_reason": "fresh_holdout_exhausted",
    }
    assert not artifact_root.exists()


def test_stable_search_rollover_cli_redacts_systemic_bridge_error_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prior_root = tmp_path / "prior-root"
    prior_root.mkdir()
    artifact_root = tmp_path / "rollover-artifacts"
    prior = _prior_evidence(prior_root)
    rollover = _rollover_manifest()

    class FakeRunner:
        def __init__(self, campaign: object) -> None:
            self.campaign = campaign

    monkeypatch.setattr(
        "korvid_prompt_lab.cli.load_prior_campaign_evidence",
        lambda root: prior,
        raising=False,
    )
    monkeypatch.setattr("korvid_prompt_lab.cli.build_baseline_candidate", lambda profile: _baseline())
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_rollover_candidates",
        lambda baseline, prior_evidence: _rollover_structured_candidates(),
        raising=False,
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_rollover_scenario_manifest",
        lambda consumed, target_per_split=6: rollover,
        raising=False,
    )
    monkeypatch.setattr("korvid_prompt_lab.cli.KorvidReadonlyRunner", FakeRunner)
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.run_stable_search",
        lambda **kwargs: (_ for _ in ()).throw(BridgeSystemError("TOKEN=secret raw_error=oops")),
    )
    monkeypatch.setenv("KORVID_READONLY_BASE_URL", "http://127.0.0.1:41001")

    exit_code, stdout, stderr = _run_cli(
        [
            "stable-search-rollover",
            "--prior-artifact-root",
            str(prior_root),
            "--artifact-root",
            str(artifact_root),
            "--json",
        ]
    )

    assert exit_code == 1
    assert "TOKEN=secret" not in stdout + stderr
    assert "raw_error=oops" not in stdout + stderr
    assert stderr == ""
    assert json.loads(stdout) == {
        "error_label": "bridge_system_error",
        "status": "system_error",
    }


def test_stable_search_rollover_cli_writes_winner_yaml_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prior_root = tmp_path / "prior-root"
    prior_root.mkdir()
    artifact_root = tmp_path / "rollover-artifacts"
    winner_output = tmp_path / "winner.yaml"
    prior = _prior_evidence(prior_root)
    candidates = _rollover_structured_candidates()
    rollover = _rollover_manifest()
    summary = {
        "campaign_id": "stable-search-korvid-small",
        "decision": {
            "status": "promote",
            "candidate_id": candidates[2].candidate.candidate_id,
            "reasons": [],
        },
    }
    summary_path = _write_json(tmp_path / "stable-search-summary.json", summary)
    calls: list[Candidate] = []

    class FakeRunner:
        def __init__(self, campaign: object) -> None:
            self.campaign = campaign

    assert hasattr(stable_rollover, "write_rollover_winner")
    real_write_rollover_winner = stable_rollover.write_rollover_winner

    def spy_write_rollover_winner(path: Path, candidate: Candidate) -> Path:
        calls.append(candidate)
        return real_write_rollover_winner(path, candidate)

    monkeypatch.setattr(
        "korvid_prompt_lab.cli.load_prior_campaign_evidence",
        lambda root: prior,
        raising=False,
    )
    monkeypatch.setattr("korvid_prompt_lab.cli.build_baseline_candidate", lambda profile: _baseline())
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_rollover_candidates",
        lambda baseline, prior_evidence: candidates,
        raising=False,
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_rollover_scenario_manifest",
        lambda consumed, target_per_split=6: rollover,
        raising=False,
    )
    monkeypatch.setattr("korvid_prompt_lab.cli.KorvidReadonlyRunner", FakeRunner)
    monkeypatch.setattr("korvid_prompt_lab.cli.run_stable_search", lambda **kwargs: _FakeArtifacts(summary_path=summary_path))
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.write_rollover_winner",
        spy_write_rollover_winner,
        raising=False,
    )
    monkeypatch.setenv("KORVID_READONLY_BASE_URL", "http://127.0.0.1:41001")

    exit_code, stdout, stderr = _run_cli(
        [
            "stable-search-rollover",
            "--prior-artifact-root",
            str(prior_root),
            "--artifact-root",
            str(artifact_root),
            "--winner-output",
            str(winner_output),
            "--json",
        ]
    )

    assert exit_code == 0, stderr
    assert stderr == ""
    assert json.loads(stdout) == summary
    assert calls == [candidates[2].candidate]
    assert yaml.safe_load(winner_output.read_text(encoding="utf-8")) == {
        "schema_version": candidates[2].candidate.schema_version,
        "candidate_id": candidates[2].candidate.candidate_id,
        "components": candidates[2].candidate.components,
        "metadata": candidates[2].candidate.metadata,
    }


def test_materialize_rollover_lineage_keeps_draft_when_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "rollover-artifacts"
    draft_path = tmp_path / ".rollover-artifacts.rollover-lineage.json"
    lineage_path = artifact_root / "rollover-lineage.json"
    draft_path.write_text("draft-lineage", encoding="utf-8")

    prior = _prior_evidence(tmp_path / "prior-root")
    rollover = _rollover_manifest()

    def fail_write_rollover_lineage(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("authoritative write failed")

    monkeypatch.setattr(
        stable_search_cli,
        "write_rollover_lineage",
        fail_write_rollover_lineage,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="authoritative write failed"):
        stable_search_cli._materialize_rollover_lineage(
            artifact_root=artifact_root,
            draft_path=draft_path,
            prior=prior,
            rollover=rollover,
            terminal_reason="stable_winner",
        )

    assert draft_path.exists()
    assert draft_path.read_text(encoding="utf-8") == "draft-lineage"
    assert not lineage_path.exists()


def test_stable_search_rollover_cli_skips_winner_yaml_when_no_stable_winner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prior_root = tmp_path / "prior-root"
    prior_root.mkdir()
    artifact_root = tmp_path / "rollover-artifacts"
    winner_output = tmp_path / "winner.yaml"
    prior = _prior_evidence(prior_root)
    rollover = _rollover_manifest()
    summary = {
        "campaign_id": "stable-search-korvid-small",
        "decision": {"status": "no_stable_winner", "candidate_id": None, "reasons": ["no_finalists"]},
    }
    summary_path = _write_json(tmp_path / "stable-search-summary.json", summary)
    calls: list[Candidate] = []

    class FakeRunner:
        def __init__(self, campaign: object) -> None:
            self.campaign = campaign

    def record_winner_write(path: Path, candidate: Candidate) -> Path:
        calls.append(candidate)
        return path

    monkeypatch.setattr(
        "korvid_prompt_lab.cli.load_prior_campaign_evidence",
        lambda root: prior,
        raising=False,
    )
    monkeypatch.setattr("korvid_prompt_lab.cli.build_baseline_candidate", lambda profile: _baseline())
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_rollover_candidates",
        lambda baseline, prior_evidence: _rollover_structured_candidates(),
        raising=False,
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.build_rollover_scenario_manifest",
        lambda consumed, target_per_split=6: rollover,
        raising=False,
    )
    monkeypatch.setattr("korvid_prompt_lab.cli.KorvidReadonlyRunner", FakeRunner)
    monkeypatch.setattr("korvid_prompt_lab.cli.run_stable_search", lambda **kwargs: _FakeArtifacts(summary_path=summary_path))
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.write_rollover_winner",
        record_winner_write,
        raising=False,
    )
    monkeypatch.setenv("KORVID_READONLY_BASE_URL", "http://127.0.0.1:41001")

    exit_code, stdout, stderr = _run_cli(
        [
            "stable-search-rollover",
            "--prior-artifact-root",
            str(prior_root),
            "--artifact-root",
            str(artifact_root),
            "--winner-output",
            str(winner_output),
            "--json",
        ]
    )

    assert exit_code == 0, stderr
    assert stderr == ""
    assert json.loads(stdout) == summary
    assert calls == []
    assert not winner_output.exists()
