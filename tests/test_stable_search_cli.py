from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.cli import main
from korvid_prompt_lab.contracts import Candidate
from korvid_prompt_lab.runner import BridgeProcessExitError
from korvid_prompt_lab.stable_candidates import StructuredCandidate
from korvid_prompt_lab.stable_scenarios import (
    ScenarioAssignment,
    ScenarioClass,
    ScenarioManifest,
    ScenarioSplitSummary,
)

ROOT = Path(__file__).resolve().parents[1]
FAKE_KORVID_EVALS = ROOT / "tests" / "fixtures" / "fake_korvid_evals.py"


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(args)
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
