from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from current_helpers import MODEL, serving

from korvid_prompt_lab.cli import build_parser, command_run, main


def experiment_spec() -> SimpleNamespace:
    return SimpleNamespace(
        runtime=serving(),
        model=SimpleNamespace(reference=MODEL),
        references=("scenarios/image-pull-typo",),
        evaluation_seed=7,
        fingerprint="f" * 64,
    )


def test_parser_exposes_only_the_current_run_command(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--experiment",
            str(tmp_path / "experiment.yaml"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--check-only",
            "--allow-capacity-changes",
        ]
    )

    assert args.command == "run"
    assert args.experiment == tmp_path / "experiment.yaml"
    assert args.artifact_root == tmp_path / "artifacts"
    assert args.check_only is True
    assert args.allow_capacity_changes is True


@pytest.mark.parametrize(
    "legacy_command",
    ["validate", "evaluate", "optimize", "native-evaluate", "native-optimize"],
)
def test_parser_rejects_removed_legacy_commands(legacy_command: str) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([legacy_command])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "legacy_flag",
    ["--candidate", "--campaign", "--model-endpoint", "--bridge-timeout-seconds"],
)
def test_run_rejects_removed_legacy_flags(
    tmp_path: Path,
    legacy_flag: str,
) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(
            [
                "run",
                "--experiment",
                str(tmp_path / "experiment.yaml"),
                legacy_flag,
                "legacy-value",
            ]
        )
    assert error.value.code == 2


def test_check_only_validates_source_without_model_or_cloud_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = experiment_spec()
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        "korvid_prompt_lab.experiment_config.load_experiment",
        lambda path: seen.setdefault("experiment_path", path) and spec,
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.source_runtime.validate_source",
        lambda path: seen.setdefault("source_root", path),
    )

    def inspect(
        runtime: Any, model: str, references: Any, *, seed: int
    ) -> dict[str, Any]:
        seen.update(
            runtime=runtime,
            model=model,
            references=references,
            seed=seed,
        )
        return {
            "prompt": {
                "baseline_equivalent": True,
                "pack_id": "low-korvid-operator",
                "source_sha256": "a" * 64,
            },
            "cases": [
                {
                    "reference": "scenarios/image-pull-typo",
                    "source_sha256": "b" * 64,
                }
            ],
        }

    monkeypatch.setattr("korvid_prompt_lab.upstream.inspect_upstream", inspect)
    monkeypatch.setattr(
        "korvid_prompt_lab.experiment.run_experiment",
        lambda *_args, **_kwargs: pytest.fail("check-only must not run the experiment"),
    )

    exit_code = main(
        [
            "run",
            "--experiment",
            str(tmp_path / "experiment.yaml"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--check-only",
        ]
    )

    assert exit_code == 0
    assert seen["experiment_path"] == tmp_path / "experiment.yaml"
    assert seen["source_root"] == Path("/reviewed/korvid")
    assert seen["runtime"].base_url == "http://127.0.0.1:1"
    assert seen["model"] == MODEL
    assert seen["references"] == spec.references
    assert seen["seed"] == 7
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "CONFIGURATION_VALID",
        "experiment_fingerprint": "f" * 64,
        "model_calls": 0,
        "cloud_calls": 0,
        "serving_preflight": "not_performed",
        "baseline_equivalent": True,
        "prompt_pack_id": "low-korvid-operator",
        "prompt_source_sha256": "a" * 64,
        "case_sources": [
            {
                "reference": "scenarios/image-pull-typo",
                "source_sha256": "b" * 64,
            }
        ],
    }


def test_check_only_fails_when_source_baseline_is_not_equivalent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "korvid_prompt_lab.experiment_config.load_experiment",
        lambda _path: experiment_spec(),
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.source_runtime.validate_source",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.inspect_upstream",
        lambda *_args, **_kwargs: {"prompt": {"baseline_equivalent": False}},
    )

    assert (
        main(
            [
                "run",
                "--experiment",
                str(tmp_path / "experiment.yaml"),
                "--check-only",
            ]
        )
        == 2
    )
    assert "experiment configuration failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [("QUALIFIED", 0), ("NOT_CONVERGED", 3), ("FAILED", 1)],
)
def test_run_forwards_current_arguments_and_maps_status_to_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    expected_exit: int,
) -> None:
    spec = experiment_spec()
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        "korvid_prompt_lab.experiment_config.load_experiment",
        lambda _path: spec,
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.source_runtime.validate_source",
        lambda path: seen.setdefault("source_root", path),
    )

    def run_experiment(
        loaded: Any,
        artifact_root: Path,
        *,
        allow_capacity_changes: bool,
    ) -> dict[str, Any]:
        seen.update(
            spec=loaded,
            artifact_root=artifact_root,
            allow_capacity_changes=allow_capacity_changes,
        )
        return {
            "status": status,
            "pipeline_completed": status != "FAILED",
            "prompt_improved": status == "QUALIFIED",
            "qualified": status == "QUALIFIED",
            "proposal_attempts": 1,
            "distinct_proposals": 1,
            "evaluations": 8,
        }

    monkeypatch.setattr(
        "korvid_prompt_lab.experiment.run_experiment",
        run_experiment,
    )
    args = build_parser().parse_args(
        [
            "run",
            "--experiment",
            str(tmp_path / "experiment.yaml"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--allow-capacity-changes",
        ]
    )

    assert command_run(args) == expected_exit
    assert seen == {
        "source_root": Path("/reviewed/korvid"),
        "spec": spec,
        "artifact_root": tmp_path / "artifacts",
        "allow_capacity_changes": True,
    }
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_run_reports_runtime_failure_without_swallowing_the_failure_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "korvid_prompt_lab.experiment_config.load_experiment",
        lambda _path: experiment_spec(),
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.source_runtime.validate_source",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.experiment.run_experiment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("runtime failed")),
    )
    artifact_root = tmp_path / "artifacts"

    assert (
        main(
            [
                "run",
                "--experiment",
                str(tmp_path / "experiment.yaml"),
                "--artifact-root",
                str(artifact_root),
            ]
        )
        == 1
    )
    assert (
        capsys.readouterr().err == "experiment failed (OSError); inspect "
        f"{artifact_root / 'experiment-summary.json'}\n"
    )
