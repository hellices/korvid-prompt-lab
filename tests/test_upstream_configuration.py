from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]
from test_experiment import experiment_mapping

from korvid_prompt_lab.experiment_config import load_experiment


def test_experiment_rejects_the_custom_navigation_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "korvid"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    data = experiment_mapping(tmp_path)
    data["runtime"]["backend"] = "korvid_native"
    data["evaluation"] = {"repetitions": 5, "seed": 0}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="korvid_upstream"):
        load_experiment(path)


def test_experiment_records_original_case_references_not_authored_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "korvid"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    data = experiment_mapping(tmp_path)
    data["runtime"]["backend"] = "korvid_upstream"
    data["evaluation"].update({
        "train": ["scenarios/image-pull-typo"],
        "validation": ["journeys/tui-follow"],
        "holdout": ["journeys/compare-namespaces"],
    })
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data))
    spec = load_experiment(path)
    assert spec.case_splits["validation"] == ("journeys/tui-follow",)
    assert spec.runtime.backend == "korvid_upstream"
    assert "native_cases" not in spec.to_mapping()
    assert "starting_seed_rules" not in spec.to_mapping()
    assert spec.to_mapping()["evaluation"]["train"] == ["scenarios/image-pull-typo"]


def test_upstream_source_serving_is_distinct_from_custom_ui_serving() -> None:
    from korvid_prompt_lab.contracts import KorvidUpstreamServing

    serving = KorvidUpstreamServing(
        backend="korvid_upstream", source_root="/korvid", base_url="http://127.0.0.1:11434",
        korvid_revision="33c483e041006eb20259a024ed85a9323e52c8f0", timeout_seconds=240,
        model_options={"native_thinking": True},
    )
    assert serving.backend == "korvid_upstream"


def test_candidate_can_hold_the_exact_original_tier_prompt() -> None:
    from korvid_prompt_lab.contracts import Candidate

    original = "Original operating prompt.\nKeep this trailing newline.\n"
    candidate = Candidate.from_mapping({
        "schema_version": 1, "candidate_id": "source-prompt",
        "components": {"tier_pack": original},
    })
    assert candidate.components["tier_pack"] == original


def test_check_only_verifies_baseline_and_original_assets_before_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from korvid_prompt_lab.cli import main

    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "korvid"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment_mapping(tmp_path)))
    seen: list[tuple[str, ...]] = []

    def inspect(serving, model, references, *, seed=0):
        seen.append(tuple(references))
        assert serving.base_url == "http://127.0.0.1:1"
        return {
            "prompt": {"baseline_equivalent": True, "pack_id": "low-korvid-operator", "source_sha256": "1" * 64},
            "cases": [{"reference": ref, "source_sha256": "2" * 64} for ref in references],
        }

    monkeypatch.setattr("korvid_prompt_lab.source_runtime.validate_source", lambda path: path)
    monkeypatch.setattr("korvid_prompt_lab.upstream.inspect_upstream", inspect)
    assert main(["run", "--experiment", str(path), "--check-only"]) == 0
    assert seen
    assert '"baseline_equivalent": true' in capsys.readouterr().out
