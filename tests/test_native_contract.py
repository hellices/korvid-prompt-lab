from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from korvid_prompt_lab.config import load_campaign, load_candidate
from korvid_prompt_lab.contracts import Candidate, KorvidNativeServing
from korvid_prompt_lab.native import initialize_native
from korvid_prompt_lab.native_contract import (
    NATIVE_KORVID_REVISION,
    native_cases,
    rules_candidate,
    rules_from_candidate,
)


def test_native_baseline_is_exactly_no_additional_rules() -> None:
    baseline = rules_candidate([])
    assert baseline.components == {"rules": "[]"}
    assert rules_from_candidate(baseline) == []
    assert "system" not in baseline.components


@pytest.mark.parametrize("rules", [[""], [" x "], ["x" * 1001], ["rule"] * 17, {"system": "x"}])
def test_native_rules_cannot_be_silently_dropped_by_korvid(rules: object) -> None:
    candidate = Candidate.from_mapping({
        "schema_version": 1, "candidate_id": "bad", "components": {"rules": json.dumps(rules)},
    })
    with pytest.raises(ValueError):
        rules_from_candidate(candidate)


def test_native_pack_matches_supported_low_scope() -> None:
    cases = native_cases()
    assert len(cases) == 15
    for split in ("train", "validation", "holdout"):
        selected = [case for case in cases if case.split == split]
        assert len(selected) == 5
        assert {case.case_id.removeprefix(f"{split}-") for case in selected} == {
            "pods", "helm", "all-pods", "logs", "describe",
        }


def test_native_init_uses_pinned_source_not_legacy_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "korvid"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    candidate_path, campaign_path = initialize_native(tmp_path / "native", model="ollama/qwen3:0.6b")
    campaign = load_campaign(campaign_path)
    assert isinstance(campaign.serving, KorvidNativeServing)
    assert campaign.serving.korvid_revision == NATIVE_KORVID_REVISION
    assert campaign.repetitions == 3
    assert len(campaign.cases) == 15
    assert rules_from_candidate(load_candidate(candidate_path)) == []
    with pytest.raises(FileExistsError):
        initialize_native(tmp_path / "native", model="ollama/qwen3:0.6b")


def test_native_campaign_cannot_evaluate_a_different_model_than_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "korvid"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    _, path = initialize_native(tmp_path / "native", model="ollama/qwen3:0.6b")
    payload = yaml.safe_load(path.read_text())
    payload["cases"][1]["models"] = ["ollama/another"]
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="model"):
        load_campaign(path)
