from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from korvid_prompt_lab.config import load_campaign
from korvid_prompt_lab.native import initialize_native
from korvid_prompt_lab.native_cli import export_native_rules, verify_rules_config
from korvid_prompt_lab.native_contract import rules_candidate


def test_failed_native_verification_never_leaves_an_applyable_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "korvid"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    _, campaign_path = initialize_native(tmp_path / "native", model="ollama/qwen3:0.6b")

    def fail(*args, **kwargs):
        raise ValueError("rules not applied")

    monkeypatch.setattr("korvid_prompt_lab.native_cli.verify_rules_config", fail)
    with pytest.raises(ValueError, match="not applied"):
        export_native_rules(rules_candidate(["Keep replies brief."]), load_campaign(campaign_path), tmp_path / "export")
    assert not (tmp_path / "export" / "korvid-config.yaml").exists()


def test_export_validates_exact_serialized_rules_and_declares_no_quality_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "korvid"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    _, campaign_path = initialize_native(tmp_path / "native", model="ollama/qwen3:0.6b")
    rules = ["Open the requested view before replying.", "Keep names and namespaces separate."]

    def verify(candidate, campaign, *, config_path):
        config = yaml.safe_load(config_path.read_text())
        assert config["agent"]["rules"] == rules
        assert config["agent"]["model_tier"] == "low"
        assert config["agent"]["follow"] is True
        assert config["readonly"] is True
        return {"prompt_fingerprint": "a" * 64, "policy": {"tier": "low"}}

    monkeypatch.setattr("korvid_prompt_lab.native_cli.verify_rules_config", verify)
    output = export_native_rules(rules_candidate(rules), load_campaign(campaign_path), tmp_path / "export")
    assert output.exists()
    manifest = json.loads((output.parent / "application-manifest.json").read_text())
    assert manifest["rules_applied"] is True
    assert manifest["qualification"] == "not_assessed"
    with pytest.raises(FileExistsError):
        export_native_rules(rules_candidate([]), load_campaign(campaign_path), output.parent)


def test_native_check_rejects_a_high_tier_even_when_rules_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "korvid"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    _, campaign_path = initialize_native(tmp_path / "native", model="ollama/qwen3:0.6b")
    monkeypatch.setattr("korvid_prompt_lab.native_cli.run_native_request", lambda *args: {
        "protocol_version": 1, "rules": [], "korvid_version": "0.4.1",
        "rules_applied": True, "ui_follow": True, "prompt_fingerprint": "not-a-hash",
        "policy": {"tier": "high", "tools": ["navigate"]},
    })
    with pytest.raises(ValueError, match="policy"):
        verify_rules_config(rules_candidate([]), load_campaign(campaign_path))
