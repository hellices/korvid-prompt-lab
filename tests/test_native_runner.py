from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from korvid_prompt_lab.config import load_campaign
from korvid_prompt_lab.native import (
    KorvidNativeRunner,
    _validate_worker_result,
    initialize_native,
    validate_native_policy,
)
from korvid_prompt_lab.native_contract import rules_candidate
from korvid_prompt_lab.native_source import validate_native_source


def test_unreviewed_source_is_rejected_before_worker_start(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    with pytest.raises(ValueError, match="revision"):
        validate_native_source(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("rules_applied", False), ("korvid_version", "0.3.0"), ("execution_mode", "scripted"),
    ("rules", ["not the requested rules"]), ("case_id", "holdout-pods"),
])
def test_worker_identity_must_match_the_real_invocation(field: str, value: object) -> None:
    payload = {
        "protocol_version": 1, "rules_applied": True, "korvid_version": "0.4.1",
        "rules": [], "case_id": "train-pods", "execution_mode": "live",
        "ui_follow": True,
        "policy": {"tier": "low"}, "prompt_fingerprint": "a" * 64,
    }
    payload[field] = value
    with pytest.raises(ValueError):
        _validate_worker_result(payload, rules=[], case_id="train-pods", execution_mode="live")


def test_self_reported_low_tier_cannot_widen_tools() -> None:
    with pytest.raises(ValueError, match="policy"):
        validate_native_policy({"tier": "low", "tools": ["navigate"]})


def test_library_case_cannot_change_the_model_measured_by_the_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "source"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    _, path = initialize_native(tmp_path / "setup", model="ollama/qwen3:0.6b")
    campaign = load_campaign(path)
    foreign_case = replace(campaign.cases[0], models=("ollama/another-model",))

    def unexpected_worker(*args, **kwargs):
        pytest.fail("a foreign case must be rejected before worker invocation")

    monkeypatch.setattr("korvid_prompt_lab.native.run_native_request", unexpected_worker)
    with pytest.raises(ValueError, match="campaign"):
        KorvidNativeRunner(campaign).run(rules_candidate([]), foreign_case, tmp_path / "run")
    assert not (tmp_path / "run").exists()
