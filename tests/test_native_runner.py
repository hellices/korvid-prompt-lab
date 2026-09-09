from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from korvid_prompt_lab.native import _validate_worker_result, validate_native_policy
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
