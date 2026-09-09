from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from korvid_prompt_lab.cli import _build_runner, build_parser, main
from korvid_prompt_lab.config import load_campaign
from korvid_prompt_lab.native import initialize_native
from korvid_prompt_lab.native_cli import native_selections


def test_primary_cli_does_not_import_a_korvid_wheel() -> None:
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'korvid' or name.startswith('korvid.'):
        raise AssertionError('primary CLI imported Korvid wheel: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from korvid_prompt_lab.cli import build_parser
build_parser()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_native_init_command_has_shipped_baseline(tmp_path: Path) -> None:
    assert main(["native-init", "--directory", str(tmp_path / "native")]) == 0
    assert "rules:" in (tmp_path / "native" / "candidate.yaml").read_text()


def test_native_default_evaluation_is_only_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "korvid"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    candidate, path = initialize_native(tmp_path / "native", model="ollama/qwen3:0.6b")
    campaign = load_campaign(path)
    args = build_parser().parse_args([
        "evaluate", "--candidate", str(candidate), "--campaign", str(path),
    ])
    native_selections(args, campaign)
    assert len(args.case_ids) == 5
    assert all(case.startswith("validation-") for case in args.case_ids)
    assert _build_runner(campaign, model_endpoint=None).__class__.__name__ == "KorvidNativeRunner"


def test_native_holdout_is_not_a_search_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "korvid"))
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    candidate, path = initialize_native(tmp_path / "native", model="ollama/qwen3:0.6b")
    campaign = load_campaign(path)
    args = build_parser().parse_args([
        "optimize", "--candidate", str(candidate), "--campaign", str(path),
        "--validation-case-id", "holdout-pods", "--max-metric-calls", "8",
    ])
    with pytest.raises(ValueError, match="validation"):
        native_selections(args, campaign)
