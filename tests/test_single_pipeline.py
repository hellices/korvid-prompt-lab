"""Guard the supported surface: one original-Korvid prompt/evaluation pipeline."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REMOVED_MODULES = (
    "baseline", "bridge", "bridge_worker", "campaign_artifacts", "campaign_cli", "experiment_cli",
    "campaigns", "comparison", "config", "korvid_pin", "korvid_readonly",
    "native", "native_cli", "native_contract", "native_source", "native_worker",
    "navigation", "navigation_cases", "navigation_cli", "navigation_harness",
    "navigation_runtime", "publish", "round_cli", "rounds",
)


def test_only_current_run_command_is_exposed() -> None:
    from korvid_prompt_lab.cli import build_parser

    parser = build_parser()
    command_action = next(action for action in parser._actions if action.dest == "command")
    assert command_action.choices is not None
    assert set(command_action.choices) == {"run"}


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_legacy_modules_are_removed_not_deprecated(name: str) -> None:
    assert importlib.util.find_spec(f"korvid_prompt_lab.{name}") is None
    assert not (ROOT / "src" / "korvid_prompt_lab" / f"{name}.py").exists()


def test_package_has_no_legacy_entry_points_or_dependency_extra() -> None:
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert set(package["project"]["scripts"]) == {"korvid-prompt-lab"}
    assert "legacy" not in package["project"].get("optional-dependencies", {})
    assert not any(
        "korvid" in dependency.lower()
        for dependency in package["project"]["optional-dependencies"]["dev"]
    )
    assert not any(
        dependency.lower().startswith("korvid")
        for dependency in package["project"]["dependencies"]
    )


def test_old_public_loader_and_execution_contracts_are_absent() -> None:
    import korvid_prompt_lab
    from korvid_prompt_lab import contracts, runner, scoring

    for name in ("load_campaign", "ProcessServing", "AKSPortForwardServing"):
        assert not hasattr(korvid_prompt_lab, name)
        assert not hasattr(contracts, name)
    for name in ("KorvidReadonlyServing", "KorvidNavigationServing", "KorvidNativeServing"):
        assert not hasattr(contracts, name)
    assert not hasattr(runner, "KorvidProcessRunner")
    for name in ("OperationGrade", "BridgeResult", "ScoredResult", "grade_quality", "score_result"):
        assert not hasattr(scoring, name)


def test_obsolete_automation_and_example_formats_are_absent() -> None:
    for directory in (
        "scripts", "infra", "examples/campaigns", "examples/candidates",
        "examples/optimization-campaigns", "docs/legacy-workflows.md",
    ):
        assert not (ROOT / directory).exists()
    for workflow in ("grounding-round.yml", "optimization-campaign.yml"):
        assert not (ROOT / ".github" / "workflows" / workflow).exists()


def test_aks_target_is_transport_only() -> None:
    from korvid_prompt_lab.aks import AKSForwardTarget

    target = AKSForwardTarget("rg", "aks", "ollama", "ollama", "qwen3:0.6b")
    assert target.model == "qwen3:0.6b"
    assert not hasattr(target, "command")
    assert not hasattr(target, "backend")
