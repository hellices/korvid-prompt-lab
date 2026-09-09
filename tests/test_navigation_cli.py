from __future__ import annotations

from pathlib import Path

import pytest

from korvid_prompt_lab.cli import _build_runner, build_parser, main
from korvid_prompt_lab.config import load_campaign
from korvid_prompt_lab.navigation import KorvidNavigationRunner, initialize_navigation
from korvid_prompt_lab.navigation_cli import navigation_selections


def test_navigation_init_is_wired_to_cli(tmp_path: Path) -> None:
    assert main(["navigation-init", "--directory", str(tmp_path / "nav"), "--model", "qwen3:0.6b"]) == 0
    assert (tmp_path / "nav" / "candidate.yaml").is_file()
    assert (tmp_path / "nav" / "campaign.yaml").is_file()


def test_default_navigation_evaluation_never_touches_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NAVIGATION_MODEL_URL", "http://127.0.0.1:11434/v1")
    candidate, path = initialize_navigation(tmp_path / "nav", model="qwen3:0.6b")
    campaign = load_campaign(path)
    args = build_parser().parse_args(["evaluate", "--candidate", str(candidate), "--campaign", str(path)])
    navigation_selections(args, campaign)
    assert args.case_ids == [case.case_id for case in campaign.cases if case.template_id == "navigation-validation"]
    assert len(args.train_case_ids) == len(args.validation_case_ids) == 8
    assert not args.milestone_case_ids
    assert isinstance(_build_runner(campaign, model_endpoint=None), KorvidNavigationRunner)


def test_navigation_search_cannot_train_or_select_on_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NAVIGATION_MODEL_URL", "http://127.0.0.1:11434/v1")
    candidate, path = initialize_navigation(tmp_path / "nav", model="qwen3:0.6b")
    campaign = load_campaign(path)
    args = build_parser().parse_args([
        "optimize", "--candidate", str(candidate), "--campaign", str(path),
        "--max-metric-calls", "8", "--reflection-model", "ollama_chat/qwen3:14b",
        "--train-case-id", "holdout-pods",
    ])
    with pytest.raises(ValueError, match="train"):
        navigation_selections(args, campaign)


def test_holdout_evaluation_requires_explicit_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NAVIGATION_MODEL_URL", "http://127.0.0.1:11434/v1")
    candidate, path = initialize_navigation(tmp_path / "nav", model="qwen3:0.6b")
    campaign = load_campaign(path)
    args = build_parser().parse_args([
        "evaluate", "--candidate", str(candidate), "--campaign", str(path),
        "--navigation-split", "holdout",
    ])
    navigation_selections(args, campaign)
    assert len(args.case_ids) == 8
    assert all(case.startswith("holdout-") for case in args.case_ids)
