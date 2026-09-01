from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.stable_scenarios import build_scenario_manifest


def test_manifest_builds_disjoint_stratified_splits() -> None:
    manifest = build_scenario_manifest()
    train = set(manifest.train)
    validation = set(manifest.validation)
    milestone = set(manifest.milestone)

    assert len(train) == len(validation) == len(milestone) == 6
    assert not train & validation
    assert not train & milestone
    assert not validation & milestone
    assert all(len(split.classes) >= 2 for split in manifest.split_summaries)
    assert any(
        split.split_name in {"validation", "milestone"}
        and "healthy-control" in split.classes
        for split in manifest.split_summaries
    )


def test_manifest_is_stable_for_the_same_installed_catalog() -> None:
    assert build_scenario_manifest() == build_scenario_manifest()
