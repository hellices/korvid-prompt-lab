from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
from korvid.evals.scenario import Scenario

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab import stable_scenarios
from korvid_prompt_lab.stable_scenarios import (
    ScenarioClass,
    build_scenario_manifest,
)


def _scenario(
    *,
    scenario_id: str,
    question: str,
    root_cause: str,
    screen: str = "screen",
) -> Scenario:
    return Scenario(
        id=scenario_id,
        question=question,
        screen=screen,
        root_cause=root_cause,
        must_mention=(("x",),),
    )


def _record(
    *,
    scenario_id: str,
    scenario_class: ScenarioClass,
    question: str,
    root_cause: str,
    sort_key: str,
) -> stable_scenarios._ScenarioRecord:
    scenario = _scenario(scenario_id=scenario_id, question=question, root_cause=root_cause)
    question_sha256, fixture_sha256 = stable_scenarios._scenario_hashes(scenario)
    return stable_scenarios._ScenarioRecord(
        scenario_id=scenario_id,
        scenario_class=scenario_class,
        sort_key=sort_key,
        question_sha256=question_sha256,
        fixture_sha256=fixture_sha256,
    )


def test_scenario_hashes_keep_question_and_fixture_separate() -> None:
    base = _scenario(scenario_id="sample", question="why?", root_cause="none")
    edited = replace(base, question="what now?")

    base_question_sha256, base_fixture_sha256 = stable_scenarios._scenario_hashes(base)
    edited_question_sha256, edited_fixture_sha256 = stable_scenarios._scenario_hashes(edited)

    assert base_question_sha256 != edited_question_sha256
    assert base_fixture_sha256 == edited_fixture_sha256


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


def test_manifest_allocates_healthy_control_to_validation_or_milestone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        _record(
            scenario_id="healthy-control-a",
            scenario_class=ScenarioClass.HEALTHY_CONTROL,
            question="healthy",
            root_cause="none",
            sort_key="0",
        ),
        _record(
            scenario_id="bad-command-a",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            question="workload 1",
            root_cause="bad_command",
            sort_key="1",
        ),
        _record(
            scenario_id="crashloop-a",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            question="workload 2",
            root_cause="crashloop_app_error",
            sort_key="2",
        ),
        _record(
            scenario_id="job-a",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            question="workload 3",
            root_cause="job_backoff_limit",
            sort_key="3",
        ),
        _record(
            scenario_id="liveness-a",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            question="workload 4",
            root_cause="liveness_probe_failing",
            sort_key="4",
        ),
        _record(
            scenario_id="oom-a",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            question="workload 5",
            root_cause="oom_killed",
            sort_key="5",
        ),
        _record(
            scenario_id="readiness-a",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            question="workload 6",
            root_cause="readiness_probe_failing",
            sort_key="6",
        ),
        _record(
            scenario_id="image-pull-a",
            scenario_class=ScenarioClass.IMAGE_CONFIG,
            question="image 1",
            root_cause="image_pull_auth",
            sort_key="7",
        ),
        _record(
            scenario_id="missing-secret-a",
            scenario_class=ScenarioClass.IMAGE_CONFIG,
            question="image 2",
            root_cause="missing_secret",
            sort_key="8",
        ),
        _record(
            scenario_id="stuck-rollout-a",
            scenario_class=ScenarioClass.IMAGE_CONFIG,
            question="image 3",
            root_cause="stuck_rollout_bad_image",
            sort_key="9",
        ),
        _record(
            scenario_id="missing-config-a",
            scenario_class=ScenarioClass.IMAGE_CONFIG,
            question="image 4",
            root_cause="missing_config",
            sort_key="10",
        ),
        _record(
            scenario_id="pending-a",
            scenario_class=ScenarioClass.SCHEDULING_RESOURCES,
            question="sched 1",
            root_cause="insufficient_resources",
            sort_key="11",
        ),
        _record(
            scenario_id="node-pressure-a",
            scenario_class=ScenarioClass.SCHEDULING_RESOURCES,
            question="sched 2",
            root_cause="node_memory_pressure_eviction",
            sort_key="12",
        ),
        _record(
            scenario_id="quota-a",
            scenario_class=ScenarioClass.SCHEDULING_RESOURCES,
            question="sched 3",
            root_cause="quota_blocked",
            sort_key="13",
        ),
        _record(
            scenario_id="service-a",
            scenario_class=ScenarioClass.NETWORKING,
            question="net 1",
            root_cause="service_selector_mismatch",
            sort_key="14",
        ),
    )
    monkeypatch.setattr(stable_scenarios, "korvid_distribution_version", lambda: "0.3.0")
    monkeypatch.setattr(stable_scenarios, "_load_catalog", lambda version: records)

    manifest = build_scenario_manifest()

    assert "healthy-control-a" not in manifest.train
    assert "healthy-control-a" in manifest.validation or "healthy-control-a" in manifest.milestone
