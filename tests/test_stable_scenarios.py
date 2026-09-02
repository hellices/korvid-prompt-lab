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


def _assignment(
    record: stable_scenarios._ScenarioRecord,
    *,
    split: stable_scenarios.ScenarioSplit,
    korvid_version: str = "0.3.0",
) -> stable_scenarios.ScenarioAssignment:
    return stable_scenarios.ScenarioAssignment(
        scenario_id=record.scenario_id,
        scenario_class=record.scenario_class,
        split=split,
        question_sha256=record.question_sha256,
        fixture_sha256=record.fixture_sha256,
        korvid_version=korvid_version,
    )


def _rollover_catalog() -> tuple[tuple[stable_scenarios._ScenarioRecord, ...], tuple[str, ...]]:
    consumed_records = (
        _record(
            scenario_id="healthy-control-a",
            scenario_class=ScenarioClass.HEALTHY_CONTROL,
            question="healthy 1",
            root_cause="none",
            sort_key="000",
        ),
        _record(
            scenario_id="healthy-control-b",
            scenario_class=ScenarioClass.HEALTHY_CONTROL,
            question="healthy 2",
            root_cause="none",
            sort_key="001",
        ),
        _record(
            scenario_id="bad-command-a",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            question="workload 1",
            root_cause="bad_command",
            sort_key="002",
        ),
        _record(
            scenario_id="crashloop-a",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            question="workload 2",
            root_cause="crashloop_app_error",
            sort_key="003",
        ),
        _record(
            scenario_id="job-a",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            question="workload 3",
            root_cause="job_backoff_limit",
            sort_key="004",
        ),
        _record(
            scenario_id="liveness-a",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            question="workload 4",
            root_cause="liveness_probe_failing",
            sort_key="005",
        ),
        _record(
            scenario_id="image-pull-a",
            scenario_class=ScenarioClass.IMAGE_CONFIG,
            question="image 1",
            root_cause="image_pull_auth",
            sort_key="006",
        ),
        _record(
            scenario_id="image-pull-b",
            scenario_class=ScenarioClass.IMAGE_CONFIG,
            question="image 2",
            root_cause="image_pull_typo",
            sort_key="007",
        ),
        _record(
            scenario_id="missing-secret-a",
            scenario_class=ScenarioClass.IMAGE_CONFIG,
            question="image 3",
            root_cause="missing_secret",
            sort_key="008",
        ),
        _record(
            scenario_id="stuck-rollout-a",
            scenario_class=ScenarioClass.IMAGE_CONFIG,
            question="image 4",
            root_cause="stuck_rollout_bad_image",
            sort_key="009",
        ),
        _record(
            scenario_id="pending-memory",
            scenario_class=ScenarioClass.SCHEDULING_RESOURCES,
            question="sched 1",
            root_cause="insufficient_resources",
            sort_key="010",
        ),
        _record(
            scenario_id="pending-disk",
            scenario_class=ScenarioClass.SCHEDULING_RESOURCES,
            question="sched 2",
            root_cause="node_selector_mismatch",
            sort_key="011",
        ),
        _record(
            scenario_id="quota-a",
            scenario_class=ScenarioClass.SCHEDULING_RESOURCES,
            question="sched 3",
            root_cause="insufficient_resources",
            sort_key="012",
        ),
        _record(
            scenario_id="service-a",
            scenario_class=ScenarioClass.NETWORKING,
            question="net 1",
            root_cause="service_selector_mismatch",
            sort_key="013",
        ),
        _record(
            scenario_id="service-b",
            scenario_class=ScenarioClass.NETWORKING,
            question="net 2",
            root_cause="service_endpoints_not_ready",
            sort_key="014",
        ),
        _record(
            scenario_id="pvc-consumed-a",
            scenario_class=ScenarioClass.STORAGE,
            question="storage 1",
            root_cause="pvc_pending_no_storageclass",
            sort_key="015",
        ),
        _record(
            scenario_id="configmap-consumed-a",
            scenario_class=ScenarioClass.STORAGE,
            question="storage 2",
            root_cause="missing_configmap",
            sort_key="016",
        ),
        _record(
            scenario_id="pvc-consumed-b",
            scenario_class=ScenarioClass.STORAGE,
            question="storage 3",
            root_cause="pvc_pending_no_storageclass",
            sort_key="017",
        ),
    )
    fresh_records = (
        _record(
            scenario_id="missing-configmap-mount",
            scenario_class=ScenarioClass.STORAGE,
            question="fresh storage 1",
            root_cause="missing_configmap",
            sort_key="018",
        ),
        _record(
            scenario_id="pvc-pending-no-storageclass",
            scenario_class=ScenarioClass.STORAGE,
            question="fresh storage 2",
            root_cause="pvc_pending_no_storageclass",
            sort_key="019",
        ),
        _record(
            scenario_id="pvc-wait-for-first-consumer",
            scenario_class=ScenarioClass.STORAGE,
            question="fresh storage 3",
            root_cause="pvc_pending_no_storageclass",
            sort_key="020",
        ),
        _record(
            scenario_id="node-pressure-eviction",
            scenario_class=ScenarioClass.SCHEDULING_RESOURCES,
            question="fresh sched 1",
            root_cause="node_memory_pressure_eviction",
            sort_key="021",
        ),
        _record(
            scenario_id="pending-insufficient-cpu",
            scenario_class=ScenarioClass.SCHEDULING_RESOURCES,
            question="fresh sched 2",
            root_cause="insufficient_resources",
            sort_key="022",
        ),
        _record(
            scenario_id="service-endpoints-not-ready",
            scenario_class=ScenarioClass.NETWORKING,
            question="fresh net 1",
            root_cause="service_endpoints_not_ready",
            sort_key="023",
        ),
        _record(
            scenario_id="service-selector-mismatch",
            scenario_class=ScenarioClass.NETWORKING,
            question="fresh net 2",
            root_cause="service_selector_mismatch",
            sort_key="024",
        ),
    )
    return consumed_records + fresh_records, tuple(record.scenario_id for record in consumed_records)


def _consumed_assignments(
    consumed_ids: tuple[str, ...],
    records: tuple[stable_scenarios._ScenarioRecord, ...],
) -> tuple[stable_scenarios.ScenarioAssignment, ...]:
    assignments: list[stable_scenarios.ScenarioAssignment] = []
    split_cycle: tuple[stable_scenarios.ScenarioSplit, ...] = ("train", "validation", "milestone")
    for index, record in enumerate(records):
        if record.scenario_id in consumed_ids:
            assignments.append(_assignment(record, split=split_cycle[index % len(split_cycle)]))
    return tuple(assignments)


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


def test_rollover_manifest_uses_fresh_balanced_holdout_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, consumed_ids = _rollover_catalog()
    consumed = _consumed_assignments(consumed_ids, records)
    monkeypatch.setattr(stable_scenarios, "korvid_distribution_version", lambda: "0.3.0")
    monkeypatch.setattr(stable_scenarios, "_load_catalog", lambda version: tuple(reversed(records)))

    rollover = stable_scenarios.build_rollover_scenario_manifest(consumed, target_per_split=6)
    repeated = stable_scenarios.build_rollover_scenario_manifest(tuple(reversed(consumed)), target_per_split=6)

    assert len(rollover.manifest.train) == 6
    assert len(rollover.manifest.validation) == 6
    assert len(rollover.manifest.milestone) == 6
    assert len(rollover.audit_reserve_ids) == 1
    assert set(rollover.manifest.milestone).isdisjoint(rollover.consumed_ids)
    assert set(rollover.manifest.train + rollover.manifest.validation) <= set(rollover.consumed_ids)
    assert set(rollover.manifest.milestone).isdisjoint(rollover.audit_reserve_ids)
    assert rollover.manifest.milestone == rollover.fresh_milestone_ids
    assert rollover == repeated

    assignments_by_id = {
        assignment.scenario_id: assignment.scenario_class for assignment in rollover.manifest.assignments
    }
    milestone_classes = [assignments_by_id[scenario_id] for scenario_id in rollover.manifest.milestone]
    assert milestone_classes.count(ScenarioClass.STORAGE) == 2
    assert milestone_classes.count(ScenarioClass.SCHEDULING_RESOURCES) == 2
    assert milestone_classes.count(ScenarioClass.NETWORKING) == 2


def test_rollover_manifest_rejects_changed_korvid_version(monkeypatch: pytest.MonkeyPatch) -> None:
    records, consumed_ids = _rollover_catalog()
    consumed = list(_consumed_assignments(consumed_ids, records))
    consumed[0] = _assignment(records[0], split="train", korvid_version="0.2.0")
    monkeypatch.setattr(stable_scenarios, "korvid_distribution_version", lambda: "0.3.0")
    monkeypatch.setattr(stable_scenarios, "_load_catalog", lambda version: records)

    with pytest.raises(ValueError, match="korvid version"):
        stable_scenarios.build_rollover_scenario_manifest(tuple(consumed), target_per_split=6)


def test_rollover_manifest_rejects_changed_question_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    records, consumed_ids = _rollover_catalog()
    consumed = list(_consumed_assignments(consumed_ids, records))
    consumed[0] = stable_scenarios.ScenarioAssignment(
        scenario_id=consumed[0].scenario_id,
        scenario_class=consumed[0].scenario_class,
        split=consumed[0].split,
        question_sha256="0" * 64,
        fixture_sha256=consumed[0].fixture_sha256,
        korvid_version=consumed[0].korvid_version,
    )
    monkeypatch.setattr(stable_scenarios, "korvid_distribution_version", lambda: "0.3.0")
    monkeypatch.setattr(stable_scenarios, "_load_catalog", lambda version: records)

    with pytest.raises(ValueError, match="question digest"):
        stable_scenarios.build_rollover_scenario_manifest(tuple(consumed), target_per_split=6)


def test_rollover_manifest_rejects_changed_fixture_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    records, consumed_ids = _rollover_catalog()
    consumed = list(_consumed_assignments(consumed_ids, records))
    consumed[0] = stable_scenarios.ScenarioAssignment(
        scenario_id=consumed[0].scenario_id,
        scenario_class=consumed[0].scenario_class,
        split=consumed[0].split,
        question_sha256=consumed[0].question_sha256,
        fixture_sha256="f" * 64,
        korvid_version=consumed[0].korvid_version,
    )
    monkeypatch.setattr(stable_scenarios, "korvid_distribution_version", lambda: "0.3.0")
    monkeypatch.setattr(stable_scenarios, "_load_catalog", lambda version: records)

    with pytest.raises(ValueError, match="fixture digest"):
        stable_scenarios.build_rollover_scenario_manifest(tuple(consumed), target_per_split=6)


def test_rollover_manifest_rejects_duplicate_consumed_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    records, consumed_ids = _rollover_catalog()
    consumed = _consumed_assignments(consumed_ids, records)
    monkeypatch.setattr(stable_scenarios, "korvid_distribution_version", lambda: "0.3.0")
    monkeypatch.setattr(stable_scenarios, "_load_catalog", lambda version: records)

    with pytest.raises(ValueError, match="duplicate"):
        stable_scenarios.build_rollover_scenario_manifest(consumed + (consumed[0],), target_per_split=6)


def test_rollover_manifest_rejects_fewer_than_twelve_consumed_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, consumed_ids = _rollover_catalog()
    consumed = _consumed_assignments(consumed_ids[:11], records)
    monkeypatch.setattr(stable_scenarios, "korvid_distribution_version", lambda: "0.3.0")
    monkeypatch.setattr(stable_scenarios, "_load_catalog", lambda version: records)

    with pytest.raises(ValueError, match="consumed"):
        stable_scenarios.build_rollover_scenario_manifest(consumed, target_per_split=6)


def test_rollover_manifest_raises_when_fresh_holdout_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, consumed_ids = _rollover_catalog()
    reduced_records = tuple(record for record in records if record.scenario_id != "pvc-wait-for-first-consumer")
    consumed = _consumed_assignments(consumed_ids, reduced_records)
    monkeypatch.setattr(stable_scenarios, "korvid_distribution_version", lambda: "0.3.0")
    monkeypatch.setattr(stable_scenarios, "_load_catalog", lambda version: reduced_records)

    with pytest.raises(stable_scenarios.FreshHoldoutExhaustedError):
        stable_scenarios.build_rollover_scenario_manifest(consumed, target_per_split=6)
