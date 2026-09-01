from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Literal

import yaml  # type: ignore[import-untyped]
from korvid.evals.scenario import Scenario, bundled_scenarios_dir, load_scenario

from .baseline import korvid_distribution_version

__all__ = [
    "ScenarioAssignment",
    "ScenarioClass",
    "ScenarioManifest",
    "ScenarioSplitSummary",
    "build_scenario_manifest",
]


class ScenarioClass(StrEnum):
    WORKLOAD_HEALTH = "workload-health"
    IMAGE_CONFIG = "image-config"
    SCHEDULING_RESOURCES = "scheduling-resources"
    NETWORKING = "networking"
    STORAGE = "storage"
    HEALTHY_CONTROL = "healthy-control"


ScenarioSplit = Literal["train", "validation", "milestone"]


@dataclass(frozen=True, slots=True)
class ScenarioAssignment:
    scenario_id: str
    scenario_class: ScenarioClass
    split: ScenarioSplit
    question_sha256: str
    fixture_sha256: str
    korvid_version: str


@dataclass(frozen=True, slots=True)
class ScenarioSplitSummary:
    split_name: ScenarioSplit
    classes: tuple[ScenarioClass, ...]
    scenario_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScenarioManifest:
    korvid_version: str
    assignments: tuple[ScenarioAssignment, ...]
    train: tuple[str, ...]
    validation: tuple[str, ...]
    milestone: tuple[str, ...]
    split_summaries: tuple[ScenarioSplitSummary, ...]


@dataclass(frozen=True, slots=True)
class _ScenarioRecord:
    scenario_id: str
    scenario_class: ScenarioClass
    sort_key: str
    question_sha256: str
    fixture_sha256: str


_SPLITS: tuple[ScenarioSplit, ...] = ("train", "validation", "milestone")
_CLASS_ORDER: tuple[ScenarioClass, ...] = (
    ScenarioClass.HEALTHY_CONTROL,
    ScenarioClass.WORKLOAD_HEALTH,
    ScenarioClass.IMAGE_CONFIG,
    ScenarioClass.SCHEDULING_RESOURCES,
    ScenarioClass.NETWORKING,
    ScenarioClass.STORAGE,
)

_ROOT_CAUSE_TO_CLASS: dict[str, ScenarioClass] = {
    "bad_command": ScenarioClass.WORKLOAD_HEALTH,
    "crashloop_app_error": ScenarioClass.WORKLOAD_HEALTH,
    "dependency_unreachable": ScenarioClass.NETWORKING,
    "missing_config": ScenarioClass.IMAGE_CONFIG,
    "missing_secret": ScenarioClass.IMAGE_CONFIG,
    "image_pull_auth": ScenarioClass.IMAGE_CONFIG,
    "image_pull_typo": ScenarioClass.IMAGE_CONFIG,
    "stuck_rollout_bad_image": ScenarioClass.IMAGE_CONFIG,
    "job_backoff_limit": ScenarioClass.WORKLOAD_HEALTH,
    "liveness_probe_failing": ScenarioClass.WORKLOAD_HEALTH,
    "missing_configmap": ScenarioClass.STORAGE,
    "node_memory_pressure_eviction": ScenarioClass.SCHEDULING_RESOURCES,
    "oom_killed": ScenarioClass.WORKLOAD_HEALTH,
    "insufficient_resources": ScenarioClass.SCHEDULING_RESOURCES,
    "node_selector_mismatch": ScenarioClass.SCHEDULING_RESOURCES,
    "pvc_pending_no_storageclass": ScenarioClass.STORAGE,
    "readiness_probe_failing": ScenarioClass.WORKLOAD_HEALTH,
    "service_endpoints_not_ready": ScenarioClass.NETWORKING,
    "service_selector_mismatch": ScenarioClass.NETWORKING,
}

_ID_PREFIX_TO_CLASS: tuple[tuple[str, ScenarioClass], ...] = (
    ("healthy-", ScenarioClass.HEALTHY_CONTROL),
    ("image-pull-", ScenarioClass.IMAGE_CONFIG),
    ("crashloop-missing-env", ScenarioClass.IMAGE_CONFIG),
    ("stuck-rollout", ScenarioClass.IMAGE_CONFIG),
    ("missing-secret-", ScenarioClass.IMAGE_CONFIG),
    ("missing-configmap-", ScenarioClass.STORAGE),
    ("pvc-", ScenarioClass.STORAGE),
    ("node-pressure-", ScenarioClass.SCHEDULING_RESOURCES),
    ("pending-", ScenarioClass.SCHEDULING_RESOURCES),
    ("quota-", ScenarioClass.SCHEDULING_RESOURCES),
    ("service-", ScenarioClass.NETWORKING),
    ("bad-command-", ScenarioClass.WORKLOAD_HEALTH),
    ("crashloop-", ScenarioClass.WORKLOAD_HEALTH),
    ("init-container-", ScenarioClass.WORKLOAD_HEALTH),
    ("job-", ScenarioClass.WORKLOAD_HEALTH),
    ("liveness-", ScenarioClass.WORKLOAD_HEALTH),
    ("oom-", ScenarioClass.WORKLOAD_HEALTH),
    ("readiness-", ScenarioClass.WORKLOAD_HEALTH),
)


def build_scenario_manifest(target_per_split: int = 6) -> ScenarioManifest:
    if isinstance(target_per_split, bool) or not isinstance(target_per_split, int):
        raise TypeError("target_per_split must be a positive integer")
    if target_per_split <= 0:
        raise ValueError("target_per_split must be a positive integer")

    korvid_version = korvid_distribution_version()
    records = tuple(_load_catalog(korvid_version))
    if len(records) < 12:
        raise ValueError("installed Korvid scenario catalog must contain at least 12 eligible scenarios")

    grouped: dict[ScenarioClass, list[_ScenarioRecord]] = {scenario_class: [] for scenario_class in _CLASS_ORDER}
    for record in records:
        grouped[record.scenario_class].append(record)

    for items in grouped.values():
        items.sort(key=lambda item: item.sort_key)

    split_buckets: dict[ScenarioSplit, list[_ScenarioRecord]] = {split: [] for split in _SPLITS}
    for scenario_class in _CLASS_ORDER:
        items = grouped[scenario_class]
        split_order: tuple[ScenarioSplit, ...] = _SPLITS
        if scenario_class is ScenarioClass.HEALTHY_CONTROL and items:
            split_order = ("validation", "milestone", "train")
        for index, record in enumerate(items):
            split_buckets[split_order[index % len(split_order)]].append(record)

    split_size = min(target_per_split, *(len(split_buckets[split]) for split in _SPLITS))
    if split_size < 4:
        raise ValueError("installed Korvid scenario catalog cannot satisfy the configured split size")

    split_assignments: dict[ScenarioSplit, tuple[ScenarioAssignment, ...]] = {}
    split_ids: dict[ScenarioSplit, tuple[str, ...]] = {}
    split_summaries: list[ScenarioSplitSummary] = []
    assignments: list[ScenarioAssignment] = []

    for split in _SPLITS:
        selected = split_buckets[split][:split_size]
        split_ids[split] = tuple(record.scenario_id for record in selected)
        split_assignments[split] = tuple(
            ScenarioAssignment(
                scenario_id=record.scenario_id,
                scenario_class=record.scenario_class,
                split=split,
                question_sha256=record.question_sha256,
                fixture_sha256=record.fixture_sha256,
                korvid_version=korvid_version,
            )
            for record in selected
        )
        assignments.extend(split_assignments[split])
        split_summaries.append(
            ScenarioSplitSummary(
                split_name=split,
                classes=_ordered_unique_classes(record.scenario_class for record in selected),
                scenario_ids=split_ids[split],
            )
        )

    return ScenarioManifest(
        korvid_version=korvid_version,
        assignments=tuple(assignments),
        train=split_ids["train"],
        validation=split_ids["validation"],
        milestone=split_ids["milestone"],
        split_summaries=tuple(split_summaries),
    )


@lru_cache(maxsize=1)
def _load_catalog(korvid_version: str) -> tuple[_ScenarioRecord, ...]:
    directory = bundled_scenarios_dir()
    if not directory.is_dir():
        raise ValueError(f"korvid bundled scenarios directory not found: {directory}")

    records: list[_ScenarioRecord] = []
    for path in sorted(directory.glob("*.yaml")):
        scenario = load_scenario(path)
        scenario_class = _scenario_class(scenario)
        if scenario_class is None:
            continue
        question_sha256, fixture_sha256 = _scenario_hashes(scenario)
        sort_key = hashlib.sha256(f"{korvid_version}:{scenario.id}".encode()).hexdigest()
        records.append(
            _ScenarioRecord(
                scenario_id=scenario.id,
                scenario_class=scenario_class,
                sort_key=sort_key,
                question_sha256=question_sha256,
                fixture_sha256=fixture_sha256,
            )
        )
    return tuple(records)


def _scenario_hashes(scenario: Scenario) -> tuple[str, str]:
    question_sha256 = hashlib.sha256(scenario.question.encode("utf-8")).hexdigest()
    canonical_fixture = asdict(scenario)
    canonical_fixture.pop("question", None)
    fixture_payload = yaml.safe_dump(canonical_fixture, sort_keys=True, allow_unicode=True)
    fixture_sha256 = hashlib.sha256(fixture_payload.encode("utf-8")).hexdigest()
    return question_sha256, fixture_sha256


def _scenario_class(scenario: Scenario) -> ScenarioClass | None:
    scenario_id = scenario.id
    if scenario_id.startswith("healthy-"):
        return ScenarioClass.HEALTHY_CONTROL

    root_cause = getattr(scenario, "root_cause", None)
    if isinstance(root_cause, str):
        mapped = _ROOT_CAUSE_TO_CLASS.get(root_cause)
        if mapped is not None:
            return mapped

    for prefix, scenario_class in _ID_PREFIX_TO_CLASS:
        if scenario_id.startswith(prefix):
            return scenario_class
    return None


def _ordered_unique_classes(classes: Iterable[ScenarioClass]) -> tuple[ScenarioClass, ...]:
    unique: dict[ScenarioClass, None] = {}
    for scenario_class in classes:
        unique[scenario_class] = None
    return tuple(unique)
