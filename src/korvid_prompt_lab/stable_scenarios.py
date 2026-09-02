from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Literal

import yaml  # type: ignore[import-untyped]
from korvid.evals.scenario import Scenario, bundled_scenarios_dir, load_scenario

from .baseline import korvid_distribution_version

__all__ = [
    "FreshHoldoutExhaustedError",
    "RolloverScenarioManifest",
    "ScenarioAssignment",
    "ScenarioClass",
    "ScenarioManifest",
    "ScenarioSplitSummary",
    "build_rollover_scenario_manifest",
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


class FreshHoldoutExhaustedError(ValueError):
    """Raised when the catalog cannot provide a fresh rollover holdout."""


@dataclass(frozen=True, slots=True)
class RolloverScenarioManifest:
    manifest: ScenarioManifest
    consumed_ids: tuple[str, ...]
    fresh_milestone_ids: tuple[str, ...]
    audit_reserve_ids: tuple[str, ...]


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
    _validate_target_per_split(target_per_split)

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


def build_rollover_scenario_manifest(
    consumed: Sequence[ScenarioAssignment],
    *,
    target_per_split: int = 6,
) -> RolloverScenarioManifest:
    _validate_target_per_split(target_per_split)

    korvid_version = korvid_distribution_version()
    records = tuple(_load_catalog(korvid_version))
    if len(records) < 12:
        raise ValueError("installed Korvid scenario catalog must contain at least 12 eligible scenarios")

    consumed_records = _validate_consumed_records(consumed, records=records, korvid_version=korvid_version)
    if len(consumed_records) < target_per_split * 2:
        raise ValueError("rollover requires at least 12 consumed scenarios")

    consumed_id_set = {record.scenario_id for record in consumed_records}
    untouched_records = tuple(record for record in records if record.scenario_id not in consumed_id_set)
    if len(untouched_records) < target_per_split + 1:
        raise FreshHoldoutExhaustedError("fresh holdout exhausted")

    fresh_buckets = _group_records_by_class(
        untouched_records,
        sort_key=lambda record: _rollover_sort_key(korvid_version, record.scenario_id),
    )
    fresh_class_order = _present_class_order(untouched_records)
    milestone_records, remaining_fresh_buckets = _take_balanced_records(
        fresh_buckets,
        count=target_per_split,
        class_order=fresh_class_order,
    )
    if len(milestone_records) < target_per_split:
        raise FreshHoldoutExhaustedError("fresh holdout exhausted")

    remaining_fresh_records = _flatten_grouped_records(remaining_fresh_buckets)
    if not remaining_fresh_records:
        raise FreshHoldoutExhaustedError("fresh holdout exhausted")

    development_buckets = _group_records_by_class(consumed_records, sort_key=lambda record: record.sort_key)
    train_records, remaining_development_buckets = _take_balanced_records(
        development_buckets,
        count=target_per_split,
        class_order=_CLASS_ORDER,
    )
    validation_records, _ = _take_balanced_records(
        remaining_development_buckets,
        count=target_per_split,
        class_order=_CLASS_ORDER,
    )

    manifest = _build_manifest(
        korvid_version,
        train=train_records,
        validation=validation_records,
        milestone=milestone_records,
    )
    return RolloverScenarioManifest(
        manifest=manifest,
        consumed_ids=tuple(record.scenario_id for record in consumed_records),
        fresh_milestone_ids=manifest.milestone,
        audit_reserve_ids=(remaining_fresh_records[0].scenario_id,),
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


def _present_class_order(records: Sequence[_ScenarioRecord]) -> tuple[ScenarioClass, ...]:
    present = {record.scenario_class for record in records}
    return tuple(scenario_class for scenario_class in _CLASS_ORDER if scenario_class in present)


def _validate_target_per_split(target_per_split: int) -> None:
    if isinstance(target_per_split, bool) or not isinstance(target_per_split, int):
        raise TypeError("target_per_split must be a positive integer")
    if target_per_split <= 0:
        raise ValueError("target_per_split must be a positive integer")


def _validate_consumed_records(
    consumed: Sequence[ScenarioAssignment],
    *,
    records: Sequence[_ScenarioRecord],
    korvid_version: str,
) -> tuple[_ScenarioRecord, ...]:
    catalog_by_id = {record.scenario_id: record for record in records}
    seen_ids: set[str] = set()
    matched: list[_ScenarioRecord] = []

    for assignment in consumed:
        if assignment.scenario_id in seen_ids:
            raise ValueError("duplicate consumed scenario ids are not allowed")
        seen_ids.add(assignment.scenario_id)

        if assignment.korvid_version != korvid_version:
            raise ValueError("consumed assignment korvid version must match the installed korvid version")

        record = catalog_by_id.get(assignment.scenario_id)
        if record is None:
            raise ValueError(f"consumed scenario {assignment.scenario_id!r} was not found in the installed catalog")
        if assignment.question_sha256 != record.question_sha256:
            raise ValueError("consumed assignment question digest must match the installed catalog")
        if assignment.fixture_sha256 != record.fixture_sha256:
            raise ValueError("consumed assignment fixture digest must match the installed catalog")
        matched.append(record)

    return tuple(sorted(matched, key=lambda record: record.sort_key))


def _rollover_sort_key(korvid_version: str, scenario_id: str) -> str:
    return hashlib.sha256(f"rollover-v1:{korvid_version}:{scenario_id}".encode()).hexdigest()


def _group_records_by_class(
    records: Sequence[_ScenarioRecord],
    *,
    sort_key: Callable[[_ScenarioRecord], str],
) -> dict[ScenarioClass, list[_ScenarioRecord]]:
    grouped: dict[ScenarioClass, list[_ScenarioRecord]] = {scenario_class: [] for scenario_class in _CLASS_ORDER}
    for record in records:
        grouped[record.scenario_class].append(record)
    for items in grouped.values():
        items.sort(key=sort_key)
    return grouped


def _take_balanced_records(
    grouped: dict[ScenarioClass, list[_ScenarioRecord]],
    *,
    count: int,
    class_order: Sequence[ScenarioClass],
) -> tuple[tuple[_ScenarioRecord, ...], dict[ScenarioClass, list[_ScenarioRecord]]]:
    remaining = {scenario_class: items.copy() for scenario_class, items in grouped.items()}
    selected: list[_ScenarioRecord] = []

    while len(selected) < count:
        progress = False
        for scenario_class in class_order:
            items = remaining[scenario_class]
            if not items:
                continue
            selected.append(items.pop(0))
            progress = True
            if len(selected) == count:
                break
        if not progress:
            break

    return tuple(selected), remaining


def _flatten_grouped_records(
    grouped: dict[ScenarioClass, list[_ScenarioRecord]],
) -> tuple[_ScenarioRecord, ...]:
    return tuple(record for scenario_class in _CLASS_ORDER for record in grouped[scenario_class])


def _build_manifest(
    korvid_version: str,
    *,
    train: Sequence[_ScenarioRecord],
    validation: Sequence[_ScenarioRecord],
    milestone: Sequence[_ScenarioRecord],
) -> ScenarioManifest:
    split_records = {
        "train": tuple(train),
        "validation": tuple(validation),
        "milestone": tuple(milestone),
    }
    split_summaries: list[ScenarioSplitSummary] = []
    assignments: list[ScenarioAssignment] = []

    for split in _SPLITS:
        selected = split_records[split]
        assignments.extend(
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
        split_summaries.append(
            ScenarioSplitSummary(
                split_name=split,
                classes=_ordered_unique_classes(record.scenario_class for record in selected),
                scenario_ids=tuple(record.scenario_id for record in selected),
            )
        )

    return ScenarioManifest(
        korvid_version=korvid_version,
        assignments=tuple(assignments),
        train=tuple(record.scenario_id for record in split_records["train"]),
        validation=tuple(record.scenario_id for record in split_records["validation"]),
        milestone=tuple(record.scenario_id for record in split_records["milestone"]),
        split_summaries=tuple(split_summaries),
    )
