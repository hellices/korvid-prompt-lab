from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

from .artifacts import write_json_artifact
from .contracts import Candidate
from .stable_scenarios import (
    RolloverScenarioManifest,
    ScenarioAssignment,
    ScenarioClass,
    ScenarioManifest,
    ScenarioSplitSummary,
)

__all__ = [
    "PriorCampaignEvidence",
    "PriorFinalistEvidence",
    "load_prior_campaign_evidence",
    "write_rollover_lineage",
    "write_rollover_winner",
]

_SPLITS: tuple[Literal["train", "validation", "milestone"], ...] = (
    "train",
    "validation",
    "milestone",
)


@dataclass(frozen=True, slots=True)
class PriorFinalistEvidence:
    candidate_id: str
    candidate_fingerprint: str
    append: str
    validation_delta: float
    milestone_delta: float


@dataclass(frozen=True, slots=True)
class PriorCampaignEvidence:
    artifact_root: Path
    campaign_id: str
    korvid_version: str
    summary_sha256: str
    scenario_manifest_sha256: str
    consumed_assignments: tuple[ScenarioAssignment, ...]
    finalist: PriorFinalistEvidence


@dataclass(frozen=True, slots=True)
class _QualificationDelta:
    candidate_id: str
    validation_delta: float
    milestone_delta: float


def load_prior_campaign_evidence(root: Path | str) -> PriorCampaignEvidence:
    artifact_root = _resolve_root(root)
    summary_path = _resolve_required_file(artifact_root, "stable-search-summary.json")
    scenario_manifest_path = _resolve_required_file(artifact_root, "scenario-manifest.json")
    candidate_manifest_path = _resolve_required_file(artifact_root, "candidate-manifest.json")
    qualification_path = _resolve_required_file(artifact_root, "stage-c/qualification-summary.json")

    summary_sha256 = _sha256_file(summary_path)
    scenario_manifest_sha256 = _sha256_file(scenario_manifest_path)
    summary = _load_json_object(summary_path)
    campaign_id = _load_summary(summary, summary_path)
    scenario_manifest = _load_scenario_manifest(_load_json_object(scenario_manifest_path), scenario_manifest_path)
    candidates = _load_candidate_manifest(_load_json_object(candidate_manifest_path), candidate_manifest_path)
    qualification = _load_qualification(_load_json_object(qualification_path), qualification_path)
    finalist = _select_finalist(qualification, candidates)

    return PriorCampaignEvidence(
        artifact_root=artifact_root,
        campaign_id=campaign_id,
        korvid_version=scenario_manifest.korvid_version,
        summary_sha256=summary_sha256,
        scenario_manifest_sha256=scenario_manifest_sha256,
        consumed_assignments=scenario_manifest.assignments,
        finalist=finalist,
    )


def write_rollover_lineage(
    path: Path | str,
    evidence: PriorCampaignEvidence,
    rollover: RolloverScenarioManifest,
    *,
    terminal_reason: str | None = None,
) -> Path:
    consumed = sorted(assignment.fixture_sha256 for assignment in evidence.consumed_assignments)
    fresh_milestone = sorted(
        assignment.fixture_sha256
        for assignment in rollover.manifest.assignments
        if assignment.split == "milestone"
    )
    return write_json_artifact(
        path,
        {
            "schema_version": 1,
            "prior": {
                "campaign_id": evidence.campaign_id,
                "decision": "no_stable_winner",
                "stable_search_summary_sha256": evidence.summary_sha256,
                "scenario_manifest_sha256": evidence.scenario_manifest_sha256,
                "finalist_id": evidence.finalist.candidate_id,
                "finalist_fingerprint": evidence.finalist.candidate_fingerprint,
            },
            "scenario_consumption": {
                "korvid_version": evidence.korvid_version,
                "consumed": consumed,
                "fresh_milestone": fresh_milestone,
                "counts": {
                    "train": len(rollover.manifest.train),
                    "validation": len(rollover.manifest.validation),
                    "milestone": len(rollover.manifest.milestone),
                    "audit_reserve": len(rollover.audit_reserve_ids),
                },
            },
            "candidate_matrix_version": "rollover-v1",
            "max_target_calls": 306,
            "terminal_reason": terminal_reason,
        },
    )


def write_rollover_winner(path: Path | str, candidate: Candidate) -> Path:
    output_path = Path(path)
    if output_path.is_symlink():
        raise FileExistsError(f"refusing to write rollover winner over a symlink: {output_path}")
    if output_path.exists():
        raise FileExistsError(f"rollover winner output already exists: {output_path}")
    if set(candidate.components) != {"system", "append"}:
        raise ValueError("rollover winner candidate components must be exactly system and append")

    payload = {
        "schema_version": candidate.schema_version,
        "candidate_id": candidate.candidate_id,
        "components": candidate.components,
        "metadata": candidate.metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    if temp_path.exists() or temp_path.is_symlink():
        raise FileExistsError(f"rollover winner temporary output already exists: {temp_path}")
    temp_fd: int | None = None
    temp_created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        temp_fd = os.open(temp_path, flags, 0o600)
        temp_created = True
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            temp_fd = None
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        os.replace(temp_path, output_path)
        temp_created = False
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if temp_created:
            temp_path.unlink(missing_ok=True)
    return output_path


def _resolve_root(root: Path | str) -> Path:
    artifact_root = Path(root)
    if artifact_root.is_symlink():
        raise ValueError(f"prior artifact root must not be a symlink: {artifact_root}")
    try:
        resolved = artifact_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"prior artifact root does not exist: {artifact_root}") from exc
    if not resolved.is_dir():
        raise ValueError(f"prior artifact root must be a directory: {artifact_root}")
    return resolved


def _resolve_required_file(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"missing required prior artifact: {relative_path}") from exc
    if not _is_confined_to_root(resolved, root):
        raise ValueError(f"required prior artifact path escapes prior root: {relative_path}")

    current = root
    if current.is_symlink():
        raise ValueError(f"required prior artifact path contains a symlink: {root}")
    for part in Path(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"required prior artifact path contains a symlink: {current}")
    if not resolved.is_file():
        raise ValueError(f"required prior artifact is not a regular file: {relative_path}")
    return resolved


def _is_confined_to_root(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name} must be a JSON object")  # noqa: TRY004 - preserve validation API
    return payload


def _load_summary(payload: Mapping[str, Any], path: Path) -> str:
    if _require_int(payload.get("schema_version"), f"{path.name}.schema_version") != 1:
        raise ValueError(f"{path.name}.schema_version must be 1")
    campaign_id = _require_string(payload.get("campaign_id"), f"{path.name}.campaign_id")
    _require_no_stable_winner(payload.get("decision"), f"{path.name}.decision")
    return campaign_id


def _load_scenario_manifest(payload: Mapping[str, Any], path: Path) -> ScenarioManifest:
    korvid_version = _require_string(payload.get("korvid_version"), f"{path.name}.korvid_version")
    assignments_value = _require_list(payload.get("assignments"), f"{path.name}.assignments")
    assignments = tuple(
        _load_assignment(item, index=index, korvid_version=korvid_version, context=path.name)
        for index, item in enumerate(assignments_value)
    )
    split_ids = {
        split: _require_string_list(payload.get(split), f"{path.name}.{split}")
        for split in _SPLITS
    }
    split_summaries = tuple(
        _load_split_summary(item, index=index, context=path.name)
        for index, item in enumerate(_require_list(payload.get("split_summaries"), f"{path.name}.split_summaries"))
    )

    assignments_by_split: dict[str, list[str]] = defaultdict(list)
    for assignment in assignments:
        assignments_by_split[assignment.split].append(assignment.scenario_id)
    for split in _SPLITS:
        if tuple(assignments_by_split.get(split, ())) != split_ids[split]:
            raise ValueError(f"{path.name} split membership is inconsistent with assignments for {split}")

    return ScenarioManifest(
        korvid_version=korvid_version,
        assignments=assignments,
        train=split_ids["train"],
        validation=split_ids["validation"],
        milestone=split_ids["milestone"],
        split_summaries=split_summaries,
    )


def _load_assignment(
    payload: Any,
    *,
    index: int,
    korvid_version: str,
    context: str,
) -> ScenarioAssignment:
    mapping = _require_mapping(payload, f"{context}.assignments[{index}]")
    scenario_id = _require_string(mapping.get("scenario_id"), f"{context}.assignments[{index}].scenario_id")
    scenario_class = _require_scenario_class(
        mapping.get("scenario_class"), f"{context}.assignments[{index}].scenario_class"
    )
    split = _require_split(mapping.get("split"), f"{context}.assignments[{index}].split")
    question_sha256 = _require_sha256(
        mapping.get("question_sha256"),
        f"{context}.assignments[{index}].question_sha256",
    )
    fixture_sha256 = _require_sha256(
        mapping.get("fixture_sha256"),
        f"{context}.assignments[{index}].fixture_sha256",
    )
    assignment_korvid_version = _require_string(
        mapping.get("korvid_version"),
        f"{context}.assignments[{index}].korvid_version",
    )
    if assignment_korvid_version != korvid_version:
        raise ValueError(f"{context}.assignments[{index}].korvid_version must match {context}.korvid_version")
    return ScenarioAssignment(
        scenario_id=scenario_id,
        scenario_class=scenario_class,
        split=split,
        question_sha256=question_sha256,
        fixture_sha256=fixture_sha256,
        korvid_version=assignment_korvid_version,
    )


def _load_split_summary(payload: Any, *, index: int, context: str) -> ScenarioSplitSummary:
    mapping = _require_mapping(payload, f"{context}.split_summaries[{index}]")
    split_name = _require_split(mapping.get("split_name"), f"{context}.split_summaries[{index}].split_name")
    classes = tuple(
        _require_scenario_class(value, f"{context}.split_summaries[{index}].classes[{class_index}]")
        for class_index, value in enumerate(
            _require_list(mapping.get("classes"), f"{context}.split_summaries[{index}].classes")
        )
    )
    scenario_ids = _require_string_list(
        mapping.get("scenario_ids"),
        f"{context}.split_summaries[{index}].scenario_ids",
    )
    return ScenarioSplitSummary(split_name=split_name, classes=classes, scenario_ids=scenario_ids)


def _load_candidate_manifest(payload: Mapping[str, Any], path: Path) -> dict[str, tuple[str, str]]:
    if _require_int(payload.get("schema_version"), f"{path.name}.schema_version") != 1:
        raise ValueError(f"{path.name}.schema_version must be 1")
    candidates: dict[str, tuple[str, str]] = {}
    for index, item in enumerate(_require_list(payload.get("candidates"), f"{path.name}.candidates")):
        candidate_id, fingerprint, append = _load_candidate_entry(item, index=index, context=path.name)
        candidates[candidate_id] = (fingerprint, append)
    if not candidates:
        raise ValueError(f"{path.name}.candidates must not be empty")
    return candidates


def _load_candidate_entry(payload: Any, *, index: int, context: str) -> tuple[str, str, str]:
    mapping = _require_mapping(payload, f"{context}.candidates[{index}]")
    axes_value = mapping.get("axes")
    if axes_value is not None:
        _require_string_list(axes_value, f"{context}.candidates[{index}].axes")

    candidate_payload: Mapping[str, Any]
    if "candidate" in mapping:
        candidate_payload = _require_mapping(mapping.get("candidate"), f"{context}.candidates[{index}].candidate")
        fingerprint = _require_sha256(
            candidate_payload.get("candidate_fingerprint"),
            f"{context}.candidates[{index}].candidate.candidate_fingerprint",
        )
    else:
        candidate_payload = mapping
        fingerprint = _require_sha256(
            mapping.get("candidate_fingerprint"),
            f"{context}.candidates[{index}].candidate_fingerprint",
        )

    candidate_id = _require_string(candidate_payload.get("candidate_id"), f"{context}.candidates[{index}].candidate_id")
    components = _require_mapping(candidate_payload.get("components"), f"{context}.candidates[{index}].components")
    if set(components) != {"system", "append"}:
        raise ValueError(f"{context}.candidates[{index}].components must be exactly system and append")

    candidate = Candidate.from_mapping(
        {
            "schema_version": _require_int(
                candidate_payload.get("schema_version"),
                f"{context}.candidates[{index}].schema_version",
            ),
            "candidate_id": candidate_id,
            "components": components,
            "metadata": _require_mapping(
                candidate_payload.get("metadata", {}),
                f"{context}.candidates[{index}].metadata",
            ),
        }
    )
    append = candidate.components["append"]
    if append != append.strip():
        raise ValueError(f"{context}.candidates[{index}].append must use canonical outer whitespace")
    if fingerprint != candidate.fingerprint:
        raise ValueError(f"{context}.candidates[{index}] fingerprint mismatch")
    return candidate_id, fingerprint, append


def _load_qualification(payload: Mapping[str, Any], path: Path) -> tuple[_QualificationDelta, ...]:
    if _require_int(payload.get("schema_version"), f"{path.name}.schema_version") != 1:
        raise ValueError(f"{path.name}.schema_version must be 1")
    stage = _require_string(payload.get("stage"), f"{path.name}.stage")
    if stage != "qualification":
        raise ValueError(f"{path.name}.stage must be qualification")
    _require_no_stable_winner(payload.get("decision"), f"{path.name}.decision")

    candidates = tuple(
        _load_qualification_candidate(item, index=index, context=path.name)
        for index, item in enumerate(_require_list(payload.get("candidates"), f"{path.name}.candidates"))
    )
    if not candidates:
        raise ValueError(f"{path.name}.candidates must not be empty")
    return candidates


def _load_qualification_candidate(payload: Any, *, index: int, context: str) -> _QualificationDelta:
    mapping = _require_mapping(payload, f"{context}.candidates[{index}]")
    candidate_id = _require_string(mapping.get("candidate_id"), f"{context}.candidates[{index}].candidate_id")
    candidate_validation = _mean_score(
        mapping.get("candidate_validation"),
        f"{context}.candidates[{index}].candidate_validation",
    )
    baseline_validation = _mean_score(
        mapping.get("baseline_validation"),
        f"{context}.candidates[{index}].baseline_validation",
    )
    candidate_milestone = _mean_score(
        mapping.get("candidate_milestone"),
        f"{context}.candidates[{index}].candidate_milestone",
    )
    baseline_milestone = _mean_score(
        mapping.get("baseline_milestone"),
        f"{context}.candidates[{index}].baseline_milestone",
    )
    return _QualificationDelta(
        candidate_id=candidate_id,
        validation_delta=candidate_validation - baseline_validation,
        milestone_delta=candidate_milestone - baseline_milestone,
    )


def _mean_score(payload: Any, context: str) -> float:
    mapping = _require_mapping(payload, context)
    return _require_finite_float(mapping.get("mean_score"), f"{context}.mean_score")


def _select_finalist(
    qualification: Sequence[_QualificationDelta],
    candidates: Mapping[str, tuple[str, str]],
) -> PriorFinalistEvidence:
    ranked = sorted(
        qualification,
        key=lambda item: (-item.validation_delta, -item.milestone_delta, item.candidate_id),
    )
    for selected in ranked:
        finalist = candidates.get(selected.candidate_id)
        if finalist is None:
            continue
        fingerprint, append = finalist
        return PriorFinalistEvidence(
            candidate_id=selected.candidate_id,
            candidate_fingerprint=fingerprint,
            append=append,
            validation_delta=selected.validation_delta,
            milestone_delta=selected.milestone_delta,
        )
    raise ValueError("no qualification candidates are present in candidate manifest")


def _require_no_stable_winner(payload: Any, context: str) -> None:
    mapping = _require_mapping(payload, context)
    status = _require_string(mapping.get("status"), f"{context}.status")
    if status != "no_stable_winner":
        raise ValueError(f"{context}.status must be no_stable_winner")
    if mapping.get("candidate_id") is not None:
        raise ValueError(f"{context}.candidate_id must be null when status is no_stable_winner")


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")  # noqa: TRY004 - preserve validation API
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")  # noqa: TRY004 - preserve validation API
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_string_list(value: Any, context: str) -> tuple[str, ...]:
    return tuple(_require_string(item, f"{context}[{index}]") for index, item in enumerate(_require_list(value, context)))


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")  # noqa: TRY004 - preserve validation API
    return value


def _require_finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")  # noqa: TRY004 - preserve validation API
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{context} must be a finite number")
    return numeric


def _require_sha256(value: Any, context: str) -> str:
    digest = _require_string(value, context)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest")
    return digest


def _require_scenario_class(value: Any, context: str) -> ScenarioClass:
    name = _require_string(value, context)
    try:
        return ScenarioClass(name)
    except ValueError as exc:
        raise ValueError(f"{context} must be a known scenario class") from exc


def _require_split(value: Any, context: str) -> Literal["train", "validation", "milestone"]:
    split = _require_string(value, context)
    if split not in _SPLITS:
        raise ValueError(f"{context} must be one of train, validation, or milestone")
    return cast(Literal["train", "validation", "milestone"], split)
