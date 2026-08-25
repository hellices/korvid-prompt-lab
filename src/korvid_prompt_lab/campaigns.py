from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .contracts import (
    Campaign,
    _ensure_keys,
    _require_mapping,
    _require_string,
    _require_unique_string_items,
)

_EXACT_OLLAMA_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class SearchStage:
    name: str
    metric_calls: int
    seeds: tuple[int, ...]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, index: int) -> SearchStage:
        _ensure_required_keys(mapping, {"name", "metric_calls", "seeds"}, f"stages[{index}]")
        return cls(
            name=_require_string(mapping.get("name"), f"stages[{index}].name"),
            metric_calls=_require_positive_int(
                mapping.get("metric_calls"), f"stages[{index}].metric_calls"
            ),
            seeds=_require_non_negative_int_items(mapping.get("seeds"), f"stages[{index}].seeds"),
        )


@dataclass(frozen=True, slots=True)
class ModelTier:
    name: str
    model: str
    digest: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, index: int) -> ModelTier:
        _ensure_required_keys(mapping, {"name", "model", "digest"}, f"model_tiers[{index}]")
        digest = _require_string(mapping.get("digest"), f"model_tiers[{index}].digest")
        if _EXACT_OLLAMA_DIGEST_PATTERN.fullmatch(digest) is None:
            raise ValueError(
                f"model_tiers[{index}].digest must be the exact 64-hex digest reported by Ollama /api/tags"
            )
        return cls(
            name=_require_string(mapping.get("name"), f"model_tiers[{index}].name"),
            model=_require_string(mapping.get("model"), f"model_tiers[{index}].model"),
            digest=digest,
        )


@dataclass(frozen=True, slots=True)
class OptimizationCampaign:
    schema_version: int
    campaign_id: str
    evaluation_campaign: str
    initial_candidate: str
    train_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
    milestone_case_ids: tuple[str, ...]
    stages: tuple[SearchStage, ...]
    model_tiers: tuple[ModelTier, ...]
    total_metric_call_limit: int
    wall_clock_limit_seconds: int
    infrastructure_retry_limit: int
    stagnation_attempt_limit: int
    confirmation_runs: int

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any], *, evaluation_campaign: Campaign
    ) -> OptimizationCampaign:
        _ensure_required_keys(
            mapping,
            {
                "schema_version",
                "campaign_id",
                "evaluation_campaign",
                "initial_candidate",
                "train_case_ids",
                "validation_case_ids",
                "milestone_case_ids",
                "stages",
                "model_tiers",
                "total_metric_call_limit",
                "wall_clock_limit_seconds",
                "infrastructure_retry_limit",
                "stagnation_attempt_limit",
                "confirmation_runs",
            },
            "optimization campaign",
        )
        if mapping.get("schema_version") != 1:
            raise ValueError("optimization campaign schema_version must be 1")

        evaluation_campaign_id = _require_string(
            mapping.get("evaluation_campaign"), "evaluation_campaign"
        )
        if evaluation_campaign_id != evaluation_campaign.campaign_id:
            raise ValueError("evaluation_campaign must match the evaluation campaign file")

        train_case_ids = _require_unique_string_items(mapping.get("train_case_ids"), "train_case_ids")
        validation_case_ids = _require_unique_string_items(
            mapping.get("validation_case_ids"), "validation_case_ids"
        )
        milestone_case_ids = _require_unique_string_items(
            mapping.get("milestone_case_ids"), "milestone_case_ids"
        )
        _validate_case_sets(
            evaluation_campaign=evaluation_campaign,
            train_case_ids=train_case_ids,
            validation_case_ids=validation_case_ids,
            milestone_case_ids=milestone_case_ids,
        )

        raw_stages = mapping.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("stages must be a non-empty list")
        stages = tuple(
            SearchStage.from_mapping(_require_mapping(item, f"stages[{index}]"), index=index)
            for index, item in enumerate(raw_stages)
        )
        _validate_stages(stages)

        raw_model_tiers = mapping.get("model_tiers")
        if not isinstance(raw_model_tiers, list) or not raw_model_tiers:
            raise ValueError("model_tiers must be a non-empty list")
        model_tiers = tuple(
            ModelTier.from_mapping(_require_mapping(item, f"model_tiers[{index}]"), index=index)
            for index, item in enumerate(raw_model_tiers)
        )
        _validate_model_tiers(model_tiers)

        total_metric_call_limit = _require_positive_int(
            mapping.get("total_metric_call_limit"), "total_metric_call_limit"
        )
        planned_metric_calls = sum(stage.metric_calls * len(stage.seeds) for stage in stages)
        if planned_metric_calls > total_metric_call_limit:
            raise ValueError("total_metric_call_limit must cover every staged search attempt")

        return cls(
            schema_version=1,
            campaign_id=_require_string(mapping.get("campaign_id"), "campaign_id"),
            evaluation_campaign=evaluation_campaign_id,
            initial_candidate=_require_string(mapping.get("initial_candidate"), "initial_candidate"),
            train_case_ids=train_case_ids,
            validation_case_ids=validation_case_ids,
            milestone_case_ids=milestone_case_ids,
            stages=stages,
            model_tiers=model_tiers,
            total_metric_call_limit=total_metric_call_limit,
            wall_clock_limit_seconds=_require_positive_int(
                mapping.get("wall_clock_limit_seconds"), "wall_clock_limit_seconds"
            ),
            infrastructure_retry_limit=_require_positive_int(
                mapping.get("infrastructure_retry_limit"), "infrastructure_retry_limit"
            ),
            stagnation_attempt_limit=_require_positive_int(
                mapping.get("stagnation_attempt_limit"), "stagnation_attempt_limit"
            ),
            confirmation_runs=_require_positive_int(
                mapping.get("confirmation_runs"), "confirmation_runs"
            ),
        )


def _load_yaml(path: Path | str) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_optimization_campaign(path: Path | str, evaluation_campaign: Campaign) -> OptimizationCampaign:
    return OptimizationCampaign.from_mapping(
        _require_mapping(_load_yaml(path), "optimization campaign"),
        evaluation_campaign=evaluation_campaign,
    )


def _http_get_json(url: str) -> Mapping[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return _require_mapping(json.load(response), "model tags probe")


def validate_model_tier_digests(
    campaign: OptimizationCampaign,
    model_endpoint: str,
    *,
    http_get_json: Callable[[str], Mapping[str, Any]] = _http_get_json,
) -> None:
    payload = _require_mapping(
        http_get_json(f"{model_endpoint.rstrip('/')}/api/tags"), "model tags probe"
    )
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("model tags probe returned an invalid payload")  # noqa: TRY004 - preserve validation API

    digests_by_model: dict[str, list[str]] = {}
    for index, item in enumerate(raw_models):
        model = _require_mapping(item, f"model tags.models[{index}]")
        name = _require_string(model.get("name"), f"model tags.models[{index}].name")
        digest = _require_string(model.get("digest"), f"model tags.models[{index}].digest")
        if _EXACT_OLLAMA_DIGEST_PATTERN.fullmatch(digest) is None:
            raise ValueError("model tags probe returned a non-immutable digest")
        digests_by_model.setdefault(name, []).append(digest)

    for tier in campaign.model_tiers:
        live_digests = digests_by_model.get(tier.model, [])
        if not live_digests:
            raise ValueError(f"live /api/tags did not advertise model {tier.model}")
        if len(live_digests) != 1:
            raise ValueError(f"live /api/tags advertised duplicate digests for model {tier.model}")
        if live_digests[0] != tier.digest:
            raise ValueError(f"live /api/tags digest mismatch for model {tier.model}")


def _ensure_required_keys(mapping: Mapping[str, Any], required: set[str], context: str) -> None:
    _ensure_keys(mapping, required, context)
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"{context} is missing required field(s): {', '.join(missing)}")


def _require_positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _require_non_negative_int_items(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of integers")  # noqa: TRY004 - preserve validation API
    if not value:
        raise ValueError(f"{context} must not be empty")
    items: list[int] = []
    seen: set[int] = set()
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{context}[{index}] must be a non-negative integer")
        if item in seen:
            raise ValueError(f"{context} contains duplicate values")
        seen.add(item)
        items.append(item)
    return tuple(items)


def _validate_case_sets(
    *,
    evaluation_campaign: Campaign,
    train_case_ids: tuple[str, ...],
    validation_case_ids: tuple[str, ...],
    milestone_case_ids: tuple[str, ...],
) -> None:
    overlaps = (
        set(train_case_ids) & set(validation_case_ids)
        or set(train_case_ids) & set(milestone_case_ids)
        or set(validation_case_ids) & set(milestone_case_ids)
    )
    if overlaps:
        raise ValueError("train_case_ids, validation_case_ids, and milestone_case_ids must be pairwise disjoint")

    evaluation_case_ids = {case.case_id for case in evaluation_campaign.cases}
    configured_case_ids = set(train_case_ids) | set(validation_case_ids) | set(milestone_case_ids)
    unknown_case_ids = sorted(configured_case_ids - evaluation_case_ids)
    if unknown_case_ids:
        raise ValueError(f"unknown case_id value(s): {', '.join(unknown_case_ids)}")
    missing_case_ids = sorted(evaluation_case_ids - configured_case_ids)
    if missing_case_ids:
        raise ValueError("train/validation/milestone case sets must cover the evaluation campaign exactly")


def _validate_stages(stages: tuple[SearchStage, ...]) -> None:
    seen_names: set[str] = set()
    seen_seeds: set[int] = set()
    for stage in stages:
        if stage.name in seen_names:
            raise ValueError(f"stages contain duplicate stage name {stage.name}")
        seen_names.add(stage.name)
        duplicate_seeds = sorted(set(stage.seeds) & seen_seeds)
        if duplicate_seeds:
            raise ValueError(f"stages contain duplicate seed values: {', '.join(str(seed) for seed in duplicate_seeds)}")
        seen_seeds.update(stage.seeds)


def _validate_model_tiers(model_tiers: tuple[ModelTier, ...]) -> None:
    seen_names: set[str] = set()
    seen_models: set[str] = set()
    seen_digests: set[str] = set()
    for tier in model_tiers:
        if tier.name in seen_names:
            raise ValueError(f"model_tiers contain duplicate tier name {tier.name}")
        if tier.model in seen_models:
            raise ValueError(f"model_tiers contain duplicate model {tier.model}")
        if tier.digest in seen_digests:
            raise ValueError(f"model_tiers contain duplicate digest {tier.digest}")
        seen_names.add(tier.name)
        seen_models.add(tier.model)
        seen_digests.add(tier.digest)
