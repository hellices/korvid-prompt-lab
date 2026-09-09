from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

CampaignEvaluationSplits = tuple[tuple[str, tuple[str, ...]], ...]
GEPA_REFLECTION_MINIBATCH_SIZE = 3
MODEL_OPTION_FIELDS = frozenset(
    {
        "native_thinking",
        "think",
        "num_ctx",
        "temperature",
        "seed",
        "keep_alive",
        "num_predict",
    }
)


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")  # noqa: TRY004
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_unique_string_items(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of strings")  # noqa: TRY004
    items: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        text = _require_string(item, f"{context}[{index}]")
        if text in seen:
            raise ValueError(f"{context} contains duplicate values")
        seen.add(text)
        items.append(text)
    if not items:
        raise ValueError(f"{context} must not be empty")
    return tuple(items)


def _ensure_keys(mapping: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _ensure_exact_keys(
    mapping: Mapping[str, Any],
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )
    _ensure_keys(mapping, required, context)


def _require_timeout(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a positive number")  # noqa: TRY004
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(f"{context} must be a positive number")
    return timeout


def _immutable_model_options(
    value: Any,
    context: str = "model_options",
) -> Mapping[str, bool | int | float | str]:
    mapping = _require_mapping(value, context)
    _ensure_keys(mapping, set(MODEL_OPTION_FIELDS), context)
    normalized: dict[str, bool | int | float | str] = {}
    for name, option in mapping.items():
        option_context = f"{context}.{name}"
        if name in {"native_thinking", "think"}:
            if type(option) is not bool:
                raise ValueError(f"{option_context} must be a boolean")
        elif name == "num_ctx":
            if isinstance(option, bool) or not isinstance(option, int) or option <= 0:
                raise ValueError(f"{option_context} must be a positive integer")
        elif name == "seed":
            if isinstance(option, bool) or not isinstance(option, int) or option < 0:
                raise ValueError(f"{option_context} must be a non-negative integer")
        elif name == "num_predict":
            if isinstance(option, bool) or not isinstance(option, int):
                raise ValueError(f"{option_context} must be an integer")
        elif name == "temperature":
            if (
                isinstance(option, bool)
                or not isinstance(option, (int, float))
                or not math.isfinite(float(option))
            ):
                raise ValueError(f"{option_context} must be a finite number")
        elif name == "keep_alive":
            invalid = (
                isinstance(option, bool)
                or not isinstance(option, (int, float, str))
                or isinstance(option, float)
                and not math.isfinite(option)
                or isinstance(option, str)
                and not option.strip()
            )
            if invalid:
                raise ValueError(
                    f"{option_context} must be a finite number or non-empty string"
                )
        normalized[name] = option
    return MappingProxyType(normalized)


def _require_positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _require_non_negative_int_items(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list of non-negative integers")
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


@dataclass(frozen=True, slots=True)
class Candidate:
    schema_version: int
    candidate_id: str
    _components: tuple[tuple[str, str], ...]
    _metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("candidate schema_version must be 1")
        _require_string(self.candidate_id, "candidate_id")
        if len(self._components) != 1 or self._components[0][0] != "tier_pack":
            raise ValueError("candidate components must contain exactly tier_pack")
        _require_string(self._components[0][1], "component tier_pack")
        metadata: dict[str, str] = {}
        for key, value in self._metadata:
            key = _require_string(key, "metadata key")
            if key in metadata:
                raise ValueError("metadata contains duplicate keys")
            metadata[key] = _require_string(value, "metadata value")
        object.__setattr__(self, "_components", (("tier_pack", self._components[0][1]),))
        object.__setattr__(self, "_metadata", tuple(sorted(metadata.items())))

    @property
    def components(self) -> dict[str, str]:
        return dict(self._components)

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "components": self.components,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Candidate:
        data = _require_mapping(mapping, "candidate")
        _ensure_keys(
            data,
            {"schema_version", "candidate_id", "components", "metadata"},
            "candidate",
        )
        if data.get("schema_version") != 1:
            raise ValueError("candidate schema_version must be 1")
        components = _require_mapping(data.get("components"), "components")
        if set(components) != {"tier_pack"}:
            raise ValueError("candidate components must contain exactly tier_pack")
        metadata = _require_mapping(data.get("metadata", {}), "metadata")
        return cls(
            schema_version=1,
            candidate_id=_require_string(data.get("candidate_id"), "candidate_id"),
            _components=(
                (
                    "tier_pack",
                    _require_string(components["tier_pack"], "component tier_pack"),
                ),
            ),
            _metadata=tuple(
                (
                    _require_string(key, "metadata key"),
                    _require_string(value, "metadata value"),
                )
                for key, value in metadata.items()
            ),
        )


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    template_id: str
    prompt: str
    models: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_string(self.case_id, "case_id")
        if self.template_id not in {"korvid-scenario", "korvid-journey"}:
            raise ValueError(
                "template_id must identify a Korvid scenario or journey source"
            )
        _require_string(self.prompt, "prompt")
        if len(self.models) != 1:
            raise ValueError("source eval case must target exactly one model")
        _require_string(self.models[0], "model")


@dataclass(frozen=True, slots=True)
class KorvidUpstreamServing:
    backend: str
    source_root: str
    base_url: str
    korvid_revision: str
    timeout_seconds: float
    model_options: Mapping[str, bool | int | float | str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if self.backend != "korvid_upstream":
            raise ValueError("serving backend must be korvid_upstream")
        _require_string(self.source_root, "serving.source_root")
        if self.base_url != "":
            _require_string(self.base_url, "serving.base_url")
        _require_string(self.korvid_revision, "serving.korvid_revision")
        object.__setattr__(
            self,
            "timeout_seconds",
            _require_timeout(self.timeout_seconds, "serving.timeout_seconds"),
        )
        object.__setattr__(
            self,
            "model_options",
            _immutable_model_options(self.model_options, "serving.model_options"),
        )


@dataclass(frozen=True, slots=True)
class Campaign:
    campaign_id: str
    repetitions: int
    models: tuple[str, ...]
    cases: tuple[EvalCase, ...]
    serving: KorvidUpstreamServing
    evaluation_splits: CampaignEvaluationSplits = ()

    def __post_init__(self) -> None:
        _require_string(self.campaign_id, "campaign_id")
        _require_positive_int(self.repetitions, "repetitions")
        if len(self.models) != 1:
            raise ValueError("campaign models must contain exactly one model")
        _require_string(self.models[0], "campaign model")
        if not self.cases:
            raise ValueError("campaign cases must not be empty")
        if type(self.serving) is not KorvidUpstreamServing:
            raise ValueError("campaign serving must be KorvidUpstreamServing")
        case_ids: set[str] = set()
        for case in self.cases:
            if not isinstance(case, EvalCase):
                raise ValueError(  # noqa: TRY004
                    "campaign cases must contain EvalCase values"
                )
            if case.case_id in case_ids:
                raise ValueError("campaign cases contain duplicate case ids")
            case_ids.add(case.case_id)
            if case.models != self.models:
                raise ValueError("campaign case models must match the campaign model")
        split_names: set[str] = set()
        for split_name, split_case_ids in self.evaluation_splits:
            split_name = _require_string(split_name, "evaluation split name")
            if split_name in split_names:
                raise ValueError("evaluation splits contain duplicate names")
            split_names.add(split_name)
            if not split_case_ids:
                raise ValueError("evaluation split case ids must not be empty")
            if len(split_case_ids) != len(set(split_case_ids)):
                raise ValueError("evaluation split contains duplicate case ids")
            unknown = sorted(set(split_case_ids) - case_ids)
            if unknown:
                raise ValueError(
                    f"evaluation split references unknown case ids: {', '.join(unknown)}"
                )


@dataclass(frozen=True, slots=True)
class SearchStage:
    name: str
    metric_calls: int
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_string(self.name, "search stage name")
        _require_positive_int(self.metric_calls, "search stage metric_calls")
        if not self.seeds:
            raise ValueError("search stage seeds must not be empty")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("search stage seeds contain duplicate values")
        for seed in self.seeds:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("search stage seeds must be non-negative integers")

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        index: int,
    ) -> SearchStage:
        data = _require_mapping(mapping, f"stages[{index}]")
        _ensure_exact_keys(
            data,
            {"name", "metric_calls", "seeds"},
            f"stages[{index}]",
        )
        return cls(
            name=_require_string(data.get("name"), f"stages[{index}].name"),
            metric_calls=_require_positive_int(
                data.get("metric_calls"),
                f"stages[{index}].metric_calls",
            ),
            seeds=_require_non_negative_int_items(
                data.get("seeds"),
                f"stages[{index}].seeds",
            ),
        )
