"""Strict unified native experiment configuration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import yaml  # type: ignore[import-untyped]

from .contracts import (
    Campaign,
    KorvidUpstreamServing,
    SearchStage,
    _immutable_model_options,
    _require_mapping,
    _require_string,
    _require_timeout,
)
from .runner import _require_loopback_endpoint
from .upstream_contract import KORVID_REVISION, KORVID_VERSION

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_REFLECTION_OPTION_FIELDS = frozenset(
    {"max_tokens", "num_ctx", "temperature", "reasoning_effort"}
)


def _exact_keys(
    mapping: Mapping[str, Any],
    required: set[str],
    context: str,
    *,
    optional: frozenset[str] | set[str] = frozenset(),
) -> None:
    missing = sorted(required - set(mapping))
    unknown = sorted(set(mapping) - required - optional)
    if missing:
        raise ValueError(f"{context} is missing required field(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _canonical_digest(value: Any, context: str) -> str:
    digest = _require_string(value, context)
    if _DIGEST_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{context} must be sha256:<64 lowercase hex>")
    return digest


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _explicit_model_tag(reference: Any, prefix: str, context: str) -> tuple[str, str]:
    text = _require_string(reference, context)
    if text != text.strip() or not text.startswith(prefix):
        raise ValueError(f"{context} must be an explicit {prefix}<model> reference")
    tag = text.removeprefix(prefix)
    if not tag.strip():
        raise ValueError(f"{context} must be an explicit {prefix}<model> reference")
    return text, tag


@dataclass(frozen=True, slots=True)
class ModelProfile:
    reference: str
    digest: str
    options: Mapping[str, bool | int | float | str]

    def __post_init__(self) -> None:
        reference, _ = _explicit_model_tag(self.reference, "ollama/", "model.reference")
        options = _immutable_model_options(self.options, "model.options")
        if options.get("native_thinking") is not True:
            raise ValueError("model.options.native_thinking must be true")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "digest", _canonical_digest(self.digest, "model.digest"))
        object.__setattr__(self, "options", options)

    @property
    def model_tag(self) -> str:
        return self.reference.removeprefix("ollama/")


def _immutable_reflection_options(
    value: Any,
) -> Mapping[str, int | float | str]:
    mapping = _require_mapping(value, "reflection.options")
    unknown = sorted(set(mapping) - _REFLECTION_OPTION_FIELDS)
    if unknown:
        raise ValueError(
            f"reflection.options has unknown field(s): {', '.join(unknown)}"
        )
    normalized: dict[str, int | float | str] = {}
    for name, option in mapping.items():
        context = f"reflection.options.{name}"
        if name in {"max_tokens", "num_ctx"}:
            normalized[name] = _positive_int(option, context)
        elif name == "temperature":
            if (
                isinstance(option, bool)
                or not isinstance(option, (int, float))
                or not math.isfinite(float(option))
            ):
                raise ValueError(f"{context} must be a finite number")
            normalized[name] = option
        else:
            normalized[name] = _require_string(option, context)
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ReflectionProfile:
    reference: str
    digest: str
    timeout_seconds: int
    options: Mapping[str, int | float | str]

    def __post_init__(self) -> None:
        reference, _ = _explicit_model_tag(
            self.reference, "ollama_chat/", "reflection.reference"
        )
        object.__setattr__(self, "reference", reference)
        object.__setattr__(
            self, "digest", _canonical_digest(self.digest, "reflection.digest")
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_int(self.timeout_seconds, "reflection.timeout_seconds"),
        )
        object.__setattr__(
            self, "options", _immutable_reflection_options(self.options)
        )

    @property
    def model_tag(self) -> str:
        return self.reference.removeprefix("ollama_chat/")


@dataclass(frozen=True, slots=True)
class LoopbackConnection:
    backend: str
    base_url: str

    def __post_init__(self) -> None:
        if self.backend != "loopback":
            raise ValueError("loopback serving backend must be loopback")
        parsed = urlsplit(self.base_url)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("loopback base_url must not contain credentials")
        if parsed.scheme != "http":
            raise ValueError("loopback base_url must use HTTP")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("loopback base_url must be an HTTP root without /v1")
        normalized = self.base_url.rstrip("/")
        _require_loopback_endpoint(normalized)
        object.__setattr__(self, "base_url", normalized)


@dataclass(frozen=True, slots=True)
class AKSConnection:
    backend: str
    resource_group: str
    cluster_name: str
    namespace: str
    service: str
    node_pool: str | None = None

    def __post_init__(self) -> None:
        if self.backend != "aks_port_forward":
            raise ValueError("AKS serving backend must be aks_port_forward")
        for name in ("resource_group", "cluster_name", "namespace", "service"):
            object.__setattr__(
                self, name, _require_string(getattr(self, name), f"serving.{name}")
            )
        if self.node_pool is not None:
            object.__setattr__(
                self,
                "node_pool",
                _require_string(self.node_pool, "serving.node_pool"),
            )


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    campaign_id: str
    runtime: KorvidUpstreamServing
    serving: LoopbackConnection | AKSConnection
    model: ModelProfile
    reflection: ReflectionProfile
    repetitions: int
    evaluation_seed: int
    stages: tuple[SearchStage, ...]
    total_metric_calls: int
    max_evaluations: int
    max_proposals: int
    wall_clock_seconds: int
    stagnation_attempt_limit: int
    case_splits: Mapping[str, tuple[str, ...]]

    @property
    def manifest(self) -> dict[str, Any]:
        serving: dict[str, Any]
        if isinstance(self.serving, LoopbackConnection):
            serving = {
                "backend": self.serving.backend,
                "base_url": self.serving.base_url,
            }
        else:
            serving = {
                "backend": self.serving.backend,
                "resource_group": self.serving.resource_group,
                "cluster_name": self.serving.cluster_name,
                "namespace": self.serving.namespace,
                "service": self.serving.service,
                "node_pool": self.serving.node_pool,
            }
        return {
            "schema_version": 2,
            "campaign_id": self.campaign_id,
            "runtime": {
                "backend": self.runtime.backend,
                "source_root": self.runtime.source_root,
                "korvid_revision": self.runtime.korvid_revision,
                "korvid_version": KORVID_VERSION,
                "timeout_seconds": self.runtime.timeout_seconds,
            },
            "serving": serving,
            "model": {
                "reference": self.model.reference,
                "digest": self.model.digest,
                "options": dict(self.model.options),
            },
            "reflection": {
                "reference": self.reflection.reference,
                "digest": self.reflection.digest,
                "timeout_seconds": self.reflection.timeout_seconds,
                "options": dict(self.reflection.options),
            },
            "evaluation": {
                "repetitions": self.repetitions,
                "seed": self.evaluation_seed,
                **{name: list(self.case_splits[name]) for name in ("train", "validation", "holdout")},
            },
            "search": {
                "stages": [
                    {
                        "name": stage.name,
                        "metric_calls": stage.metric_calls,
                        "seeds": list(stage.seeds),
                    }
                    for stage in self.stages
                ],
                "total_metric_calls": self.total_metric_calls,
                "max_evaluations": self.max_evaluations,
                "max_proposals": self.max_proposals,
                "wall_clock_seconds": self.wall_clock_seconds,
                "stagnation_attempt_limit": self.stagnation_attempt_limit,
            },
            "baseline": "unchanged_korvid_prompt_pack",
        }

    def to_mapping(self) -> dict[str, Any]:
        return self.manifest

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def campaign(self, base_url: str) -> Campaign:
        from .upstream_contract import load_source_cases

        endpoint = LoopbackConnection("loopback", base_url).base_url
        model = self.model.reference
        source_cases = load_source_cases(Path(self.runtime.source_root), self.references)
        return Campaign(
            campaign_id=self.campaign_id,
            repetitions=self.repetitions,
            models=(model,),
            cases=tuple(case.eval_case(model) for case in source_cases),
            serving=replace(self.runtime, base_url=endpoint),
            evaluation_splits=tuple(
                (split, tuple(ref.split("/", 1)[1] for ref in self.case_splits[split]))
                for split in ("train", "validation", "holdout")
            ),
        )

    @property
    def references(self) -> tuple[str, ...]:
        return tuple(ref for split in ("train", "validation", "holdout") for ref in self.case_splits[split])


def _required_env(value: Any, expected: str, context: str) -> str:
    reference = _require_string(value, context)
    if reference != f"env:{expected}":
        raise ValueError(f"{context} must be env:{expected}")
    from os import getenv

    resolved = getenv(expected)
    if resolved is None or not resolved.strip():
        raise ValueError(f"{context} references missing environment variable {expected}")
    return resolved


def _parse_runtime(mapping: Mapping[str, Any]) -> tuple[str, float]:
    _exact_keys(
        mapping,
        {
            "backend",
            "source_root",
            "korvid_revision",
            "timeout_seconds",
        },
        "runtime",
    )
    if mapping.get("backend") != "korvid_upstream":
        raise ValueError(
            "runtime.backend must be korvid_upstream"
        )
    revision = _require_string(mapping.get("korvid_revision"), "runtime.korvid_revision")
    if revision != KORVID_REVISION:
        raise ValueError("runtime.korvid_revision must use the reviewed Korvid v0.4.1 revision")
    source_root = _required_env(
        mapping.get("source_root"), "KORVID_NATIVE_SOURCE_ROOT", "runtime.source_root"
    )
    timeout = _require_timeout(
        mapping.get("timeout_seconds"), "runtime.timeout_seconds"
    )
    return source_root, timeout


def _parse_serving(mapping: Mapping[str, Any]) -> LoopbackConnection | AKSConnection:
    backend = _require_string(mapping.get("backend"), "serving.backend")
    if backend == "loopback":
        _exact_keys(mapping, {"backend", "base_url"}, "serving.loopback")
        return LoopbackConnection(
            backend="loopback",
            base_url=_required_env(
                mapping.get("base_url"),
                "KORVID_NATIVE_MODEL_URL",
                "serving.base_url",
            ),
        )
    if backend == "aks_port_forward":
        _exact_keys(
            mapping,
            {"backend", "resource_group", "cluster_name", "namespace", "service"},
            "serving.aks_port_forward",
            optional={"node_pool"},
        )
        return AKSConnection(
            backend=backend,
            resource_group=_require_string(
                mapping.get("resource_group"), "serving.resource_group"
            ),
            cluster_name=_require_string(
                mapping.get("cluster_name"), "serving.cluster_name"
            ),
            namespace=_require_string(mapping.get("namespace"), "serving.namespace"),
            service=_require_string(mapping.get("service"), "serving.service"),
            node_pool=(
                _require_string(mapping.get("node_pool"), "serving.node_pool")
                if mapping.get("node_pool") is not None
                else None
            ),
        )
    raise ValueError("serving.backend must be loopback or aks_port_forward")


def _parse_stages(value: Any) -> tuple[SearchStage, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("search.stages must be a non-empty list")
    stages = tuple(
        SearchStage.from_mapping(
            _require_mapping(item, f"search.stages[{index}]"), index=index
        )
        for index, item in enumerate(value)
    )
    names: set[str] = set()
    seeds: set[int] = set()
    for stage in stages:
        if stage.name in names:
            raise ValueError(f"search.stages contains duplicate stage name {stage.name}")
        duplicates = sorted(seeds.intersection(stage.seeds))
        if duplicates:
            raise ValueError(
                "search.stages contains duplicate seed values: "
                + ", ".join(str(seed) for seed in duplicates)
            )
        names.add(stage.name)
        seeds.update(stage.seeds)
    return stages


def load_experiment(path: Path | str) -> ExperimentSpec:
    data = _require_mapping(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")), "experiment"
    )
    _exact_keys(
        data,
        {
            "schema_version",
            "campaign_id",
            "runtime",
            "serving",
            "model",
            "reflection",
            "evaluation",
            "search",
        },
        "experiment",
    )
    if data.get("schema_version") != 2:
        raise ValueError("experiment schema_version must be 2")
    runtime_source_root, runtime_timeout = _parse_runtime(
        _require_mapping(data.get("runtime"), "runtime")
    )

    model_mapping = _require_mapping(data.get("model"), "model")
    _exact_keys(model_mapping, {"reference", "digest", "options"}, "model")
    model = ModelProfile(
        reference=_require_string(model_mapping.get("reference"), "model.reference"),
        digest=_require_string(model_mapping.get("digest"), "model.digest"),
        options=_require_mapping(model_mapping.get("options"), "model.options"),
    )

    reflection_mapping = _require_mapping(data.get("reflection"), "reflection")
    _exact_keys(
        reflection_mapping,
        {"reference", "digest", "timeout_seconds", "options"},
        "reflection",
    )
    reflection = ReflectionProfile(
        reference=_require_string(
            reflection_mapping.get("reference"), "reflection.reference"
        ),
        digest=_require_string(reflection_mapping.get("digest"), "reflection.digest"),
        timeout_seconds=_positive_int(
            reflection_mapping.get("timeout_seconds"), "reflection.timeout_seconds"
        ),
        options=_require_mapping(
            reflection_mapping.get("options"), "reflection.options"
        ),
    )

    evaluation = _require_mapping(data.get("evaluation"), "evaluation")
    _exact_keys(evaluation, {"repetitions", "seed", "train", "validation", "holdout"}, "evaluation")
    case_splits: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    seen_ids: set[str] = set()
    for split in ("train", "validation", "holdout"):
        refs = evaluation[split]
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"evaluation.{split} must contain original Korvid case references")
        selected: list[str] = []
        for ref in refs:
            if not isinstance(ref, str) or re.fullmatch(r"(scenarios|journeys)/[a-z0-9][a-z0-9_-]*", ref) is None:
                raise ValueError("case references must be scenarios/<original-id> or journeys/<original-id>")
            case_id = ref.split("/", 1)[1]
            if ref in seen or case_id in seen_ids:
                raise ValueError("original case IDs must be unique and disjoint across splits")
            seen.add(ref)
            seen_ids.add(case_id)
            selected.append(ref)
        case_splits[split] = tuple(selected)
    repetitions = _positive_int(
        evaluation.get("repetitions"), "evaluation.repetitions"
    )
    if repetitions < 5:
        raise ValueError("evaluation.repetitions must be at least 5")

    search = _require_mapping(data.get("search"), "search")
    _exact_keys(
        search,
        {
            "stages",
            "total_metric_calls",
            "max_evaluations",
            "max_proposals",
            "wall_clock_seconds",
            "stagnation_attempt_limit",
        },
        "search",
    )
    stages = _parse_stages(search.get("stages"))
    total_metric_calls = _positive_int(
        search.get("total_metric_calls"), "search.total_metric_calls"
    )
    planned_metric_calls = sum(
        stage.metric_calls * len(stage.seeds) for stage in stages
    )
    if planned_metric_calls > total_metric_calls:
        raise ValueError("search.total_metric_calls must cover all configured stages")

    return ExperimentSpec(
        campaign_id=_require_string(data.get("campaign_id"), "campaign_id"),
        runtime=KorvidUpstreamServing(
            backend="korvid_upstream",
            source_root=runtime_source_root,
            base_url="",
            korvid_revision=KORVID_REVISION,
            timeout_seconds=runtime_timeout,
            model_options=model.options,
        ),
        serving=_parse_serving(_require_mapping(data.get("serving"), "serving")),
        model=model,
        reflection=reflection,
        repetitions=repetitions,
        evaluation_seed=_non_negative_int(evaluation.get("seed"), "evaluation.seed"),
        stages=stages,
        total_metric_calls=total_metric_calls,
        max_evaluations=_positive_int(
            search.get("max_evaluations"), "search.max_evaluations"
        ),
        max_proposals=_positive_int(
            search.get("max_proposals"), "search.max_proposals"
        ),
        wall_clock_seconds=_positive_int(
            search.get("wall_clock_seconds"), "search.wall_clock_seconds"
        ),
        stagnation_attempt_limit=_positive_int(
            search.get("stagnation_attempt_limit"),
            "search.stagnation_attempt_limit",
        ),
        case_splits=MappingProxyType(case_splits),
    )
