from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .contracts import (
    DEFAULT_BRIDGE_TIMEOUT_SECONDS,
    AKSPortForwardServing,
    Campaign,
    Candidate,
    EvalCase,
    KorvidReadonlyServing,
    ProcessServing,
    _ensure_keys,
    _require_bridge_timeout,
    _require_mapping,
    _require_string,
    _require_unique_string_items,
    _resolve_env_string,
)

#: Closed vocabulary for `serving.provider` on the `korvid_readonly` backend.
KORVID_READONLY_PROVIDERS = frozenset({"ollama", "openai-compat"})

#: Closed vocabulary for `serving.profile` on the `korvid_readonly` backend,
#: matching the installed `korvid.evals` CLI's `--profile` choices.
KORVID_READONLY_PROFILES = frozenset({"small", "full"})


def _load_yaml(path: Path | str) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _resolve_string_items(value: Any, context: str) -> tuple[str, ...]:
    items = _require_unique_string_items(value, context)
    return tuple(_resolve_env_string(item, context) for item in items)


def _resolve_required_env_string(value: Any, context: str) -> str:
    reference = _require_string(value, context)
    if not reference.startswith("env:"):
        raise ValueError(f"{context} must be an env: reference")
    return _resolve_env_string(reference, context)


def load_candidate(path: Path | str) -> Candidate:
    return Candidate.from_mapping(_load_yaml(path))


def _parse_case(mapping: Mapping[str, Any]) -> EvalCase:
    _ensure_keys(mapping, {"case_id", "template_id", "prompt", "models"}, "case")
    return EvalCase(
        case_id=_require_string(mapping.get("case_id"), "case_id"),
        template_id=_require_string(mapping.get("template_id"), "template_id"),
        prompt=_require_string(mapping.get("prompt"), "prompt"),
        models=_resolve_string_items(mapping.get("models"), "case.models"),
    )


def _parse_bridge_command(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list")
    tokens = tuple(_require_string(item, f"{context} item") for item in value)
    if any(token.startswith("env:") for token in tokens):
        raise ValueError(f"{context} must not use env: interpolation")
    missing = [placeholder for placeholder in ("{request}", "{response}") if placeholder not in tokens]
    if missing:
        raise ValueError(f"{context} must include {' and '.join(missing)}")
    return tokens


def _parse_process_serving(mapping: Mapping[str, Any]) -> ProcessServing:
    _ensure_keys(mapping, {"backend", "command"}, "serving.process")
    return ProcessServing(
        backend="process",
        command=_parse_bridge_command(mapping.get("command"), "serving.command"),
    )


def _parse_aks_serving(mapping: Mapping[str, Any]) -> AKSPortForwardServing:
    _ensure_keys(
        mapping,
        {"backend", "resource_group", "cluster_name", "namespace", "service", "model", "command"},
        "serving.aks_port_forward",
    )
    return AKSPortForwardServing(
        backend="aks_port_forward",
        resource_group=_require_string(mapping.get("resource_group"), "serving.resource_group"),
        cluster_name=_require_string(mapping.get("cluster_name"), "serving.cluster_name"),
        namespace=_resolve_env_string(mapping.get("namespace"), "serving.namespace"),
        service=_resolve_env_string(mapping.get("service"), "serving.service"),
        model=_resolve_env_string(mapping.get("model"), "serving.model"),
        command=_parse_bridge_command(mapping.get("command"), "serving.command"),
    )


def _parse_korvid_readonly_serving(mapping: Mapping[str, Any]) -> KorvidReadonlyServing:
    _ensure_keys(
        mapping,
        {"backend", "provider", "base_url", "profile", "timeout_seconds"},
        "serving.korvid_readonly",
    )
    provider = _require_string(mapping.get("provider"), "serving.provider")
    if provider not in KORVID_READONLY_PROVIDERS:
        raise ValueError("serving.provider must be ollama or openai-compat")
    profile = _require_string(mapping.get("profile"), "serving.profile")
    if profile not in KORVID_READONLY_PROFILES:
        raise ValueError("serving.profile must be small or full")
    timeout_seconds = _require_bridge_timeout(mapping.get("timeout_seconds"), "serving.timeout_seconds")
    return KorvidReadonlyServing(
        backend="korvid_readonly",
        provider=provider,
        base_url=_resolve_required_env_string(
            mapping.get("base_url"), "serving.base_url"
        ),
        profile=profile,
        timeout_seconds=timeout_seconds,
    )


def load_campaign(path: Path | str) -> Campaign:
    data = _require_mapping(_load_yaml(path), "campaign")
    _ensure_keys(
        data,
        {
            "schema_version",
            "campaign_id",
            "repetitions",
            "bridge_timeout_seconds",
            "models",
            "cases",
            "serving",
        },
        "campaign",
    )

    if data.get("schema_version") != 1:
        raise ValueError("campaign schema_version must be 1")

    repetitions = data.get("repetitions")
    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("campaign repetitions must be a positive integer")

    bridge_timeout_seconds = _require_bridge_timeout(
        data.get("bridge_timeout_seconds", DEFAULT_BRIDGE_TIMEOUT_SECONDS),
        "campaign bridge_timeout_seconds",
    )

    models = _resolve_string_items(data.get("models"), "campaign.models")

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("campaign cases must be a non-empty list")
    cases = tuple(_parse_case(_require_mapping(item, "case")) for item in raw_cases)
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("campaign cases contain duplicate case_id values")

    covered_models = {model for case in cases for model in case.models}
    missing_models = [model for model in models if model not in covered_models]
    if missing_models:
        raise ValueError("campaign model coverage is incomplete")

    serving_mapping = _require_mapping(data.get("serving"), "serving")
    backend = _require_string(serving_mapping.get("backend"), "serving.backend")
    serving: ProcessServing | AKSPortForwardServing | KorvidReadonlyServing
    if backend == "process":
        serving = _parse_process_serving(serving_mapping)
    elif backend == "aks_port_forward":
        serving = _parse_aks_serving(serving_mapping)
    elif backend == "korvid_readonly":
        serving = _parse_korvid_readonly_serving(serving_mapping)
    else:
        raise ValueError("serving backend must be process, aks_port_forward, or korvid_readonly")

    return Campaign(
        schema_version=1,
        campaign_id=_require_string(data.get("campaign_id"), "campaign_id"),
        repetitions=repetitions,
        models=models,
        cases=cases,
        serving=serving,
        bridge_timeout_seconds=bridge_timeout_seconds,
    )
