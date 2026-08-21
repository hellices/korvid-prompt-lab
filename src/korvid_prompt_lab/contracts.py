from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")  # noqa: TRY004 - preserve validation API
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_unique_string_items(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of strings")  # noqa: TRY004 - preserve validation API
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


def _resolve_env_string(value: Any, context: str) -> str:
    text = _require_string(value, context)
    if text.startswith("env:"):
        env_name = _require_string(text[4:], context)
        from os import getenv

        resolved = getenv(env_name)
        if resolved is None or not resolved.strip():
            raise ValueError(f"{context} references missing environment variable {env_name}")
        return resolved
    return text


@dataclass(frozen=True, slots=True)
class Candidate:
    schema_version: int
    candidate_id: str
    _components: tuple[tuple[str, str], ...]
    _metadata: tuple[tuple[str, str], ...] = ()

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
            "components": dict(self._components),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Candidate:
        data = _require_mapping(mapping, "candidate")
        _ensure_keys(data, {"schema_version", "candidate_id", "components", "metadata"}, "candidate")

        if data.get("schema_version") != 1:
            raise ValueError("candidate schema_version must be 1")

        candidate_id = _require_string(data.get("candidate_id"), "candidate_id")

        components_value = _require_mapping(data.get("components"), "components")
        if not components_value:
            raise ValueError("components must not be empty")

        components: list[tuple[str, str]] = []
        for key, value in components_value.items():
            key_text = _require_string(key, "component key")
            is_tool_key = key_text.startswith("tool.") and len(key_text) > len("tool.")
            if not (key_text in {"system", "append"} or is_tool_key):
                raise ValueError("component key must be system, append, or tool.<tool-name>")
            components.append((key_text, _require_string(value, f"component {key_text}")))

        metadata_items: list[tuple[str, str]] = []
        metadata_value = data.get("metadata", {})
        if metadata_value:
            metadata_mapping = _require_mapping(metadata_value, "metadata")
            for key, value in metadata_mapping.items():
                metadata_items.append((_require_string(key, "metadata key"), _require_string(value, "metadata value")))

        return cls(
            schema_version=1,
            candidate_id=candidate_id,
            _components=tuple(components),
            _metadata=tuple(metadata_items),
        )


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    template_id: str
    prompt: str
    models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProcessServing:
    backend: str
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AKSPortForwardServing:
    backend: str
    resource_group: str
    cluster_name: str
    namespace: str
    service: str
    model: str


@dataclass(frozen=True, slots=True)
class Campaign:
    schema_version: int
    campaign_id: str
    repetitions: int
    models: tuple[str, ...]
    cases: tuple[EvalCase, ...]
    serving: ProcessServing | AKSPortForwardServing
