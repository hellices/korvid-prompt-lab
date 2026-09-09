"""Prompt Lab's additive-rules contract for the unchanged Korvid release."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from .contracts import Candidate
from .navigation_cases import NavigationCase, navigation_cases

NATIVE_KORVID_VERSION = "0.4.1"
NATIVE_KORVID_REVISION = "33c483e041006eb20259a024ed85a9323e52c8f0"
NATIVE_PROTOCOL_VERSION = 1
_NATIVE_TASKS = frozenset({"pods", "helm", "all-pods", "logs", "describe"})


def validate_rules(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError("agent.rules must be a JSON list of at most 16 rules")
    for rule in value:
        if not isinstance(rule, str) or not rule.strip() or rule != rule.strip() or len(rule) > 1000:
            raise ValueError("each agent rule must be a canonical non-blank string of at most 1000 characters")
    return list(value)


def rules_from_candidate(candidate: Candidate) -> list[str]:
    if set(candidate.components) != {"rules"}:
        raise ValueError("native candidates support only the rules component; system/overlay/tool overrides are not applied")
    try:
        value = json.loads(candidate.components["rules"])
    except json.JSONDecodeError as exc:
        raise ValueError("native rules component must encode a JSON string array") from exc
    return validate_rules(value)


def rules_candidate(rules: list[str]) -> Candidate:
    return Candidate.from_mapping({
        "schema_version": 1,
        "candidate_id": "korvid-native-rules",
        "components": {"rules": json.dumps(validate_rules(rules), ensure_ascii=False)},
        "metadata": {
            "scope": "native-low-ui", "korvid_version": NATIVE_KORVID_VERSION,
            "korvid_revision": NATIVE_KORVID_REVISION,
        },
    })


def native_cases() -> tuple[NavigationCase, ...]:
    return tuple(
        replace(case, expected=tuple(sorted({"logs": "", "describe": "", **dict(case.expected)}.items())))
        for case in navigation_cases() if case.case_id.split("-", 1)[1] in _NATIVE_TASKS
    )


def find_native_case(case_id: str) -> NavigationCase:
    for case in native_cases():
        if case.case_id == case_id:
            return case
    raise ValueError(f"case is not supported by the native low UI pack: {case_id}")


def require_native_pack(case_ids: Sequence[str]) -> None:
    if len(case_ids) != 15 or set(case_ids) != {case.case_id for case in native_cases()}:
        raise ValueError("native campaign requires the complete 15-case low-tier UI pack")
