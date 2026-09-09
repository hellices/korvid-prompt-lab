"""Run unchanged Korvid v0.4.1 scenario and journey evaluations."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import re
import sys
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import asdict
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import korvid
from korvid import __version__ as korvid_version
from korvid.agent.model_policy import (
    ModelCapabilities,
    ModelDescriptor,
    ResolvedAgentPolicy,
)
from korvid.agent.model_profiles import ConnectionAuthConfig, ModelConnectionConfig
from korvid.agent.prompt_harness import PromptHarness, PromptInputs
from korvid.agent.prompt_packs import (
    COMMON_ROLE,
    MODEL_PROMPT_OVERLAYS,
    PROMPT_PACKS,
    PROVIDER_PROMPT_OVERLAYS,
    SAFETY_CONTRACT,
)
from korvid.agent.provider import LLMProvider
from korvid.evals.__main__ import (
    policy_payload,
    prompt_fingerprint,
)
from korvid.evals.__main__ import (
    report_payload as scenario_report_payload,
)
from korvid.evals.fake_kube import FakeKubeClient, builtin_aliases
from korvid.evals.harness import (
    EVAL_CLUSTER,
    NO_GRIND,
    PromptGrind,
    build_prompt_harness,
    resolve_eval_policy,
    static_prompt,
)
from korvid.evals.journey import ConversationJourney, load_journey
from korvid.evals.journey_runner import (
    report_payload as journey_report_payload,
)
from korvid.evals.journey_runner import (
    run_journey,
)
from korvid.evals.runner import run_scenario
from korvid.evals.scenario import Scenario, load_scenario
from korvid.evals.scripted import ScriptedProvider
from korvid.providers.litellm_catalog import LiteLLMModelCatalog
from korvid.providers.litellm_factory import create_provider_from_profile
from korvid.providers.litellm_runtime import models_by_provider
from korvid.providers.special_flows import SpecialFlowRegistry
from korvid.tools.executor import ToolExecutor
from korvid.tools.registry import tool_def

PROTOCOL_VERSION = 1
if korvid.__file__ is None:
    raise RuntimeError("upstream worker requires a filesystem Korvid package")
_SOURCE_ROOT = Path(korvid.__file__).resolve().parents[2]
_REFERENCE = re.compile(r"^(scenarios|journeys)/([a-z0-9][a-z0-9-]*)$")
_MODEL_BOUND_ERRORS = (
    re.compile(r"^iteration limit reached \(\d+\)"),
    re.compile(r"^history budget exceeded mid-turn \(\d+ chars\)"),
)
_RUNTIME_DISTRIBUTIONS = ("PyYAML", "boto3", "httpx", "korvid", "litellm", "textual")
_TRACE_ARGUMENT_LIMIT = 512


class UpstreamWorkerError(RuntimeError):
    """The unchanged upstream runtime could not produce scoreable evidence."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_value(value: Any) -> str:
    encoded = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_metadata() -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        if not name or not distribution.version:
            raise UpstreamWorkerError("upstream runtime dependency metadata is incomplete")
        dependencies[name.lower().replace("_", "-")] = distribution.version
    if any(name.lower() not in dependencies for name in _RUNTIME_DISTRIBUTIONS):
        raise UpstreamWorkerError("upstream runtime is missing required distribution metadata")
    if dependencies["korvid"] != korvid_version:
        raise UpstreamWorkerError("upstream Korvid source and distribution versions differ")
    identity = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "dependencies": dependencies,
        "lock_parity": "not-asserted",
    }
    return {**identity, "fingerprint": _sha_value(identity)}


def _model(value: Any) -> tuple[dict[str, Any], ModelConnectionConfig]:
    if not isinstance(value, dict):
        raise TypeError("model must be an object")
    reference, endpoint, options = (
        value.get("reference"),
        value.get("endpoint"),
        value.get("options"),
    )
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("model.reference must be a non-blank string")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("model.endpoint must be a non-blank string")
    if not isinstance(options, dict):
        raise TypeError("model.options must be an object")
    profile = ModelConnectionConfig(
        model=reference,
        endpoint=endpoint,
        auth=ConnectionAuthConfig(method="none"),
        options=options,
    )
    if profile.config_error is not None:
        raise ValueError(f"invalid model profile: {profile.config_error}")
    normalized = {
        "reference": reference,
        "endpoint": endpoint,
        "options": _plain(profile.options),
    }
    return normalized, profile


def _script(value: Any) -> list[list[dict[str, Any]]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(batch, list) for batch in value):
        raise TypeError("script must be a list of event lists")
    if any(not isinstance(event, dict) for batch in value for event in batch):
        raise TypeError("every scripted event must be an object")
    return value


def _live_provider(profile: ModelConnectionConfig) -> LLMProvider:
    flows = SpecialFlowRegistry.from_entry_points(reserved_prefixes=models_by_provider())
    provider = create_provider_from_profile(
        profile,
        catalog=LiteLLMModelCatalog(flows=flows),
        flows=flows,
        ca_bundle=None,
    )
    if provider is None:
        raise UpstreamWorkerError("upstream production provider factory refused the model")
    return provider


class _ObservedProvider(LLMProvider):
    def __init__(self, delegate: LLMProvider) -> None:
        self._delegate = delegate
        self.calls: list[dict[str, Any]] = []
        self.completions = 0

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._delegate.descriptor

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._delegate.capabilities

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._delegate.prepare_messages(messages)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        self.completions += 1
        async for event in self._delegate.complete(messages, tools, stream=stream):
            if isinstance(event, Mapping) and event.get("type") == "tool_call":
                raw_arguments = event.get("arguments", "{}")
                name = event.get("name") if isinstance(event.get("name"), str) else ""
                if not isinstance(raw_arguments, str):
                    self.calls.append(
                        {
                            "name": name,
                            "arguments_valid": False,
                            "arguments_raw": _bounded_trace(repr(raw_arguments)),
                            "error_label": "arguments_not_string",
                        }
                    )
                else:
                    try:
                        arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        self.calls.append(
                            {
                                "name": name,
                                "arguments_valid": False,
                                "arguments_raw": _bounded_trace(raw_arguments),
                                "error_label": "malformed_arguments",
                            }
                        )
                    else:
                        if isinstance(arguments, dict):
                            self.calls.append(
                                {
                                    "name": name,
                                    "arguments": arguments,
                                    "arguments_valid": True,
                                }
                            )
                        else:
                            self.calls.append(
                                {
                                    "name": name,
                                    "arguments_valid": False,
                                    "arguments_raw": _bounded_trace(raw_arguments),
                                    "error_label": "non_object_arguments",
                                }
                            )
            yield event

    async def aclose(self) -> None:
        await self._delegate.aclose()


def _provider_factory(
    profile: ModelConnectionConfig,
    script: list[list[dict[str, Any]]] | None,
    observed: list[_ObservedProvider],
) -> Callable[[], _ObservedProvider]:
    def factory() -> _ObservedProvider:
        delegate = ScriptedProvider(script) if script is not None else _live_provider(profile)
        provider = _ObservedProvider(delegate)
        observed.append(provider)
        return provider

    return factory


def _bounded_trace(value: str) -> str:
    return "".join(char if char.isprintable() else " " for char in value)[
        :_TRACE_ARGUMENT_LIMIT
    ]


def _diagnostic_names(policy: ResolvedAgentPolicy) -> frozenset[str]:
    names: set[str] = set()
    for schema in policy.tools:
        name = str(schema["function"]["name"])
        definition = tool_def(name)
        if definition is not None and definition.effect in (
            "cluster_read",
            "external_read",
        ):
            names.add(name)
    return frozenset(names)


def _source_path(reference: Any) -> tuple[str, str, Path]:
    if not isinstance(reference, str):
        raise TypeError("reference must be a string")
    match = _REFERENCE.fullmatch(reference)
    if match is None:
        raise ValueError("reference must name one scenarios/<id> or journeys/<id>")
    directory, case_id = match.groups()
    path = _SOURCE_ROOT / "src" / "korvid" / "evals" / directory / f"{case_id}.yaml"
    if not path.is_file():
        raise ValueError(f"unknown upstream source reference: {reference}")
    return directory, case_id, path


def _load_reference(reference: Any, expected_sha256: Any = None) -> tuple[str, Any, Path, str]:
    directory, case_id, path = _source_path(reference)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError("upstream source hash differs from the catalog")
    loaded = load_scenario(path) if directory == "scenarios" else load_journey(path)
    if loaded.id != case_id:
        raise ValueError("upstream source filename and declared id differ")
    return ("scenario" if directory == "scenarios" else "journey"), loaded, path, digest


def _policy(value: ResolvedAgentPolicy) -> dict[str, Any]:
    return {
        **policy_payload(value),
        "max_iterations": value.max_iterations,
        "max_history_chars": value.max_history_chars,
        "max_result_chars": value.max_result_chars,
        "max_tool_calls_per_iteration": value.max_tool_calls_per_iteration,
        "tools": sorted(str(tool["function"]["name"]) for tool in value.tools),
    }


async def _resolved_policy(
    profile: ModelConnectionConfig,
    script: list[list[dict[str, Any]]] | None,
) -> ResolvedAgentPolicy:
    provider = ScriptedProvider(script or [[{"type": "done"}]]) if script is not None else _live_provider(profile)
    try:
        return resolve_eval_policy(provider, model_tier="low")
    finally:
        await provider.aclose()


def _asset(text_or_value: Any) -> dict[str, Any]:
    key = "text" if isinstance(text_or_value, str) else "value"
    return {key: _plain(text_or_value), "sha256": _sha_value(text_or_value)}


def _authored_contexts(
    references: Sequence[Any],
    policy: ResolvedAgentPolicy,
    tier_pack: str,
) -> list[dict[str, Any]]:
    baseline = PromptHarness()
    copied = build_prompt_harness(policy, PromptGrind(tier_pack=tier_pack))
    contexts: list[dict[str, Any]] = []
    for reference in references:
        kind, source, _path, _digest = _load_reference(reference)
        authored: list[tuple[int, str, Any]]
        if kind == "scenario":
            authored = [(1, source.question, source.interaction)]
        else:
            authored = [(1, source.turns[0].user, source.interaction)]
            authored.extend(
                (index, turn.user, turn.interaction)
                for index, turn in enumerate(source.turns[1:], 2)
                if turn.interaction is not None
            )
        for turn, question, interaction in authored:
            inputs = PromptInputs(
                policy=policy,
                interaction=interaction,
                cluster=EVAL_CLUSTER,
                previous_interaction=None,
            )
            original = baseline.compose(question, inputs)
            replacement = copied.compose(question, inputs)
            original_text = f"{original.system_message}\0{original.user_message}"
            copied_text = f"{replacement.system_message}\0{replacement.user_message}"
            contexts.append(
                {
                    "reference": reference,
                    "turn": turn,
                    "original_system_message": original.system_message,
                    "original_user_message": original.user_message,
                    "baseline_sha256": _sha_text(original_text),
                    "copied_pack_sha256": _sha_text(copied_text),
                    "equivalent": original == replacement,
                }
            )
    return contexts


async def _inspect(payload: Mapping[str, Any]) -> dict[str, Any]:
    model, profile = _model(payload.get("model"))
    script = _script(payload.get("script"))
    references = payload.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("references must be a non-empty list")
    policy = await _resolved_policy(profile, script)
    pack_id = policy.prompt_pack_id
    tier_pack = PROMPT_PACKS[pack_id]
    copied = PromptGrind(tier_pack=tier_pack)
    original_static = static_prompt(policy, NO_GRIND)
    copied_static = static_prompt(policy, copied)
    from korvid.agent import prompt_harness as prompt_harness_module
    from korvid.agent import prompt_packs as prompt_packs_module
    from korvid.evals import harness as eval_harness_module

    source_assets = {}
    for name, module in (
        ("prompt_packs", prompt_packs_module),
        ("prompt_harness", prompt_harness_module),
        ("eval_harness", eval_harness_module),
    ):
        if module.__file__ is None:
            raise UpstreamWorkerError(f"{name} source path is unavailable")
        path = Path(module.__file__).resolve()
        source_assets[name] = {
            "source_path": str(path.relative_to(_SOURCE_ROOT)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "operation": "inspect",
        "model": model,
        "policy": _policy(policy),
        "prompt": {
            "text": tier_pack,
            "pack_id": pack_id,
            **source_assets["prompt_packs"],
            "baseline_equivalent": original_static == copied_static,
            "original_composed": {
                "text": original_static,
                "rendering": "canonical inspection rendering (source static_prompt)",
                "baseline_sha256": _sha_text(original_static),
                "copied_pack_sha256": _sha_text(copied_static),
                "fingerprint": prompt_fingerprint(policy, grind=NO_GRIND),
            },
            "authored_contexts": _authored_contexts(references, policy, tier_pack),
            "assets": {
                "safety_contract": _asset(SAFETY_CONTRACT),
                "common_role": _asset(COMMON_ROLE),
                "provider_overlays": _asset(PROVIDER_PROMPT_OVERLAYS),
                "model_overlays": _asset(MODEL_PROMPT_OVERLAYS),
                "sources": source_assets,
            },
        },
        "runtime": _runtime_metadata(),
    }
    candidate = payload.get("tier_pack")
    if candidate is not None:
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError("tier_pack must be a non-blank string")
        static_prompt(policy, PromptGrind(tier_pack=candidate))
        contexts = _authored_contexts(references, policy, candidate)
        response["candidate_validation"] = {
            "valid": True,
            "tier_pack_sha256": _sha_text(candidate),
            "authored_contexts": len(contexts),
        }
    return response


def _model_bound(error: str | None) -> bool:
    return error is not None and any(pattern.match(error) for pattern in _MODEL_BOUND_ERRORS)


def _scenario_upstream(run: Any) -> dict[str, Any]:
    if run.error is not None and not _model_bound(run.error):
        raise UpstreamWorkerError("upstream provider/runtime failed")
    return {
        "outcome": run.outcome,
        "failure_class": run.failure_class,
        "grade": _plain(asdict(run.grade)),
        "citations": _plain(asdict(run.citations)),
        "iterations": run.iterations,
        "tool_calls": run.tool_calls,
        "resolvable_tool_calls": run.resolvable_tool_calls,
        "on_target_tool_calls": run.on_target_tool_calls,
        "malformed_tool_calls": run.malformed_tool_calls,
        "write_attempts": run.write_attempts,
        "safety_violations": run.safety_violations,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "tokens_estimated": run.tokens_estimated,
        "wall_time_seconds": run.wall_time_s,
        "error_label": "model_bound" if _model_bound(run.error) else None,
    }


def _journey_upstream(run: Any) -> dict[str, Any]:
    for turn in run.turns:
        if turn.error is not None and not _model_bound(turn.error):
            raise UpstreamWorkerError("upstream provider/runtime failed")
    return {
        "success": run.success,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "tokens_estimated": run.tokens_estimated,
        "turns": [
            {
                "outcome": turn.outcome,
                "failure_class": turn.failure_class,
                "grade": _plain(asdict(turn.grade)),
                "tool_calls": turn.tool_calls,
                "tool_names": list(turn.tool_names),
                "malformed_tool_calls": turn.malformed_tool_calls,
                "write_attempts": turn.write_attempts,
                "safety_violations": turn.safety_violations,
                "forbidden_target_calls": turn.forbidden_target_calls,
                "wrong_namespace_calls": turn.wrong_namespace_calls,
                "wall_time_seconds": turn.wall_time_s,
                "error_label": "model_bound" if _model_bound(turn.error) else None,
            }
            for turn in run.turns
        ],
    }


async def _evaluate(payload: Mapping[str, Any]) -> dict[str, Any]:
    model, profile = _model(payload.get("model"))
    script = _script(payload.get("script"))
    tier_pack = payload.get("tier_pack")
    if not isinstance(tier_pack, str) or not tier_pack.strip():
        raise ValueError("tier_pack must be a non-blank string")
    prompt_path = payload.get("prompt_path")
    prompt_path_verified = False
    if prompt_path is not None:
        if not isinstance(prompt_path, str) or not prompt_path:
            raise TypeError("prompt_path must be a non-blank string")
        if Path(prompt_path).read_text(encoding="utf-8") != tier_pack:
            raise ValueError("reloaded optimized prompt differs from the candidate")
        prompt_path_verified = True
    kind, source, path, source_sha256 = _load_reference(
        payload.get("reference"), payload.get("source_sha256")
    )
    observed: list[_ObservedProvider] = []
    factory = _provider_factory(profile, script, observed)
    policy = await _resolved_policy(profile, script)
    grind = PromptGrind(tier_pack=tier_pack)
    if kind == "scenario":
        scenario: Scenario = source
        scenario_report = await run_scenario(
            scenario,
            provider_factory=factory,
            executor_factory=lambda: ToolExecutor(
                FakeKubeClient(scenario), builtin_aliases()
            ),
            repetitions=1,
            policy=policy,
            model_tier="low",
            grind=grind,
        )
        scenario_run = scenario_report.runs[0]
        upstream = _scenario_upstream(scenario_run)
        source_report = scenario_report_payload([scenario_report])[0]
        success = scenario_run.outcome == "success"
        safety_violations = scenario_run.safety_violations
        questions = [scenario.question]
    else:
        journey: ConversationJourney = source
        journey_report = await run_journey(
            journey,
            provider_factory=factory,
            executor_factory=lambda fixture: ToolExecutor(
                FakeKubeClient(fixture), builtin_aliases()
            ),
            repetitions=1,
            policy=policy,
            model_tier="low",
            grind=grind,
        )
        journey_run = journey_report.runs[0]
        upstream = _journey_upstream(journey_run)
        source_report = journey_report_payload([journey_report])[0]
        success = journey_run.success
        safety_violations = sum(
            turn.safety_violations for turn in journey_run.turns
        )
        questions = [turn.user for turn in journey.turns]
    calls = [call for provider in observed for call in provider.calls]
    diagnostic_names = _diagnostic_names(policy)
    diagnostic_calls = sum(
        1 for call in calls if call.get("name") in diagnostic_names
    )
    completions = sum(provider.completions for provider in observed)
    wall_time = (
        upstream["wall_time_seconds"]
        if kind == "scenario"
        else sum(turn["wall_time_seconds"] for turn in upstream["turns"])
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "operation": "evaluate",
        "execution_mode": "scripted" if script is not None else "live",
        "reference": payload["reference"],
        "case_id": source.id,
        "kind": kind,
        "source_path": str(path.relative_to(_SOURCE_ROOT)),
        "source_sha256": source_sha256,
        "questions": questions,
        "tier_pack_sha256": _sha_text(tier_pack),
        "prompt_path_verified": prompt_path_verified,
        "model": model,
        "policy": _policy(policy),
        "prompt": prompt_fingerprint(policy, grind=grind),
        "runtime": _runtime_metadata(),
        "success": success,
        "hard_failures": (
            ["upstream_safety_violation"] if safety_violations else []
        ),
        "source_report": source_report,
        "upstream": upstream,
        "calls": calls,
        "usage": {
            "iterations": completions,
            "tool_calls": len(calls),
            "diagnostic_calls": diagnostic_calls,
            "wall_time_seconds": wall_time,
        },
    }


def run_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")
    if korvid_version != "0.4.1":
        raise RuntimeError(f"upstream worker requires Korvid 0.4.1, found {korvid_version}")
    operation = payload.get("operation")
    if operation == "inspect":
        return asyncio.run(_inspect(payload))
    if operation == "evaluate":
        return asyncio.run(_evaluate(payload))
    raise ValueError("operation must be 'inspect' or 'evaluate'")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.request.resolve() == args.response.resolve():
            raise ValueError("request and response paths must differ")
        args.response.unlink(missing_ok=True)
        raw = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("request root must be an object")
        response = run_request(raw)
        args.response.write_text(
            json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        label = "".join(char if char.isprintable() else " " for char in str(exc))[:240]
        print(f"upstream-worker: {label}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
