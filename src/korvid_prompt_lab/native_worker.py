"""Standalone Korvid v0.4.1 native UI evaluation worker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]
from korvid import __version__ as korvid_version
from korvid.__main__ import (
    _AgentToolUIBridgeProxy,
    _AgentUiBridgeProxy,
    _create_provider_from_active_profile,
)
from korvid.agent.events import (
    AgentError,
    AgentEvent,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from korvid.agent.interaction import InteractionContext, PaneContext
from korvid.agent.model_policy import (
    ModelCapabilities,
    ModelDescriptor,
    ResolvedAgentPolicy,
)
from korvid.agent.prompt_harness import PromptHarness, PromptInputs, _user_rule_layer
from korvid.agent.provider import LLMProvider
from korvid.core.config import (
    ConnectionAuthConfig,
    KorvidConfig,
    ModelConnectionConfig,
    _parse_agent_rules,
    load_config,
)
from korvid.core.store import ALL_NAMESPACES, ResourceStore, Summary
from korvid.core.watch import WatchManager
from korvid.evals.fake_kube import FakeKubeClient, builtin_aliases
from korvid.evals.harness import EVAL_CLUSTER, build_eval_harness, resolve_eval_policy
from korvid.evals.scenario import ContainerLogs
from korvid.evals.scripted import ScriptedProvider
from korvid.k8s.discovery import ResourceMeta
from korvid.k8s.helm import (
    HELM_RELEASES_META,
    HELM_REVISIONS_META,
    revision_from_secret,
)
from korvid.k8s.logs import LogLine
from korvid.k8s.models import PodSummary
from korvid.tools.executor import ToolExecutor
from korvid.ui.app import AppUIBridge, KorvidApp
from korvid.ui.widgets.describe_screen import DescribeScreen
from navigation_cases import (
    NavigationCase,
    find_navigation_case,
    fixture_objects,
)

PROTOCOL_VERSION = 1
SUPPORTED_CASE_TASKS = frozenset({"pods", "helm", "all-pods", "logs", "describe"})
_ERROR_LABEL_LIMIT = 240
_RUNTIME_DISTRIBUTIONS = ("PyYAML", "boto3", "korvid", "litellm", "textual")


class NativeWorkerError(RuntimeError):
    """The native runtime failed after request validation."""


@dataclass(frozen=True)
class _Fixture:
    objects: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    logs: dict[str, ContainerLogs]
    forbidden: tuple[dict[str, str], ...] = ()


class _RecordingPanel:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.events: list[AgentEvent] = []

    def apply_event(self, event: AgentEvent) -> None:
        self.events.append(event)
        self._delegate.apply_event(event)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _ProviderObserver(LLMProvider):
    """Delegate provider calls while counting actual model rounds."""

    def __init__(self, delegate: LLMProvider) -> None:
        self._delegate = delegate
        self.iterations = 0

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._delegate.descriptor

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._delegate.capabilities

    def prepare_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return self._delegate.prepare_messages(messages)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        self.iterations += 1
        async for event in self._delegate.complete(messages, tools, stream=stream):
            yield event

    async def aclose(self) -> None:
        await self._delegate.aclose()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


def _bounded_label(value: object) -> str:
    text = "".join(char if char.isprintable() else " " for char in str(value))
    return text[:_ERROR_LABEL_LIMIT]


def _runtime_metadata() -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        if not name or not distribution.version:
            raise NativeWorkerError("native runtime dependency metadata is incomplete")
        dependencies[name.lower().replace("_", "-")] = distribution.version
    if any(name.lower() not in dependencies for name in _RUNTIME_DISTRIBUTIONS):
        raise NativeWorkerError("native runtime is missing required distribution metadata")
    if dependencies["korvid"] != korvid_version:
        raise NativeWorkerError("native Korvid source and installed distribution versions differ")
    identity = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "dependencies": dependencies,
        "lock_parity": "not-asserted",
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**identity, "fingerprint": fingerprint}


def _validated_rules(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError("rules must be a list")
    parsed, warnings = _parse_agent_rules(value)
    if warnings or list(parsed) != value:
        detail = "; ".join(warnings) or "rules must already be stripped strings"
        raise ValueError(f"invalid agent.rules: {detail}")
    return parsed


def _model_profile(value: Any) -> tuple[dict[str, Any], ModelConnectionConfig]:
    if not isinstance(value, dict):
        raise TypeError("model must be an object")
    reference = value.get("reference")
    endpoint = value.get("endpoint")
    options = value.get("options")
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
    model = {
        "reference": reference,
        "endpoint": endpoint,
        "options": _plain(profile.options),
    }
    return model, profile


def _provider(
    payload: Mapping[str, Any], profile: ModelConnectionConfig
) -> tuple[str, LLMProvider]:
    script = payload.get("script")
    if script is not None:
        if not isinstance(script, list) or any(
            not isinstance(batch, list) for batch in script
        ):
            raise ValueError("script must be null or a list of event lists")
        for batch in script:
            if any(not isinstance(event, dict) for event in batch):
                raise ValueError("every scripted event must be an object")
        return "scripted", ScriptedProvider(script)
    provider = _create_provider_from_active_profile(profile, None, None)
    if provider is None:
        raise RuntimeError("production provider factory refused the model profile")
    return "live", provider


def _policy_payload(policy: ResolvedAgentPolicy) -> dict[str, Any]:
    return {
        "tier": policy.tier.value,
        "tools": sorted(str(tool["function"]["name"]) for tool in policy.tools),
        "max_iterations": policy.max_iterations,
        "max_history_chars": policy.max_history_chars,
        "max_result_chars": policy.max_result_chars,
        "max_tool_calls_per_iteration": policy.max_tool_calls_per_iteration,
        "prompt_pack_id": policy.prompt_pack_id,
    }


def _prompt_digest(system_message: str, tools: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(system_message.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(
        json.dumps(
            [_plain(tool) for tool in tools], sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _rules_once(system_message: str, rules: tuple[str, ...]) -> bool:
    layers = _user_rule_layer(rules)
    return all(system_message.count(layer) == 1 for layer in layers)


def _initial_interaction(case: NavigationCase) -> InteractionContext:
    return InteractionContext(
        kube_context=None,
        context_epoch=0,
        focused_pane=PaneContext(
            kind=case.initial_kind,
            scope=case.initial_scope,
            filter_pattern=case.initial_filter or None,
            selected=None,
        ),
        secondary_pane=None,
        timeline_cursor=None,
    )


def _static_identity(
    policy: ResolvedAgentPolicy,
    rules: tuple[str, ...],
    case: NavigationCase,
) -> tuple[str, bool]:
    prompts = PromptHarness()
    composed = prompts.compose(
        case.prompt,
        PromptInputs(
            policy=policy,
            interaction=_initial_interaction(case),
            cluster=EVAL_CLUSTER,
            user_rules=rules,
            previous_interaction=None,
        ),
    )
    return _prompt_digest(composed.system_message, policy.tools), _rules_once(
        composed.system_message, rules
    )


def _case(payload: Mapping[str, Any]) -> NavigationCase:
    case_id = payload.get("case_id")
    if not isinstance(case_id, str):
        raise TypeError("case_id must be a string")
    case = find_navigation_case(case_id)
    task = case_id.split("-", 1)[1] if "-" in case_id else ""
    if task not in SUPPORTED_CASE_TASKS:
        raise ValueError(f"unsupported native UI case: {case_id}")
    return case


def _aliases() -> dict[str, ResourceMeta]:
    aliases = builtin_aliases()
    for meta in (HELM_RELEASES_META, HELM_REVISIONS_META):
        for alias in (meta.plural, meta.kind.lower(), *meta.shortnames):
            aliases[alias] = meta
    return aliases


def _fixture(case: NavigationCase) -> _Fixture:
    return _Fixture(
        objects=fixture_objects(case),
        events=(),
        logs={
            f"{case.namespace}/{case.pod}/main": ContainerLogs(
                current=("fixture ready",)
            )
        },
    )


def _watch_source(
    kube: FakeKubeClient,
    aliases: Mapping[str, ResourceMeta],
    objects: tuple[dict[str, Any], ...],
) -> Any:
    async def source(kind: str, scope: str) -> AsyncIterator[tuple[str, Summary]]:
        namespace = None if scope == ALL_NAMESPACES else scope
        rows: Sequence[Summary]
        if kind == "pods":
            rows = [
                PodSummary.from_manifest(obj)
                for obj in objects
                if obj.get("kind") == "Pod"
                and (
                    namespace is None
                    or obj.get("metadata", {}).get("namespace") == namespace
                )
            ]
        elif kind == HELM_RELEASES_META.plural:
            rows = await kube.list_helm_releases(namespace)
        elif kind == HELM_REVISIONS_META.plural:
            rows = [
                revision_from_secret(obj)
                for obj in objects
                if obj.get("type") == "helm.sh/release.v1"
                and (
                    namespace is None
                    or obj.get("metadata", {}).get("namespace") == namespace
                )
            ]
        else:
            meta = aliases.get(kind)
            if meta is None:
                raise ValueError(f"unknown fixture kind: {kind}")
            rows = await kube.list_objects(meta, namespace)
        for row in rows:
            yield "ADDED", row
        await asyncio.Event().wait()

    return source


def _get_manifest(kube: FakeKubeClient, aliases: Mapping[str, ResourceMeta]) -> Any:
    async def get_manifest(
        kind: str, namespace: str | None, name: str
    ) -> dict[str, Any]:
        meta = aliases.get(kind) or aliases.get(kind.lower())
        if meta is None:
            raise ValueError(f"unknown fixture kind: {kind}")
        return await kube.get_object(meta, namespace, name)

    return get_manifest


def _stream_logs(kube: FakeKubeClient) -> Any:
    async def stream_logs(
        namespace: str,
        pod: str,
        container: str,
        **kwargs: Any,
    ) -> AsyncIterator[LogLine]:
        async for line in kube.stream_logs(namespace, pod, container, **kwargs):
            yield line
        await asyncio.Event().wait()

    return stream_logs


async def _fixture_namespaces() -> list[str]:
    return ["default", "shop", "monitoring", "sandbox", "other"]


def _normalize_kind(kind: str, aliases: Mapping[str, ResourceMeta]) -> str:
    meta = aliases.get(kind) or aliases.get(kind.lower())
    return meta.plural if meta is not None else kind.lower()


def _observe(app: KorvidApp, aliases: Mapping[str, ResourceMeta]) -> dict[str, str]:
    context = app.agent_ui.workspace_bridge.snapshot()
    pane = context.focused_pane
    logs = ""
    triples = app._logs.current_triples if app._logs.mode else []
    if triples:
        logs = ",".join("/".join(triple) for triple in triples)
    describe = ""
    if isinstance(app.screen, DescribeScreen):
        identity = app.screen.resource_identity
        if identity is not None:
            describe = (
                f"{_normalize_kind(identity.kind, aliases)}/"
                f"{identity.namespace or ''}/{identity.name}"
            )
    return {
        "kind": _normalize_kind(pane.kind, aliases),
        "scope": pane.scope,
        "filter": pane.filter_pattern or "",
        "logs": logs,
        "describe": describe,
    }


def _expected(case: NavigationCase) -> dict[str, str]:
    return {"logs": "", "describe": "", **dict(case.expected)}


def _missing(expected: Mapping[str, str], observed: Mapping[str, str]) -> list[str]:
    return [
        f"{key}: expected {value!r}, observed {observed.get(key, '')!r}"
        for key, value in expected.items()
        if observed.get(key) != value
    ]


def _calls(events: Sequence[AgentEvent]) -> tuple[list[dict[str, Any]], list[str]]:
    started: dict[str, tuple[str, dict[str, Any]]] = {}
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, event in enumerate(events):
        if isinstance(event, AgentError):
            followed_by_completion = (
                index + 1 < len(events)
                and isinstance(events[index + 1], TurnComplete)
            )
            if followed_by_completion and re.fullmatch(
                r"iteration limit reached \(\d+\) — refine the question",
                event.message,
            ):
                errors.append("agent:iteration-limit")
                continue
            if followed_by_completion and re.fullmatch(
                r"history budget exceeded mid-turn \(\d+ chars\) — turn ended early",
                event.message,
            ):
                errors.append("agent:history-budget-exceeded")
                continue
            raise NativeWorkerError("native agent turn failed")
        if isinstance(event, ToolCallStarted):
            arguments: dict[str, Any] = {}
            try:
                decoded = json.loads(event.arguments) if event.arguments else {}
            except json.JSONDecodeError:
                errors.append(f"tool:{event.name}:malformed-arguments")
            else:
                if isinstance(decoded, dict):
                    arguments = decoded
                else:
                    errors.append(f"tool:{event.name}:non-object-arguments")
            started[event.call_id] = (event.name, arguments)
        elif isinstance(event, ToolCallFinished):
            name, arguments = started.pop(event.call_id, (event.name, {}))
            calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "ok": event.ok,
                    "result": event.summary,
                }
            )
            if not event.ok:
                errors.append(f"tool:{name}:failed")
    for name, _arguments in started.values():
        errors.append(f"tool:{name}:unfinished")
    return calls, [_bounded_label(error) for error in errors[:16]]


def _actual_prompt_identity(
    app: KorvidApp,
    rules: tuple[str, ...],
) -> tuple[str, bool]:
    session = app.agent_session
    if session is None or session.latest_outbound_payload is None:
        raise RuntimeError("native agent produced no outbound payload")
    payload = json.loads(session.latest_outbound_payload.payload_json)
    messages = payload.get("messages")
    tools = payload.get("tools")
    if not isinstance(messages, list) or not isinstance(tools, list):
        raise NativeWorkerError("native outbound payload has an invalid shape")
    system = next(
        (
            message.get("content")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "system"
        ),
        None,
    )
    if not isinstance(system, str):
        raise NativeWorkerError("native outbound payload has no system message")
    return _prompt_digest(system, tools), _rules_once(system, rules)


async def _evaluate(
    payload: Mapping[str, Any],
    rules: tuple[str, ...],
    model: dict[str, Any],
    profile: ModelConnectionConfig,
    case: NavigationCase,
) -> dict[str, Any]:
    config_path = payload.get("config_path")
    if config_path is None:
        config = KorvidConfig(
            namespace=case.initial_scope,
            readonly=True,
            agent_model_tier="low",
            agent_rules=rules,
            agent_follow=True,
        )
    else:
        if not isinstance(config_path, str) or not config_path:
            raise TypeError("config_path must be a non-blank string when provided")
        config = load_config(Path(config_path))
        _assert_loaded_config(config, rules, profile)
    execution_mode, raw_provider = _provider(payload, profile)
    provider = _ProviderObserver(raw_provider)
    fixture = _fixture(case)
    kube = FakeKubeClient(fixture)
    aliases = _aliases()
    store = ResourceStore()
    watches = WatchManager(store, _watch_source(kube, aliases, fixture.objects))
    tool_proxy = _AgentToolUIBridgeProxy()
    agent_bridge = _AgentUiBridgeProxy()
    executor = ToolExecutor(kube, aliases, ui=tool_proxy)
    harness = build_eval_harness(
        provider=provider,
        execution=executor,
        bridge=agent_bridge,
        model_tier="low",
        user_rules=config.agent_rules,
    )
    app = KorvidApp(
        config=config,
        store=store,
        watch_manager=watches,
        list_namespaces=_fixture_namespaces,
        aliases=aliases,
        get_manifest=_get_manifest(kube, aliases),
        stream_logs=_stream_logs(kube),
        agent_session=harness.session,
        agent_model_name=profile.model,
        agent_follow_bridge=tool_proxy,
    )
    tool_proxy.target = AppUIBridge(app)
    agent_bridge.target = app.agent_ui.workspace_bridge
    panel = _RecordingPanel(app.agent_ui._panel)
    app.agent_ui._panel = cast(Any, panel)
    started: float | None = None
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            setup = await tool_proxy.agent_navigate(
                case.initial_kind, case.initial_scope
            )
            if setup.startswith("ERROR:"):
                raise RuntimeError(
                    f"native UI fixture setup failed: {_bounded_label(setup)}"
                )
            if case.initial_filter:
                setup = await tool_proxy.agent_set_filter(case.initial_filter)
                if setup.startswith("ERROR:"):
                    raise RuntimeError(
                        f"native UI fixture setup failed: {_bounded_label(setup)}"
                    )
            await pilot.pause()
            initial = _observe(app, aliases)
            started = time.monotonic()
            await app.agent_ui.run_turn(case.prompt)
            await pilot.pause()
            observed = _observe(app, aliases)
            fingerprint, rules_applied = _actual_prompt_identity(app, rules)
    finally:
        await provider.aclose()
    if started is None:
        raise NativeWorkerError("native agent turn did not start")
    wall_time = time.monotonic() - started
    expected = _expected(case)
    calls, errors = _calls(panel.events)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "execution_mode": execution_mode,
        "rules": list(rules),
        "korvid_version": korvid_version,
        "case_id": case.case_id,
        "initial": initial,
        "expected": expected,
        "observed": observed,
        "missing_postconditions": _missing(expected, observed),
        "calls": calls,
        "errors": errors,
        "policy": _policy_payload(harness.policy),
        "prompt_fingerprint": fingerprint,
        "rules_applied": rules_applied and harness.user_rules == rules,
        "ui_follow": config.agent_follow,
        "model": model,
        "runtime": _runtime_metadata(),
        "wall_time_seconds": wall_time,
        "iterations": provider.iterations,
    }


def _verify_config(
    payload: Mapping[str, Any],
    rules: tuple[str, ...],
    model: dict[str, Any],
    profile: ModelConnectionConfig,
    case: NavigationCase,
) -> dict[str, Any]:
    with _verification_config(payload.get("config_path"), rules, profile) as config:
        _assert_loaded_config(config, rules, profile)
    execution_mode, provider = _provider(payload, profile)
    try:
        policy = resolve_eval_policy(provider, model_tier="low")
        fingerprint, rules_applied = _static_identity(policy, rules, case)
    finally:
        asyncio.run(provider.aclose())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "operation": "verify-config",
        "execution_mode": execution_mode,
        "rules": list(rules),
        "korvid_version": korvid_version,
        "model": model,
        "policy": _policy_payload(policy),
        "prompt_fingerprint": fingerprint,
        "rules_applied": rules_applied,
        "ui_follow": config.agent_follow,
        "runtime": _runtime_metadata(),
    }


@contextmanager
def _verification_config(
    config_path: Any,
    rules: tuple[str, ...],
    profile: ModelConnectionConfig,
) -> Generator[KorvidConfig, None, None]:
    if config_path is not None:
        if not isinstance(config_path, str) or not config_path:
            raise TypeError("config_path must be a non-blank string when provided")
        yield load_config(Path(config_path))
        return

    document = yaml.safe_dump(
        {
            "readonly": True,
            "agent": {
                "active": "prompt-lab",
                "model_tier": "low",
                "follow": True,
                "rules": list(rules),
                "profiles": {
                    "prompt-lab": {
                        "model": profile.model,
                        "endpoint": profile.endpoint,
                        "auth": {"method": "none"},
                        "options": _plain(profile.options),
                    }
                },
            },
        },
        allow_unicode=True,
        sort_keys=True,
    )
    with tempfile.TemporaryDirectory(prefix="korvid-native-verify-") as temporary:
        path = Path(temporary) / "config.yaml"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        yield load_config(path)


def _assert_loaded_config(
    config: KorvidConfig,
    rules: tuple[str, ...],
    expected_profile: ModelConnectionConfig,
) -> None:
    rule_warnings = [warning for warning in config.warnings if "agent.rules" in warning]
    if rule_warnings:
        raise ValueError(f"invalid agent.rules: {'; '.join(rule_warnings)}")
    if config.agent_rules != rules:
        raise ValueError("loaded agent.rules do not exactly match requested rules")
    if not config.readonly:
        raise ValueError("loaded config must set readonly: true")
    if config.agent_model_tier != "low":
        raise ValueError("loaded config must set agent.model_tier: low")
    if not config.agent_follow:
        raise ValueError("loaded config must set agent.follow: true")
    loaded_profile = config.model_connections.active_profile
    if loaded_profile is None:
        raise ValueError("loaded config has no active model profile")
    if (
        loaded_profile.model != expected_profile.model
        or loaded_profile.endpoint != expected_profile.endpoint
        or loaded_profile.auth.method != "none"
        or _plain(loaded_profile.auth.settings) != {}
        or _plain(loaded_profile.options) != _plain(expected_profile.options)
    ):
        raise ValueError(
            "loaded config model profile does not exactly match requested model"
        )


def run_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and execute one worker request."""
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")
    if korvid_version != "0.4.1":
        raise RuntimeError(
            f"native worker requires Korvid 0.4.1, found {korvid_version}"
        )
    operation = payload.get("operation")
    if operation not in {"evaluate", "verify-config"}:
        raise ValueError("operation must be 'evaluate' or 'verify-config'")
    rules = _validated_rules(payload.get("rules"))
    model, profile = _model_profile(payload.get("model"))
    case = _case(payload)
    if operation == "verify-config":
        return _verify_config(payload, rules, model, profile, case)
    return asyncio.run(_evaluate(payload, rules, model, profile, case))


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
    except (OSError, TypeError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"native-worker: {_bounded_label(exc)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
