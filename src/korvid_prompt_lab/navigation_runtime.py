from __future__ import annotations

import asyncio
import copy
import json
import math
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

import anyio
import httpx2
from korvid.agent.events import (
    AgentError,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from korvid.agent.outbound import OutboundPolicyError, sanitize_recorded_tool_result
from korvid.agent.provider import LLMProvider
from korvid.agent.runtime import AgentRuntime, _is_recorded_read, _parsed_arguments
from korvid.core.redaction import RedactionRecord
from korvid.tools.executor import RecordedExecution, ToolOutcome
from korvid.tools.registry import TOOL_DEFS, ToolDef
from mcp import types as mcp_types
from mcp.client.extension import UnexpectedClaimedResult
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from .contracts import Candidate, _require_string

NAVIGATION_TOOL_NAMES = frozenset(
    {
        "navigate",
        "set_filter",
        "drill_down",
        "open_logs",
        "open_describe",
        "list_resources",
        "helm_list_releases",
    }
)

_NAVIGATION_TOOL_ORDER = (
    "list_resources",
    "helm_list_releases",
    "navigate",
    "set_filter",
    "open_logs",
    "open_describe",
    "drill_down",
)
_NAVIGATION_EFFECTS = frozenset({"cluster_read", "ui_only"})
_MAX_TOOL_CALLS_PER_ITERATION = 2
_MALFORMED_ARGUMENTS_RESULT = "ERROR: bad arguments"
_FORBIDDEN_TOOL_RESULT = "ERROR: tool not allowed"
_TOOL_EXECUTION_BLOCKED_RESULT = "ERROR: tool execution blocked"
_DISCARDED_TOOL_CALL_SUMMARY = "discarded: too many tool calls in one response"
_MCP_TRANSPORT_ERRORS = (
    OSError,
    httpx2.HTTPError,
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
    anyio.EndOfStream,
)
_MCP_PROTOCOL_ERRORS = (MCPError, UnexpectedClaimedResult, ValidationError)
_MCP_RUNTIME_ERRORS = _MCP_PROTOCOL_ERRORS + _MCP_TRANSPORT_ERRORS


class NavigationRuntimeError(RuntimeError):
    """Base error for the bounded MCP navigation runtime."""


class NavigationTimeoutError(NavigationRuntimeError):
    """The navigation turn exceeded its outer wall-clock budget."""


class NavigationMCPError(NavigationRuntimeError):
    """The MCP transport failed while discovering or executing tools."""


class NavigationProtocolError(NavigationRuntimeError):
    """The MCP server advertised an incompatible or unusable contract."""


@dataclass(frozen=True, slots=True)
class NavigationCall:
    name: str
    arguments: dict[str, Any]
    result: str
    ok: bool


@dataclass(frozen=True, slots=True)
class NavigationTurn:
    answer: str
    calls: tuple[NavigationCall, ...]
    errors: tuple[str, ...]
    blocked_tools: tuple[str, ...]
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0


class _IterationProvider(LLMProvider):
    def __init__(self, delegate: LLMProvider) -> None:
        self.delegate = delegate
        self.iterations = 0

    @property
    def name(self) -> str:
        return self.delegate.name

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.delegate.prepare_messages(messages)

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, stream: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        self.iterations += 1
        async for event in self.delegate.complete(messages, tools, stream=stream):
            yield event

    async def aclose(self) -> None:
        await self.delegate.aclose()


@dataclass(frozen=True, slots=True)
class _RecordedCall:
    call: NavigationCall
    error_label: str | None = None
    blocked: bool = False


class _NavigationSession(Protocol):
    async def list_tools(
        self,
        *,
        params: mcp_types.PaginatedRequestParams | None = None,
    ) -> mcp_types.ListToolsResult: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> mcp_types.CallToolResult: ...


class _SystemicToolFailure(OutboundPolicyError):
    headline = "navigation runtime aborted"


class _NavigationAgentRuntime(AgentRuntime):
    async def _tool_result(
        self, name: str, arguments: str
    ) -> tuple[str, tuple[RedactionRecord, ...], bool, str | None]:
        produced: tuple[RedactionRecord, ...] = ()
        errored = True
        incarnation: str | None = None
        container: str | None = None
        parsed = _parsed_arguments(arguments)
        if parsed is None:
            result = _MALFORMED_ARGUMENTS_RESULT
        else:
            (
                result,
                produced,
                errored,
                incarnation,
                container,
            ) = await self._execute_tool(name, parsed)
        text, records = sanitize_recorded_tool_result(
            name,
            result,
            produced,
            max_chars=self._max_result_chars,
            error=errored,
            result_format=self._result_formats.get(name),
        )
        ref = None
        if _is_recorded_read(name):
            ref = self._evidence.record(
                name,
                parsed or {},
                text,
                error=errored,
                incarnation=incarnation,
                container=container,
            )
        return text, records, errored, ref


def _classify_mcp_exception(exc: BaseException, *, phase: str) -> NavigationRuntimeError | None:
    if isinstance(exc, asyncio.CancelledError):
        return None
    if isinstance(exc, BaseExceptionGroup):
        nested = [_classify_mcp_exception(child, phase=phase) for child in exc.exceptions]
        if any(item is None for item in nested):
            return None
        if any(isinstance(item, NavigationProtocolError) for item in nested):
            return NavigationProtocolError(f"navigation MCP protocol failed during {phase}")
        return NavigationMCPError(f"navigation MCP transport failed during {phase}")
    if isinstance(exc, _MCP_PROTOCOL_ERRORS):
        return NavigationProtocolError(f"navigation MCP protocol failed during {phase}")
    if isinstance(exc, _MCP_TRANSPORT_ERRORS):
        return NavigationMCPError(f"navigation MCP transport failed during {phase}")
    return None


def _require_classified_mcp_exception(exc: BaseException, *, phase: str) -> NavigationRuntimeError:
    classified = _classify_mcp_exception(exc, phase=phase)
    if classified is None:
        raise TypeError(f"unclassifiable MCP exception for phase {phase!r}: {type(exc).__name__}")
    return classified


def _load_navigation_tool_defs() -> tuple[ToolDef, ...]:
    by_name = {definition.name: definition for definition in TOOL_DEFS if definition.name in NAVIGATION_TOOL_NAMES}
    missing = NAVIGATION_TOOL_NAMES - set(by_name)
    if missing:
        raise NavigationProtocolError(
            f"installed korvid registry is missing navigation tool(s): {', '.join(sorted(missing))}"
        )
    ordered: list[ToolDef] = []
    for name in _NAVIGATION_TOOL_ORDER:
        definition = by_name[name]
        if definition.effect not in _NAVIGATION_EFFECTS:
            raise NavigationProtocolError(
                f"installed korvid registry exposes {name!r} with forbidden effect {definition.effect!r}"
            )
        if "mcp" not in definition.surfaces:
            raise NavigationProtocolError(f"installed korvid registry does not expose {name!r} on MCP")
        ordered.append(definition)
    return tuple(ordered)


_NAVIGATION_TOOL_DEFS = _load_navigation_tool_defs()


def get_navigation_tools() -> list[dict[str, Any]]:
    """Korvid's MCP navigation tools, filtered to the bounded allowlist."""
    return [copy.deepcopy(definition.schema) for definition in _NAVIGATION_TOOL_DEFS]


def _validate_navigation_candidate(candidate: Candidate) -> tuple[str, dict[str, str]]:
    components = candidate.components
    system = components.get("system")
    if system is None:
        raise ValueError("navigation candidate requires a system component")
    system_prompt = _require_string(system, "navigation candidate system")
    append = components.get("append")
    if append is not None:
        system_prompt = f"{system_prompt} {_require_string(append, 'navigation candidate append')}"

    tool_overrides: dict[str, str] = {}
    for key, value in components.items():
        if key in {"system", "append"}:
            continue
        if not key.startswith("tool.") or len(key) <= len("tool."):
            raise ValueError(
                f"navigation candidate component key must be system, append, or tool.<tool-name>: {key!r}"
            )
        tool_name = key[len("tool.") :]
        if tool_name not in NAVIGATION_TOOL_NAMES:
            raise ValueError(f"navigation candidate does not support component {key!r}")
        tool_overrides[tool_name] = _require_string(value, f"navigation candidate component {key}")
    return system_prompt, tool_overrides


def _schema_contract(schema: Any) -> Any:
    if not isinstance(schema, Mapping):
        raise NavigationProtocolError("tool schema must be a JSON object")
    contract: dict[str, Any] = {}
    for key in (
        "type",
        "enum",
        "const",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
    ):
        if key in schema:
            contract[key] = copy.deepcopy(schema[key])
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise NavigationProtocolError("tool schema required must be a list of strings")
        contract["required"] = sorted(required)
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, Mapping):
            raise NavigationProtocolError("tool schema properties must be a mapping")
        contract["properties"] = {name: _schema_contract(value) for name, value in sorted(properties.items())}
    if "items" in schema:
        items = schema["items"]
        contract["items"] = _schema_contract(items) if isinstance(items, Mapping) else copy.deepcopy(items)
    for key in ("oneOf", "anyOf", "allOf"):
        if key in schema:
            options = schema[key]
            if not isinstance(options, list):
                raise NavigationProtocolError(f"tool schema {key} must be a list")
            contract[key] = [
                _schema_contract(option) if isinstance(option, Mapping) else copy.deepcopy(option)
                for option in options
            ]
    return contract


def _tool_schema_parameters(tool_schema: Mapping[str, Any]) -> Mapping[str, Any]:
    function = tool_schema.get("function")
    if not isinstance(function, Mapping):
        raise NavigationProtocolError("tool schema is missing function metadata")
    parameters = function.get("parameters")
    if not isinstance(parameters, Mapping):
        raise NavigationProtocolError("tool schema is missing JSON parameters")
    return parameters


def _ensure_tool_compatible(definition: ToolDef, tool: mcp_types.Tool) -> None:
    expected = _schema_contract(_tool_schema_parameters(definition.schema))
    observed = _schema_contract(tool.input_schema)
    if expected != observed:
        raise NavigationProtocolError(f"advertised schema mismatch for navigation tool {definition.name!r}")


def _openai_schema_from_mcp(tool: mcp_types.Tool, definition: ToolDef) -> dict[str, Any]:
    description = tool.description or definition.schema["function"].get("description", "")
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": description,
            "parameters": copy.deepcopy(tool.input_schema),
        },
    }


async def _list_advertised_tools(session: _NavigationSession) -> list[mcp_types.Tool]:
    tools: list[mcp_types.Tool] = []
    seen_cursors: set[str] = set()
    cursor: str | None = None
    while True:
        params = mcp_types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None
        try:
            result = await session.list_tools(params=params)
        except asyncio.CancelledError:
            raise
        except BaseExceptionGroup as exc:
            classified = _classify_mcp_exception(exc, phase="tools/list")
            if classified is not None:
                raise classified from exc
            raise
        except _MCP_RUNTIME_ERRORS as exc:
            raise _require_classified_mcp_exception(exc, phase="tools/list") from exc
        if not isinstance(result, mcp_types.ListToolsResult):
            raise NavigationProtocolError(
                f"tools/list returned {type(result).__name__}, expected ListToolsResult"
            )
        tools.extend(result.tools)
        cursor = result.next_cursor
        if cursor is None:
            return tools
        if cursor in seen_cursors:
            raise NavigationProtocolError("tools/list repeated a pagination cursor")
        seen_cursors.add(cursor)


async def _discover_navigation_tools(session: _NavigationSession) -> list[dict[str, Any]]:
    advertised = await _list_advertised_tools(session)
    advertised_by_name: dict[str, mcp_types.Tool] = {}
    for tool in advertised:
        if tool.name in advertised_by_name and tool.name in NAVIGATION_TOOL_NAMES:
            raise NavigationProtocolError(f"tools/list advertised {tool.name!r} more than once")
        advertised_by_name.setdefault(tool.name, tool)
    missing = [name for name in _NAVIGATION_TOOL_ORDER if name not in advertised_by_name]
    if missing:
        raise NavigationProtocolError(
            f"tools/list missing required navigation tool(s): {', '.join(missing)}"
        )

    discovered: list[dict[str, Any]] = []
    for definition in _NAVIGATION_TOOL_DEFS:
        advertised_tool = advertised_by_name.get(definition.name)
        if advertised_tool is None:
            continue
        _ensure_tool_compatible(definition, advertised_tool)
        discovered.append(_openai_schema_from_mcp(advertised_tool, definition))
    return discovered


def _apply_tool_overrides(
    tools: Sequence[Mapping[str, Any]],
    descriptions: Mapping[str, str],
) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for tool in tools:
        copied = cast(dict[str, Any], copy.deepcopy(tool))
        name = copied["function"]["name"]
        if name in descriptions:
            copied["function"]["description"] = descriptions[name]
        rewritten.append(copied)
    return rewritten


def _type_names(schema: Mapping[str, Any]) -> tuple[str, ...]:
    raw = schema.get("type")
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list) and raw and all(isinstance(item, str) for item in raw):
        return tuple(raw)
    raise NavigationProtocolError("tool schema type must be a string or a list of strings")


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        return math.isfinite(float(value))
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True


def _validate_value(path: str, value: Any, schema: Mapping[str, Any]) -> None:
    expected = _type_names(schema)
    if expected and not any(_matches_type(value, name) for name in expected):
        raise ValueError(f"{path}: invalid type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is outside the advertised enum")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: value does not match the advertised constant")
    if "minimum" in schema and isinstance(value, int | float) and float(value) < float(schema["minimum"]):
        raise ValueError(f"{path}: value is below the advertised minimum")
    if "maximum" in schema and isinstance(value, int | float) and float(value) > float(schema["maximum"]):
        raise ValueError(f"{path}: value is above the advertised maximum")
    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ValueError(f"{path}: string is shorter than the advertised minimum length")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ValueError(f"{path}: string is longer than the advertised maximum length")
    if isinstance(value, Mapping):
        _validate_object(path, value, schema)
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                _validate_value(f"{path}[{index}]", item, items)


def _validate_object(path: str, value: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    properties_raw = schema.get("properties", {})
    if not isinstance(properties_raw, Mapping):
        raise NavigationProtocolError("tool schema properties must be a mapping")
    properties = {str(name): prop for name, prop in properties_raw.items()}
    unknown = sorted(key for key in value if key not in properties)
    if unknown:
        raise ValueError(f"{path}: unexpected argument(s): {', '.join(unknown)}")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
        raise NavigationProtocolError("tool schema required must be a list of strings")
    missing = [item for item in required if item not in value]
    if missing:
        raise ValueError(f"{path}: missing required argument(s): {', '.join(missing)}")
    for key, item in value.items():
        prop_schema = properties.get(key)
        if not isinstance(prop_schema, Mapping):
            raise NavigationProtocolError(f"tool schema for {path}.{key} must be a JSON object")
        _validate_value(f"{path}.{key}", item, prop_schema)


def _validated_arguments(name: str, arguments: dict[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        raise TypeError(f"{name}: arguments must be a JSON object")
    if "object" not in _type_names(schema):
        raise NavigationProtocolError(f"tool schema for {name!r} is not an object schema")
    validated = copy.deepcopy(dict(arguments))
    _validate_object(name, validated, schema)
    return validated


def _json_object_or_none(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text or "{}")
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _model_dump(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", by_alias=True)
    return value


def _render_tool_result(result: mcp_types.CallToolResult) -> str:
    parts: list[str] = []
    for item in result.content:
        kind = getattr(item, "type", "")
        if kind == "text":
            parts.append(str(getattr(item, "text", "")))
            continue
        if kind == "resource":
            resource = getattr(item, "resource", None)
            text = getattr(resource, "text", None)
            if isinstance(text, str):
                parts.append(text)
            else:
                parts.append(json.dumps(_model_dump(resource), ensure_ascii=False, sort_keys=True))
            continue
        parts.append(json.dumps(_model_dump(item), ensure_ascii=False, sort_keys=True))
    if parts:
        return "\n".join(parts)
    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False, sort_keys=True)
    return ""


class _NavigationExecutor(RecordedExecution):
    def __init__(self, *, session: _NavigationSession, tool_schemas: Mapping[str, Mapping[str, Any]]) -> None:
        self._session = session
        self._tool_schemas = {name: copy.deepcopy(schema) for name, schema in tool_schemas.items()}
        self.records: list[_RecordedCall] = []
        self.systemic_error: NavigationRuntimeError | None = None

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        return (await self.execute_recorded(name, arguments)).text

    async def execute_recorded(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        schema = self._tool_schemas.get(name)
        copied_arguments = copy.deepcopy(arguments)
        if schema is None:
            self.records.append(
                _RecordedCall(
                    call=NavigationCall(
                        name=name,
                        arguments=copied_arguments,
                        result=_FORBIDDEN_TOOL_RESULT,
                        ok=False,
                    ),
                    error_label=f"{name}:blocked-tool",
                    blocked=True,
                )
            )
            return ToolOutcome(text=_FORBIDDEN_TOOL_RESULT, error=True)
        try:
            validated = _validated_arguments(name, copied_arguments, schema)
        except ValueError as exc:
            result = f"ERROR: invalid arguments ({exc})"
            self.records.append(
                _RecordedCall(
                    call=NavigationCall(name=name, arguments=copied_arguments, result=result, ok=False),
                    error_label=f"{name}:invalid-arguments",
                )
            )
            return ToolOutcome(text=result, error=True)
        try:
            outcome = await self._session.call_tool(name, arguments=copy.deepcopy(validated))
        except asyncio.CancelledError:
            raise
        except BaseExceptionGroup as exc:
            classified = _classify_mcp_exception(exc, phase=f"tools/call {name}")
            if classified is None:
                raise
            self.systemic_error = classified
            raise _SystemicToolFailure("navigation MCP tool execution aborted") from exc
        except _MCP_RUNTIME_ERRORS as exc:
            self.systemic_error = _require_classified_mcp_exception(exc, phase=f"tools/call {name}")
            raise _SystemicToolFailure("navigation MCP tool execution aborted") from exc
        if not isinstance(outcome, mcp_types.CallToolResult):
            self.systemic_error = NavigationProtocolError(
                f"tools/call {name!r} returned {type(outcome).__name__}, expected CallToolResult"
            )
            raise _SystemicToolFailure("navigation MCP tool execution returned an unsupported result type")
        result = _render_tool_result(outcome)
        ok = not outcome.is_error
        self.records.append(
            _RecordedCall(
                call=NavigationCall(name=name, arguments=validated, result=result, ok=ok),
                error_label=None if ok else f"{name}:tool-error",
            )
        )
        return ToolOutcome(text=result, error=not ok)


def _synthetic_call(
    name: str,
    raw_arguments: str,
    event: ToolCallFinished,
) -> tuple[NavigationCall, str | None, bool]:
    arguments = _json_object_or_none(raw_arguments) or {}
    if event.summary == _DISCARDED_TOOL_CALL_SUMMARY:
        return (
            NavigationCall(name=name, arguments=arguments, result=event.summary, ok=False),
            f"{name}:discarded-extra-tool-call",
            True,
        )
    if event.summary == _MALFORMED_ARGUMENTS_RESULT:
        return (
            NavigationCall(name=name, arguments=arguments, result=_MALFORMED_ARGUMENTS_RESULT, ok=False),
            f"{name}:malformed-arguments",
            False,
        )
    if event.summary == "blocked":
        return (
            NavigationCall(name=name, arguments=arguments, result=_TOOL_EXECUTION_BLOCKED_RESULT, ok=False),
            None,
            False,
        )
    return NavigationCall(name=name, arguments=arguments, result=event.summary, ok=event.ok), None, False


def _classify_agent_error(message: str) -> str:
    lowered = message.lower()
    if lowered.startswith("iteration limit reached"):
        return "iteration-limit-reached"
    if lowered.startswith("history budget exceeded"):
        return "history-budget-exceeded"
    if lowered.startswith("request too large"):
        return "request-too-large"
    if lowered.startswith("the turn stopped before its next provider request"):
        return "request-blocked"
    if lowered.startswith("outbound policy blocked the provider request"):
        return "request-blocked"
    return "model-runtime-error"


def _validate_turn_limits(max_iterations: int, timeout_seconds: float) -> None:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int | float):
        raise TypeError("timeout_seconds must be a positive number")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_seconds must be a positive number")


async def run_navigation_turn(
    *,
    provider: LLMProvider,
    session: _NavigationSession,
    candidate: Candidate,
    prompt: str,
    screen_context: str = "",
    max_iterations: int = 4,
    timeout_seconds: float = 60.0,
) -> NavigationTurn:
    _require_string(prompt, "prompt")
    if not isinstance(screen_context, str):
        raise TypeError("screen_context must be a string")
    _validate_turn_limits(max_iterations, timeout_seconds)
    system_prompt, tool_overrides = _validate_navigation_candidate(candidate)

    try:
        async with asyncio.timeout(float(timeout_seconds)):
            tools = _apply_tool_overrides(await _discover_navigation_tools(session), tool_overrides)
            tool_schemas = {
                tool["function"]["name"]: _tool_schema_parameters(tool) for tool in tools
            }
            executor = _NavigationExecutor(session=session, tool_schemas=tool_schemas)
            counted_provider = _IterationProvider(provider)
            runtime = _NavigationAgentRuntime(
                provider=counted_provider,
                executor=executor,
                tools=tools,
                max_iterations=max_iterations,
                max_tool_calls_per_iteration=_MAX_TOOL_CALLS_PER_ITERATION,
                system_prompt=system_prompt,
            )

            answer_parts: list[str] = []
            calls: list[NavigationCall] = []
            errors: list[str] = []
            blocked_tools: list[str] = []
            pending_calls: dict[str, tuple[str, str]] = {}
            input_tokens = 0
            output_tokens = 0

            record_index = 0
            async for event in runtime.run_turn(prompt, screen_context):
                if isinstance(event, TextDelta):
                    answer_parts.append(event.text)
                elif isinstance(event, ToolCallStarted):
                    pending_calls[event.call_id] = (event.name, event.arguments)
                elif isinstance(event, ToolCallFinished):
                    name, raw_arguments = pending_calls.pop(event.call_id, (event.name, "{}"))
                    if record_index < len(executor.records):
                        record = executor.records[record_index]
                        record_index += 1
                        calls.append(record.call)
                        if record.error_label is not None:
                            errors.append(record.error_label)
                        if record.blocked:
                            blocked_tools.append(record.call.name)
                    else:
                        call, error_label, blocked = _synthetic_call(name, raw_arguments, event)
                        calls.append(call)
                        if error_label is not None:
                            errors.append(error_label)
                        if blocked:
                            blocked_tools.append(call.name)
                elif isinstance(event, AgentError):
                    if executor.systemic_error is None:
                        errors.append(_classify_agent_error(event.message))
                elif isinstance(event, TurnComplete):
                    input_tokens = event.input_tokens
                    output_tokens = event.output_tokens

            if executor.systemic_error is not None:
                raise executor.systemic_error

            return NavigationTurn(
                answer="".join(answer_parts),
                calls=tuple(calls),
                errors=tuple(errors),
                blocked_tools=tuple(blocked_tools),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                iterations=counted_provider.iterations,
            )
    except asyncio.CancelledError:
        raise
    except TimeoutError as exc:
        raise NavigationTimeoutError(f"navigation turn timed out after {timeout_seconds:.3f}s") from exc


def _validate_navigation_mcp_url(url: str) -> str:
    value = _require_string(url, "navigation MCP url")
    parts = urlsplit(value)
    host = parts.hostname
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("navigation MCP url must include a valid explicit port") from exc
    if parts.scheme != "http" or host not in {"127.0.0.1", "localhost", "::1"} or port is None:
        raise ValueError(
            "navigation MCP url must be a loopback http URL with an explicit port, for example "
            "http://127.0.0.1:41001/mcp/"
        )
    if parts.username or parts.password:
        raise ValueError("navigation MCP url must not include credentials")
    if parts.path not in {"/mcp", "/mcp/"} or parts.query or parts.fragment:
        raise ValueError("navigation MCP url must point to /mcp or /mcp/ without query or fragment")
    return parts._replace(path="/mcp/").geturl()


async def _enter_mcp_context(stack: AsyncExitStack, context_manager: Any, *, phase: str) -> Any:
    try:
        return await stack.enter_async_context(context_manager)
    except asyncio.CancelledError:
        raise
    except BaseExceptionGroup as exc:
        classified = _classify_mcp_exception(exc, phase=phase)
        if classified is not None:
            raise classified from exc
        raise
    except _MCP_RUNTIME_ERRORS as exc:
        raise _require_classified_mcp_exception(exc, phase=phase) from exc


async def _await_mcp_operation(operation: Awaitable[Any], *, phase: str) -> Any:
    try:
        return await operation
    except asyncio.CancelledError:
        raise
    except BaseExceptionGroup as exc:
        classified = _classify_mcp_exception(exc, phase=phase)
        if classified is not None:
            raise classified from exc
        raise
    except _MCP_RUNTIME_ERRORS as exc:
        raise _require_classified_mcp_exception(exc, phase=phase) from exc


async def _close_mcp_stack(stack: AsyncExitStack) -> None:
    try:
        await stack.aclose()
    except asyncio.CancelledError:
        raise
    except BaseExceptionGroup as exc:
        classified = _classify_mcp_exception(exc, phase="cleanup")
        if classified is not None:
            raise classified from exc
        raise
    except _MCP_RUNTIME_ERRORS as exc:
        raise _require_classified_mcp_exception(exc, phase="cleanup") from exc


@asynccontextmanager
async def connect_navigation_mcp(url: str) -> AsyncIterator[ClientSession]:
    validated_url = _validate_navigation_mcp_url(url)
    stack = AsyncExitStack()
    try:
        http_client = cast(
            httpx2.AsyncClient,
            await _enter_mcp_context(
                stack,
                httpx2.AsyncClient(follow_redirects=False, trust_env=False),
                phase="connect",
            ),
        )
        read_stream, write_stream = cast(
            tuple[Any, Any],
            await _enter_mcp_context(
                stack,
                streamable_http_client(
                    validated_url,
                    http_client=http_client,
                    terminate_on_close=True,
                ),
                phase="connect",
            ),
        )
        session = cast(
            ClientSession,
            await _enter_mcp_context(
                stack,
                ClientSession(read_stream, write_stream),
                phase="connect",
            ),
        )
        await _await_mcp_operation(session.initialize(), phase="initialize")
        yield session
    finally:
        await _close_mcp_stack(stack)
