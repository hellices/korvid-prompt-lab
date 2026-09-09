from __future__ import annotations

import asyncio
import copy
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pytest
from korvid.agent.profiles import build_profile
from korvid.agent.provider import LLMProvider
from korvid.tools.registry import TOOL_DEFS
from mcp import types as mcp_types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.contracts import Candidate
from korvid_prompt_lab.navigation_runtime import (
    NAVIGATION_TOOL_NAMES,
    NavigationMCPError,
    NavigationProtocolError,
    NavigationTimeoutError,
    connect_navigation_mcp,
    get_navigation_tools,
    run_navigation_turn,
)

FULL_NAVIGATION_TOOL_NAMES = tuple(tool["function"]["name"] for tool in get_navigation_tools())


def _candidate(*, components: dict[str, str] | None = None) -> Candidate:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": "navigation-candidate",
        "components": {
            "system": "You are the bounded Korvid navigation runtime.",
            "append": "Only acknowledge tool calls Korvid actually returned.",
        },
    }
    if components:
        payload["components"].update(components)
    return Candidate.from_mapping(payload)


def _tool_schema(name: str) -> dict[str, Any]:
    for definition in TOOL_DEFS:
        if definition.name == name:
            return copy.deepcopy(definition.schema)
    raise AssertionError(f"unknown tool {name!r}")


def _mcp_tool(name: str) -> mcp_types.Tool:
    schema = _tool_schema(name)
    function = schema["function"]
    return mcp_types.Tool(
        name=function["name"],
        description=function.get("description"),
        input_schema=copy.deepcopy(function["parameters"]),
    )


class FakeSession:
    def __init__(
        self,
        *,
        tool_names: tuple[str, ...],
        results: dict[str, Any] | None = None,
        list_tools_error: Exception | None = None,
    ) -> None:
        self._tools = [_mcp_tool(name) for name in tool_names]
        self._results = results or {}
        self._list_tools_error = list_tools_error
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def list_tools(
        self,
        *,
        params: mcp_types.PaginatedRequestParams | None = None,
    ) -> mcp_types.ListToolsResult:
        if self._list_tools_error is not None:
            raise self._list_tools_error
        if params is not None and params.cursor is not None:
            raise AssertionError("tests only exercise a single tools/list page")
        return mcp_types.ListToolsResult(tools=copy.deepcopy(self._tools))

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> mcp_types.CallToolResult:
        snapshot = copy.deepcopy(arguments)
        self.calls.append((name, snapshot))
        outcome = self._results.get(name)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            return mcp_types.CallToolResult(content=[mcp_types.TextContent(text=f"{name} ok")])
        if callable(outcome):
            return outcome(name, snapshot)
        return copy.deepcopy(outcome)


class ScriptedProvider(LLMProvider):
    def __init__(self, scripts: list[list[dict[str, Any]] | BaseException]) -> None:
        self._scripts = scripts
        self._index = 0
        self.seen_messages: list[list[dict[str, Any]]] = []
        self.seen_tools: list[list[dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return "scripted-local"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        del stream
        self.seen_messages.append(copy.deepcopy(messages))
        self.seen_tools.append(copy.deepcopy(tools))

        async def _iterate() -> AsyncIterator[dict[str, Any]]:
            if self._index >= len(self._scripts):
                raise AssertionError("unexpected completion request")
            script = self._scripts[self._index]
            self._index += 1
            if isinstance(script, BaseException):
                raise script
            for event in script:
                yield copy.deepcopy(event)

        return _iterate()


class HangingProvider(LLMProvider):
    @property
    def name(self) -> str:
        return "hanging-local"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        del messages, tools, stream

        async def _iterate() -> AsyncIterator[dict[str, Any]]:
            while True:
                await asyncio.sleep(60)
                if False:  # pragma: no cover - keeps this an async generator
                    yield {}

        return _iterate()


def test_get_navigation_tools_uses_installed_mcp_registry_surface() -> None:
    small = build_profile("small", readonly=True, resize_supported=False)
    small_names = {tool["function"]["name"] for tool in small.tools}

    tools = get_navigation_tools()
    tool_names = [tool["function"]["name"] for tool in tools]

    assert "navigate" not in small_names
    assert tool_names == [
        "list_resources",
        "helm_list_releases",
        "navigate",
        "set_filter",
        "open_logs",
        "open_describe",
        "drill_down",
    ]
    assert set(tool_names) == NAVIGATION_TOOL_NAMES
    assert "get_logs" not in tool_names
    assert "get_resource" not in tool_names
    assert "diagnose_pod" not in tool_names
    assert "propose_write" not in tool_names
    assert "delete_resource" not in tool_names
    assert all(tool["type"] == "function" for tool in tools)


def test_run_navigation_turn_discovers_only_navigation_tools_and_applies_overrides() -> None:
    session = FakeSession(
        tool_names=(
            "list_resources",
            "helm_list_releases",
            "navigate",
            "set_filter",
            "open_logs",
            "open_describe",
            "drill_down",
            "get_logs",
            "get_resource",
            "diagnose_pod",
            "propose_write",
            "delete_resource",
        )
    )
    provider = ScriptedProvider(
        [[{"type": "text_delta", "text": "ready"}, {"type": "usage", "input_tokens": 2, "output_tokens": 1}]]
    )
    candidate = _candidate(
        components={"tool.navigate": "Switch to the table the user asked for before answering."}
    )

    turn = asyncio.run(
        run_navigation_turn(
            provider=provider,
            session=session,
            candidate=candidate,
            prompt="show me the relevant screen",
        )
    )

    seen_names = [tool["function"]["name"] for tool in provider.seen_tools[0]]
    navigate_tool = next(tool for tool in provider.seen_tools[0] if tool["function"]["name"] == "navigate")

    assert seen_names == [
        "list_resources",
        "helm_list_releases",
        "navigate",
        "set_filter",
        "open_logs",
        "open_describe",
        "drill_down",
    ]
    assert navigate_tool["function"]["description"] == (
        "Switch to the table the user asked for before answering."
    )
    assert turn.answer == "ready"
    assert turn.calls == ()
    assert turn.errors == ()
    assert turn.blocked_tools == ()


def test_run_navigation_turn_requires_full_navigation_surface_before_provider_contact() -> None:
    session = FakeSession(
        tool_names=("list_resources", "helm_list_releases", "navigate", "set_filter", "open_logs", "open_describe")
    )
    provider = ScriptedProvider([[{"type": "text_delta", "text": "unused"}]])

    with pytest.raises(NavigationProtocolError, match="drill_down"):
        asyncio.run(
            run_navigation_turn(
                provider=provider,
                session=session,
                candidate=_candidate(),
                prompt="show me the right screen",
            )
        )

    assert provider.seen_tools == []


def test_run_navigation_turn_records_success_calls_and_namespace_arguments() -> None:
    session = FakeSession(
        tool_names=FULL_NAVIGATION_TOOL_NAMES,
        results={
            "list_resources": mcp_types.CallToolResult(
                content=[mcp_types.TextContent(text="shop/worker-7 ready=0/1 restarts=3")]
            )
        },
    )
    provider = ScriptedProvider(
        [
            [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "list_resources",
                    "arguments": '{"kind":"pods","namespace":"shop"}',
                },
                {"type": "usage", "input_tokens": 5, "output_tokens": 2},
            ],
            [{"type": "text_delta", "text": "I opened the pods view."}, {"type": "usage", "input_tokens": 3, "output_tokens": 4}],
        ]
    )

    turn = asyncio.run(
        run_navigation_turn(
            provider=provider,
            session=session,
            candidate=_candidate(),
            prompt="find the failing pod",
        )
    )

    assert session.calls == [("list_resources", {"kind": "pods", "namespace": "shop"})]
    assert turn.answer == "I opened the pods view."
    assert turn.calls[0].name == "list_resources"
    assert turn.calls[0].arguments == {"kind": "pods", "namespace": "shop"}
    assert turn.calls[0].result == "shop/worker-7 ready=0/1 restarts=3"
    assert turn.calls[0].ok is True
    assert turn.errors == ()
    assert turn.blocked_tools == ()
    assert turn.input_tokens == 8
    assert turn.output_tokens == 6


def test_run_navigation_turn_handles_malformed_json_without_forwarding() -> None:
    session = FakeSession(tool_names=FULL_NAVIGATION_TOOL_NAMES)
    provider = ScriptedProvider(
        [
            [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "navigate",
                    "arguments": '{"view":"pods"',
                },
                {"type": "usage", "input_tokens": 4, "output_tokens": 1},
            ],
            [{"type": "text_delta", "text": "Need a valid tool call."}, {"type": "usage", "input_tokens": 2, "output_tokens": 3}],
        ]
    )

    turn = asyncio.run(
        run_navigation_turn(
            provider=provider,
            session=session,
            candidate=_candidate(),
            prompt="navigate to pods",
        )
    )

    assert session.calls == []
    assert turn.answer == "Need a valid tool call."
    assert len(turn.calls) == 1
    assert turn.calls[0].name == "navigate"
    assert turn.calls[0].arguments == {}
    assert turn.calls[0].result == "ERROR: bad arguments"
    assert turn.calls[0].ok is False
    assert turn.errors == ("navigate:malformed-arguments",)
    assert turn.blocked_tools == ()


def test_run_navigation_turn_blocks_forbidden_tools_without_forwarding() -> None:
    session = FakeSession(tool_names=(*FULL_NAVIGATION_TOOL_NAMES, "get_logs"))
    provider = ScriptedProvider(
        [
            [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "get_logs",
                    "arguments": '{"pod":"worker-7","namespace":"shop"}',
                },
                {"type": "usage", "input_tokens": 3, "output_tokens": 1},
            ],
            [{"type": "text_delta", "text": "That tool is unavailable here."}, {"type": "usage", "input_tokens": 2, "output_tokens": 4}],
        ]
    )

    turn = asyncio.run(
        run_navigation_turn(
            provider=provider,
            session=session,
            candidate=_candidate(),
            prompt="show logs",
        )
    )

    assert session.calls == []
    assert turn.calls[0].name == "get_logs"
    assert turn.calls[0].arguments == {"pod": "worker-7", "namespace": "shop"}
    assert turn.calls[0].result == "ERROR: tool not allowed"
    assert turn.calls[0].ok is False
    assert turn.errors == ("get_logs:blocked-tool",)
    assert turn.blocked_tools == ("get_logs",)


def test_run_navigation_turn_preserves_mcp_is_error() -> None:
    session = FakeSession(
        tool_names=FULL_NAVIGATION_TOOL_NAMES,
        results={
            "navigate": mcp_types.CallToolResult(
                content=[mcp_types.TextContent(text="unknown view alias")],
                is_error=True,
            )
        },
    )
    provider = ScriptedProvider(
        [
            [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "navigate",
                    "arguments": '{"view":"not-a-real-view"}',
                },
                {"type": "usage", "input_tokens": 3, "output_tokens": 1},
            ],
            [{"type": "text_delta", "text": "Korvid rejected that navigation request."}, {"type": "usage", "input_tokens": 2, "output_tokens": 5}],
        ]
    )

    turn = asyncio.run(
        run_navigation_turn(
            provider=provider,
            session=session,
            candidate=_candidate(),
            prompt="open the made-up view",
        )
    )

    assert turn.calls[0].ok is False
    assert turn.calls[0].result == "unknown view alias"
    assert turn.errors == ("navigate:tool-error",)
    assert turn.blocked_tools == ()


def test_run_navigation_turn_reports_iteration_limit() -> None:
    session = FakeSession(tool_names=FULL_NAVIGATION_TOOL_NAMES)
    provider = ScriptedProvider(
        [
            [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "navigate",
                    "arguments": '{"view":"pods"}',
                },
                {"type": "usage", "input_tokens": 3, "output_tokens": 1},
            ],
            [
                {
                    "type": "tool_call",
                    "id": "call-2",
                    "name": "navigate",
                    "arguments": '{"view":"pods"}',
                },
                {"type": "usage", "input_tokens": 3, "output_tokens": 1},
            ],
        ]
    )

    turn = asyncio.run(
        run_navigation_turn(
            provider=provider,
            session=session,
            candidate=_candidate(),
            prompt="keep navigating forever",
            max_iterations=2,
        )
    )

    assert turn.answer == ""
    assert [call.name for call in turn.calls] == ["navigate", "navigate"]
    assert turn.errors == ("iteration-limit-reached",)
    assert turn.blocked_tools == ()


def test_run_navigation_turn_rejects_blank_candidate() -> None:
    session = FakeSession(tool_names=("navigate",))
    provider = ScriptedProvider([[{"type": "text_delta", "text": "unused"}]])
    candidate = Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "missing-system",
            "components": {"append": "Only this append exists."},
        }
    )

    with pytest.raises(ValueError, match="system"):
        asyncio.run(
            run_navigation_turn(
                provider=provider,
                session=session,
                candidate=candidate,
                prompt="open pods",
            )
        )


def test_run_navigation_turn_rejects_unsupported_candidate_tool_override() -> None:
    session = FakeSession(tool_names=("navigate",))
    provider = ScriptedProvider([[{"type": "text_delta", "text": "unused"}]])
    candidate = _candidate(components={"tool.get_logs": "Forbidden in navigation mode."})

    with pytest.raises(ValueError, match="tool.get_logs"):
        asyncio.run(
            run_navigation_turn(
                provider=provider,
                session=session,
                candidate=candidate,
                prompt="open pods",
            )
        )


def test_run_navigation_turn_rejects_incompatible_server_schema() -> None:
    bad_navigate = _mcp_tool("navigate")
    bad_navigate.input_schema = {"type": "object", "properties": {"namespace": {"type": "string"}}, "required": []}
    session = FakeSession(tool_names=FULL_NAVIGATION_TOOL_NAMES)
    session._tools = [bad_navigate if tool.name == "navigate" else tool for tool in session._tools]
    provider = ScriptedProvider([[{"type": "text_delta", "text": "unused"}]])

    with pytest.raises(NavigationProtocolError, match="navigate"):
        asyncio.run(
            run_navigation_turn(
                provider=provider,
                session=session,
                candidate=_candidate(),
                prompt="open pods",
            )
        )


def test_run_navigation_turn_propagates_programming_error_from_tools_list() -> None:
    provider = ScriptedProvider([[{"type": "text_delta", "text": "unused"}]])
    session = FakeSession(
        tool_names=(),
        list_tools_error=ValueError("unexpected tools/list bug"),
    )

    with pytest.raises(ValueError, match="unexpected tools/list bug"):
        asyncio.run(
            run_navigation_turn(
                provider=provider,
                session=session,
                candidate=_candidate(),
                prompt="open pods",
            )
        )

    assert provider.seen_tools == []


def test_run_navigation_turn_times_out() -> None:
    session = FakeSession(tool_names=FULL_NAVIGATION_TOOL_NAMES)

    with pytest.raises(NavigationTimeoutError):
        asyncio.run(
            run_navigation_turn(
                provider=HangingProvider(),
                session=session,
                candidate=_candidate(),
                prompt="open pods",
                timeout_seconds=0.05,
            )
        )


def test_run_navigation_turn_raises_disconnect_error() -> None:
    session = FakeSession(
        tool_names=FULL_NAVIGATION_TOOL_NAMES,
        results={"navigate": ConnectionError("socket closed")},
    )
    provider = ScriptedProvider(
        [
            [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "navigate",
                    "arguments": '{"view":"pods"}',
                },
                {"type": "usage", "input_tokens": 2, "output_tokens": 1},
            ]
        ]
    )

    with pytest.raises(NavigationMCPError, match="tools/call"):
        asyncio.run(
            run_navigation_turn(
                provider=provider,
                session=session,
                candidate=_candidate(),
                prompt="open pods",
            )
        )


def test_run_navigation_turn_propagates_programming_error_from_tool_call() -> None:
    session = FakeSession(
        tool_names=FULL_NAVIGATION_TOOL_NAMES,
        results={"navigate": ValueError("unexpected tools/call bug")},
    )
    provider = ScriptedProvider(
        [
            [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "navigate",
                    "arguments": '{"view":"pods"}',
                },
                {"type": "usage", "input_tokens": 2, "output_tokens": 1},
            ]
        ]
    )

    with pytest.raises(ValueError, match="unexpected tools/call bug"):
        asyncio.run(
            run_navigation_turn(
                provider=provider,
                session=session,
                candidate=_candidate(),
                prompt="open pods",
            )
        )


def test_connect_navigation_mcp_initializes_and_closes_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class DummyAsyncClient:
        def __init__(self, *, follow_redirects: bool = True, trust_env: bool = True) -> None:
            assert follow_redirects is False
            assert trust_env is False
            events.append("http-client:init")

        async def __aenter__(self) -> Self:
            events.append("http-client:enter")
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            del exc_type, exc, tb
            events.append("http-client:exit")

    @asynccontextmanager
    async def fake_streamable_http_client(
        url: str,
        *,
        http_client: Any = None,
        terminate_on_close: bool = True,
    ) -> AsyncIterator[tuple[str, str]]:
        assert url == "http://127.0.0.1:41001/mcp/"
        assert http_client is not None
        assert terminate_on_close is True
        events.append("transport:enter")
        try:
            yield ("reader", "writer")
        finally:
            events.append("transport:exit")

    class DummySession:
        def __init__(self, read_stream: str, write_stream: str) -> None:
            assert read_stream == "reader"
            assert write_stream == "writer"
            events.append("session:init")

        async def __aenter__(self) -> Self:
            events.append("session:enter")
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            del exc_type, exc, tb
            events.append("session:exit")

        async def initialize(self) -> mcp_types.InitializeResult:
            events.append("session:initialize")
            return mcp_types.InitializeResult(
                protocol_version="2026-03-26",
                capabilities=mcp_types.ServerCapabilities(),
                server_info=mcp_types.Implementation(name="fake-korvid", version="0.0.0"),
            )

    monkeypatch.setattr("korvid_prompt_lab.navigation_runtime.httpx2.AsyncClient", DummyAsyncClient)
    monkeypatch.setattr(
        "korvid_prompt_lab.navigation_runtime.streamable_http_client",
        fake_streamable_http_client,
    )
    monkeypatch.setattr("korvid_prompt_lab.navigation_runtime.ClientSession", DummySession)

    async def exercise() -> None:
        async with connect_navigation_mcp("http://127.0.0.1:41001/mcp") as session:
            assert isinstance(session, DummySession)

    asyncio.run(exercise())

    assert events == [
        "http-client:init",
        "http-client:enter",
        "transport:enter",
        "session:init",
        "session:enter",
        "session:initialize",
        "session:exit",
        "transport:exit",
        "http-client:exit",
    ]


def test_connect_navigation_mcp_wraps_initialize_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyAsyncClient:
        def __init__(self, *, follow_redirects: bool = True, trust_env: bool = True) -> None:
            assert follow_redirects is False
            assert trust_env is False

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            del exc_type, exc, tb

    @asynccontextmanager
    async def fake_streamable_http_client(
        url: str,
        *,
        http_client: Any = None,
        terminate_on_close: bool = True,
    ) -> AsyncIterator[tuple[str, str]]:
        assert url == "http://127.0.0.1:41001/mcp/"
        assert http_client is not None
        assert terminate_on_close is True
        yield ("reader", "writer")

    class DummySession:
        def __init__(self, read_stream: str, write_stream: str) -> None:
            assert read_stream == "reader"
            assert write_stream == "writer"

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            del exc_type, exc, tb

        async def initialize(self) -> mcp_types.InitializeResult:
            raise OSError("connection reset during initialize")

    monkeypatch.setattr("korvid_prompt_lab.navigation_runtime.httpx2.AsyncClient", DummyAsyncClient)
    monkeypatch.setattr(
        "korvid_prompt_lab.navigation_runtime.streamable_http_client",
        fake_streamable_http_client,
    )
    monkeypatch.setattr("korvid_prompt_lab.navigation_runtime.ClientSession", DummySession)

    async def exercise() -> None:
        async with connect_navigation_mcp("http://127.0.0.1:41001/mcp"):
            raise AssertionError("unreachable")

    with pytest.raises(NavigationMCPError, match="initialize"):
        asyncio.run(exercise())


def test_connect_navigation_mcp_accepts_trailing_slash_input(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_urls: list[str] = []

    class DummyAsyncClient:
        def __init__(self, *, follow_redirects: bool = True, trust_env: bool = True) -> None:
            assert follow_redirects is False
            assert trust_env is False

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            del exc_type, exc, tb

    @asynccontextmanager
    async def fake_streamable_http_client(
        url: str,
        *,
        http_client: Any = None,
        terminate_on_close: bool = True,
    ) -> AsyncIterator[tuple[str, str]]:
        assert http_client is not None
        assert terminate_on_close is True
        seen_urls.append(url)
        yield ("reader", "writer")

    class DummySession:
        def __init__(self, read_stream: str, write_stream: str) -> None:
            assert read_stream == "reader"
            assert write_stream == "writer"

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            del exc_type, exc, tb

        async def initialize(self) -> mcp_types.InitializeResult:
            return mcp_types.InitializeResult(
                protocol_version="2026-03-26",
                capabilities=mcp_types.ServerCapabilities(),
                server_info=mcp_types.Implementation(name="fake-korvid", version="0.0.0"),
            )

    monkeypatch.setattr("korvid_prompt_lab.navigation_runtime.httpx2.AsyncClient", DummyAsyncClient)
    monkeypatch.setattr(
        "korvid_prompt_lab.navigation_runtime.streamable_http_client",
        fake_streamable_http_client,
    )
    monkeypatch.setattr("korvid_prompt_lab.navigation_runtime.ClientSession", DummySession)

    async def exercise() -> None:
        async with connect_navigation_mcp("http://127.0.0.1:41001/mcp/"):
            return

    asyncio.run(exercise())

    assert seen_urls == ["http://127.0.0.1:41001/mcp/"]


def test_connect_navigation_mcp_rejects_non_loopback_urls() -> None:
    async def exercise() -> None:
        async with connect_navigation_mcp("https://example.com/mcp"):
            raise AssertionError("unreachable")

    with pytest.raises(ValueError, match="loopback http"):
        asyncio.run(exercise())
