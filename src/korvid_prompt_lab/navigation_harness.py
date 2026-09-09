"""Real Korvid screens and MCP transport, backed only by synthetic cluster data."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from korvid.core.config import KorvidConfig
from korvid.core.store import ResourceStore, Summary
from korvid.core.watch import WatchManager
from korvid.evals.fake_kube import FakeKubeClient, builtin_aliases
from korvid.evals.scenario import ContainerLogs, Scenario
from korvid.k8s.helm import (
    HELM_RELEASES_META,
    HELM_REVISIONS_META,
    revision_from_secret,
)
from korvid.k8s.logs import LogLine
from korvid.k8s.models import PodSummary
from korvid.mcp.server import KorvidMCPServer
from korvid.tools.executor import ToolExecutor
from korvid.tools.registry import mcp_tool_schemas
from korvid.ui.app import AppUIBridge, KorvidApp
from korvid.ui.widgets.describe_screen import DescribeScreen
from korvid.ui.widgets.log_pane import LogPane
from mcp import ClientSession
from textual.pilot import Pilot

from .navigation_cases import NavigationCase, fixture_objects
from .navigation_runtime import connect_navigation_mcp


@dataclass
class NavigationWorkspace:
    case: NavigationCase
    app: KorvidApp
    pilot: Pilot[None]
    session: ClientSession

    async def settle(self) -> None:
        await self.pilot.pause()

    def observe(self) -> dict[str, str]:
        # Korvid 0.3 has no public screen snapshot API. Keep version-specific
        # observation here; never infer screen success from an MCP response.
        logs = self.app.query_one(LogPane)
        screen = self.app.screen
        return {
            "kind": self.app.current_kind,
            "scope": self.app.current_scope,
            "filter": self.app.filter_pattern,
            "drill_parent": self.app._drill.parent_uid or "",
            "describe": screen._title if isinstance(screen, DescribeScreen) else "",
            "logs": ",".join("/".join(row) for row in self.app._current_log_triples)
            if logs.display else "",
        }

    def missing_postconditions(self) -> tuple[str, ...]:
        observed = self.observe()
        return tuple(
            f"{key}: expected {value!r}, observed {observed[key]!r}"
            for key, value in self.case.expected
            if observed[key] != value
        )

    def screen_context(self) -> str:
        return self.app._screen_context()


@asynccontextmanager
async def navigation_workspace(case: NavigationCase) -> AsyncIterator[NavigationWorkspace]:
    objects = fixture_objects(case)
    fixture = Scenario(
        id=case.case_id, question=case.prompt, screen="", root_cause="navigation",
        must_mention=(), objects=objects,
        logs={f"{case.namespace}/{case.pod}/main": ContainerLogs(current=("fixture ready",))},
    )
    kube = FakeKubeClient(fixture)
    aliases = builtin_aliases()
    for meta in (HELM_RELEASES_META, HELM_REVISIONS_META):
        for alias in (meta.plural, meta.kind.lower(), *meta.shortnames):
            aliases[alias] = meta

    async def source(kind: str, scope: str) -> AsyncIterator[tuple[str, Summary]]:
        namespace = None if scope == "*" else scope
        rows: Sequence[Summary]
        if kind == "pods":
            rows = [
                PodSummary.from_manifest(obj) for obj in objects
                if obj["kind"] == "Pod"
                and (namespace is None or obj["metadata"]["namespace"] == namespace)
            ]
        elif kind == "helmreleases":
            rows = await kube.list_helm_releases(namespace)
        elif kind == "helmrevisions":
            rows = [
                revision_from_secret(obj) for obj in objects
                if obj.get("type") == "helm.sh/release.v1"
                and (namespace is None or obj["metadata"]["namespace"] == namespace)
            ]
        else:
            rows = await kube.list_objects(aliases[kind], namespace)
        for row in rows:
            yield "ADDED", row
        await asyncio.Event().wait()

    async def get_manifest(kind: str, namespace: str | None, name: str) -> dict[str, Any]:
        return await kube.get_object(aliases[kind], namespace, name)

    async def stream_logs(
        namespace: str, pod: str, container: str, **kwargs: Any,
    ) -> AsyncIterator[LogLine]:
        async for line in kube.stream_logs(namespace, pod, container, **kwargs):
            yield line
        await asyncio.Event().wait()

    store = ResourceStore()
    watches = WatchManager(store, source)
    app = KorvidApp(
        config=KorvidConfig(namespace=case.initial_scope, readonly=True),
        store=store, watch_manager=watches, aliases=aliases,
        get_manifest=get_manifest, stream_logs=stream_logs,
        list_namespaces=_fixture_namespaces,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        bridge = AppUIBridge(app)
        result = await bridge.agent_navigate(case.initial_kind, case.initial_scope)
        if result.startswith("ERROR:"):
            raise RuntimeError(f"navigation fixture setup failed: {result}")
        if case.initial_filter:
            result = await bridge.agent_set_filter(case.initial_filter)
            if result.startswith("ERROR:"):
                raise RuntimeError(f"navigation fixture setup failed: {result}")
        await pilot.pause()
        server = KorvidMCPServer(
            ToolExecutor(kube, aliases, ui=bridge), mcp_tool_schemas(), port=0,
        )
        task = asyncio.create_task(server.run())
        try:
            port = await asyncio.wait_for(server.wait_started(), 10)
            async with connect_navigation_mcp(f"http://127.0.0.1:{port}/mcp") as session:
                yield NavigationWorkspace(case, app, pilot, session)
        finally:
            server.request_shutdown()
            try:
                await asyncio.wait_for(task, 10)
            finally:
                await watches.stop_all()


async def _fixture_namespaces() -> list[str]:
    return ["default", "shop", "monitoring", "sandbox", "other"]
