from __future__ import annotations

import asyncio

import pytest

from korvid_prompt_lab.navigation_cases import navigation_cases
from korvid_prompt_lab.navigation_harness import navigation_workspace


def test_navigation_pack_has_balanced_disjoint_splits() -> None:
    cases = navigation_cases()
    assert len(cases) == 24
    assert len({case.case_id for case in cases}) == len(cases)
    assert {case.split for case in cases} == {"train", "validation", "holdout"}
    for split in ("train", "validation", "holdout"):
        assert len([case for case in cases if case.split == split]) == 8
    assert len({case.prompt for case in cases}) == len(cases)


def test_navigation_identity_includes_cluster_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    case = navigation_cases()[0]
    original = case.fingerprint
    monkeypatch.setattr("korvid_prompt_lab.navigation_cases.fixture_objects", lambda case: ())
    assert case.fingerprint != original


def test_real_mcp_navigation_changes_actual_korvid_view() -> None:
    async def check() -> None:
        case = next(case for case in navigation_cases() if case.case_id == "train-helm")
        async with navigation_workspace(case) as workspace:
            assert workspace.observe()["kind"] == "pods"
            before = workspace.missing_postconditions()
            assert before
            result = await workspace.session.call_tool(
                "navigate", {"view": "helm", "namespace": case.namespace}
            )
            assert not result.is_error
            await workspace.settle()
            assert workspace.observe()["kind"] == "helmreleases"
            assert workspace.missing_postconditions() == ()

    asyncio.run(check())


def test_failed_navigation_does_not_change_or_satisfy_state() -> None:
    async def check() -> None:
        case = next(case for case in navigation_cases() if case.case_id == "train-helm")
        async with navigation_workspace(case) as workspace:
            initial = workspace.observe()
            result = await workspace.session.call_tool("navigate", {"view": "imaginary"})
            assert result.is_error
            assert workspace.observe() == initial
            assert workspace.missing_postconditions()

    asyncio.run(check())


def test_real_helm_drill_tracks_parent_and_visible_revisions() -> None:
    async def check() -> None:
        case = next(case for case in navigation_cases() if case.case_id == "train-history")
        async with navigation_workspace(case) as workspace:
            await workspace.session.call_tool(
                "navigate", {"view": "helm", "namespace": case.namespace}
            )
            await workspace.settle()
            result = await workspace.session.call_tool("drill_down", {"name": case.release})
            assert not result.is_error
            await workspace.settle()
            assert workspace.missing_postconditions() == ()
            assert workspace.observe()["kind"] == "helmrevisions"

    asyncio.run(check())


def test_real_log_and_describe_panes_observe_target_not_model_claim() -> None:
    async def check(task: str) -> None:
        case = next(case for case in navigation_cases() if case.case_id == f"train-{task}")
        async with navigation_workspace(case) as workspace:
            assert workspace.missing_postconditions()
            name = "open_logs" if task == "logs" else "open_describe"
            args = (
                {"pod": case.pod, "namespace": case.namespace}
                if task == "logs"
                else {"kind": "pods", "name": case.pod, "namespace": case.namespace}
            )
            result = await workspace.session.call_tool(name, args)
            assert not result.is_error
            await workspace.settle()
            assert workspace.missing_postconditions() == ()

    asyncio.run(check("logs"))
    asyncio.run(check("describe"))


@pytest.mark.parametrize("case", navigation_cases(), ids=lambda case: case.case_id)
def test_all_authored_targets_are_reachable_through_real_mcp(case) -> None:
    async def check() -> None:
        async with navigation_workspace(case) as workspace:
            task = case.case_id.split("-", 1)[1]
            if task in {"pods", "all-pods", "helm", "history"}:
                await workspace.session.call_tool("navigate", {
                    "view": "helm" if task in {"helm", "history"} else "pods",
                    "namespace": "all" if task == "all-pods" else case.namespace,
                })
                await workspace.settle()
            if task in {"filter", "clear"}:
                await workspace.session.call_tool("set_filter", {
                    "pattern": case.pod if task == "filter" else "",
                })
            if task == "logs":
                await workspace.session.call_tool("open_logs", {"pod": case.pod, "namespace": case.namespace})
            if task == "describe":
                await workspace.session.call_tool("open_describe", {
                    "kind": "pods", "name": case.pod, "namespace": case.namespace,
                })
            if task == "history":
                await workspace.session.call_tool("drill_down", {"name": case.release})
            await workspace.settle()
            assert workspace.missing_postconditions() == ()

    asyncio.run(check())
