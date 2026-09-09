"""Convenience commands and split policy for the navigation-only task."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .config import load_campaign, load_candidate
from .contracts import Campaign, KorvidNativeServing, KorvidNavigationServing
from .navigation_cases import find_navigation_case, require_complete_navigation_pack


def add_navigation_commands(subparsers: argparse._SubParsersAction) -> None:
    initialize = subparsers.add_parser(
        "navigation-init", help="Create a compact navigation prompt and disjoint 24-case MCP evaluation pack.",
    )
    initialize.add_argument("--directory", type=Path, required=True)
    initialize.add_argument("--model", default="qwen3:0.6b")
    initialize.add_argument("--repetitions", type=int, default=3)
    initialize.set_defaults(func=command_navigation_init)
    assist = subparsers.add_parser(
        "navigation-assist", help="Perform one UI-assistance request on a running Korvid MCP server.",
    )
    assist.add_argument("--candidate", type=Path, required=True)
    assist.add_argument("--campaign", type=Path, required=True)
    assist.add_argument("--prompt", required=True)
    assist.add_argument(
        "--mcp-url", default=None,
        help="Explicit loopback Korvid MCP URL, or set KORVID_NAVIGATION_MCP_URL. No automatic instance selection.",
    )
    assist.set_defaults(func=command_navigation_assist)


def command_navigation_init(args: argparse.Namespace) -> int:
    from .navigation import initialize_navigation

    try:
        candidate, campaign = initialize_navigation(
            args.directory, model=args.model, repetitions=args.repetitions,
        )
    except (OSError, ValueError) as exc:
        print(f"navigation initialization failed: {exc}", file=sys.stderr)
        return 2
    print(f"candidate={candidate}\ncampaign={campaign}")
    return 0


def navigation_selections(args: argparse.Namespace, campaign: Campaign) -> None:
    if isinstance(campaign.serving, KorvidNativeServing):
        return
    split = getattr(args, "navigation_split", None)
    if not isinstance(campaign.serving, KorvidNavigationServing):
        if split is not None:
            raise ValueError("--navigation-split requires a korvid_navigation campaign")
        return
    require_complete_navigation_pack([case.case_id for case in campaign.cases])
    groups = {
        group: [case.case_id for case in campaign.cases if find_navigation_case(case.case_id).split == group]
        for group in ("train", "validation", "holdout")
    }
    for group in ("train", "validation"):
        attribute = f"{group}_case_ids"
        selected = getattr(args, attribute)
        if not selected:
            setattr(args, attribute, groups[group])
        elif not set(selected).issubset(groups[group]):
            raise ValueError(f"navigation {group} cases must come from the authored {group} split")
    if args.command == "evaluate":
        if args.case_ids and split is not None:
            raise ValueError("--case-id and --navigation-split cannot be combined")
        if not args.case_ids:
            args.case_ids = groups[split or "validation"]
        elif any(find_navigation_case(case_id).split == "holdout" for case_id in args.case_ids):
            raise ValueError("use --navigation-split holdout to explicitly evaluate the held-out pack")
        if args.milestone_case_ids:
            raise ValueError("navigation evaluation is not write-operation publication evidence")


async def _assist(args: argparse.Namespace):
    from .navigation import local_model_provider
    from .navigation_runtime import connect_navigation_mcp, run_navigation_turn

    candidate = load_candidate(args.candidate)
    campaign = load_campaign(args.campaign)
    serving = campaign.serving
    if not isinstance(serving, KorvidNavigationServing) or len(campaign.models) != 1:
        raise ValueError("navigation-assist requires a single-model korvid_navigation campaign")
    endpoint = args.mcp_url or os.environ.get("KORVID_NAVIGATION_MCP_URL", "")
    if not endpoint:
        raise ValueError("provide --mcp-url or KORVID_NAVIGATION_MCP_URL for the running Korvid instance")
    async with (
        local_model_provider(serving, campaign.models[0]) as provider,
        connect_navigation_mcp(endpoint) as session,
    ):
        return await run_navigation_turn(
            provider=provider, session=session, candidate=candidate, prompt=args.prompt,
            max_iterations=serving.max_iterations, timeout_seconds=serving.timeout_seconds,
        )


def command_navigation_assist(args: argparse.Namespace) -> int:
    from .navigation_runtime import NavigationRuntimeError

    try:
        turn = asyncio.run(_assist(args))
    except (OSError, ValueError, NavigationRuntimeError) as exc:
        print(f"navigation assist failed: {exc}", file=sys.stderr)
        return 2
    print(turn.answer)
    print(json.dumps({
        "acknowledged_actions": [call.name for call in turn.calls if call.ok],
        "failed_actions": [call.name for call in turn.calls if not call.ok],
        "errors": list(turn.errors),
        "blocked_tools": list(turn.blocked_tools),
        "screen_state_verified": False,
    }, ensure_ascii=False))
    return 0 if turn.calls and not (turn.errors or turn.blocked_tools) and all(
        call.ok for call in turn.calls
    ) else 1
