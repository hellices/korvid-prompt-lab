"""Native rules setup, split selection, and verified Korvid config export."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .artifacts import write_json_artifact
from .config import load_campaign, load_candidate
from .contracts import Campaign, Candidate, KorvidNativeServing
from .native import initialize_native, native_model, validate_native_prompt_identity
from .native_contract import (
    NATIVE_KORVID_REVISION,
    NATIVE_KORVID_VERSION,
    find_native_case,
    require_native_pack,
    rules_from_candidate,
)
from .native_source import run_native_request
from .runner import BridgeSystemError


def add_native_commands(subparsers: argparse._SubParsersAction) -> None:
    initialize = subparsers.add_parser(
        "native-init", help="Create a Korvid v0.4.1 native low-tier rules campaign (recommended).",
    )
    initialize.add_argument("--directory", type=Path, required=True)
    initialize.add_argument("--model", default="ollama/qwen3:0.6b")
    initialize.add_argument("--repetitions", type=int, default=3)
    initialize.set_defaults(func=command_native_init)
    check = subparsers.add_parser(
        "native-check", help="Verify pinned native source, production rules loading and resolved low policy.",
    )
    check.add_argument("--campaign", type=Path, required=True)
    check.add_argument("--candidate", type=Path, required=True)
    check.set_defaults(func=command_native_check)
    export = subparsers.add_parser(
        "native-export", help="Export additive agent.rules and verify them with Korvid's own config loader.",
    )
    export.add_argument("--campaign", type=Path, required=True)
    export.add_argument("--candidate", type=Path, required=True)
    export.add_argument("--directory", type=Path, required=True)
    export.set_defaults(func=command_native_export)


def command_native_init(args: argparse.Namespace) -> int:
    try:
        candidate, campaign = initialize_native(args.directory, model=args.model, repetitions=args.repetitions)
    except (OSError, ValueError) as exc:
        print(f"native initialization failed: {exc}", file=sys.stderr)
        return 2
    print(f"candidate={candidate}\ncampaign={campaign}")
    return 0


def native_selections(args: argparse.Namespace, campaign: Campaign) -> None:
    if not isinstance(campaign.serving, KorvidNativeServing):
        return
    require_native_pack([case.case_id for case in campaign.cases])
    groups = {
        split: [case.case_id for case in campaign.cases if find_native_case(case.case_id).split == split]
        for split in ("train", "validation", "holdout")
    }
    for split in ("train", "validation"):
        key = f"{split}_case_ids"
        if not getattr(args, key):
            setattr(args, key, groups[split])
        elif not set(getattr(args, key)).issubset(groups[split]):
            raise ValueError(f"native {split} IDs must be from the authored {split} split")
    if args.command == "evaluate":
        evaluation_split = getattr(args, "navigation_split", None)
        if args.case_ids and evaluation_split is not None:
            raise ValueError("--case-id and --navigation-split cannot be combined")
        if not args.case_ids:
            args.case_ids = groups[evaluation_split or "validation"]
        elif any(find_native_case(case_id).split == "holdout" for case_id in args.case_ids):
            raise ValueError("native holdout requires --navigation-split holdout")
        if args.milestone_case_ids:
            raise ValueError("native UI rules do not qualify write-operation publication")


def verify_rules_config(
    candidate: Candidate, campaign: Campaign, *, config_path: Path | None = None,
) -> dict[str, Any]:
    serving = campaign.serving
    if not isinstance(serving, KorvidNativeServing) or len(campaign.models) != 1:
        raise ValueError("native rules verification requires a single-model korvid_native campaign")
    rules = rules_from_candidate(candidate)
    request = {
        "protocol_version": 1, "operation": "verify-config", "rules": rules,
        "case_id": "train-pods",
        "model": native_model(campaign.models[0], serving, 0), "script": None,
        **({"config_path": str(config_path.resolve())} if config_path is not None else {}),
    }
    result = run_native_request(serving, request)
    if (
        result.get("protocol_version") != 1 or result.get("rules") != rules
        or result.get("korvid_version") != NATIVE_KORVID_VERSION
        or result.get("rules_applied") is not True or result.get("ui_follow") is not True
    ):
        raise ValueError("native Korvid did not load exactly the exported rules and follow configuration")
    validate_native_prompt_identity(result)
    if result.get("model") != request["model"]:
        raise ValueError("native exported model configuration differs from the evaluated connection")
    return result


def export_native_rules(candidate: Candidate, campaign: Campaign, directory: Path) -> Path:
    serving = campaign.serving
    if not isinstance(serving, KorvidNativeServing) or len(campaign.models) != 1:
        raise ValueError("native export requires a single-model korvid_native campaign")
    rules = rules_from_candidate(candidate)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "korvid-config.yaml"
    model = native_model(campaign.models[0], serving, 0)
    config = {
        "readonly": True,
        "agent": {
            "active": "prompt-lab", "model_tier": "low", "follow": True, "rules": rules,
            "profiles": {"prompt-lab": {
                "model": model["reference"], "endpoint": model["endpoint"],
                "auth": {"method": "none"}, "options": model["options"],
            }},
        },
    }
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    with tempfile.TemporaryDirectory(prefix="korvid-rule-export-") as temporary:
        staged = Path(temporary) / "korvid-config.yaml"
        staged.write_text(text, encoding="utf-8")
        verified = verify_rules_config(candidate, campaign, config_path=staged)
    path.write_text(text, encoding="utf-8")
    write_json_artifact(directory / "application-manifest.json", {
        "schema_version": 1, "korvid_version": NATIVE_KORVID_VERSION,
        "korvid_revision": NATIVE_KORVID_REVISION, "candidate_fingerprint": candidate.fingerprint,
        "campaign_id": campaign.campaign_id, "rules_applied": True,
        "prompt_fingerprint": verified["prompt_fingerprint"],
        "policy": verified["policy"], "ui_follow": True,
        "runtime": verified.get("runtime"),
        "qualification": "not_assessed", "config": path.name,
    })
    return path


def command_native_check(args: argparse.Namespace) -> int:
    try:
        result = verify_rules_config(load_candidate(args.candidate), load_campaign(args.campaign))
    except (OSError, ValueError, BridgeSystemError) as exc:
        print(f"native check failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def command_native_export(args: argparse.Namespace) -> int:
    try:
        path = export_native_rules(load_candidate(args.candidate), load_campaign(args.campaign), args.directory)
    except (OSError, ValueError, BridgeSystemError) as exc:
        print(f"native export failed: {exc}", file=sys.stderr)
        return 2
    print(f"config={path}\nRules loading verified; model-quality qualification is not implied.")
    return 0
