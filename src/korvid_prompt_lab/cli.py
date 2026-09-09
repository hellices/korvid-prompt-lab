"""One explicit entry point for native model connection, search, and verification."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import FrameType

import httpx
from litellm.exceptions import APIError

from .aks import AKSPortForwardError
from .model_session import ModelSessionError
from .reflection import ProposalProviderError
from .runner import BridgeSystemError


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="korvid-prompt-lab")
    subparsers = root.add_subparsers(dest="command", required=True)
    parser = subparsers.add_parser(
        "run", help="Run the complete native Korvid experiment from one version-2 manifest (recommended).",
    )
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/native-experiment"))
    parser.add_argument("--check-only", action="store_true", help="Validate configuration and pinned source without cloud or model calls.")
    parser.add_argument("--allow-capacity-changes", action="store_true", help="Explicitly permit an owned AKS node-pool 0-to-1 transition and restoration.")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    return command_run(build_parser().parse_args(argv))


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def command_run(args: argparse.Namespace) -> int:
    from .experiment import run_experiment
    from .experiment_config import load_experiment
    from .source_runtime import validate_source
    from .upstream import inspect_upstream

    try:
        spec = load_experiment(args.experiment)
        validate_source(Path(spec.runtime.source_root))
        if args.check_only:
            snapshot = inspect_upstream(
                replace(spec.runtime, base_url="http://127.0.0.1:1"),
                spec.model.reference, spec.references, seed=spec.evaluation_seed,
            )
            if snapshot["prompt"]["baseline_equivalent"] is not True:
                raise ValueError("copied baseline is not the original Korvid prompt")
    except (OSError, ValueError, BridgeSystemError, subprocess.SubprocessError) as exc:
        print(f"experiment configuration failed: {exc}", file=sys.stderr)
        return 2
    if args.check_only:
        print(json.dumps({
            "status": "CONFIGURATION_VALID", "experiment_fingerprint": spec.fingerprint,
            "model_calls": 0, "cloud_calls": 0, "serving_preflight": "not_performed",
            "baseline_equivalent": snapshot["prompt"]["baseline_equivalent"],
            "prompt_pack_id": snapshot["prompt"]["pack_id"],
            "prompt_source_sha256": snapshot["prompt"]["source_sha256"],
            "case_sources": [
                {"reference": case["reference"], "source_sha256": case["source_sha256"]}
                for case in snapshot["cases"]
            ],
        }, indent=2))
        return 0
    previous = signal.signal(signal.SIGTERM, _interrupt)
    try:
        report = run_experiment(spec, args.artifact_root, allow_capacity_changes=args.allow_capacity_changes)
    except KeyboardInterrupt:
        print("experiment cancelled; inspect experiment-summary.json and serving cleanup evidence", file=sys.stderr)
        return 130
    except (OSError, ValueError, BridgeSystemError, AKSPortForwardError, ModelSessionError, ProposalProviderError, subprocess.SubprocessError, httpx.HTTPError, APIError, ExceptionGroup) as exc:
        print(
            f"experiment failed ({type(exc).__name__}); inspect {args.artifact_root / 'experiment-summary.json'}",
            file=sys.stderr,
        )
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps({
        key: report[key] for key in (
            "status", "pipeline_completed", "prompt_improved", "qualified",
            "proposal_attempts", "distinct_proposals", "evaluations",
        )
    }, indent=2))
    return 0 if report["status"] == "QUALIFIED" else 3 if report["status"] == "NOT_CONVERGED" else 1
