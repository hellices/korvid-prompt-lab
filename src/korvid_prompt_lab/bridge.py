"""The ``korvid-bridge`` entry point the shipped campaigns invoke.

Korvid's operation harness needs Korvid itself, its bundled operation pack,
and Textual. None of that belongs in this control plane, so the bridge is a
thin launcher: it resolves a Korvid source checkout from the ``KORVID_SOURCE_ROOT``
runtime policy variable and runs :mod:`korvid_prompt_lab.bridge_worker` inside
that checkout's own ``uv`` environment.

The checkout is treated as read-only: ``uv run --no-sync`` never installs into
it and ``PYTHONDONTWRITEBYTECODE`` keeps the import machinery from writing
caches there. The source root is runtime policy only — it is read from the
environment, never from candidate text or the request artifact.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .bridge_worker import (
    DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    DEFAULT_PROFILE,
    DEFAULT_TURN_TIMEOUT_SECONDS,
    EXIT_SYSTEMIC_FAILURE,
    sanitize_error,
)

SOURCE_ROOT_ENV = "KORVID_SOURCE_ROOT"
UV_BIN_ENV = "KORVID_UV_BIN"

WORKER_MODULE_PATH = Path(__file__).resolve().with_name("bridge_worker.py")

#: Files that must exist for a directory to be a usable Korvid source checkout.
_CHECKOUT_MARKERS = (
    Path("pyproject.toml"),
    Path("src") / "korvid" / "__init__.py",
    Path("tests") / "evals" / "operation_app.py",
)


class BridgeConfigurationError(Exception):
    """The bridge cannot be launched; the runner must treat this as systemic."""


def resolve_source_root(env: Mapping[str, str]) -> Path:
    """Resolve the Korvid source checkout named by ``KORVID_SOURCE_ROOT``."""
    raw = env.get(SOURCE_ROOT_ENV, "")
    if not raw or not raw.strip():
        raise BridgeConfigurationError(
            f"{SOURCE_ROOT_ENV} must name the Korvid source checkout to evaluate against"
        )

    root = Path(raw.strip()).expanduser().resolve()
    missing = [str(marker) for marker in _CHECKOUT_MARKERS if not (root / marker).exists()]
    if missing:
        raise BridgeConfigurationError(
            f"{SOURCE_ROOT_ENV}={root} is not a Korvid source checkout (missing {', '.join(missing)})"
        )
    return root


def _resolve_uv(env: Mapping[str, str]) -> str:
    configured = env.get(UV_BIN_ENV, "").strip()
    if configured:
        return configured
    found = shutil.which("uv", path=env.get("PATH"))
    if not found:
        raise BridgeConfigurationError(
            f"uv was not found on PATH; install uv or set {UV_BIN_ENV} to its absolute path"
        )
    return found


def build_worker_invocation(
    *,
    source_root: Path,
    request_path: Path,
    response_path: Path,
    env: Mapping[str, str],
    scripted: bool = False,
    profile: str = DEFAULT_PROFILE,
    approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    turn_timeout: float = DEFAULT_TURN_TIMEOUT_SECONDS,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Build the ``uv`` command and environment that run the worker in *source_root*."""
    command = [
        _resolve_uv(env),
        "run",
        "--project",
        str(source_root),
        # The checkout is authoritative and read-only: never sync it.
        "--no-sync",
        "python",
        str(WORKER_MODULE_PATH),
        "--request",
        str(request_path),
        "--response",
        str(response_path),
        "--profile",
        profile,
        "--approval-timeout",
        str(float(approval_timeout)),
        "--turn-timeout",
        str(float(turn_timeout)),
    ]
    if scripted:
        command.append("--scripted")

    worker_env = dict(env)
    existing = worker_env.get("PYTHONPATH", "")
    # `tests.evals.operation_app` lives in the checkout, outside its installed package.
    worker_env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{existing}" if existing else str(source_root)
    )
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"
    return tuple(command), worker_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="korvid-bridge",
        description=(
            "Run one graded Korvid operation journey for a prompt candidate inside the"
            f" Korvid source checkout named by {SOURCE_ROOT_ENV}."
        ),
    )
    parser.add_argument("--request", required=True, type=Path, help="Path to the bridge request JSON.")
    parser.add_argument("--response", required=True, type=Path, help="Path to write the bridge response JSON.")
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Deterministic self-test: use Korvid's operation scripts instead of the model endpoint.",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Korvid agent profile to arm.")
    parser.add_argument(
        "--approval-timeout",
        type=float,
        default=DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        help="Approval window injected into the Korvid app.",
    )
    parser.add_argument(
        "--turn-timeout",
        type=float,
        default=DEFAULT_TURN_TIMEOUT_SECONDS,
        help="Upper bound on one turn reaching a dialog or ending.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ)

    try:
        command, worker_env = build_worker_invocation(
            source_root=resolve_source_root(env),
            request_path=args.request,
            response_path=args.response,
            env=env,
            scripted=args.scripted,
            profile=args.profile,
            approval_timeout=args.approval_timeout,
            turn_timeout=args.turn_timeout,
        )
    except BridgeConfigurationError as exc:
        print(f"korvid-bridge: {sanitize_error(exc, env)}", file=sys.stderr)
        return EXIT_SYSTEMIC_FAILURE

    try:
        completed = subprocess.run(command, env=worker_env, shell=False, check=False)
    except FileNotFoundError as exc:
        print(f"korvid-bridge: uv could not be launched: {sanitize_error(exc, env)}", file=sys.stderr)
        return EXIT_SYSTEMIC_FAILURE
    except OSError as exc:
        print(f"korvid-bridge: uv could not be executed: {sanitize_error(exc, env)}", file=sys.stderr)
        return EXIT_SYSTEMIC_FAILURE
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
