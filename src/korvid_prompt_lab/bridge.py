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
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bridge_worker import (
    DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    DEFAULT_PROFILE,
    DEFAULT_TURN_TIMEOUT_SECONDS,
    EXIT_SYSTEMIC_FAILURE,
    sanitize_error,
)

SOURCE_ROOT_ENV = "KORVID_SOURCE_ROOT"
UV_BIN_ENV = "KORVID_UV_BIN"

#: Runtime policy set by :class:`~korvid_prompt_lab.runner.KorvidProcessRunner`: the
#: wall-clock budget this launcher owns, always strictly inside the runner's own
#: campaign timeout so the launcher can tear its worker down before the runner
#: stops waiting. Absent means a hand-run bridge with no campaign policy.
BRIDGE_TIMEOUT_ENV = "KORVID_BRIDGE_TIMEOUT_SECONDS"

#: How long each termination signal is given before the next escalation step.
WORKER_TERMINATION_GRACE_SECONDS = 2.0

#: Worst-case wall time :func:`_terminate_process_group` needs: SIGTERM plus SIGKILL.
#: :class:`~korvid_prompt_lab.runner.KorvidProcessRunner` reserves at least this much
#: of the campaign timeout so the launcher can finish before the runner starts killing.
WORKER_TEARDOWN_BUDGET_SECONDS = 2 * WORKER_TERMINATION_GRACE_SECONDS

#: How often the launcher re-checks whether the worker process group has drained.
_GROUP_POLL_INTERVAL_SECONDS = 0.05

#: Signals that mean "stop", and that must be handed on to the worker group.
_TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)

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


def _build_worker_environment(source_root: Path, env: Mapping[str, str]) -> dict[str, str]:
    worker_env = dict(env)
    existing = worker_env.get("PYTHONPATH", "")
    # `tests.evals.operation_app` lives in the checkout, outside its installed package.
    worker_env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{existing}" if existing else str(source_root)
    )
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"
    return worker_env


def _build_worker_command(source_root: Path, env: Mapping[str, str]) -> list[str]:
    return [
        _resolve_uv(env),
        "run",
        "--project",
        str(source_root),
        # The checkout is authoritative and read-only: never sync it.
        "--no-sync",
        "python",
        str(WORKER_MODULE_PATH),
    ]


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
    command = _build_worker_command(source_root, env)
    command.extend(
        [
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
    )
    if scripted:
        command.append("--scripted")

    return tuple(command), _build_worker_environment(source_root, env)


def build_worker_import_check(
    *,
    source_root: Path,
    env: Mapping[str, str],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Build the worker preflight that proves Korvid's runtime imports resolve."""
    command = _build_worker_command(source_root, env)
    command.append("--check-imports")
    return tuple(command), _build_worker_environment(source_root, env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="korvid-bridge",
        description=(
            "Run one graded Korvid operation journey for a prompt candidate inside the"
            f" Korvid source checkout named by {SOURCE_ROOT_ENV}."
        ),
    )
    parser.add_argument(
        "--check-imports",
        action="store_true",
        help="Resolve Korvid's runtime imports inside the checked-out Korvid environment.",
    )
    parser.add_argument("--request", type=Path, help="Path to the bridge request JSON.")
    parser.add_argument("--response", type=Path, help="Path to write the bridge response JSON.")
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


def resolve_timeout_budget(env: Mapping[str, str]) -> float | None:
    """Return the wall-clock budget this launcher owns, or ``None`` when unset.

    The value is runtime policy: the runner derives it from the campaign's
    ``bridge_timeout_seconds`` and always leaves itself a margin, so the launcher
    can terminate its whole worker process group and report a systemic failure
    before the runner stops waiting. A present-but-unusable value is refused
    rather than ignored: silently running unbounded is exactly the orphaned-worker
    failure this budget exists to prevent.
    """
    raw = env.get(BRIDGE_TIMEOUT_ENV)
    if raw is None:
        return None
    try:
        budget = float(raw)
    except ValueError as exc:
        raise BridgeConfigurationError(
            f"{BRIDGE_TIMEOUT_ENV} must be a positive number of seconds"
        ) from exc
    if not math.isfinite(budget) or budget <= 0.0:
        raise BridgeConfigurationError(
            f"{BRIDGE_TIMEOUT_ENV} must be a positive number of seconds"
        )
    return budget


class LauncherTerminated(BaseException):
    """The launcher itself was signalled; its worker must not outlive it.

    Deliberately a :class:`BaseException`: it is a termination request, not a
    program error, and it must not be swallowed by an ``except Exception``.
    """


def _process_group_is_populated(group: int) -> bool:
    """Return whether any process is still a member of *group*."""
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A member exists that we may not signal — most often our own unreaped
        # zombie, which the caller clears with poll() before the next probe.
        return True
    except OSError:  # pragma: no cover - defensive
        return False
    return True


def _await_process_group_exit(process: subprocess.Popen[bytes], group: int, timeout: float) -> bool:
    """Poll until *group* has drained, reaping the direct child so it stops holding it open."""
    deadline = time.monotonic() + timeout
    while True:
        # An unreaped zombie is still a group member, so reap before testing.
        process.poll()
        if not _process_group_is_populated(group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_GROUP_POLL_INTERVAL_SECONDS)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Kill every process in the worker's group, not just the direct child.

    ``uv run`` execs a grandchild worker and dies on SIGTERM immediately, so
    treating "the direct child was reaped" as "the group is gone" would skip the
    SIGKILL escalation and leave the real grader running — free to write a late
    ``response.json`` into a run directory the control plane has abandoned. The
    escalation is therefore gated on the *group* draining. The worker was started
    with ``start_new_session=True``, so its process-group id equals its pid, the
    launcher is not a member, and no signal sent here can reach this process.

    Termination signals are ignored for the duration. This launcher and its runner
    tear down on overlapping deadlines, so a runner-initiated SIGTERM landing here
    would otherwise unwind the escalation before SIGKILL is sent — and nothing else
    can reach the worker, because it lives in this launcher's private session.
    """
    if os.name != "posix":  # pragma: no cover - POSIX-only process groups
        process.kill()
        process.wait()
        return

    previous = _set_termination_handlers(signal.SIG_IGN)
    try:
        try:
            group = os.getpgid(process.pid)
        except (OSError, ProcessLookupError):  # pragma: no cover - already reaped
            group = process.pid

        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(group, sig)
            except (OSError, ProcessLookupError):
                break
            if _await_process_group_exit(process, group, WORKER_TERMINATION_GRACE_SECONDS):
                break

        try:
            process.wait(timeout=WORKER_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL is not refusable
            pass
    finally:
        _restore_termination_handlers(previous)


def _set_termination_handlers(handler: Any) -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for sig in _TERMINATION_SIGNALS:
        try:
            previous[sig] = signal.signal(sig, handler)
        except (OSError, ValueError, AttributeError):  # pragma: no cover - platform dependent
            continue
    return previous


def _install_termination_handlers() -> dict[int, Any]:
    """Turn a termination signal into an exception so the worker group is torn down.

    The runner signals *this* launcher's process group, which does not contain the
    worker: the worker deliberately lives in its own session so the launcher's own
    kills stay scoped to it. Exiting on the signal without passing that termination
    on would leave the worker running.
    """
    def _raise(signum: int, _frame: Any) -> None:
        raise LauncherTerminated(f"korvid-bridge received signal {signum}")

    return _set_termination_handlers(_raise)


def _restore_termination_handlers(previous: Mapping[int, Any]) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (OSError, ValueError):  # pragma: no cover - platform dependent
            continue


def _run_worker_process_group(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout: float | None,
) -> subprocess.CompletedProcess[bytes]:
    """Run *command* as its own process group and never leave a descendant behind."""
    posix = os.name == "posix"
    # `Popen.__init__` returns long after the fork, so for most of it the worker is
    # already running while this function still has no handle on it. A raising
    # handler there would unwind past the only code that can reach the worker's
    # private session, so signals are latched until `process` is bound.
    pending: list[int] = []

    def _latch(signum: int, _frame: Any) -> None:
        pending.append(signum)

    previous_handlers = _set_termination_handlers(_latch) if posix else {}
    try:
        process = subprocess.Popen(
            list(command),
            env=dict(env),
            shell=False,
            # A fresh session makes this launcher the sole owner of the worker subtree,
            # so the kills below can never reach this process or whatever launched it.
            start_new_session=posix,
        )
    except BaseException:
        _restore_termination_handlers(previous_handlers)
        raise

    try:
        if posix:
            _install_termination_handlers()
        if pending:
            raise LauncherTerminated(f"korvid-bridge received signal {pending[0]}")
        process.wait(timeout=timeout)
    except BaseException:  # a timeout or a signal must never orphan the worker
        _terminate_process_group(process)
        raise
    finally:
        _restore_termination_handlers(previous_handlers)
    return subprocess.CompletedProcess(list(command), process.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ)

    try:
        timeout = resolve_timeout_budget(env)
        source_root = resolve_source_root(env)
        if args.check_imports:
            if args.request is not None or args.response is not None:
                raise BridgeConfigurationError(
                    "--check-imports does not accept --request or --response"
                )
            command, worker_env = build_worker_import_check(source_root=source_root, env=env)
        else:
            if args.request is None or args.response is None:
                raise BridgeConfigurationError(
                    "--request and --response are required unless --check-imports is used"
                )
            command, worker_env = build_worker_invocation(
                source_root=source_root,
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
        completed = _run_worker_process_group(command, env=worker_env, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(
            f"korvid-bridge: worker timed out after {timeout} seconds; the worker process"
            " group was terminated",
            file=sys.stderr,
        )
        return EXIT_SYSTEMIC_FAILURE
    except LauncherTerminated as exc:
        print(f"korvid-bridge: {exc}; the worker process group was terminated", file=sys.stderr)
        return EXIT_SYSTEMIC_FAILURE
    except KeyboardInterrupt:  # pragma: no cover - interactive use only
        print("korvid-bridge: interrupted; the worker process group was terminated", file=sys.stderr)
        return EXIT_SYSTEMIC_FAILURE
    except FileNotFoundError as exc:
        print(f"korvid-bridge: uv could not be launched: {sanitize_error(exc, env)}", file=sys.stderr)
        return EXIT_SYSTEMIC_FAILURE
    except OSError as exc:
        print(f"korvid-bridge: uv could not be executed: {sanitize_error(exc, env)}", file=sys.stderr)
        return EXIT_SYSTEMIC_FAILURE
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
