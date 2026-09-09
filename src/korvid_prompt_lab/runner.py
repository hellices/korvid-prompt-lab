from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from .contracts import Campaign, Candidate, EvalCase
from .scoring import EvaluationResult

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
PROCESS_GROUP_TERMINATION_GRACE_SECONDS = 5.0
_GROUP_POLL_INTERVAL_SECONDS = 0.05


def _require_loopback_endpoint(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    if (
        parts.scheme != "http"
        or parts.hostname not in _LOOPBACK_HOSTS
        or parts.port is None
    ):
        raise ValueError(
            "model_endpoint must be a loopback http URL with an explicit port, "
            "for example http://127.0.0.1:41001"
        )
    if parts.path not in {"", "/"} or parts.query or parts.fragment:
        raise ValueError(
            "model_endpoint must be a loopback base URL without path, query, or fragment"
        )
    return endpoint


class BridgeSystemError(RuntimeError):
    """Base class for subprocess bridge failures that abort evaluation."""


class BridgeInvocationError(BridgeSystemError):
    """Raised when the current subprocess bridge cannot produce evidence."""


class BridgeExecutionModeError(BridgeSystemError):
    """Raised when subprocess bridge evidence uses a forbidden execution mode."""


@runtime_checkable
class KorvidRunner(Protocol):
    @property
    def campaign(self) -> Campaign: ...

    def run(
        self,
        candidate: Candidate,
        case: EvalCase,
        run_dir: Path | str,
        *,
        repetition: int = 1,
        seed: int = 0,
    ) -> EvaluationResult: ...


def _process_group_is_populated(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _await_process_group_exit(
    process: subprocess.Popen[bytes],
    group: int,
    timeout: float,
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        process.poll()
        if not _process_group_is_populated(group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_GROUP_POLL_INTERVAL_SECONDS)


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdout, process.stderr, process.stdin):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            continue


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name != "posix":
        process.kill()
        process.wait()
        _close_process_pipes(process)
        return

    try:
        group = os.getpgid(process.pid)
    except (OSError, ProcessLookupError):
        group = process.pid

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except (OSError, ProcessLookupError):
            break
        if _await_process_group_exit(
            process,
            group,
            PROCESS_GROUP_TERMINATION_GRACE_SECONDS,
        ):
            break

    try:
        process.wait(timeout=PROCESS_GROUP_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    _close_process_pipes(process)
