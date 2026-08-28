from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from .artifacts import write_json_artifact
from .bridge_worker import EXECUTION_MODE_LIVE, EXECUTION_MODES, PROTOCOL_VERSION
from .contracts import (
    AKSPortForwardServing,
    Campaign,
    Candidate,
    EvalCase,
    ProcessServing,
    _ensure_keys,
    _require_bridge_timeout,
    _require_mapping,
    _require_string,
)
from .scoring import BridgeResult, OperationGrade

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

#: Runtime policy handed to the bridge launcher so it can tear its own worker
#: process group down before this runner stops waiting. Never read from a
#: candidate or the optimizer: it is derived from campaign policy alone.
BRIDGE_TIMEOUT_ENV = "KORVID_BRIDGE_TIMEOUT_SECONDS"

#: Share of the campaign timeout reserved for the launcher's own cleanup.
LAUNCHER_GRACE_FRACTION = 0.1
#: Upper bound on that reservation, so a long campaign budget is not wasted.
LAUNCHER_MAX_GRACE_SECONDS = 10.0
#: Lower bound on that reservation: the launcher's own worst-case teardown window
#: (:data:`korvid_prompt_lab.bridge.WORKER_TEARDOWN_BUDGET_SECONDS`). Reserving less
#: would let this runner start killing while the launcher is mid-escalation.
LAUNCHER_TEARDOWN_RESERVATION_SECONDS = 4.0
#: How long each termination signal is given before the next escalation step.
PROCESS_GROUP_TERMINATION_GRACE_SECONDS = 5.0
#: How often the runner re-checks whether the bridge process group has drained.
_GROUP_POLL_INTERVAL_SECONDS = 0.05


def launcher_timeout_seconds(timeout_seconds: float) -> float:
    """Return the launcher's budget: strictly inside the campaign timeout.

    The launcher owns its own worker process group, so it must be able to run a
    full SIGTERM/SIGKILL teardown and report a systemic failure before this runner
    gives up and starts killing. The reservation is therefore at least the
    launcher's worst-case teardown window, proportional above that so a long
    campaign is not cut short, and capped so a 900-second budget does not donate
    90 seconds to cleanup. It is finally clamped to half the timeout so a very
    short campaign still leaves the launcher a usable budget; the two deadlines
    then overlap, which is safe because the launcher ignores termination signals
    for the duration of its own teardown.
    """
    grace = min(
        max(timeout_seconds * LAUNCHER_GRACE_FRACTION, LAUNCHER_TEARDOWN_RESERVATION_SECONDS),
        LAUNCHER_MAX_GRACE_SECONDS,
    )
    return timeout_seconds - min(grace, timeout_seconds * 0.5)


def _require_loopback_endpoint(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    if parts.scheme != "http" or parts.hostname not in _LOOPBACK_HOSTS or parts.port is None:
        raise ValueError(
            "model_endpoint must be a loopback http URL with an explicit port, for example http://127.0.0.1:41001"
        )
    if parts.path not in {"", "/"} or parts.query or parts.fragment:
        raise ValueError("model_endpoint must be a loopback base URL without path, query, or fragment")
    return endpoint


class BridgeSystemError(RuntimeError):
    """Base class for runner failures that should abort optimization."""


class BridgeTimeoutError(BridgeSystemError):
    """Raised when the bridge command does not finish before the timeout."""


class BridgeArtifactError(BridgeSystemError):
    """Raised when runner artifacts cannot be prepared or cleaned up."""


class BridgeInvocationError(BridgeSystemError):
    """Raised when the bridge command cannot be launched."""


class BridgeProcessExitError(BridgeSystemError):
    """Raised when the bridge command exits non-zero."""


class BridgeMissingOutputError(BridgeSystemError):
    """Raised when the bridge command does not create a response file."""


class BridgeMalformedOutputError(BridgeSystemError):
    """Raised when the bridge response is not valid protocol JSON."""


class BridgeProtocolMismatchError(BridgeSystemError):
    """Raised when the bridge response uses an unsupported protocol version."""


class BridgeFingerprintMismatchError(BridgeSystemError):
    """Raised when the bridge response does not match the candidate fingerprint."""


class BridgeIdentityMismatchError(BridgeSystemError):
    """Raised when the bridge response does not match the requested case identity."""


class BridgeStatusError(BridgeSystemError):
    """Raised when the bridge returns a systemic status."""


class BridgeExecutionModeError(BridgeSystemError):
    """Raised when the bridge evidence was not produced the way the campaign requires."""


@runtime_checkable
class KorvidRunner(Protocol):
    """The shape the adapter, optimizer, and CLI depend on for any evidence source.

    :class:`KorvidProcessRunner` (write/approval, via the bridge subprocess) and
    :class:`~korvid_prompt_lab.korvid_readonly.KorvidReadonlyRunner` (read-only,
    via the installed ``korvid.evals`` CLI) both satisfy this structurally, with
    no inheritance or import of this module required from the read-only runner.
    Callers must select a concrete runner by ``campaign.serving.backend`` before
    depending on anything backend-specific; afterwards only this shared shape
    may be relied upon.
    """

    @property
    def campaign(self) -> Campaign:
        # Read-only: both concrete runners are frozen dataclasses, so the
        # Protocol must declare `campaign` as a getter, not a settable field.
        ...

    def run(
        self,
        candidate: Candidate,
        case: EvalCase,
        run_dir: Path | str,
        *,
        repetition: int = 1,
        seed: int = 0,
    ) -> BridgeResult: ...


@dataclass(frozen=True, slots=True)
class KorvidProcessRunner:
    campaign: Campaign
    timeout_seconds: float | None = None
    model_endpoint: str | None = None

    def __post_init__(self) -> None:
        serving = self.campaign.serving
        if not isinstance(serving, (ProcessServing, AKSPortForwardServing)):
            raise ValueError("KorvidProcessRunner requires process or aks_port_forward serving")  # noqa: TRY004 - preserve validation API
        if self.timeout_seconds is None:
            # Runtime policy belongs to the campaign, never to a candidate or the optimizer.
            object.__setattr__(self, "timeout_seconds", self.campaign.bridge_timeout_seconds)
        _require_bridge_timeout(self.timeout_seconds, "timeout_seconds")
        if isinstance(serving, AKSPortForwardServing):
            if self.model_endpoint is None:
                raise ValueError("aks_port_forward serving requires a model_endpoint")
            _require_loopback_endpoint(self.model_endpoint)
        elif self.model_endpoint is not None:
            raise ValueError("process serving must not receive a model_endpoint")

    def run(
        self,
        candidate: Candidate,
        case: EvalCase,
        run_dir: Path | str,
        *,
        repetition: int = 1,
        seed: int = 0,
    ) -> BridgeResult:
        run_path = Path(run_dir)
        request_path = run_path / "request.json"
        response_path = run_path / "response.json"

        if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition <= 0:
            raise ValueError("repetition must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if repetition > self.campaign.repetitions:
            raise ValueError("repetition must not exceed campaign.repetitions")

        try:
            if response_path.exists():
                response_path.unlink()
        except OSError as exc:
            raise BridgeArtifactError(f"runner could not clean previous response artifact: {response_path}") from exc

        try:
            write_json_artifact(request_path, self._build_request(candidate, case, run_path, repetition, seed))
        except OSError as exc:
            raise BridgeArtifactError(f"runner could not write request artifact: {request_path}") from exc
        command = self._expand_command(request_path, response_path)
        if not command:
            raise BridgeInvocationError("bridge command must not be empty")

        try:
            completed = self._run_bridge_process_group(command)
        except FileNotFoundError as exc:
            raise BridgeInvocationError(f"bridge command could not be launched: {command[0]}") from exc
        except OSError as exc:
            raise BridgeInvocationError(f"bridge command could not be executed: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise BridgeTimeoutError(f"bridge timed out after {self.timeout_seconds} seconds") from exc

        if completed.returncode != 0:
            detail = _decode_process_output(completed.stderr) or _decode_process_output(completed.stdout)
            detail = detail or f"exit code {completed.returncode}"
            raise BridgeProcessExitError(f"bridge exited non-zero: {detail}")

        if not response_path.exists():
            raise BridgeMissingOutputError("bridge did not create a response.json artifact")

        payload = self._load_response(response_path)
        protocol_version = _require_response_int(payload, "protocol_version")
        if protocol_version != PROTOCOL_VERSION:
            raise BridgeProtocolMismatchError(
                f"bridge response protocol_version must be {PROTOCOL_VERSION}"
            )

        response_fingerprint = _require_response_string(payload, "candidate_fingerprint")
        if response_fingerprint != candidate.fingerprint:
            raise BridgeFingerprintMismatchError("bridge response fingerprint does not match the candidate")

        self._validate_request_identity(payload, case, repetition, seed)
        status = _require_response_string(payload, "status")
        if status not in {"completed", "model_failure"}:
            raise BridgeStatusError(f"bridge returned systemic status: {status}")

        execution_mode = self._require_execution_mode(payload)
        grade = self._parse_grade(payload, status)
        return BridgeResult(
            protocol_version=protocol_version,
            status=status,
            execution_mode=execution_mode,
            candidate_fingerprint=candidate.fingerprint,
            grade=grade,
            answer=_require_response_text(payload, "answer"),
            journal=_require_response_mapping(payload, "journal"),
            usage=_require_response_mapping(payload, "usage"),
            error=_require_optional_response_string(payload, "error"),
        )

    def _require_execution_mode(self, payload: Mapping[str, Any]) -> str:
        """Return how this evidence was produced, refusing anything a live campaign forbids."""
        execution_mode = _require_response_string(payload, "execution_mode")
        if execution_mode not in EXECUTION_MODES:
            raise BridgeExecutionModeError(
                f"bridge response execution_mode must be one of {', '.join(sorted(EXECUTION_MODES))}"
            )
        # A campaign that stood up a model endpoint is a live campaign. Scripted
        # evidence is model-free by construction, so accepting it here would let a
        # perfect score be published for a model that was never contacted.
        if self.model_endpoint is not None and execution_mode != EXECUTION_MODE_LIVE:
            raise BridgeExecutionModeError(
                "a campaign serving a model endpoint requires live bridge evidence,"
                f" but the bridge reported execution_mode {execution_mode}"
            )
        return execution_mode

    def _bridge_environment(self) -> dict[str, str]:
        """Campaign runtime policy for the launcher; never candidate or optimizer text."""
        timeout = float(self.timeout_seconds if self.timeout_seconds is not None else 0.0)
        return {
            **os.environ,
            BRIDGE_TIMEOUT_ENV: repr(launcher_timeout_seconds(timeout)),
        }

    def _run_bridge_process_group(self, command: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        """Run the bridge as its own process group and never leave a descendant behind.

        ``subprocess.run(timeout=...)`` kills only the direct child, so a launcher that
        execs ``uv``, which execs the worker, leaves a live grader that can still write
        a late ``response.json`` into this run directory. Owning the group means one
        signal reaches the whole subtree.
        """
        process = subprocess.Popen(
            list(command),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=self._bridge_environment(),
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except BaseException:  # a timeout or an interrupt must never orphan the bridge
            _terminate_process_group(process)
            raise
        return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)

    def _build_request(
        self,
        candidate: Candidate,
        case: EvalCase,
        run_dir: Path,
        repetition: int,
        seed: int,
    ) -> dict[str, Any]:
        if len(case.models) != 1:
            raise ValueError("KorvidProcessRunner requires exactly one model per case")

        return {
            "protocol_version": PROTOCOL_VERSION,
            "candidate_fingerprint": candidate.fingerprint,
            "candidate": {
                "schema_version": candidate.schema_version,
                "candidate_id": candidate.candidate_id,
                "components": candidate.components,
                "metadata": candidate.metadata,
            },
            "case": {
                "case_id": case.case_id,
                "template_id": case.template_id,
                "prompt": case.prompt,
                "model": case.models[0],
                "repetition": repetition,
                "seed": seed,
            },
            "runtime": {
                "campaign_id": self.campaign.campaign_id,
                "repetitions": self.campaign.repetitions,
                "artifact_dir": str(run_dir),
                "model_endpoint": self.model_endpoint,
            },
        }

    def _expand_command(self, request_path: Path, response_path: Path) -> tuple[str, ...]:
        expanded: list[str] = []
        serving = self.campaign.serving
        if not isinstance(serving, (ProcessServing, AKSPortForwardServing)):
            raise ValueError("KorvidProcessRunner requires process or aks_port_forward serving")  # noqa: TRY004 - preserve validation API
        for token in serving.command:
            if token == "{request}":
                expanded.append(str(request_path))
            elif token == "{response}":
                expanded.append(str(response_path))
            else:
                expanded.append(token)
        return tuple(expanded)

    def _load_response(self, path: Path) -> Mapping[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise BridgeArtifactError(f"runner could not read response artifact: {path}") from exc
        except UnicodeDecodeError as exc:
            raise BridgeMalformedOutputError("bridge response is not valid UTF-8 JSON") from exc
        except json.JSONDecodeError as exc:
            raise BridgeMalformedOutputError("bridge response is not valid JSON") from exc

        try:
            mapping = _require_mapping(payload, "bridge response")
            _ensure_keys(
                mapping,
                {
                    "protocol_version",
                    "status",
                    "execution_mode",
                    "candidate_fingerprint",
                    "request_identity",
                    "grade",
                    "answer",
                    "journal",
                    "usage",
                    "error",
                },
                "bridge response",
            )
            return mapping
        except ValueError as exc:
            raise BridgeMalformedOutputError(str(exc)) from exc

    def _parse_grade(self, payload: Mapping[str, Any], status: str) -> OperationGrade | None:
        if "grade" not in payload:
            raise BridgeMalformedOutputError("bridge response missing grade")

        grade_payload = payload["grade"]
        if grade_payload is None:
            if status == "completed":
                raise BridgeMalformedOutputError("completed bridge responses must include a grade")
            return None
        if status != "completed":
            raise BridgeMalformedOutputError("only completed bridge responses may include a grade")

        try:
            grade_mapping = _require_mapping(grade_payload, "grade")
            _ensure_keys(grade_mapping, {"completion", "verification", "efficiency", "hard_failures"}, "grade")
            if "hard_failures" not in grade_mapping:
                raise ValueError("grade missing hard_failures")
            hard_failures = grade_mapping["hard_failures"]
            if not isinstance(hard_failures, list):
                raise ValueError("hard_failures must be a list")  # noqa: TRY004 - preserve validation API
            return OperationGrade(
                completion=_coerce_metric(grade_mapping["completion"], "completion"),
                verification=_coerce_metric(grade_mapping["verification"], "verification"),
                efficiency=_coerce_metric(grade_mapping["efficiency"], "efficiency"),
                hard_failures=tuple(_require_string(item, "hard_failure") for item in hard_failures),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BridgeMalformedOutputError("bridge grade is malformed") from exc

    def _validate_request_identity(
        self,
        payload: Mapping[str, Any],
        case: EvalCase,
        repetition: int,
        seed: int,
    ) -> None:
        identity = _require_response_mapping(payload, "request_identity")
        try:
            _ensure_keys(identity, {"case_id", "template_id", "model", "repetition", "seed"}, "request_identity")
        except ValueError as exc:
            raise BridgeMalformedOutputError(str(exc)) from exc

        expected = {
            "case_id": case.case_id,
            "template_id": case.template_id,
            "model": case.models[0],
            "repetition": repetition,
            "seed": seed,
        }
        actual = {
            "case_id": _require_response_string(identity, "case_id"),
            "template_id": _require_response_string(identity, "template_id"),
            "model": _require_response_string(identity, "model"),
            "repetition": _require_response_int(identity, "repetition"),
            "seed": _require_response_int(identity, "seed"),
        }
        if actual != expected:
            raise BridgeIdentityMismatchError("bridge response identity does not match the request")


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
    """Poll until *group* drains, reaping the bridge so it stops holding it open."""
    deadline = time.monotonic() + timeout
    while True:
        # An unreaped zombie is still a group member, so reap before testing.
        process.poll()
        if not _process_group_is_populated(group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_GROUP_POLL_INTERVAL_SECONDS)


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    """Close this side of the bridge's pipes; a survivor may still hold the other end."""
    for stream in (process.stdout, process.stderr, process.stdin):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:  # pragma: no cover - defensive
            continue


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Signal the bridge's whole process group, escalating to SIGKILL.

    The bridge was started with ``start_new_session=True``, so its process-group id
    equals its pid and every descendant that did not start its own session is in it.
    SIGTERM first gives the launcher a chance to hand the termination on to its own
    worker group and report a systemic failure; SIGKILL then guarantees nothing here
    survives to write a late response artifact.

    The escalation is gated on the *group* draining, not on the bridge process being
    reaped: a launcher that exits immediately on SIGTERM would otherwise end the
    escalation while its descendants are still running. ``communicate`` is avoided
    for the same reason — a surviving descendant inherits the pipes and would block
    it until its own deadline.
    """
    if os.name != "posix":  # pragma: no cover - POSIX-only process groups
        process.kill()
        process.wait()
        _close_process_pipes(process)
        return

    try:
        group = os.getpgid(process.pid)
    except (OSError, ProcessLookupError):  # pragma: no cover - already reaped
        group = process.pid

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except (OSError, ProcessLookupError):
            break
        if _await_process_group_exit(process, group, PROCESS_GROUP_TERMINATION_GRACE_SECONDS):
            break

    try:
        process.wait(timeout=PROCESS_GROUP_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL is not refusable
        pass
    _close_process_pipes(process)


def _coerce_metric(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")  # noqa: TRY004 - preserve validation API
    return float(value)


def _require_response_mapping(payload: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if field_name not in payload:
        raise BridgeMalformedOutputError(f"bridge response missing {field_name}")
    try:
        return _require_mapping(payload[field_name], field_name)
    except ValueError as exc:
        raise BridgeMalformedOutputError(f"{field_name} must be an object") from exc


def _require_response_text(payload: Mapping[str, Any], field_name: str) -> str:
    if field_name not in payload:
        raise BridgeMalformedOutputError(f"bridge response missing {field_name}")
    value = payload[field_name]
    if not isinstance(value, str):
        raise BridgeMalformedOutputError(f"{field_name} must be a string")
    return value


def _require_optional_response_string(payload: Mapping[str, Any], field_name: str) -> str | None:
    if field_name not in payload:
        raise BridgeMalformedOutputError(f"bridge response missing {field_name}")
    value = payload[field_name]
    if value is None:
        return None
    try:
        return _require_string(value, field_name)
    except ValueError as exc:
        raise BridgeMalformedOutputError(f"{field_name} must be a non-empty string when present") from exc


def _require_response_string(payload: Mapping[str, Any], field_name: str) -> str:
    if field_name not in payload:
        raise BridgeMalformedOutputError(f"bridge response missing {field_name}")
    try:
        return _require_string(payload[field_name], field_name)
    except ValueError as exc:
        raise BridgeMalformedOutputError(f"{field_name} must be a non-empty string") from exc


def _require_response_int(payload: Mapping[str, Any], field_name: str) -> int:
    if field_name not in payload:
        raise BridgeMalformedOutputError(f"bridge response missing {field_name}")
    value = payload[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise BridgeMalformedOutputError(f"{field_name} must be an integer")
    return value


def _decode_process_output(value: bytes | None) -> str:
    if not value:
        return ""
    return value.decode("utf-8", errors="replace").strip()
