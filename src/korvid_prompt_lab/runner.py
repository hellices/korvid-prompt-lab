from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import write_json_artifact
from .contracts import Candidate, Campaign, EvalCase, ProcessServing, _ensure_keys, _require_mapping, _require_string
from .scoring import BridgeResult, OperationGrade


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


@dataclass(frozen=True, slots=True)
class KorvidProcessRunner:
    campaign: Campaign
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.campaign.serving, ProcessServing):
            raise ValueError("KorvidProcessRunner requires process serving")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

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
            completed = subprocess.run(
                command,
                shell=False,
                timeout=self.timeout_seconds,
                capture_output=True,
                text=False,
                check=False,
            )
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
        if protocol_version != 1:
            raise BridgeProtocolMismatchError("bridge response protocol_version must be 1")

        response_fingerprint = _require_response_string(payload, "candidate_fingerprint")
        if response_fingerprint != candidate.fingerprint:
            raise BridgeFingerprintMismatchError("bridge response fingerprint does not match the candidate")

        self._validate_request_identity(payload, case, repetition, seed)
        status = _require_response_string(payload, "status")
        if status not in {"completed", "model_failure"}:
            raise BridgeStatusError(f"bridge returned systemic status: {status}")

        grade = self._parse_grade(payload, status)
        return BridgeResult(
            protocol_version=protocol_version,
            status=status,
            candidate_fingerprint=candidate.fingerprint,
            grade=grade,
            answer=_require_response_text(payload, "answer"),
            journal=_require_response_mapping(payload, "journal"),
            usage=_require_response_mapping(payload, "usage"),
            error=_require_optional_response_string(payload, "error"),
        )

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
            "protocol_version": 1,
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
            },
        }

    def _expand_command(self, request_path: Path, response_path: Path) -> tuple[str, ...]:
        expanded: list[str] = []
        for token in self.campaign.serving.command:
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
            hard_failures = grade_mapping.get("hard_failures", [])
            if not isinstance(hard_failures, list):
                raise ValueError("hard_failures must be a list")
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


def _coerce_metric(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
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
