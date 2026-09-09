"""Model endpoint lifecycle and immutable digest verification."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from .aks import (
    AKSForwardTarget,
    AKSPortForward,
    AKSPortForwardError,
    AKSPreflightTransientError,
)
from .artifacts import write_json_artifact
from .contracts import _require_mapping, _require_string
from .experiment_budget import BudgetExhausted, ExperimentBudget
from .experiment_config import AKSConnection, ExperimentSpec, LoopbackConnection

_LIVE_DIGEST_PATTERN = re.compile(r"(?:sha256:)?([0-9a-f]{64})")


class ModelSessionError(RuntimeError):
    """Raised when model serving identity or owned capacity cannot be verified."""


@dataclass(frozen=True, slots=True)
class ResolvedModelSession:
    base_url: str
    evidence: dict[str, Any]
    _verify_callback: Callable[[], dict[str, Any]] = field(
        repr=False, compare=False
    )

    def verify(self) -> dict[str, Any]:
        """Recheck both configured model digests against the live endpoint."""
        return self._verify_callback()


def _canonical_live_digest(value: Any, context: str) -> str:
    digest = _require_string(value, context)
    match = _LIVE_DIGEST_PATTERN.fullmatch(digest)
    if match is None:
        raise ValueError(f"{context} is not a canonical SHA-256 digest")
    return f"sha256:{match.group(1)}"


def _model_digests(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("model tags probe returned an invalid models list")  # noqa: TRY004
    digests: dict[str, list[str]] = {}
    for index, item in enumerate(raw_models):
        model = _require_mapping(item, f"model tags.models[{index}]")
        raw_name = model.get("name", model.get("model"))
        name = _require_string(raw_name, f"model tags.models[{index}].name")
        digest = _canonical_live_digest(
            model.get("digest"), f"model tags.models[{index}].digest"
        )
        digests.setdefault(name, []).append(digest)
    return digests


def _verified_identity(
    digests: Mapping[str, list[str]], *, role: str, model_tag: str, expected: str
) -> dict[str, Any]:
    live = digests.get(model_tag, [])
    if not live:
        raise ValueError(f"live /api/tags did not advertise {role} model {model_tag}")
    if len(live) != 1:
        raise ValueError(
            f"live /api/tags advertised duplicate digests for {role} model {model_tag}"
        )
    if live[0] != expected:
        raise ValueError(f"live /api/tags digest mismatch for {role} model {model_tag}")
    return {
        "reference": model_tag,
        "expected_digest": expected,
        "live_digest": live[0],
        "verified": True,
    }


def _bounded_timeout(
    budget: ExperimentBudget | None, maximum: float
) -> float:
    if budget is None:
        return maximum
    budget.check()
    return min(maximum, budget.remaining_seconds)


def _verify_endpoint(
    base_url: str,
    spec: ExperimentSpec,
    *,
    budget: ExperimentBudget | None = None,
) -> dict[str, Any]:
    timeout = _bounded_timeout(
        budget, min(float(spec.runtime.timeout_seconds), 30.0)
    )
    try:
        with httpx.Client(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
        ) as client:
            tags_response = client.get(f"{base_url.rstrip('/')}/api/tags")
            tags_response.raise_for_status()
            tags = _require_mapping(tags_response.json(), "model tags probe")
            digests = _model_digests(tags)
            models = {
                "target": _verified_identity(
                    digests,
                    role="target",
                    model_tag=spec.model.model_tag,
                    expected=spec.model.digest,
                ),
                "reflection": _verified_identity(
                    digests,
                    role="reflection",
                    model_tag=spec.reflection.model_tag,
                    expected=spec.reflection.digest,
                ),
            }
            version_response = client.get(f"{base_url.rstrip('/')}/api/version")
            version_response.raise_for_status()
            version_payload = _require_mapping(
                version_response.json(), "model version probe"
            )
            version = _require_string(
                version_payload.get("version"), "model version probe.version"
            )
    except httpx.HTTPError as exc:
        raise ModelSessionError("model identity probe failed") from exc
    except ValueError as exc:
        raise ModelSessionError(str(exc)) from exc
    return {"models": models, "api_version": version}


def _node_pool_command(
    connection: AKSConnection, subscription_id: str, *args: str
) -> tuple[str, ...]:
    assert connection.node_pool is not None
    return (
        "az",
        "aks",
        "nodepool",
        *args,
        "--resource-group",
        connection.resource_group,
        "--cluster-name",
        connection.cluster_name,
        "--name",
        connection.node_pool,
        "--subscription",
        subscription_id,
    )


def _run_az(
    args: tuple[str, ...],
    *,
    timeout: float,
    budget: ExperimentBudget | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=_bounded_timeout(budget, timeout),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ModelSessionError("AKS node pool command failed") from exc
    if result.returncode != 0:
        raise ModelSessionError("AKS node pool command failed")
    return result


def _node_pool_count(
    connection: AKSConnection,
    subscription_id: str,
    *,
    budget: ExperimentBudget | None = None,
) -> int:
    result = _run_az(
        _node_pool_command(
            connection,
            subscription_id,
            "show",
            "--query",
            "count",
            "--output",
            "tsv",
            "--only-show-errors",
        ),
        timeout=30,
        budget=budget,
    )
    try:
        count = int(result.stdout.strip())
    except ValueError as exc:
        raise ModelSessionError("AKS node pool count was not an integer") from exc
    if count < 0:
        raise ModelSessionError("AKS node pool count was negative")
    return count


def _scale_node_pool(
    connection: AKSConnection,
    subscription_id: str,
    count: int,
    *,
    budget: ExperimentBudget | None = None,
) -> None:
    _run_az(
        _node_pool_command(
            connection,
            subscription_id,
            "scale",
            "--node-count",
            str(count),
            "--only-show-errors",
        ),
        timeout=600,
        budget=budget,
    )


def _run_aks_preflight_command(
    args: tuple[str, ...], *, budget: ExperimentBudget | None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=_bounded_timeout(budget, 30.0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ModelSessionError("AKS preflight command failed") from exc


def _resolve_subscription(*, budget: ExperimentBudget | None) -> str:
    result = _run_aks_preflight_command(
        (
            "az", "account", "show", "--query", "id",
            "--output", "tsv", "--only-show-errors",
        ),
        budget=budget,
    )
    if result.returncode != 0:
        raise ModelSessionError("AKS subscription lookup failed")
    try:
        return str(UUID(result.stdout.strip()))
    except ValueError as exc:
        raise ModelSessionError("AKS subscription lookup returned an invalid ID") from exc


def _aks_preflight_http(
    url: str, *, budget: ExperimentBudget | None
) -> Mapping[str, Any]:
    with httpx.Client(
        trust_env=False,
        follow_redirects=False,
        timeout=httpx.Timeout(_bounded_timeout(budget, 5.0)),
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        return _require_mapping(response.json(), "AKS models probe")


def _open_aks_forward(
    spec: ExperimentSpec,
    connection: AKSConnection,
    work_dir: Path,
    *,
    subscription_id: str,
    budget: ExperimentBudget | None = None,
) -> AKSPortForward:
    serving = AKSForwardTarget(
        resource_group=connection.resource_group,
        cluster_name=connection.cluster_name,
        namespace=connection.namespace,
        service=connection.service,
        model=spec.model.model_tag,
    )
    startup_limit = _bounded_timeout(
        budget, min(float(spec.wall_clock_seconds), 600.0)
    )
    deadline = time.monotonic() + startup_limit
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelSessionError("AKS model readiness deadline exhausted")
        forward = AKSPortForward(
            serving,
            workspace_dir=work_dir,
            command_runner=lambda args: _run_aks_preflight_command(
                (*args, "--subscription", subscription_id) if args[0] == "az" else args,
                budget=budget,
            ),
            http_get_json=lambda url: _aks_preflight_http(url, budget=budget),
            port_forward_ready_timeout_seconds=_bounded_timeout(
                budget, min(10.0, remaining)
            ),
        )
        try:
            forward.__enter__()
        except AKSPreflightTransientError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelSessionError(
                    "AKS model readiness deadline exhausted"
                ) from exc
            time.sleep(min(1.0, remaining))
            continue
        return forward


@contextmanager
def model_session(
    spec: ExperimentSpec,
    work_dir: Path,
    *,
    allow_capacity_changes: bool = False,
    budget: ExperimentBudget | None = None,
) -> Iterator[ResolvedModelSession]:
    """Resolve one endpoint for the campaign.

    ``budget`` clips bootstrap commands, probes, and readiness waits. Owned
    process and capacity cleanup deliberately runs outside that deadline.
    """
    directory = Path(work_dir)
    artifact_path = directory / "model-session.json"
    connection = spec.serving
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "experiment_fingerprint": spec.fingerprint,
        "status": "preparing",
        "connection": {
            "backend": connection.backend,
            "owned_process": isinstance(connection, AKSConnection),
        },
        "runtime": {
            "backend": spec.runtime.backend,
            "korvid_revision": spec.runtime.korvid_revision,
        },
    }
    if isinstance(connection, LoopbackConnection):
        evidence["connection"]["configured_base_url"] = connection.base_url
    else:
        evidence["connection"].update(
            {
                "resource_group": connection.resource_group,
                "cluster_name": connection.cluster_name,
                "namespace": connection.namespace,
                "service": connection.service,
            }
        )
    write_json_artifact(artifact_path, evidence)

    forward: AKSPortForward | None = None
    base_url: str | None = None
    original_count: int | None = None
    subscription_id: str | None = None
    scaled_from_zero = False
    cleanup_errors: list[Exception] = []

    try:
        if isinstance(connection, AKSConnection):
            subscription_id = _resolve_subscription(budget=budget)
            evidence["connection"]["subscription_id"] = subscription_id
            write_json_artifact(artifact_path, evidence)
            if connection.node_pool is not None:
                original_count = _node_pool_count(
                    connection, subscription_id, budget=budget
                )
                evidence["node_pool"] = {
                    "name": connection.node_pool,
                    "original_count": original_count,
                    "capacity_changed": False,
                    "final_count": None,
                }
                if original_count == 0:
                    if not allow_capacity_changes:
                        raise PermissionError(
                            "allow_capacity_changes=True is required to scale a zero-node model pool"
                        )
                    scaled_from_zero = True
                    evidence["node_pool"]["capacity_changed"] = True
                    _scale_node_pool(connection, subscription_id, 1, budget=budget)
            forward = _open_aks_forward(
                spec, connection, directory,
                subscription_id=subscription_id, budget=budget,
            )
            base_url = forward.base_url
            evidence["connection"]["resolved_base_url"] = base_url
        else:
            base_url = connection.base_url

        verification = _verify_endpoint(base_url, spec, budget=budget)
        evidence.update(verification)
        evidence["rechecks"] = []
        evidence["status"] = "ready"
        write_json_artifact(artifact_path, evidence)

        def verify_again() -> dict[str, Any]:
            try:
                recheck = _verify_endpoint(base_url, spec, budget=budget)
            except (BudgetExhausted, ModelSessionError) as exc:
                evidence["rechecks"].append(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                write_json_artifact(artifact_path, evidence)
                raise
            entry = {"status": "verified", **recheck}
            evidence["rechecks"].append(entry)
            write_json_artifact(artifact_path, evidence)
            if recheck["api_version"] != verification["api_version"]:
                entry["status"] = "failed"
                entry["reason"] = "api_version_changed"
                write_json_artifact(artifact_path, evidence)
                raise ModelSessionError(
                    "Ollama API version changed during the model session"
                )
            return recheck

        resolved = ResolvedModelSession(
            base_url=base_url,
            evidence=evidence,
            _verify_callback=verify_again,
        )
        yield resolved
    finally:
        primary = sys.exc_info()[1]
        if primary is not None:
            evidence["status"] = "failed"
            evidence["error_type"] = type(primary).__name__
        if forward is not None:
            try:
                forward.__exit__(
                    type(primary) if primary is not None else None,
                    primary,
                    primary.__traceback__ if primary is not None else None,
                )
            except (AKSPortForwardError, OSError, subprocess.SubprocessError) as exc:
                cleanup_errors.append(exc)
        if (
            isinstance(connection, AKSConnection)
            and connection.node_pool is not None
            and original_count is not None
        ):
            assert subscription_id is not None
            if scaled_from_zero:
                try:
                    _scale_node_pool(connection, subscription_id, 0)
                except ModelSessionError as exc:
                    cleanup_errors.append(exc)
            try:
                evidence["node_pool"]["final_count"] = _node_pool_count(
                    connection, subscription_id
                )
            except ModelSessionError as exc:
                cleanup_errors.append(exc)
            else:
                if evidence["node_pool"]["final_count"] != original_count:
                    cleanup_errors.append(
                        ModelSessionError(
                            "AKS node pool was not restored to its original count"
                        )
                    )
        if primary is None and not cleanup_errors:
            evidence["status"] = "closed"
        elif primary is None:
            evidence["status"] = "cleanup_failed"
        try:
            write_json_artifact(artifact_path, evidence)
        except OSError as exc:
            cleanup_errors.append(exc)
        if primary is not None and cleanup_errors:
            raise BaseExceptionGroup(
                "model session failed and cleanup also failed",
                [primary, *cleanup_errors],
            )
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0]
        if cleanup_errors:
            raise ExceptionGroup("model session cleanup failed", cleanup_errors)
