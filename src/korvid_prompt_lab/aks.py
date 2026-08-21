from __future__ import annotations

import json
import re
import select
import subprocess
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from .contracts import AKSPortForwardServing, _require_mapping


class AKSPortForwardError(RuntimeError):
    """Raised when AKS discovery or port-forward preparation fails."""


class _CommandResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


class _CommandRunner(Protocol):
    def __call__(self, args: tuple[str, ...]) -> _CommandResult: ...


class _PortForwardProcess(Protocol):
    def read_line(self) -> str: ...

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class _ProcessLauncher(Protocol):
    def __call__(self, args: tuple[str, ...]) -> _PortForwardProcess: ...


class _HttpGetJson(Protocol):
    def __call__(self, url: str) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class _SubprocessPortForward:
    process: subprocess.Popen[str]

    def read_line(self) -> str:
        if self.process.stdout is None:
            return ""
        ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
        if not ready:
            return ""
        return self.process.stdout.readline()

    def poll(self) -> int | None:
        return self.process.poll()

    def terminate(self) -> None:
        self.process.terminate()

    def wait(self, timeout: float | None = None) -> int:
        if timeout is None:
            return self.process.wait()
        return self.process.wait(timeout=timeout)


def _run_command(args: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )


def _launch_process(args: tuple[str, ...]) -> _SubprocessPortForward:
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return _SubprocessPortForward(process)


def _http_get_json(url: str) -> Mapping[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return _require_mapping(json.load(response), "models probe")


@dataclass(slots=True)
class AKSPortForward:
    serving: AKSPortForwardServing
    workspace_dir: Path | str = field(default_factory=Path.cwd)
    command_runner: _CommandRunner = _run_command
    process_launcher: _ProcessLauncher = _launch_process
    http_get_json: _HttpGetJson = _http_get_json
    _process: _PortForwardProcess | None = field(init=False, default=None)
    _kubeconfig_path: Path | None = field(init=False, default=None)
    _base_url: str | None = field(init=False, default=None)

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise AKSPortForwardError("port-forward is not active")
        return self._base_url

    def __enter__(self) -> AKSPortForward:
        self._kubeconfig_path = self._create_kubeconfig_path()
        try:
            self._validate_cluster()
            self._write_kubeconfig()
            service_port = self._validate_service()
            self._validate_endpoints()
            self._start_port_forward(service_port)
            self._probe_models()
            return self
        except BaseException:
            self._cleanup()
            raise

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._cleanup()
        return None

    def _create_kubeconfig_path(self) -> Path:
        workspace_dir = Path(self.workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        kubeconfig_path = workspace_dir / f".kubeconfig-{uuid.uuid4().hex}.yaml"
        kubeconfig_path.touch(mode=0o600, exist_ok=False)
        return kubeconfig_path

    def _validate_cluster(self) -> None:
        result = self.command_runner(
            (
                "az",
                "aks",
                "show",
                "--resource-group",
                self.serving.resource_group,
                "--name",
                self.serving.cluster_name,
                "--output",
                "json",
                "--only-show-errors",
            )
        )
        if result.returncode != 0:
            raise AKSPortForwardError("AKS cluster lookup failed")
        payload = self._load_json(result.stdout, "AKS cluster metadata")
        if payload.get("name") != self.serving.cluster_name or payload.get("resourceGroup") != self.serving.resource_group:
            raise AKSPortForwardError("AKS cluster validation failed")

    def _write_kubeconfig(self) -> None:
        assert self._kubeconfig_path is not None
        result = self.command_runner(
            (
                "az",
                "aks",
                "get-credentials",
                "--resource-group",
                self.serving.resource_group,
                "--name",
                self.serving.cluster_name,
                "--file",
                str(self._kubeconfig_path),
                "--overwrite-existing",
                "--only-show-errors",
            )
        )
        if result.returncode != 0:
            raise AKSPortForwardError("AKS kubeconfig acquisition failed")

    def _validate_service(self) -> int:
        payload = self._kubectl_get("service", self.serving.service, "AKS Service lookup failed")
        metadata = _require_mapping(payload.get("metadata"), "service.metadata")
        if metadata.get("name") != self.serving.service or metadata.get("namespace") != self.serving.namespace:
            raise AKSPortForwardError("AKS Service validation failed")
        spec = _require_mapping(payload.get("spec"), "service.spec")
        raw_ports = spec.get("ports")
        if not isinstance(raw_ports, list) or not raw_ports:
            raise AKSPortForwardError("AKS Service does not expose a port")
        for item in raw_ports:
            port_mapping = _require_mapping(item, "service port")
            if port_mapping.get("protocol", "TCP") != "TCP":
                continue
            port = port_mapping.get("port")
            if isinstance(port, int) and port > 0:
                return port
        raise AKSPortForwardError("AKS Service does not expose a TCP port")

    def _validate_endpoints(self) -> None:
        payload = self._kubectl_get("endpoints", self.serving.service, "AKS endpoints lookup failed")
        metadata = _require_mapping(payload.get("metadata"), "endpoints.metadata")
        if metadata.get("name") != self.serving.service or metadata.get("namespace") != self.serving.namespace:
            raise AKSPortForwardError("AKS endpoints validation failed")
        raw_subsets = payload.get("subsets")
        if not isinstance(raw_subsets, list) or not raw_subsets:
            raise AKSPortForwardError("AKS Service must expose Ready endpoints")
        ready_addresses = 0
        for item in raw_subsets:
            subset = _require_mapping(item, "endpoints subset")
            addresses = subset.get("addresses", [])
            not_ready = subset.get("notReadyAddresses", [])
            if not isinstance(addresses, list) or not isinstance(not_ready, list):
                raise AKSPortForwardError("AKS endpoints payload is invalid")
            if not_ready:
                raise AKSPortForwardError("AKS Service must expose Ready endpoints only")
            ready_addresses += len(addresses)
        if ready_addresses == 0:
            raise AKSPortForwardError("AKS Service must expose Ready endpoints")

    def _kubectl_get(self, resource_type: str, resource_name: str, error_message: str) -> Mapping[str, Any]:
        assert self._kubeconfig_path is not None
        result = self.command_runner(
            (
                "kubectl",
                "--kubeconfig",
                str(self._kubeconfig_path),
                "--namespace",
                self.serving.namespace,
                "get",
                resource_type,
                resource_name,
                "--output",
                "json",
            )
        )
        if result.returncode != 0:
            raise AKSPortForwardError(error_message)
        return self._load_json(result.stdout, resource_type)

    def _start_port_forward(self, service_port: int) -> None:
        assert self._kubeconfig_path is not None
        self._process = self.process_launcher(
            (
                "kubectl",
                "--kubeconfig",
                str(self._kubeconfig_path),
                "--namespace",
                self.serving.namespace,
                "port-forward",
                "--address",
                "127.0.0.1",
                f"service/{self.serving.service}",
                f":{service_port}",
            )
        )
        local_port = self._wait_for_port_forward()
        self._base_url = f"http://127.0.0.1:{local_port}"

    def _wait_for_port_forward(self) -> int:
        assert self._process is not None
        while True:
            line = self._process.read_line()
            match = re.search(r"Forwarding from 127\.0\.0\.1:(\d+) -> \d+", line)
            if match is not None:
                return int(match.group(1))
            returncode = self._process.poll()
            if returncode is not None:
                raise AKSPortForwardError("port-forward exited before it became ready")

    def _probe_models(self) -> None:
        try:
            payload = _require_mapping(self.http_get_json(f"{self.base_url}/v1/models"), "models probe")
        except AKSPortForwardError:
            raise
        except Exception as exc:
            raise AKSPortForwardError("model probe failed") from exc
        raw_models = payload.get("data")
        if not isinstance(raw_models, list):
            raise AKSPortForwardError("model probe returned an invalid payload")
        model_ids = {
            item.get("id")
            for item in (_require_mapping(model, "model entry") for model in raw_models)
            if isinstance(item.get("id"), str)
        }
        if self.serving.model not in model_ids:
            raise AKSPortForwardError(f"model probe did not advertise model {self.serving.model}")

    def _load_json(self, text: str, context: str) -> Mapping[str, Any]:
        try:
            return _require_mapping(json.loads(text), context)
        except json.JSONDecodeError as exc:
            raise AKSPortForwardError(f"{context} returned invalid JSON") from exc
        except ValueError as exc:
            raise AKSPortForwardError(f"{context} returned an invalid payload") from exc

    def _cleanup(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
            self._process.wait()
            self._process = None
        self._base_url = None
        if self._kubeconfig_path is not None:
            self._kubeconfig_path.unlink(missing_ok=True)
            self._kubeconfig_path = None
