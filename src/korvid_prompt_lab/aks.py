from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol, Self

from .contracts import AKSPortForwardServing, _require_mapping

_MAX_BUFFERED_OUTPUT_BYTES = 65536
_READER_THREAD_NAME = "korvid-aks-port-forward-reader"
_OUTPUT_POLL_INTERVAL_SECONDS = 0.1
_READER_JOIN_TIMEOUT_SECONDS = 5.0
_READ_CHUNK_BYTES = 4096


class AKSPortForwardError(RuntimeError):
    """Raised when AKS discovery or port-forward preparation fails."""


class AKSPortForwardTimeoutError(AKSPortForwardError):
    """Raised when the AKS port-forward does not become ready before a deadline."""


class _CommandResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


class _CommandRunner(Protocol):
    def __call__(self, args: tuple[str, ...]) -> _CommandResult: ...


class _PortForwardProcess(Protocol):
    def read_output(self) -> str: ...

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def close(self) -> None: ...


class _ProcessLauncher(Protocol):
    def __call__(self, args: tuple[str, ...]) -> _PortForwardProcess: ...


class _HttpGetJson(Protocol):
    def __call__(self, url: str) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class _SubprocessPortForward:
    """Owns one port-forward child and keeps its merged output pipe drained.

    ``kubectl port-forward`` logs a line per proxied connection, so an unread pipe
    fills within a few thousand requests and blocks the child mid-campaign. A daemon
    reader keeps the pipe empty for the whole forward lifetime and retains only the
    most recent bytes, which is all readiness parsing needs.
    """

    process: subprocess.Popen[bytes]
    max_buffered_bytes: int = _MAX_BUFFERED_OUTPUT_BYTES
    output_poll_interval_seconds: float = _OUTPUT_POLL_INTERVAL_SECONDS
    reader_join_timeout_seconds: float = _READER_JOIN_TIMEOUT_SECONDS
    _lock: threading.Lock = field(init=False, repr=False, compare=False, default_factory=threading.Lock)
    _buffer: bytearray = field(init=False, repr=False, compare=False, default_factory=bytearray)
    _output_ready: threading.Event = field(
        init=False, repr=False, compare=False, default_factory=threading.Event
    )
    _closing: threading.Event = field(
        init=False, repr=False, compare=False, default_factory=threading.Event
    )
    _failure: BaseException | None = field(init=False, repr=False, compare=False, default=None)
    _reader: threading.Thread | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        stream = self.process.stdout
        if stream is None:
            return
        reader = threading.Thread(
            target=self._drain_output,
            args=(stream,),
            name=_READER_THREAD_NAME,
            daemon=True,
        )
        self._reader = reader
        reader.start()

    def read_output(self) -> str:
        self._output_ready.wait(self.output_poll_interval_seconds)
        with self._lock:
            chunk = bytes(self._buffer)
            self._buffer.clear()
            self._output_ready.clear()
            failure = None if chunk else self._failure
            if failure is not None:
                self._failure = None
        if failure is not None and not self._closing.is_set():
            raise AKSPortForwardError("port-forward output reader failed") from failure
        return chunk.decode("utf-8", errors="replace")

    def poll(self) -> int | None:
        return self.process.poll()

    def terminate(self) -> None:
        self.process.terminate()

    def kill(self) -> None:
        self.process.kill()

    def wait(self, timeout: float | None = None) -> int:
        if timeout is None:
            return self.process.wait()
        return self.process.wait(timeout=timeout)

    def close(self) -> None:
        self._closing.set()
        reader = self._reader
        if reader is None or reader is threading.current_thread():
            return
        reader.join(timeout=self.reader_join_timeout_seconds)
        if not reader.is_alive():
            self._reader = None

    def _drain_output(self, stream: IO[bytes]) -> None:
        try:
            fileno = stream.fileno()
            while True:
                chunk = os.read(fileno, _READ_CHUNK_BYTES)
                if not chunk:
                    return
                self._store_output(chunk)
        except (OSError, ValueError) as exc:
            self._record_failure(exc)
        finally:
            with self._lock:
                self._output_ready.set()
            with suppress(OSError, ValueError):
                stream.close()

    def _store_output(self, chunk: bytes) -> None:
        with self._lock:
            self._buffer.extend(chunk)
            overflow = len(self._buffer) - self.max_buffered_bytes
            if overflow > 0:
                del self._buffer[:overflow]
            self._output_ready.set()

    def _record_failure(self, exc: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = exc


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
        bufsize=0,
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
    monotonic_clock: Callable[[], float] = time.monotonic
    port_forward_ready_timeout_seconds: float = 10.0
    cleanup_wait_timeout_seconds: float = 5.0
    _process: _PortForwardProcess | None = field(init=False, default=None)
    _kubeconfig_path: Path | None = field(init=False, default=None)
    _base_url: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.port_forward_ready_timeout_seconds <= 0:
            raise ValueError("port_forward_ready_timeout_seconds must be positive")
        if self.cleanup_wait_timeout_seconds <= 0:
            raise ValueError("cleanup_wait_timeout_seconds must be positive")

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise AKSPortForwardError("port-forward is not active")
        return self._base_url

    def __enter__(self) -> Self:
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
        deadline = self.monotonic_clock() + self.port_forward_ready_timeout_seconds
        output = ""
        while True:
            output += self._process.read_output()
            match = re.search(r"Forwarding from 127\.0\.0\.1:(\d+) -> \d+", output)
            if match is not None:
                return int(match.group(1))
            returncode = self._process.poll()
            if returncode is not None:
                raise AKSPortForwardError("port-forward exited before it became ready")
            if self.monotonic_clock() >= deadline:
                raise AKSPortForwardTimeoutError("timed out waiting for port-forward readiness")
            if len(output) > 4096:
                output = output[-4096:]

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
            self._cleanup_process(self._process)
            self._process = None
        self._base_url = None
        if self._kubeconfig_path is not None:
            self._kubeconfig_path.unlink(missing_ok=True)
            self._kubeconfig_path = None

    def _cleanup_process(self, process: _PortForwardProcess) -> None:
        try:
            self._stop_process(process)
        finally:
            process.close()

    def _stop_process(self, process: _PortForwardProcess) -> None:
        if process.poll() is not None:
            self._reap_process(process)
            return
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        if self._wait_for_process_exit(process, self.cleanup_wait_timeout_seconds):
            self._reap_process(process)
            return
        try:
            process.kill()
        except ProcessLookupError:
            pass
        if self._wait_for_process_exit(process, self.cleanup_wait_timeout_seconds):
            self._reap_process(process)
            return
        if process.poll() is not None:
            self._reap_process(process)

    def _wait_for_process_exit(self, process: _PortForwardProcess, timeout: float) -> bool:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _reap_process(self, process: _PortForwardProcess) -> None:
        process.wait()
