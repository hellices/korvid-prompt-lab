from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.aks import (
    AKSPortForward,
    AKSPortForwardError,
    AKSPortForwardTimeoutError,
    _CommandResult,
    _CommandRunner,
    _HttpGetJson,
    _ProcessLauncher,
)
from korvid_prompt_lab.contracts import AKSPortForwardServing


@dataclass(slots=True)
class FakeCompletedCommand:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeCommandRunner(_CommandRunner):
    def __init__(self, results: list[FakeCompletedCommand]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args: tuple[str, ...]) -> _CommandResult:
        self.calls.append(tuple(args))
        if not self._results:
            raise AssertionError(f"unexpected command: {args}")
        return self._results.pop(0)


class FakeProcess:
    def __init__(
        self,
        *,
        chunks: tuple[str, ...] = (),
        lines: tuple[str, ...] = (),
        poll_values: tuple[int | None, ...] = (),
        wait_returncode: int = 0,
        wait_effects: tuple[int | BaseException | None, ...] = (),
    ) -> None:
        self._chunks = list(chunks)
        self._lines = list(lines)
        self._poll_values = list(poll_values)
        self._wait_returncode = wait_returncode
        self._wait_effects = list(wait_effects)
        self._current_returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def read_line(self) -> str:
        if self._lines:
            return self._lines.pop(0)
        return ""

    def read_output(self) -> str:
        if self._chunks:
            return self._chunks.pop(0)
        return self.read_line()

    def poll(self) -> int | None:
        if self._poll_values:
            self._current_returncode = self._poll_values.pop(0)
            return self._current_returncode
        return self._current_returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self._wait_effects:
            effect = self._wait_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            if effect is not None:
                self._current_returncode = effect
                return effect
        return self._wait_returncode


class FakeClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = list(values)
        self.calls = 0

    def monotonic(self) -> float:
        self.calls += 1
        if self._values:
            return self._values.pop(0)
        raise AssertionError("unexpected monotonic clock read")


class PartialOutputProcess(FakeProcess):
    def read_line(self) -> str:
        raise AssertionError("read_line should not be used for partial port-forward output")


class FakeProcessLauncher(_ProcessLauncher):
    def __init__(self, processes: list[FakeProcess]) -> None:
        self._processes = list(processes)
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args: tuple[str, ...]) -> FakeProcess:
        self.calls.append(tuple(args))
        if not self._processes:
            raise AssertionError(f"unexpected process launch: {args}")
        return self._processes.pop(0)


class FakeHttpGet(_HttpGetJson):
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self._payloads = list(payloads)
        self.urls: list[str] = []

    def __call__(self, url: str) -> Mapping[str, Any]:
        self.urls.append(url)
        if not self._payloads:
            raise AssertionError(f"unexpected probe: {url}")
        return self._payloads.pop(0)


def _serving() -> AKSPortForwardServing:
    return AKSPortForwardServing(
        backend="aks_port_forward",
        resource_group="rg-team-a",
        cluster_name="cluster-one",
        namespace="korvid",
        service="korvid-api",
        model="qwen3-4b",
    )


def _successful_results(
    *,
    service_name: str = "korvid-api",
    namespace: str = "korvid",
    cluster_name: str = "cluster-one",
    resource_group: str = "rg-team-a",
    ready_addresses: tuple[str, ...] = ("10.0.0.12",),
    not_ready_addresses: tuple[str, ...] = (),
    service_port: int = 8080,
) -> list[FakeCompletedCommand]:
    return [
        FakeCompletedCommand(
            stdout=json.dumps({"name": cluster_name, "resourceGroup": resource_group}),
        ),
        FakeCompletedCommand(stdout="merged"),
        FakeCompletedCommand(
            stdout=json.dumps(
                {
                    "metadata": {"name": service_name, "namespace": namespace},
                    "spec": {"ports": [{"port": service_port, "protocol": "TCP"}]},
                }
            )
        ),
        FakeCompletedCommand(
            stdout=json.dumps(
                {
                    "metadata": {"name": service_name, "namespace": namespace},
                    "subsets": [
                        {
                            "addresses": [{"ip": address} for address in ready_addresses],
                            "notReadyAddresses": [{"ip": address} for address in not_ready_addresses],
                        }
                    ],
                }
            )
        ),
    ]


def test_aks_port_forward_rejects_missing_resource_group(tmp_path: Path) -> None:
    commands = FakeCommandRunner(
        [FakeCompletedCommand(returncode=3, stderr="ResourceGroupNotFound")]
    )

    with pytest.raises(AKSPortForwardError, match="AKS cluster lookup failed"), AKSPortForward(
        _serving(),
        workspace_dir=tmp_path,
        command_runner=commands,
    ):
        pytest.fail("context should not open")

    assert len(commands.calls) == 1
    assert commands.calls[0] == (
        "az",
        "aks",
        "show",
        "--resource-group",
        "rg-team-a",
        "--name",
        "cluster-one",
        "--output",
        "json",
        "--only-show-errors",
    )


def test_aks_port_forward_rejects_wrong_cluster_response(tmp_path: Path) -> None:
    commands = FakeCommandRunner(_successful_results(cluster_name="cluster-two"))

    with pytest.raises(AKSPortForwardError, match="AKS cluster validation failed"), AKSPortForward(
        _serving(),
        workspace_dir=tmp_path,
        command_runner=commands,
    ):
        pytest.fail("context should not open")


def test_aks_port_forward_rejects_non_ready_endpoints(tmp_path: Path) -> None:
    commands = FakeCommandRunner(
        _successful_results(not_ready_addresses=("10.0.0.99",), ready_addresses=())
    )

    with pytest.raises(AKSPortForwardError, match="Ready endpoints"), AKSPortForward(
        _serving(),
        workspace_dir=tmp_path,
        command_runner=commands,
    ):
        pytest.fail("context should not open")


def test_aks_port_forward_rejects_port_forward_early_exit(tmp_path: Path) -> None:
    commands = FakeCommandRunner(_successful_results())
    process = FakeProcess(poll_values=(9,))
    launcher = FakeProcessLauncher([process])

    with pytest.raises(AKSPortForwardError, match="port-forward exited before it became ready"), AKSPortForward(
        _serving(),
        workspace_dir=tmp_path,
        command_runner=commands,
        process_launcher=launcher,
    ):
        pytest.fail("context should not open")

    assert process.terminate_calls == 0
    assert process.wait_calls == [None]


def test_aks_port_forward_stalled_live_process_times_out_and_cleans_up(tmp_path: Path) -> None:
    commands = FakeCommandRunner(_successful_results())
    process = FakeProcess(poll_values=(None, None, None))
    launcher = FakeProcessLauncher([process])
    clock = FakeClock((100.0, 100.2, 100.5, 101.2))
    cleanup_timeout = 0.5

    with pytest.raises(
        AKSPortForwardTimeoutError, match="timed out waiting for port-forward readiness"
    ), AKSPortForward(
        _serving(),
        workspace_dir=tmp_path,
        command_runner=commands,
        process_launcher=launcher,
        monotonic_clock=clock.monotonic,
        port_forward_ready_timeout_seconds=1.0,
        cleanup_wait_timeout_seconds=cleanup_timeout,
    ):
        pytest.fail("context should not open")

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_calls == [cleanup_timeout, None]


def test_aks_port_forward_partial_output_still_honors_timeout(tmp_path: Path) -> None:
    commands = FakeCommandRunner(_successful_results())
    process = PartialOutputProcess(
        chunks=("Forwarding from 127.0.0.1:", "", ""),
        poll_values=(None, None, None),
    )
    launcher = FakeProcessLauncher([process])
    clock = FakeClock((200.0, 200.3, 200.7, 201.2))
    cleanup_timeout = 0.5

    with pytest.raises(
        AKSPortForwardTimeoutError, match="timed out waiting for port-forward readiness"
    ), AKSPortForward(
        _serving(),
        workspace_dir=tmp_path,
        command_runner=commands,
        process_launcher=launcher,
        monotonic_clock=clock.monotonic,
        port_forward_ready_timeout_seconds=1.0,
        cleanup_wait_timeout_seconds=cleanup_timeout,
    ):
        pytest.fail("context should not open")

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_calls == [cleanup_timeout, None]


def test_aks_port_forward_rejects_model_probe_mismatch(tmp_path: Path) -> None:
    commands = FakeCommandRunner(_successful_results())
    process = FakeProcess(lines=("Forwarding from 127.0.0.1:41001 -> 8080\n",))
    launcher = FakeProcessLauncher([process])
    http_get = FakeHttpGet([{"data": [{"id": "qwen3-14b"}]}])
    cleanup_timeout = 0.5

    with pytest.raises(AKSPortForwardError, match="did not advertise model qwen3-4b"), AKSPortForward(
        _serving(),
        workspace_dir=tmp_path,
        command_runner=commands,
        process_launcher=launcher,
        http_get_json=http_get,
        cleanup_wait_timeout_seconds=cleanup_timeout,
    ):
        pytest.fail("context should not open")

    assert http_get.urls == ["http://127.0.0.1:41001/v1/models"]
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_calls == [cleanup_timeout, None]


def test_aks_port_forward_binds_loopback_uses_unique_kubeconfigs_and_cleans_only_owned_processes(
    tmp_path: Path,
) -> None:
    commands = FakeCommandRunner(_successful_results() + _successful_results())
    first_process = FakeProcess(lines=("Forwarding from 127.0.0.1:41001 -> 8080\n",))
    second_process = FakeProcess(lines=("Forwarding from 127.0.0.1:41002 -> 8080\n",))
    launcher = FakeProcessLauncher([first_process, second_process])
    cleanup_timeout = 0.5
    http_get = FakeHttpGet(
        [
            {"data": [{"id": "qwen3-4b"}]},
            {"data": [{"id": "qwen3-4b"}]},
        ]
    )

    with AKSPortForward(
        _serving(),
        workspace_dir=tmp_path,
        command_runner=commands,
        process_launcher=launcher,
        http_get_json=http_get,
        cleanup_wait_timeout_seconds=cleanup_timeout,
    ) as first_forward:
        assert first_forward.base_url == "http://127.0.0.1:41001"
        assert first_process.terminate_calls == 0
        assert second_process.terminate_calls == 0

    first_kubeconfig = Path(commands.calls[1][commands.calls[1].index("--file") + 1])
    assert not first_kubeconfig.exists()
    assert first_process.terminate_calls == 1
    assert first_process.kill_calls == 0
    assert first_process.wait_calls == [cleanup_timeout, None]
    assert second_process.terminate_calls == 0

    with AKSPortForward(
        _serving(),
        workspace_dir=tmp_path,
        command_runner=commands,
        process_launcher=launcher,
        http_get_json=http_get,
        cleanup_wait_timeout_seconds=cleanup_timeout,
    ) as second_forward:
        assert second_forward.base_url == "http://127.0.0.1:41002"

    second_kubeconfig = Path(commands.calls[5][commands.calls[5].index("--file") + 1])
    assert first_kubeconfig != second_kubeconfig
    assert not second_kubeconfig.exists()
    assert second_process.terminate_calls == 1
    assert second_process.kill_calls == 0
    assert second_process.wait_calls == [cleanup_timeout, None]
    assert launcher.calls[0][-4:] == ("--address", "127.0.0.1", "service/korvid-api", ":8080")
    assert launcher.calls[1][-4:] == ("--address", "127.0.0.1", "service/korvid-api", ":8080")


def test_aks_port_forward_cleans_up_on_base_exception_during_startup(tmp_path: Path) -> None:
    commands = FakeCommandRunner(_successful_results())
    process = FakeProcess(lines=("Forwarding from 127.0.0.1:41001 -> 8080\n",))
    launcher = FakeProcessLauncher([process])
    cleanup_timeout = 0.5

    def raising_probe(_url: str) -> Mapping[str, Any]:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt), AKSPortForward(
        _serving(),
        workspace_dir=tmp_path,
        command_runner=commands,
        process_launcher=launcher,
        http_get_json=cast(_HttpGetJson, raising_probe),
        cleanup_wait_timeout_seconds=cleanup_timeout,
    ):
        pytest.fail("context should not open")

    kubeconfig = Path(commands.calls[1][commands.calls[1].index("--file") + 1])
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_calls == [cleanup_timeout, None]
    assert not kubeconfig.exists()


def test_aks_port_forward_cleanup_kills_ignores_terminate_process(tmp_path: Path) -> None:
    commands = FakeCommandRunner(_successful_results())
    process = FakeProcess(
        lines=("Forwarding from 127.0.0.1:41001 -> 8080\n",),
        wait_effects=(
            subprocess.TimeoutExpired(cmd=("kubectl", "port-forward"), timeout=0.25),
            137,
        ),
    )
    launcher = FakeProcessLauncher([process])
    http_get = FakeHttpGet([{"data": [{"id": "qwen3-4b"}]}])

    with AKSPortForward(
        _serving(),
        workspace_dir=tmp_path,
        command_runner=commands,
        process_launcher=launcher,
        http_get_json=http_get,
        cleanup_wait_timeout_seconds=0.25,
    ):
        pass

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == [0.25, 0.25, None]
