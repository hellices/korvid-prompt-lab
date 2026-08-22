from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.bridge import (
    BRIDGE_TIMEOUT_ENV,
    WORKER_MODULE_PATH,
    BridgeConfigurationError,
    LauncherTerminated,
    _run_worker_process_group,
    build_worker_invocation,
    main,
    resolve_source_root,
)

ROOT = Path(__file__).resolve().parents[1]


def _fake_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "korvid-checkout"
    (root / "src" / "korvid").mkdir(parents=True)
    (root / "tests" / "evals").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='korvid'\n", encoding="utf-8")
    (root / "src" / "korvid" / "__init__.py").write_text("", encoding="utf-8")
    (root / "tests" / "evals" / "operation_app.py").write_text("", encoding="utf-8")
    return root


def test_korvid_bridge_console_entry_point_is_declared_and_loadable() -> None:
    """RED proof: the shipped AKS campaign invokes `korvid-bridge`, which must exist."""
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = manifest["project"]["scripts"]

    assert scripts.get("korvid-bridge") == "korvid_prompt_lab.bridge:main"

    from importlib.metadata import entry_points

    console_scripts = {entry.name: entry for entry in entry_points(group="console_scripts")}
    assert "korvid-bridge" in console_scripts, "korvid-bridge is not an installed console script"
    assert console_scripts["korvid-bridge"].load() is main


def test_korvid_bridge_console_script_is_executable() -> None:
    script = Path(sys.executable).parent / "korvid-bridge"
    assert script.exists(), f"console script not installed at {script}"

    completed = subprocess.run(
        [str(script), "--help"], capture_output=True, text=True, timeout=120, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert "--request" in completed.stdout
    assert "--response" in completed.stdout


def test_resolve_source_root_requires_the_runtime_policy_variable() -> None:
    with pytest.raises(BridgeConfigurationError, match="KORVID_SOURCE_ROOT"):
        resolve_source_root({})
    with pytest.raises(BridgeConfigurationError, match="KORVID_SOURCE_ROOT"):
        resolve_source_root({"KORVID_SOURCE_ROOT": "   "})


def test_resolve_source_root_rejects_a_directory_that_is_not_a_korvid_checkout(tmp_path: Path) -> None:
    empty = tmp_path / "not-korvid"
    empty.mkdir()

    with pytest.raises(BridgeConfigurationError, match="Korvid source checkout"):
        resolve_source_root({"KORVID_SOURCE_ROOT": str(empty)})

    with pytest.raises(BridgeConfigurationError, match="Korvid source checkout"):
        resolve_source_root({"KORVID_SOURCE_ROOT": str(tmp_path / "missing")})


def test_resolve_source_root_returns_the_resolved_checkout(tmp_path: Path) -> None:
    root = _fake_source_root(tmp_path)

    resolved = resolve_source_root({"KORVID_SOURCE_ROOT": f"{root}/./"})

    assert resolved == root.resolve()


def test_build_worker_invocation_runs_the_worker_in_the_checkout_uv_environment(tmp_path: Path) -> None:
    root = _fake_source_root(tmp_path)

    command, env = build_worker_invocation(
        source_root=root,
        request_path=tmp_path / "request.json",
        response_path=tmp_path / "response.json",
        env={"KORVID_UV_BIN": "/opt/bin/uv", "PATH": "/usr/bin"},
    )

    assert command == (
        "/opt/bin/uv",
        "run",
        "--project",
        str(root),
        "--no-sync",
        "python",
        str(WORKER_MODULE_PATH),
        "--request",
        str(tmp_path / "request.json"),
        "--response",
        str(tmp_path / "response.json"),
        "--profile",
        "small",
        "--approval-timeout",
        "5.0",
        "--turn-timeout",
        "120.0",
    )
    assert env["PYTHONPATH"].split(":")[0] == str(root)
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_build_worker_invocation_prepends_the_checkout_to_an_existing_pythonpath(tmp_path: Path) -> None:
    root = _fake_source_root(tmp_path)

    _, env = build_worker_invocation(
        source_root=root,
        request_path=tmp_path / "request.json",
        response_path=tmp_path / "response.json",
        env={"KORVID_UV_BIN": "uv", "PYTHONPATH": "/existing/path"},
    )

    assert env["PYTHONPATH"] == f"{root}:/existing/path"


def test_build_worker_invocation_forwards_runtime_policy_flags(tmp_path: Path) -> None:
    root = _fake_source_root(tmp_path)

    command, _ = build_worker_invocation(
        source_root=root,
        request_path=tmp_path / "request.json",
        response_path=tmp_path / "response.json",
        env={"KORVID_UV_BIN": "uv"},
        scripted=True,
        profile="full",
        approval_timeout=1.5,
        turn_timeout=30.0,
    )

    assert "--scripted" in command
    assert command[command.index("--profile") + 1] == "full"
    assert command[command.index("--approval-timeout") + 1] == "1.5"
    assert command[command.index("--turn-timeout") + 1] == "30.0"


def test_build_worker_invocation_requires_a_uv_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fake_source_root(tmp_path)
    monkeypatch.setattr("korvid_prompt_lab.bridge.shutil.which", lambda _name, path=None: None)

    with pytest.raises(BridgeConfigurationError, match="uv"):
        build_worker_invocation(
            source_root=root,
            request_path=tmp_path / "request.json",
            response_path=tmp_path / "response.json",
            env={},
        )


def test_source_root_is_runtime_policy_and_never_read_from_the_request(tmp_path: Path) -> None:
    root = _fake_source_root(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"candidate": {"components": {"system": "/attacker/korvid"}}}),
        encoding="utf-8",
    )

    command, env = build_worker_invocation(
        source_root=root,
        request_path=request_path,
        response_path=tmp_path / "response.json",
        env={"KORVID_UV_BIN": "uv"},
    )

    assert "/attacker/korvid" not in " ".join(command)
    assert "/attacker/korvid" not in env["PYTHONPATH"]


def test_main_returns_the_worker_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fake_source_root(tmp_path)
    recorded: dict[str, Any] = {}

    class _Completed:
        returncode = 3

    def _fake_run(command: Any, **kwargs: Any) -> Any:
        recorded["command"] = tuple(command)
        recorded["env"] = kwargs["env"]
        return _Completed()

    monkeypatch.setattr("korvid_prompt_lab.bridge._run_worker_process_group", _fake_run)
    monkeypatch.setenv("KORVID_SOURCE_ROOT", str(root))
    monkeypatch.setenv("KORVID_UV_BIN", "uv")

    exit_code = main(
        [
            "--request",
            str(tmp_path / "request.json"),
            "--response",
            str(tmp_path / "response.json"),
        ]
    )

    assert exit_code == 3
    assert recorded["command"][0] == "uv"
    assert recorded["env"]["PYTHONPATH"].startswith(str(root))


def test_main_fails_closed_without_leaking_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The failure text echoes the configured source root, so a root that happens to
    # carry the credential value proves the launcher redacts rather than reports it.
    monkeypatch.setenv("KORVID_SOURCE_ROOT", str(tmp_path / "zzz9q1w2e3" / "missing"))
    monkeypatch.setenv("KORVID_EVAL_API_KEY", "zzz9q1w2e3")

    exit_code = main(
        [
            "--request",
            str(tmp_path / "request.json"),
            "--response",
            str(tmp_path / "response.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Korvid source checkout" in captured.err
    assert "zzz9q1w2e3" not in captured.err
    assert "zzz9q1w2e3" not in captured.out
    assert "***" in captured.err


def test_main_reports_a_missing_uv_launcher_as_a_nonzero_configuration_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _fake_source_root(tmp_path)

    def _explode(command: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory", command[0])

    monkeypatch.setattr("korvid_prompt_lab.bridge._run_worker_process_group", _explode)
    monkeypatch.setenv("KORVID_SOURCE_ROOT", str(root))
    monkeypatch.setenv("KORVID_UV_BIN", "uv")

    exit_code = main(
        [
            "--request",
            str(tmp_path / "request.json"),
            "--response",
            str(tmp_path / "response.json"),
        ]
    )

    assert exit_code != 0
    assert "uv" in capsys.readouterr().err


# --- worker process-group ownership ------------------------------------------------


FAKE_PROCESS_TREE = ROOT / "tests" / "fixtures" / "fake_process_tree.py"


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - alive but owned by another user
        return True
    return True


def _recorded_pids(pid_file: Path) -> dict[str, int]:
    entries: dict[str, int] = {}
    for line in pid_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label, _, pid = line.partition(":")
        entries[label] = int(pid)
    return entries


def _await_process_exit(pids: dict[str, int], *, timeout: float = 15.0) -> dict[str, bool]:
    deadline = time.monotonic() + timeout
    alive = {label: _process_is_alive(pid) for label, pid in pids.items()}
    while time.monotonic() < deadline and any(alive.values()):
        time.sleep(0.05)
        alive = {label: _process_is_alive(pid) for label, pid in pids.items()}
    return alive


@pytest.mark.skipif(os.name != "posix", reason="process groups are a POSIX guarantee")
def test_launcher_terminates_the_whole_worker_process_group_on_its_own_timeout(
    tmp_path: Path,
) -> None:
    """`uv` spawns the worker, so killing only `uv` would orphan a live grader."""
    root = _fake_source_root(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pid_file = tmp_path / "pids.txt"
    pid_file.touch()
    response_path = run_dir / "response.json"

    env = dict(os.environ)
    env.update(
        {
            "KORVID_SOURCE_ROOT": str(root),
            "KORVID_UV_BIN": str(FAKE_PROCESS_TREE),
            BRIDGE_TIMEOUT_ENV: "1.0",
            "FAKE_TREE_PID_FILE": str(pid_file),
            "FAKE_TREE_DEPTH": "2",
            "FAKE_TREE_SLEEP": "5",
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
            ),
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "korvid_prompt_lab.bridge",
            "--request",
            str(run_dir / "request.json"),
            "--response",
            str(response_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode != 0
    assert "timed out" in completed.stderr

    pids = _recorded_pids(pid_file)
    assert sorted(pids) == ["level-0", "level-1", "level-2", "parent"]
    descendants = {label: pid for label, pid in pids.items() if label != "parent"}

    assert _await_process_exit(descendants) == dict.fromkeys(descendants, False)

    time.sleep(6.0)
    assert not response_path.exists()


def test_launcher_without_a_configured_budget_still_runs_to_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-run bridge has no campaign policy, so it must not invent a deadline."""
    root = _fake_source_root(tmp_path)
    recorded: dict[str, Any] = {}

    class _Completed:
        returncode = 0

    def _fake_run(command: Any, **kwargs: Any) -> Any:
        recorded["timeout"] = kwargs.get("timeout")
        return _Completed()

    monkeypatch.setattr("korvid_prompt_lab.bridge._run_worker_process_group", _fake_run)
    monkeypatch.setenv("KORVID_SOURCE_ROOT", str(root))
    monkeypatch.setenv("KORVID_UV_BIN", "uv")
    monkeypatch.delenv(BRIDGE_TIMEOUT_ENV, raising=False)

    exit_code = main(
        ["--request", str(tmp_path / "request.json"), "--response", str(tmp_path / "response.json")]
    )

    assert exit_code == 0
    assert recorded["timeout"] is None


@pytest.mark.parametrize("raw", ["", "   ", "0", "-1", "not-a-number", "nan", "inf"])
def test_launcher_refuses_an_unusable_timeout_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], raw: str
) -> None:
    """A malformed budget is runtime policy that cannot be honoured: fail closed."""
    root = _fake_source_root(tmp_path)
    monkeypatch.setenv("KORVID_SOURCE_ROOT", str(root))
    monkeypatch.setenv("KORVID_UV_BIN", "uv")
    monkeypatch.setenv(BRIDGE_TIMEOUT_ENV, raw)

    exit_code = main(
        ["--request", str(tmp_path / "request.json"), "--response", str(tmp_path / "response.json")]
    )

    assert exit_code != 0
    assert BRIDGE_TIMEOUT_ENV in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="process groups are a POSIX guarantee")
def test_launcher_escalates_to_sigkill_for_a_worker_that_ignores_sigterm(tmp_path: Path) -> None:
    """`uv` dies on SIGTERM at once; the grader underneath it may not.

    Deciding "the group is gone" from the direct child having exited would skip the
    SIGKILL escalation and leave the real grader running.
    """
    root = _fake_source_root(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pid_file = tmp_path / "pids.txt"
    pid_file.touch()
    response_path = run_dir / "response.json"

    env = dict(os.environ)
    env.update(
        {
            "KORVID_SOURCE_ROOT": str(root),
            "KORVID_UV_BIN": str(FAKE_PROCESS_TREE),
            BRIDGE_TIMEOUT_ENV: "1.0",
            "FAKE_TREE_PID_FILE": str(pid_file),
            "FAKE_TREE_DEPTH": "1",
            "FAKE_TREE_SLEEP": "20",
            "FAKE_TREE_IGNORE_SIGTERM": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
            ),
        }
    )

    started = time.monotonic()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "korvid_prompt_lab.bridge",
            "--request",
            str(run_dir / "request.json"),
            "--response",
            str(response_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode != 0
    pids = _recorded_pids(pid_file)
    descendants = {label: pid for label, pid in pids.items() if label != "parent"}
    assert _await_process_exit(descendants) == dict.fromkeys(descendants, False)
    # Dead well before it could reach its own write, so the escalation really ran.
    assert time.monotonic() - started < 20.0
    assert not response_path.exists()


@pytest.mark.skipif(os.name != "posix", reason="process groups are a POSIX guarantee")
def test_launcher_tears_down_its_worker_group_when_it_is_itself_signalled(tmp_path: Path) -> None:
    """The runner signals the launcher's group, which does not contain the worker.

    The launcher gives its worker its own session so its own kills stay scoped, so
    it must hand that termination on when it is signalled — otherwise a runner-side
    teardown leaves the worker running.
    """
    root = _fake_source_root(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pid_file = tmp_path / "pids.txt"
    pid_file.touch()
    response_path = run_dir / "response.json"

    env = dict(os.environ)
    env.update(
        {
            "KORVID_SOURCE_ROOT": str(root),
            "KORVID_UV_BIN": str(FAKE_PROCESS_TREE),
            "FAKE_TREE_PID_FILE": str(pid_file),
            "FAKE_TREE_DEPTH": "2",
            "FAKE_TREE_SLEEP": "5",
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
            ),
        }
    )
    env.pop(BRIDGE_TIMEOUT_ENV, None)

    launcher = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "korvid_prompt_lab.bridge",
            "--request",
            str(run_dir / "request.json"),
            "--response",
            str(response_path),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and len(_recorded_pids(pid_file)) < 4:
            time.sleep(0.05)
        assert sorted(_recorded_pids(pid_file)) == ["level-0", "level-1", "level-2", "parent"]

        launcher.send_signal(signal.SIGTERM)
        assert launcher.wait(timeout=20.0) != 0
    finally:
        if launcher.poll() is None:  # pragma: no cover - only on an unexpected hang
            launcher.kill()
        launcher.communicate()

    pids = _recorded_pids(pid_file)
    descendants = {label: pid for label, pid in pids.items() if label != "parent"}
    assert _await_process_exit(descendants) == dict.fromkeys(descendants, False)

    time.sleep(6.0)
    assert not response_path.exists()


@pytest.mark.skipif(os.name != "posix", reason="process groups are a POSIX guarantee")
def test_launcher_tears_down_a_worker_signalled_during_its_own_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Popen.__init__` returns long after the fork, and the worker is already live.

    A termination signal delivered in that window must not unwind past the point
    where the launcher still holds the only handle on its own private session.
    """
    pid_file = tmp_path / "pids.txt"
    pid_file.touch()
    response_path = tmp_path / "response.json"

    env = {
        **os.environ,
        "FAKE_TREE_PID_FILE": str(pid_file),
        "FAKE_TREE_DEPTH": "1",
        "FAKE_TREE_SLEEP": "30",
    }

    real_popen = subprocess.Popen

    class _SignalledDuringLaunch(subprocess.Popen):  # type: ignore[type-arg]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            real_popen.__init__(self, *args, **kwargs)  # type: ignore[misc]
            # The child is alive and has spawned its own child, but the caller
            # still has no handle on it: this is the window `Popen.__init__` owns.
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and len(_recorded_pids(pid_file)) < 3:
                time.sleep(0.02)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.5)

    monkeypatch.setattr("korvid_prompt_lab.bridge.subprocess.Popen", _SignalledDuringLaunch)

    started = time.monotonic()
    with pytest.raises(LauncherTerminated):
        _run_worker_process_group(
            [
                sys.executable,
                str(FAKE_PROCESS_TREE),
                "--request",
                str(tmp_path / "request.json"),
                "--response",
                str(response_path),
            ],
            env=env,
            timeout=30.0,
        )

    pids = _recorded_pids(pid_file)
    descendants = {label: pid for label, pid in pids.items() if label != "parent"}
    assert descendants, "the fake worker chain never started"

    assert _await_process_exit(descendants) == dict.fromkeys(descendants, False)
    assert time.monotonic() - started < 30.0
    assert not response_path.exists()
