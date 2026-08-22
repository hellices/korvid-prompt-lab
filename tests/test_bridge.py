from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.bridge import (
    WORKER_MODULE_PATH,
    BridgeConfigurationError,
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

    monkeypatch.setattr("korvid_prompt_lab.bridge.subprocess.run", _fake_run)
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

    monkeypatch.setattr("korvid_prompt_lab.bridge.subprocess.run", _explode)
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
