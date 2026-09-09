from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from korvid_prompt_lab.contracts import KorvidUpstreamServing
from korvid_prompt_lab.runner import BridgeInvocationError


def _source_repo(root: Path) -> str:
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    for name in (
        "src/korvid/evals/harness.py",
        "src/korvid/ui/agent_ui_controller.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# reviewed fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )
    python = root / ".venv" / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _serving(
    root: Path,
    *,
    base_url: str = "http://127.0.0.1:11434",
) -> KorvidUpstreamServing:
    return KorvidUpstreamServing(
        backend="korvid_upstream",
        source_root=str(root),
        base_url=base_url,
        korvid_revision="33c483e041006eb20259a024ed85a9323e52c8f0",
        timeout_seconds=3,
        model_options=MappingProxyType({}),
    )


def test_native_source_module_and_launcher_aliases_are_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("korvid_prompt_lab.native_source")

    from korvid_prompt_lab import source_runtime

    assert hasattr(source_runtime, "validate_source")
    assert hasattr(source_runtime, "run_upstream_request")
    assert not hasattr(source_runtime, "validate_native_source")
    assert not hasattr(source_runtime, "run_native_request")
    assert not hasattr(source_runtime, "run_source")


def test_unreviewed_source_is_rejected_before_worker_start(tmp_path: Path) -> None:
    from korvid_prompt_lab.source_runtime import validate_source

    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    with pytest.raises(ValueError, match="revision"):
        validate_source(tmp_path)


def test_inherited_git_variables_cannot_attest_a_modified_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from korvid_prompt_lab import source_runtime

    trusted = tmp_path / "trusted"
    revision = _source_repo(trusted)
    actual = tmp_path / "actual"
    subprocess.run(["git", "clone", "--quiet", str(trusted), str(actual)], check=True)
    python = actual / ".venv" / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    (actual / "src/korvid/evals/harness.py").write_text(
        "# modified product\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(source_runtime, "KORVID_REVISION", revision)
    monkeypatch.setenv("GIT_DIR", str(trusted / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(trusted))

    with pytest.raises(ValueError, match="unchanged"):
        source_runtime.validate_source(actual)


def test_source_must_be_the_attested_repository_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from korvid_prompt_lab import source_runtime

    trusted = tmp_path / "trusted"
    revision = _source_repo(trusted)
    nested = trusted / ".venv" / "nested"
    for relative in (
        "src/korvid/evals/harness.py",
        "src/korvid/ui/agent_ui_controller.py",
        ".venv/bin/python"
        if sys.platform != "win32"
        else ".venv/Scripts/python.exe",
    ):
        path = nested / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# not reviewed\n", encoding="utf-8")
    monkeypatch.setattr(source_runtime, "KORVID_REVISION", revision)

    with pytest.raises(ValueError, match="root"):
        source_runtime.validate_source(nested)


def test_source_launcher_can_only_select_upstream_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from korvid_prompt_lab import source_runtime

    root = tmp_path / "source"
    root.mkdir()
    python = root / "python"
    python.write_text("", encoding="utf-8")
    observed: dict[str, Any] = {}

    class Process:
        returncode = 0

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            observed["command"] = command
            observed["env"] = kwargs["env"]
            response = Path(command[command.index("--response") + 1])
            response.write_text(json.dumps({"ok": True}), encoding="utf-8")

        def communicate(self, timeout: float) -> tuple[bytes, bytes]:
            observed["timeout"] = timeout
            return b"", b""

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(source_runtime, "validate_source", lambda _: python)
    monkeypatch.setattr(source_runtime.subprocess, "Popen", Process)
    result = source_runtime.run_upstream_request(_serving(root), {"request": True})

    command = observed["command"]
    assert Path(command[1]).name == "upstream_worker.py"
    assert "native_worker.py" not in command
    assert result == {"ok": True}
    assert observed["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert observed["env"]["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    private = Path(observed["env"]["TMPDIR"])
    assert not private.exists()


def test_source_launcher_requires_a_bound_loopback_endpoint_before_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from korvid_prompt_lab import source_runtime

    root = tmp_path / "source"
    root.mkdir()
    serving = _serving(root, base_url="")
    monkeypatch.setattr(
        source_runtime,
        "validate_source",
        lambda _: pytest.fail("unbound serving must fail before source validation"),
    )

    with pytest.raises(ValueError, match="loopback"):
        source_runtime.run_upstream_request(serving, {"request": True})


def test_source_timeout_is_systemic_and_cleans_the_owned_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from korvid_prompt_lab import source_runtime

    root = tmp_path / "source"
    root.mkdir()
    python = root / "python"
    python.write_text("", encoding="utf-8")
    terminated: list[object] = []

    class Process:
        returncode = None
        stdout = None
        stderr = None
        stdin = None

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def communicate(self, timeout: float) -> tuple[bytes, bytes]:
            raise subprocess.TimeoutExpired("upstream_worker.py", timeout)

        def poll(self) -> None:
            return None

    monkeypatch.setattr(source_runtime, "validate_source", lambda _: python)
    monkeypatch.setattr(source_runtime.subprocess, "Popen", Process)
    monkeypatch.setattr(
        source_runtime,
        "_terminate_process_group",
        lambda process: terminated.append(process),
    )

    with pytest.raises(BridgeInvocationError, match="timed out"):
        source_runtime.run_upstream_request(_serving(root), {"request": True})
    assert len(terminated) == 1


def test_source_worker_launch_failures_are_typed_system_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from korvid_prompt_lab import source_runtime

    root = tmp_path / "source"
    root.mkdir()
    python = root / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(source_runtime, "validate_source", lambda _: python)
    monkeypatch.setattr(
        source_runtime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("worker interpreter unavailable")
        ),
    )

    with pytest.raises(BridgeInvocationError, match="could not be launched"):
        source_runtime.run_upstream_request(_serving(root), {"request": True})
