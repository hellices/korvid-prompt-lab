"""Execute a pinned Korvid source environment without importing its legacy wheel."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import write_json_artifact
from .contracts import KorvidNativeServing
from .native_contract import NATIVE_KORVID_REVISION
from .runner import BridgeInvocationError, _terminate_process_group


def _native_environment() -> dict[str, str]:
    return {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }


def validate_native_source(root: Path) -> Path:
    root = root.resolve(strict=True)
    env = _native_environment()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=15, check=False, env=env,
    )
    if revision.returncode or revision.stdout.strip() != NATIVE_KORVID_REVISION:
        raise ValueError("native Korvid source revision must match the reviewed v0.4.1 commit")
    checkout = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, timeout=15, check=False, env=env,
    )
    if checkout.returncode or Path(checkout.stdout.rstrip("\n")).resolve() != root:
        raise ValueError("native source must be the attested Git repository root")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True, text=True, timeout=15, check=False, env=env,
    )
    if status.returncode or status.stdout.strip():
        raise ValueError("native Korvid source must be unchanged; do not patch the product for evaluation")
    for relative in ("src/korvid/evals/harness.py", "src/korvid/ui/agent_ui_controller.py"):
        if not (root / relative).is_file():
            raise ValueError(f"native Korvid source is missing {relative}")
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise ValueError("native Korvid source requires its own prepared .venv; see native-check setup")
    return python


def run_native_request(serving: KorvidNativeServing, payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(serving.source_root).resolve(strict=True)
    python = validate_native_source(root)
    with tempfile.TemporaryDirectory(prefix="korvid-native-") as temporary:
        private = Path(temporary)
        request_path = write_json_artifact(private / "request.json", payload)
        response_path = private / "response.json"
        env = _native_environment()
        for key in list(env):
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
                del env[key]
        env["NO_PROXY"] = "127.0.0.1,localhost,::1"
        env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(Path(__file__).resolve().parents[1])))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TMPDIR"] = str(private)
        env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        env["DO_NOT_TRACK"] = "1"
        command = [
            str(python), str(Path(__file__).with_name("native_worker.py")),
            "--request", str(request_path), "--response", str(response_path),
        ]
        process = subprocess.Popen(
            command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        try:
            _, stderr = process.communicate(timeout=serving.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise BridgeInvocationError("native Korvid worker timed out; no score recorded") from exc
        finally:
            if process.poll() is None:
                _terminate_process_group(process)
        if process.returncode:
            # Worker diagnostics are bounded labels; never include provider errors,
            # model text, or raw cluster output in public control-plane messages.
            detail = stderr.decode("utf-8", errors="replace").strip().splitlines()
            label = detail[-1] if detail else "worker failed"
            if not label.startswith("native-worker:"):
                label = "worker failed (inspect the native environment and import contract)"
            raise BridgeInvocationError(f"native Korvid exited {process.returncode}: {label[:240]}")
        if not response_path.is_file() or response_path.stat().st_size > 2 * 1024 * 1024:
            raise BridgeInvocationError("native Korvid response is missing or too large")
        data = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("native worker response must be an object")  # noqa: TRY004
        return data
