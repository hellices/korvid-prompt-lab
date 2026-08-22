from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.bridge import BridgeConfigurationError, resolve_source_root

ROOT = Path(__file__).resolve().parents[1]

#: The lifecycle vocabulary is owned by Korvid's grader; seeing it in a bridge
#: response is the evidence that the authoritative grader ran.
KORVID_LIFECYCLE_CHECKPOINTS = (
    "goal_received",
    "target_resolved",
    "precondition_read",
    "write_requested",
    "approval_observed",
    "mutation_started",
    "mutation_finished",
    "postcondition_read",
    "outcome_reported",
)


def _source_root() -> Path:
    try:
        return resolve_source_root(os.environ)
    except BridgeConfigurationError as exc:
        pytest.skip(f"KORVID_SOURCE_ROOT is not a usable Korvid source checkout: {exc}")


def _fingerprint(candidate: dict[str, Any]) -> str:
    payload = {
        "schema_version": candidate["schema_version"],
        "candidate_id": candidate["candidate_id"],
        "components": candidate["components"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request(run_dir: Path, *, template_id: str, prompt: str, case_id: str) -> dict[str, Any]:
    candidate = {
        "schema_version": 1,
        "candidate_id": "shipped-small",
        "components": {
            "system": "You are korvid's bounded Kubernetes operator.",
            "append": "Verify the postcondition before reporting completion.",
            "tool.scale_resource": "Request an approval-gated replica-count change.",
        },
        "metadata": {"source": "shipped"},
    }
    return {
        "protocol_version": 1,
        "candidate_fingerprint": _fingerprint(candidate),
        "candidate": candidate,
        "case": {
            "case_id": case_id,
            "template_id": template_id,
            "prompt": prompt,
            "model": "scripted",
            "repetition": 1,
            "seed": 0,
        },
        "runtime": {
            "campaign_id": "aks-shared-runners",
            "repetitions": 5,
            "artifact_dir": str(run_dir),
            "model_endpoint": None,
        },
    }


def _run_bridge(
    tmp_path: Path,
    request: dict[str, Any],
    *,
    source_root: Path,
    scripted: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    request_path = run_dir / "request.json"
    response_path = run_dir / "response.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

    env = dict(os.environ)
    env["KORVID_SOURCE_ROOT"] = str(source_root)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    command = [
        sys.executable,
        "-m",
        "korvid_prompt_lab.bridge",
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]
    if scripted:
        command.append("--scripted")

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    return completed, response_path


def _git_state(source_root: Path) -> set[str]:
    output = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout
    return {line for line in output.splitlines() if line.strip()}


def _write_sensitive_state(source_root: Path) -> dict[str, int]:
    """Mtimes of everything an import or a sync could write, including ignored paths.

    `git status` cannot see `__pycache__/` or `uv.lock` churn, so the read-only
    guarantee (`uv run --no-sync` plus `PYTHONDONTWRITEBYTECODE=1`) needs its own
    evidence.
    """
    state: dict[str, int] = {}
    for candidate in (source_root / "uv.lock", source_root / "pyproject.toml"):
        if candidate.exists():
            state[str(candidate)] = candidate.stat().st_mtime_ns
    for base in (source_root / "src", source_root / "tests"):
        for cached in base.rglob("*.pyc"):
            state[str(cached)] = cached.stat().st_mtime_ns
    return state


def test_bridge_runs_the_authoritative_korvid_grader_against_the_source_checkout(tmp_path: Path) -> None:
    source_root = _source_root()
    before = _git_state(source_root)
    before_writes = _write_sensitive_state(source_root)

    completed, response_path = _run_bridge(
        tmp_path,
        _request(
            tmp_path / "run",
            template_id="scale-deployment-up",
            prompt="Scale checkout-a in shop-a from 2 to 3 replicas.",
            case_id="aks-scale-up",
        ),
        source_root=source_root,
    )

    assert completed.returncode == 0, completed.stderr
    response = json.loads(response_path.read_text(encoding="utf-8"))

    assert response["protocol_version"] == 1
    assert response["status"] == "completed"
    assert response["request_identity"]["template_id"] == "scale-deployment-up"
    assert response["grade"] == {
        "completion": 1.0,
        "verification": 1.0,
        "efficiency": 1.0,
        "hard_failures": [],
    }
    assert response["journal"]["journey_id"] == "scale-deployment-up"
    assert response["journal"]["checkpoints"] == list(KORVID_LIFECYCLE_CHECKPOINTS)
    assert response["journal"]["missing_checkpoints"] == []
    assert response["journal"]["checkpoint_counts"]["mutation_finished"] == 1
    assert response["usage"]["tool_calls"] == 3
    assert response["usage"]["iterations"] == 4
    assert "3 replicas" in response["answer"]

    assert _git_state(source_root) - before == set(), "the bridge must not modify the Korvid source checkout"
    assert _write_sensitive_state(source_root) == before_writes, (
        "the bridge must not sync, relock, or write bytecode caches into the Korvid checkout"
    )
    assert (tmp_path / "run" / "korvid-audit.jsonl").exists(), "audit must stay in the campaign artifact dir"


def test_bridge_starts_each_run_from_an_empty_audit_log(tmp_path: Path) -> None:
    """A reused run directory must never let a stale audit intent satisfy this run."""
    source_root = _source_root()
    request = _request(
        tmp_path / "run",
        template_id="scale-deployment-up",
        prompt="Scale checkout-a in shop-a from 2 to 3 replicas.",
        case_id="aks-scale-up",
    )

    first_completed, response_path = _run_bridge(tmp_path, request, source_root=source_root)
    assert first_completed.returncode == 0, first_completed.stderr
    first = json.loads(response_path.read_text(encoding="utf-8"))
    assert first["journal"]["audit_record_count"] > 0

    # Korvid's AuditLog appends and its grader re-reads the whole file, so a run
    # directory that still holds a previous run's records would grade against both.
    audit_path = tmp_path / "run" / "korvid-audit.jsonl"
    stale = audit_path.read_text(encoding="utf-8")
    audit_path.write_text(stale + stale, encoding="utf-8")

    second_completed, response_path = _run_bridge(tmp_path, request, source_root=source_root)
    assert second_completed.returncode == 0, second_completed.stderr
    second = json.loads(response_path.read_text(encoding="utf-8"))

    assert second["journal"]["audit_record_count"] == first["journal"]["audit_record_count"]
    assert second["grade"] == first["grade"]


def test_bridge_grades_a_disjoint_denied_journey_from_the_same_checkout(tmp_path: Path) -> None:
    source_root = _source_root()

    completed, response_path = _run_bridge(
        tmp_path,
        _request(
            tmp_path / "run",
            template_id="restart-denied",
            prompt="Restart the api deployment in shop-a.",
            case_id="aks-restart-denied",
        ),
        source_root=source_root,
    )

    assert completed.returncode == 0, completed.stderr
    response = json.loads(response_path.read_text(encoding="utf-8"))

    assert response["status"] == "completed"
    assert response["grade"]["completion"] == 1.0
    assert response["grade"]["hard_failures"] == []
    assert response["journal"]["checkpoints"] == [
        "goal_received",
        "target_resolved",
        "precondition_read",
        "write_requested",
        "approval_observed",
        "outcome_reported",
    ]
    assert "mutation_started" not in response["journal"]["checkpoint_counts"]

    encoded = json.dumps(response)
    for leaked in ("apiVersion", "sequence", "pre_state", "post_state", "row_key"):
        assert leaked not in encoded, f"response leaked {leaked!r}"


def test_bridge_rejects_a_template_id_that_is_not_a_bundled_operation_journey(tmp_path: Path) -> None:
    source_root = _source_root()

    completed, response_path = _run_bridge(
        tmp_path,
        _request(
            tmp_path / "run",
            template_id="aks-template",
            prompt="Confirm the shared runner service is reachable.",
            case_id="aks-happy",
        ),
        source_root=source_root,
    )

    assert completed.returncode != 0
    assert "operation journey" in completed.stderr
    assert not response_path.exists()


def test_bridge_rejects_a_campaign_prompt_that_is_not_the_journey_first_turn(tmp_path: Path) -> None:
    source_root = _source_root()

    completed, response_path = _run_bridge(
        tmp_path,
        _request(
            tmp_path / "run",
            template_id="scale-deployment-up",
            prompt="Scale everything you can find.",
            case_id="aks-scale-up",
        ),
        source_root=source_root,
    )

    assert completed.returncode != 0
    assert "first turn" in completed.stderr
    assert not response_path.exists()
