"""Process-boundary tests for scripts/run-grounding-round.sh.

Each test spins up a minimal fake PATH containing az, kubectl,
korvid-prompt-lab, and korvid-grounding-report shims, then runs the
real script as a subprocess.  Call records are captured via a shared
``calls.txt`` file so we can assert ordering without mocking internals.
"""

from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run-grounding-round.sh"

_BASE_ENV: dict[str, str] = {
    "GROUNDING_MODEL": "qwen3:1.7b",
    "GROUNDING_CANDIDATE": "seed",
    "GROUNDING_ROUND_TYPE": "evaluate",
    "KORVID_SOURCE_ROOT": "/fake/korvid",
    "GROUNDING_ARTIFACT_ROOT": "",  # overridden per test
    "WORKFLOW_RUN_URL": "https://github.com/org/repo/actions/runs/1",
    "PROMPT_LAB_REVISION": "abc123",
    "KORVID_REVISION": "def456",
    "GROUNDING_REFLECTION_MODEL": "qwen3:0.6b",
    "GROUNDING_REFLECTION_CREDENTIAL": "fake-token",
}


def _make_fake_bin(
    fake_bin_dir: Path,
    *,
    node_count: int,
    evaluation_exit: int,
    preflight_exit: int,
    calls_file: Path,
) -> None:
    """Write shim executables into *fake_bin_dir*."""
    fake_bin_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, body: str) -> None:
        p = fake_bin_dir / name
        p.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body), encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    _write(
        "az",
        f"""\
        CALLS="{calls_file}"
        if [[ "$*" == *"nodepool show"* ]]; then
            echo {node_count}
        elif [[ "$*" == *"nodepool scale"*"--node-count 1"* ]]; then
            echo "scale:1" >> "$CALLS"
        elif [[ "$*" == *"nodepool scale"*"--node-count 0"* ]]; then
            echo "scale:0" >> "$CALLS"
        fi
        """,
    )

    _write(
        "kubectl",
        """\
        : # no-op kubectl shim
        """,
    )

    _write(
        "korvid-prompt-lab",
        f"""\
        CALLS="{calls_file}"
        if [[ "$1" == "aks-check" ]]; then
            echo "aks-check" >> "$CALLS"
            exit {preflight_exit}
        elif [[ "$1" == "evaluate" ]]; then
            echo "evaluate" >> "$CALLS"
            exit {evaluation_exit}
        elif [[ "$1" == "optimize" ]]; then
            echo "optimize" >> "$CALLS"
            mkdir -p "${{GROUNDING_ARTIFACT_ROOT}}/optimize"
            echo "candidate: optimized" > "${{GROUNDING_ARTIFACT_ROOT}}/optimize/best-candidate.yaml"
            exit 0
        fi
        """,
    )

    _write(
        "korvid-grounding-report",
        f"""\
        CALLS="{calls_file}"
        echo "report" >> "$CALLS"
        exit 0
        """,
    )


def run_script(
    tmp_path: Path,
    *,
    original_count: int = 0,
    evaluation_exit: int = 0,
    preflight_exit: int = 0,
    round_type: str = "evaluate",
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    calls_file = tmp_path / "calls.txt"
    calls_file.touch()

    fake_bin = tmp_path / "bin"
    _make_fake_bin(
        fake_bin,
        node_count=original_count,
        evaluation_exit=evaluation_exit,
        preflight_exit=preflight_exit,
        calls_file=calls_file,
    )

    env = dict(os.environ)
    env.update(_BASE_ENV)
    env["GROUNDING_ARTIFACT_ROOT"] = str(artifact_root)
    env["GROUNDING_ROUND_TYPE"] = round_type
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["_AKS_CHECK_POLL_INTERVAL"] = "0"  # no sleep in tests
    env["_AKS_CHECK_DEADLINE_SECONDS"] = "1"  # fast timeout in tests
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=30,
    )
    calls = [line for line in calls_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return result, calls


# ---------------------------------------------------------------------------
# Tests (RED phase: all fail until script exists)
# ---------------------------------------------------------------------------


def test_round_script_restores_zero_count_after_unsafe_evaluation(tmp_path: Path) -> None:
    """evaluate exit=1 must still run report and restore node-count to 0."""
    result, calls = run_script(tmp_path, original_count=0, evaluation_exit=1)

    assert result.returncode == 1
    assert calls.index("scale:1") < calls.index("evaluate")
    assert calls[-1] == "scale:0"
    assert "report" in calls


def test_round_script_never_scales_down_preexisting_capacity(tmp_path: Path) -> None:
    """When the pool already has 1 node the script must not touch scaling."""
    result, calls = run_script(tmp_path, original_count=1, evaluation_exit=0)

    assert result.returncode == 0
    assert "scale:1" not in calls
    assert "scale:0" not in calls


def test_round_script_restores_pool_on_signal_or_systemic_failure(tmp_path: Path) -> None:
    """When aks-check (preflight) fails the pool must be restored to 0."""
    result, calls = run_script(tmp_path, original_count=0, preflight_exit=1)

    assert result.returncode != 0
    assert calls[-1] == "scale:0"


def test_round_script_rejects_unsupported_model(tmp_path: Path) -> None:
    """GROUNDING_MODEL values outside the allowlist must exit 2."""
    result, _calls = run_script(
        tmp_path,
        original_count=0,
        extra_env={"GROUNDING_MODEL": "gpt-99"},
    )

    assert result.returncode == 2


def test_round_script_exits_nonzero_when_required_env_absent(tmp_path: Path) -> None:
    """Missing required env vars must cause non-zero exit before any cloud call."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        extra_env={"GROUNDING_MODEL": ""},  # unset by overriding with empty
    )
    # bash :? expansion exits 1 for empty/unset
    assert result.returncode != 0
    assert "scale:1" not in calls


def test_round_script_preserves_evaluate_exit_1_as_final_exit(tmp_path: Path) -> None:
    """exit 1 from evaluate is an expected safety signal, not a systemic error."""
    result, calls = run_script(tmp_path, original_count=0, evaluation_exit=1)

    assert result.returncode == 1
    assert "report" in calls


def test_round_script_optimize_evaluate_runs_optimize_then_evaluate(tmp_path: Path) -> None:
    """optimize-evaluate round must call optimize then evaluate."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        round_type="optimize-evaluate",
    )

    assert result.returncode == 0
    assert "optimize" in calls
    assert "evaluate" in calls
    assert calls.index("optimize") < calls.index("evaluate")
