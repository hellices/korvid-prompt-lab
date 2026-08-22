"""Process-boundary tests for scripts/run-grounding-round.sh.

Each test spins up a minimal fake PATH containing az, kubectl,
korvid-prompt-lab, and korvid-grounding-report shims, then runs the
real script as a subprocess.  Call records are captured via a shared
``calls.txt`` file so we can assert ordering without mocking internals.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import textwrap
import time
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


def _make_fake_bin_blocking(
    fake_bin_dir: Path,
    *,
    node_count: int,
    calls_file: Path,
    ready_file: Path,
) -> None:
    """Write shim executables where aks-check blocks until killed."""
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

    _write("kubectl", ": # no-op kubectl shim\n")

    _write(
        "korvid-prompt-lab",
        f"""\
        CALLS="{calls_file}"
        if [[ "$1" == "aks-check" ]]; then
            echo "aks-check" >> "$CALLS"
            # Signal readiness then block until killed
            touch "{ready_file}"
            sleep 60
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


def _run_script_with_signal(
    tmp_path: Path,
    sig: signal.Signals,
    *,
    original_count: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run script, wait for aks-check to start, then send *sig* to the script."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    calls_file = tmp_path / "calls.txt"
    calls_file.touch()
    ready_file = tmp_path / "ready"

    fake_bin = tmp_path / "bin"
    _make_fake_bin_blocking(
        fake_bin,
        node_count=original_count,
        calls_file=calls_file,
        ready_file=ready_file,
    )

    env = dict(os.environ)
    env.update(_BASE_ENV)
    env["GROUNDING_ARTIFACT_ROOT"] = str(artifact_root)
    env["GROUNDING_ROUND_TYPE"] = "evaluate"
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["_AKS_CHECK_POLL_INTERVAL"] = "0"
    env["_AKS_CHECK_DEADLINE_SECONDS"] = "120"

    proc = subprocess.Popen(
        ["bash", str(_SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        preexec_fn=os.setpgrp,  # isolate in a new process group so the signal
        # reaches bash and all its children (e.g. the blocking sleep) without
        # propagating back to the pytest runner
    )
    # Wait for aks-check shim to signal readiness (up to 10 s)
    deadline = time.monotonic() + 10
    while not ready_file.exists():
        if time.monotonic() > deadline:
            proc.kill()
            proc.wait()
            raise TimeoutError("aks-check shim never wrote ready_file")
        time.sleep(0.05)

    # Kill the entire process group: this lets the blocking child (sleep) die
    # first so bash can deliver the pending signal to its INT trap.
    os.killpg(proc.pid, sig)
    stdout, stderr = proc.communicate(timeout=15)
    calls = [line for line in calls_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr), calls


def _make_fake_bin(
    fake_bin_dir: Path,
    *,
    node_count: int,
    evaluation_exit: int,
    preflight_exit: int,
    calls_file: Path,
    optimize_mode: str = "actual-layout",
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
        _record_args() {{
            local prefix="$1"
            shift
            printf '%s argc=%s\\n' "$prefix" "$#" >> "$CALLS"
            for arg in "$@"; do
                printf '%s arg=%s\\n' "$prefix" "$arg" >> "$CALLS"
            done
        }}
        if [[ "$1" == "aks-check" ]]; then
            echo "aks-check" >> "$CALLS"
            exit {preflight_exit}
        elif [[ "$1" == "evaluate" ]]; then
            echo "evaluate" >> "$CALLS"
            _record_args "evaluate" "$@"
            exit {evaluation_exit}
        elif [[ "$1" == "optimize" ]]; then
            echo "optimize" >> "$CALLS"
            _record_args "optimize" "$@"
            case "{optimize_mode}" in
                actual-layout)
                    mkdir -p "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-1"
                    echo "candidate: optimized" > "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-1/best-candidate.yaml"
                    echo "{{}}" > "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-1/optimization-summary.json"
                    ;;
                missing-best-candidate)
                    mkdir -p "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-1"
                    echo "{{}}" > "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-1/optimization-summary.json"
                    ;;
                ambiguous-best-candidate)
                    mkdir -p "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-1"
                    mkdir -p "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-2"
                    echo "candidate: optimized-1" > "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-1/best-candidate.yaml"
                    echo "{{}}" > "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-1/optimization-summary.json"
                    echo "candidate: optimized-2" > "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-2/best-candidate.yaml"
                    echo "{{}}" > "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-2/optimization-summary.json"
                    ;;
                *)
                    echo "unexpected optimize mode: {optimize_mode}" >&2
                    exit 97
                    ;;
            esac
            exit 0
        fi
        """,
    )

    _write(
        "korvid-grounding-report",
        f"""\
        CALLS="{calls_file}"
        echo "report" >> "$CALLS"
        printf 'report argc=%s\\n' "$#" >> "$CALLS"
        for arg in "$@"; do
            printf 'report arg=%s\\n' "$arg" >> "$CALLS"
        done
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
    optimize_mode: str = "actual-layout",
    artifact_root_name: str = "artifacts",
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    artifact_root = tmp_path / artifact_root_name
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
        optimize_mode=optimize_mode,
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


def test_round_script_optimize_evaluate_uses_invocation_artifacts_and_preserves_spaced_paths(tmp_path: Path) -> None:
    """optimize-evaluate must use the invocation-layout artifacts and keep spaced paths as one argv token."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        round_type="optimize-evaluate",
        artifact_root_name="artifacts with spaces",
    )

    assert result.returncode == 0, result.stderr
    assert f"evaluate arg={tmp_path / 'artifacts with spaces' / 'optimize' / 'invocations' / 'opt-run-1' / 'best-candidate.yaml'}" in calls
    assert f"report arg={tmp_path / 'artifacts with spaces' / 'evaluate'}" in calls
    assert f"report arg={tmp_path / 'artifacts with spaces' / 'optimize' / 'invocations' / 'opt-run-1'}" in calls


def test_round_script_optimize_evaluate_rejects_missing_best_candidate(tmp_path: Path) -> None:
    """optimize-evaluate must fail when optimize does not produce exactly one best-candidate file."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        round_type="optimize-evaluate",
        optimize_mode="missing-best-candidate",
    )

    assert result.returncode == 1
    assert "did not produce exactly one regular best-candidate.yaml" in result.stderr
    assert "evaluate" not in calls
    assert "report" not in calls


def test_round_script_optimize_evaluate_rejects_ambiguous_best_candidate(tmp_path: Path) -> None:
    """optimize-evaluate must fail when optimize leaves multiple invocation best-candidate files."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        round_type="optimize-evaluate",
        optimize_mode="ambiguous-best-candidate",
    )

    assert result.returncode == 1
    assert "did not produce exactly one regular best-candidate.yaml" in result.stderr
    assert "evaluate" not in calls
    assert "report" not in calls


# ---------------------------------------------------------------------------
# Issue 1: Signal cancellation — exactly one cleanup and conventional exit code
# ---------------------------------------------------------------------------


def test_round_script_sigint_exits_130_and_scales_down_exactly_once(tmp_path: Path) -> None:
    """SIGINT must exit 130 and execute nodepool scale-down exactly once."""
    result, calls = _run_script_with_signal(tmp_path, signal.SIGINT, original_count=0)

    assert result.returncode == 130, f"expected 130, got {result.returncode}\nstderr: {result.stderr}"
    assert calls.count("scale:0") == 1, f"expected exactly 1 scale:0, got {calls.count('scale:0')}\ncalls: {calls}"


def test_round_script_sigterm_exits_143_and_scales_down_exactly_once(tmp_path: Path) -> None:
    """SIGTERM must exit 143 and execute nodepool scale-down exactly once."""
    result, calls = _run_script_with_signal(tmp_path, signal.SIGTERM, original_count=0)

    assert result.returncode == 143, f"expected 143, got {result.returncode}\nstderr: {result.stderr}"
    assert calls.count("scale:0") == 1, f"expected exactly 1 scale:0, got {calls.count('scale:0')}\ncalls: {calls}"


def test_round_script_sigint_no_scale_when_pool_already_had_capacity(tmp_path: Path) -> None:
    """SIGINT on a pre-existing pool must not trigger any scale-down."""
    result, calls = _run_script_with_signal(tmp_path, signal.SIGINT, original_count=1)

    assert result.returncode == 130
    assert "scale:0" not in calls


# ---------------------------------------------------------------------------
# Issue 2: Reflection credential must never appear in optimize argv
# ---------------------------------------------------------------------------


def test_round_script_optimize_credential_not_in_argv(tmp_path: Path) -> None:
    """GROUNDING_REFLECTION_CREDENTIAL must not be passed as a CLI arg to optimize."""
    credential = _BASE_ENV["GROUNDING_REFLECTION_CREDENTIAL"]
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        round_type="optimize-evaluate",
    )

    assert result.returncode == 0, result.stderr
    assert "optimize" in calls
    # The credential value must not appear in any recorded argument
    leaked = [line for line in calls if line.startswith("optimize arg=") and credential in line]
    assert not leaked, f"credential leaked into optimize argv: {leaked}"


def test_round_script_optimize_credential_not_in_report_args(tmp_path: Path) -> None:
    """GROUNDING_REFLECTION_CREDENTIAL must not appear in report args either."""
    credential = _BASE_ENV["GROUNDING_REFLECTION_CREDENTIAL"]
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        round_type="optimize-evaluate",
    )

    assert result.returncode == 0, result.stderr
    leaked = [line for line in calls if line.startswith("report arg=") and credential in line]
    assert not leaked, f"credential leaked into report argv: {leaked}"
