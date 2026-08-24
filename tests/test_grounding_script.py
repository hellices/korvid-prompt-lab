"""Process-boundary tests for scripts/run-grounding-round.sh.

Each test spins up a minimal fake PATH containing az, kubectl,
korvid-prompt-lab, and korvid-grounding-report shims, then runs the
real script as a subprocess.  Call records are captured via a shared
``calls.txt`` file so we can assert ordering without mocking internals.

The ``korvid-prompt-lab`` shim is a *strict* fake parser: it accepts exactly the
options the real :func:`korvid_prompt_lab.cli.build_parser` accepts for each
subcommand, enforces the same required options, and exits 2 with an argparse-
shaped usage error otherwise.  Every argv the orchestrator emits is additionally
replayed through the **real** parser, so an argv-incompatible invocation can
never pass these tests again.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import textwrap
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from korvid_prompt_lab.cli import build_parser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run-grounding-round.sh"

CAMPAIGN_PATH = "examples/campaigns/aks-shared-runners.yaml"
TRAIN_CASE_ID = "aks-scale-deployment-up"
VALIDATION_CASE_ID = "aks-restart-denied"
MILESTONE_CASE_IDS = "aks-scale-deployment-up,aks-restart-denied"
MAX_METRIC_CALLS = "7"
SEED = "3"
CANDIDATE_PATH = "examples/candidates/shipped-small.yaml"

#: Deterministic 64-hex-char fingerprints for the fake optimize shim.  Real
#: fingerprints are content hashes; these stand in for "seed" and "an
#: optimized-and-changed candidate" so `optimization_changed()` in the script
#: under test can validate them with the same regex it uses in production.
_SEED_FINGERPRINT = hashlib.sha256(b"seed-candidate-content").hexdigest()
_CHANGED_FINGERPRINT = hashlib.sha256(b"optimized-candidate-content").hexdigest()

_BASE_ENV: dict[str, str] = {
    "GROUNDING_MODEL": "qwen3:1.7b",
    "GROUNDING_CANDIDATE": CANDIDATE_PATH,
    "GROUNDING_CAMPAIGN": CAMPAIGN_PATH,
    "GROUNDING_ROUND_TYPE": "evaluate",
    "GROUNDING_TRAIN_CASE_ID": TRAIN_CASE_ID,
    "GROUNDING_VALIDATION_CASE_ID": VALIDATION_CASE_ID,
    "GROUNDING_MILESTONE_CASE_IDS": MILESTONE_CASE_IDS,
    "GROUNDING_MAX_METRIC_CALLS": MAX_METRIC_CALLS,
    "GROUNDING_SEED": SEED,
    "KORVID_SOURCE_ROOT": "/fake/korvid",
    "KORVID_AKS_MODEL": "qwen3:1.7b",
    "KORVID_AKS_NAMESPACE": "ollama",
    "KORVID_AKS_SERVICE": "ollama",
    "GROUNDING_ARTIFACT_ROOT": "",  # overridden per test
    "WORKFLOW_RUN_URL": "https://github.com/org/repo/actions/runs/1",
    "PROMPT_LAB_REVISION": "abc123",
    "KORVID_REVISION": "def456",
    "GROUNDING_REFLECTION_MODEL": "openai/gpt-4.1-mini",
    "GROUNDING_REFLECTION_CREDENTIAL": "fake-token",
}

#: Options the strict fake parser accepts, mirroring ``build_parser()``.  The
#: ``test_fake_parser_*`` tests below prove this table has not drifted from the
#: real CLI, so a shim that silently swallows a nonexistent flag fails loudly.
_FAKE_PARSER_SPEC: dict[str, dict[str, str]] = {
    "aks-check": {
        "value": "--campaign --artifact-root",
        "flag": "",
        "required": "--campaign",
    },
    "evaluate": {
        "value": (
            "--candidate --campaign --artifact-root --case-id --train-case-id "
            "--validation-case-id --milestone-case-id --bundle-kind"
        ),
        "flag": "--json",
        "required": "--candidate --campaign",
    },
    "optimize": {
        "value": (
            "--candidate --campaign --artifact-root --max-metric-calls "
            "--reflection-model --seed --train-case-id --validation-case-id"
        ),
        "flag": "",
        "required": "--candidate --campaign --max-metric-calls",
    },
}


def recorded_argv_all(calls: Sequence[str], prefix: str) -> list[list[str]]:
    """Reconstruct every argv the orchestrator passed for one subcommand, in
    call order. A subcommand may run more than once per round (e.g.
    ``evaluate`` for both the seed and best-candidate comparison), so this
    returns one argv list per invocation."""
    counts_marker = f"{prefix} argc="
    arg_marker = f"{prefix} arg="
    filtered = [line for line in calls if line.startswith((counts_marker, arg_marker))]

    invocations: list[list[str]] = []
    current: list[str] | None = None
    expected_argc = 0
    for line in filtered:
        if line.startswith(counts_marker):
            if current is not None:
                assert len(current) == expected_argc, (
                    f"{prefix} argc={expected_argc} but recorded {len(current)} args"
                )
                invocations.append(current)
            expected_argc = int(line[len(counts_marker) :])
            current = []
        else:
            assert current is not None, f"{prefix} arg= recorded before any argc= line"
            current.append(line[len(arg_marker) :])
    if current is not None:
        assert len(current) == expected_argc, (
            f"{prefix} argc={expected_argc} but recorded {len(current)} args"
        )
        invocations.append(current)
    return invocations


def recorded_argv(calls: Sequence[str], prefix: str) -> list[str]:
    """Reconstruct the argv the orchestrator passed for the single expected
    invocation of one subcommand."""
    invocations = recorded_argv_all(calls, prefix)
    assert len(invocations) == 1, (
        f"expected exactly one {prefix!r} invocation, got {len(invocations)}"
    )
    return invocations[0]


def option_value(argv: Sequence[str], option: str) -> str:
    assert option in argv, f"{option} missing from argv {list(argv)}"
    return argv[argv.index(option) + 1]


def option_values(argv: Sequence[str], option: str) -> list[str]:
    return [argv[index + 1] for index, token in enumerate(argv) if token == option]


def assert_report_arg(calls: Sequence[str], option: str) -> None:
    """Assert that the recorded report invocation carries a bare flag/option."""
    argv = recorded_argv(calls, "report")
    assert option in argv, f"{option} missing from report argv {argv}"


def assert_report_value(calls: Sequence[str], option: str, expected: str) -> None:
    """Assert that the recorded report invocation carries option=expected."""
    argv = recorded_argv(calls, "report")
    assert option_value(argv, option) == expected, (
        f"expected {option}={expected!r} in report argv {argv}"
    )


def artifact_path(tmp_path: Path, *segments: str, artifact_root_name: str = "artifacts") -> str:
    """Build the expected string form of a path under the run's artifact root."""
    return str(tmp_path.joinpath(artifact_root_name, *segments))


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
        start_new_session=True,  # isolate in a new session/process group so the
        # signal reaches bash and all its children (e.g. the blocking sleep)
        # without propagating back to the pytest runner
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
    calls = [
        line
        for line in calls_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return subprocess.CompletedProcess(
        proc.args, proc.returncode, stdout, stderr
    ), calls


def _make_fake_bin(
    fake_bin_dir: Path,
    *,
    node_count: int,
    evaluation_exit: int,
    preflight_exit: int,
    calls_file: Path,
    optimize_mode: str = "actual-layout",
    preflight_success_after: int = 0,
    emit_evaluation_summary: bool = True,
    optimize_changed: bool = True,
    evaluation_exits: Sequence[int] | None = None,
    evaluation_systemic_failures: Sequence[int] | None = None,
) -> None:
    """Write shim executables into *fake_bin_dir*."""
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    attempts_file = fake_bin_dir.parent / "aks-check-attempts"
    eval_attempts_file = fake_bin_dir.parent / "evaluate-attempts"

    # Each evaluate invocation (seed "before" eval, then best-candidate "after"
    # eval) can be configured independently; a shorter list reuses its last
    # entry for every later invocation, so a single-element list (the common
    # case) applies uniformly to both.
    _exits = list(evaluation_exits) if evaluation_exits is not None else [evaluation_exit]
    _systemic = (
        list(evaluation_systemic_failures) if evaluation_systemic_failures is not None else [0]
    )
    eval_exits_literal = " ".join(str(value) for value in _exits)
    eval_systemic_literal = " ".join(str(value) for value in _systemic)
    emit_evaluation_summary_bash = "true" if emit_evaluation_summary else "false"

    optimization_summary_json = json.dumps(
        {
            "seed_candidate_fingerprint": _SEED_FINGERPRINT,
            "best_candidate_fingerprint": (
                _CHANGED_FINGERPRINT if optimize_changed else _SEED_FINGERPRINT
            ),
            "best_candidate_differs_from_seed": optimize_changed,
        }
    )

    def _write(name: str, body: str) -> None:
        p = fake_bin_dir / name
        p.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body), encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    _write(
        "az",
        f"""\
        CALLS="{calls_file}"
        if [[ "$*" == *"nodepool show"* ]]; then
            echo "nodepool-show" >> "$CALLS"
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

        # --- strict fake parser -------------------------------------------
        # Mirrors korvid_prompt_lab.cli.build_parser(): unknown options and
        # missing required options exit 2 exactly like argparse, so an
        # argv-incompatible orchestrator can never look healthy here.
        _subcommand="${{1:-}}"
        _usage_error() {{
            echo "korvid-prompt-lab ${{_subcommand}}: error: $1" >&2
            exit 2
        }}

        case "$_subcommand" in
            aks-check)
                _value_opts="{_FAKE_PARSER_SPEC["aks-check"]["value"]}"
                _flag_opts="{_FAKE_PARSER_SPEC["aks-check"]["flag"]}"
                _required_opts="{_FAKE_PARSER_SPEC["aks-check"]["required"]}"
                ;;
            evaluate)
                _value_opts="{_FAKE_PARSER_SPEC["evaluate"]["value"]}"
                _flag_opts="{_FAKE_PARSER_SPEC["evaluate"]["flag"]}"
                _required_opts="{_FAKE_PARSER_SPEC["evaluate"]["required"]}"
                ;;
            optimize)
                _value_opts="{_FAKE_PARSER_SPEC["optimize"]["value"]}"
                _flag_opts="{_FAKE_PARSER_SPEC["optimize"]["flag"]}"
                _required_opts="{_FAKE_PARSER_SPEC["optimize"]["required"]}"
                ;;
            *)
                echo "korvid-prompt-lab: error: argument command: invalid choice: '$_subcommand'" >&2
                exit 2
                ;;
        esac

        echo "$_subcommand" >> "$CALLS"
        if [[ "$_subcommand" == "aks-check" ]]; then
            _attempts_file="{attempts_file}"
            _attempts=$(( $(cat "$_attempts_file" 2>/dev/null || echo 0) + 1 ))
            echo "$_attempts" > "$_attempts_file"
            # Record argv once: a retried preflight would otherwise bury the
            # single interesting argv under hundreds of identical records.
            if (( _attempts == 1 )); then
                _record_args "$_subcommand" "$@"
            fi
        else
            _record_args "$_subcommand" "$@"
            if [[ "$_subcommand" == "optimize" ]]; then
                printf 'optimize env OPENAI_API_KEY=%s\n' "${{OPENAI_API_KEY:+set}}" >> "$CALLS"
                printf 'optimize env ANTHROPIC_API_KEY=%s\n' "${{ANTHROPIC_API_KEY:+set}}" >> "$CALLS"
                printf 'optimize env OLLAMA_API_BASE=%s\n' "${{OLLAMA_API_BASE:-}}" >> "$CALLS"
            fi
        fi

        _candidate_arg=""
        _artifact_root_arg=""
        _seen=""
        shift
        while (( $# )); do
            case " $_value_opts " in
                *" $1 "*)
                    if (( $# < 2 )); then
                        _usage_error "argument $1: expected one argument"
                    fi
                    case "$1" in
                        --candidate) _candidate_arg="$2" ;;
                        --artifact-root) _artifact_root_arg="$2" ;;
                    esac
                    _seen="$_seen $1"
                    shift 2
                    continue
                    ;;
            esac
            case " $_flag_opts " in
                *" $1 "*)
                    _seen="$_seen $1"
                    shift
                    continue
                    ;;
            esac
            _usage_error "unrecognized arguments: $1"
        done
        for _req in $_required_opts; do
            case " $_seen " in
                *" $_req "*) ;;
                *) _usage_error "the following arguments are required: $_req" ;;
            esac
        done
        # --- end strict fake parser ---------------------------------------

        if [[ "$_subcommand" == "aks-check" ]]; then
            if (( {preflight_success_after} > 0 && _attempts > {preflight_success_after} )); then
                exit 0
            fi
            exit {preflight_exit}
        elif [[ "$_subcommand" == "evaluate" ]]; then
            _eval_attempts_file="{eval_attempts_file}"
            _eval_attempt=$(( $(cat "$_eval_attempts_file" 2>/dev/null || echo 0) + 1 ))
            echo "$_eval_attempt" > "$_eval_attempts_file"

            _eval_exits=({eval_exits_literal})
            _eval_systemic=({eval_systemic_literal})
            _exit_idx=$(( _eval_attempt - 1 ))
            if (( _exit_idx >= ${{#_eval_exits[@]}} )); then
                _exit_idx=$(( ${{#_eval_exits[@]}} - 1 ))
            fi
            _systemic_idx=$(( _eval_attempt - 1 ))
            if (( _systemic_idx >= ${{#_eval_systemic[@]}} )); then
                _systemic_idx=$(( ${{#_eval_systemic[@]}} - 1 ))
            fi
            _this_exit="${{_eval_exits[$_exit_idx]}}"
            _this_systemic="${{_eval_systemic[$_systemic_idx]}}"
            _this_hard=0
            if (( _this_exit == 1 )); then
                _this_hard=1
            fi

            printf 'evaluate candidate=%s artifact_root=%s\\n' "$_candidate_arg" "$_artifact_root_arg" >> "$CALLS"
            if {emit_evaluation_summary_bash}; then
                mkdir -p "$_artifact_root_arg"
                printf '{{"hard_safety_failures": %s, "systemic_failures": %s}}\\n' \\
                    "$_this_hard" "$_this_systemic" > "$_artifact_root_arg/evaluation-summary.json"
            fi
            exit "$_this_exit"
        elif [[ "$_subcommand" == "optimize" ]]; then
            case "{optimize_mode}" in
                actual-layout)
                    mkdir -p "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-1"
                    echo "candidate: optimized" > "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-1/best-candidate.yaml"
                    printf '%s\\n' '{optimization_summary_json}' \\
                        > "${{GROUNDING_ARTIFACT_ROOT}}/optimize/invocations/opt-run-1/optimization-summary.json"
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
    preflight_success_after: int = 0,
    emit_evaluation_summary: bool = True,
    optimize_changed: bool = True,
    evaluation_exits: Sequence[int] | None = None,
    evaluation_systemic_failures: Sequence[int] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    artifact_root = tmp_path / artifact_root_name
    artifact_root.mkdir(parents=True)
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
        preflight_success_after=preflight_success_after,
        emit_evaluation_summary=emit_evaluation_summary,
        optimize_changed=optimize_changed,
        evaluation_exits=evaluation_exits,
        evaluation_systemic_failures=evaluation_systemic_failures,
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
    calls = [
        line
        for line in calls_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return result, calls


# ---------------------------------------------------------------------------
# Tests (RED phase: all fail until script exists)
# ---------------------------------------------------------------------------


def test_round_script_restores_zero_count_after_unsafe_evaluation(
    tmp_path: Path,
) -> None:
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


def test_round_script_restores_pool_on_signal_or_systemic_failure(
    tmp_path: Path,
) -> None:
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


def test_round_script_optimize_evaluate_runs_optimize_then_evaluate(
    tmp_path: Path,
) -> None:
    """optimize-evaluate must evaluate the seed candidate, then optimize, then
    evaluate the changed best candidate — seed evidence always precedes
    optimization, and the after-evaluation always follows it."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        round_type="optimize-evaluate",
    )

    assert result.returncode == 0
    assert "optimize" in calls
    evaluate_indexes = [index for index, call in enumerate(calls) if call == "evaluate"]
    assert len(evaluate_indexes) == 2
    optimize_index = calls.index("optimize")
    assert evaluate_indexes[0] < optimize_index < evaluate_indexes[1]


def test_round_script_optimize_evaluate_uses_invocation_artifacts_and_preserves_spaced_paths(
    tmp_path: Path,
) -> None:
    """optimize-evaluate must use the invocation-layout artifacts and keep spaced paths as one argv token."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        round_type="optimize-evaluate",
        artifact_root_name="artifacts with spaces",
    )

    assert result.returncode == 0, result.stderr
    assert (
        f"evaluate arg={tmp_path / 'artifacts with spaces' / 'optimize' / 'invocations' / 'opt-run-1' / 'best-candidate.yaml'}"
        in calls
    )
    assert f"report arg={tmp_path / 'artifacts with spaces' / 'evaluate'}" in calls
    assert (
        f"report arg={tmp_path / 'artifacts with spaces' / 'optimize' / 'invocations' / 'opt-run-1'}"
        in calls
    )


def test_round_script_optimize_evaluate_rejects_missing_best_candidate(
    tmp_path: Path,
) -> None:
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
    # The seed evaluation runs before optimize and so has already completed,
    # but the best-candidate evaluation must never run when optimize fails.
    assert sum(call.startswith("evaluate candidate=") for call in calls) == 1
    assert "report" not in calls


def test_round_script_optimize_evaluate_rejects_ambiguous_best_candidate(
    tmp_path: Path,
) -> None:
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
    # The seed evaluation runs before optimize and so has already completed,
    # but the best-candidate evaluation must never run when optimize fails.
    assert sum(call.startswith("evaluate candidate=") for call in calls) == 1
    assert "report" not in calls


# ---------------------------------------------------------------------------
# Issue 1: Signal cancellation — exactly one cleanup and conventional exit code
# ---------------------------------------------------------------------------


def test_round_script_sigint_exits_130_and_scales_down_exactly_once(
    tmp_path: Path,
) -> None:
    """SIGINT must exit 130 and execute nodepool scale-down exactly once."""
    result, calls = _run_script_with_signal(tmp_path, signal.SIGINT, original_count=0)

    assert result.returncode == 130, (
        f"expected 130, got {result.returncode}\nstderr: {result.stderr}"
    )
    assert calls.count("scale:0") == 1, (
        f"expected exactly 1 scale:0, got {calls.count('scale:0')}\ncalls: {calls}"
    )


def test_round_script_sigterm_exits_143_and_scales_down_exactly_once(
    tmp_path: Path,
) -> None:
    """SIGTERM must exit 143 and execute nodepool scale-down exactly once."""
    result, calls = _run_script_with_signal(tmp_path, signal.SIGTERM, original_count=0)

    assert result.returncode == 143, (
        f"expected 143, got {result.returncode}\nstderr: {result.stderr}"
    )
    assert calls.count("scale:0") == 1, (
        f"expected exactly 1 scale:0, got {calls.count('scale:0')}\ncalls: {calls}"
    )


def test_round_script_sigint_no_scale_when_pool_already_had_capacity(
    tmp_path: Path,
) -> None:
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
    leaked = [
        line
        for line in calls
        if line.startswith("optimize arg=") and credential in line
    ]
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
    leaked = [
        line for line in calls if line.startswith("report arg=") and credential in line
    ]
    assert not leaked, f"credential leaked into report argv: {leaked}"


def test_round_script_rejects_optimize_without_reflection_model_before_any_cloud_call(
    tmp_path: Path,
) -> None:
    """Missing reflection config must fail before the pool is read or scaled up."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        round_type="optimize-evaluate",
        extra_env={"GROUNDING_REFLECTION_MODEL": ""},
    )

    assert result.returncode != 0
    assert "GROUNDING_REFLECTION_MODEL" in result.stderr
    assert calls == [], (
        "misconfiguration must not read, scale, or wait on the modeleval pool"
    )


def test_round_script_rejects_optimize_without_reflection_credential_before_any_cloud_call(
    tmp_path: Path,
) -> None:
    """A missing reflection credential must not cost a GPU node or a 15-minute wait."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        round_type="optimize-evaluate",
        extra_env={"GROUNDING_REFLECTION_CREDENTIAL": ""},
    )

    assert result.returncode != 0
    assert "GROUNDING_REFLECTION_CREDENTIAL" in result.stderr
    assert calls == []


def test_round_script_ollama_reflection_needs_no_credential_and_uses_cluster_dns(
    tmp_path: Path,
) -> None:
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        round_type="optimize-evaluate",
        extra_env={
            "GROUNDING_REFLECTION_MODEL": "ollama_chat/qwen3:14b",
            "GROUNDING_REFLECTION_CREDENTIAL": "",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "optimize env OPENAI_API_KEY=" in calls
    assert "optimize env ANTHROPIC_API_KEY=" in calls
    assert (
        "optimize env OLLAMA_API_BASE="
        "http://ollama.ollama.svc.cluster.local:11434"
    ) in calls


def test_round_script_hosted_reflection_still_requires_credential(
    tmp_path: Path,
) -> None:
    result, calls = run_script(
        tmp_path,
        original_count=0,
        round_type="optimize-evaluate",
        extra_env={
            "GROUNDING_REFLECTION_MODEL": "openai/gpt-4.1-mini",
            "GROUNDING_REFLECTION_CREDENTIAL": "",
        },
    )

    assert result.returncode != 0
    assert "GROUNDING_REFLECTION_CREDENTIAL" in result.stderr
    assert calls == []


@pytest.mark.parametrize(
    "model",
    ["ollama_chat", "ollama_chat/", "ollama chat/qwen3:14b", "ollama_chat/qwen3:14b;env"],
)
def test_round_script_rejects_malformed_reflection_model_before_cloud(
    tmp_path: Path,
    model: str,
) -> None:
    result, calls = run_script(
        tmp_path,
        original_count=0,
        round_type="optimize-evaluate",
        extra_env={
            "GROUNDING_REFLECTION_MODEL": model,
            "GROUNDING_REFLECTION_CREDENTIAL": "",
        },
    )

    assert result.returncode == 2
    assert "invalid reflection model" in result.stderr
    assert calls == []


def test_round_script_evaluate_round_ignores_absent_reflection_config(
    tmp_path: Path,
) -> None:
    """Evaluate-only rounds must run without any reflection credential."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        extra_env={
            "GROUNDING_REFLECTION_MODEL": "",
            "GROUNDING_REFLECTION_CREDENTIAL": "",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "evaluate" in calls
    assert "optimize" not in calls


# ---------------------------------------------------------------------------
# Task 3 finding 1: the orchestrator must speak the real CLI's argv
# ---------------------------------------------------------------------------


def test_fake_parser_spec_matches_the_real_cli_parser() -> None:
    """The strict shim's option table must not drift from build_parser()."""
    parser = build_parser()

    minimal = {
        "aks-check": ["aks-check", "--campaign", "c.yaml"],
        "evaluate": ["evaluate", "--candidate", "x.yaml", "--campaign", "c.yaml"],
        "optimize": [
            "optimize",
            "--candidate",
            "x.yaml",
            "--campaign",
            "c.yaml",
            "--max-metric-calls",
            "4",
        ],
    }

    for subcommand, spec in _FAKE_PARSER_SPEC.items():
        base = minimal[subcommand]
        parsed = parser.parse_args(base)
        assert parsed.command == subcommand

        for option in spec["value"].split():
            if option in base:
                continue
            probe = "common" if option == "--bundle-kind" else "1"
            parser.parse_args([*base, option, probe])
        for option in spec["flag"].split():
            parser.parse_args([*base, option])

        for required in spec["required"].split():
            stripped = list(base)
            index = stripped.index(required)
            del stripped[index : index + 2]
            with pytest.raises(SystemExit) as excinfo:
                parser.parse_args(stripped)
            assert excinfo.value.code == 2, f"{subcommand} {required} must be required"


def test_real_cli_parser_rejects_the_previous_orchestrator_argv() -> None:
    """Regression guard: the argv this task removed must still be a usage error."""
    parser = build_parser()
    removed = (
        ["aks-check", "--korvid-source-root", "/fake/korvid", "--model", "qwen3:1.7b"],
        [
            "evaluate",
            "--candidate",
            "seed",
            "--model",
            "qwen3:1.7b",
            "--korvid-source-root",
            "/fake/korvid",
            "--artifact-root",
            "a",
        ],
        [
            "optimize",
            "--candidate",
            "seed",
            "--reflection-model",
            "qwen3:0.6b",
            "--korvid-source-root",
            "/fake/korvid",
            "--artifact-root",
            "a",
        ],
    )
    for argv in removed:
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(argv)
        assert excinfo.value.code == 2


def test_round_script_argv_is_accepted_by_the_real_cli_parser(tmp_path: Path) -> None:
    """Every argv the orchestrator emits must parse against the installed CLI."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        round_type="optimize-evaluate",
    )

    assert result.returncode == 0, result.stderr

    parser = build_parser()
    for subcommand in ("aks-check", "optimize"):
        argv = recorded_argv(calls, subcommand)
        assert argv[0] == subcommand
        assert "--korvid-source-root" not in argv, (
            f"{subcommand} passes --korvid-source-root, which the CLI does not define; "
            "the source root is runtime policy read from KORVID_SOURCE_ROOT"
        )
        assert "--model" not in argv, (
            f"{subcommand} passes --model, which the CLI does not define; the model "
            "comes from the campaign"
        )
        parsed = parser.parse_args(argv)  # SystemExit(2) if argv-incompatible
        assert parsed.command == subcommand
        assert str(parsed.campaign) == CAMPAIGN_PATH

    # optimize-evaluate evaluates twice under an identical contract: the seed
    # candidate before optimization, then the best candidate after — both
    # argvs must independently parse against the real CLI.
    evaluate_invocations = recorded_argv_all(calls, "evaluate")
    assert len(evaluate_invocations) == 2
    for argv in evaluate_invocations:
        assert argv[0] == "evaluate"
        assert "--korvid-source-root" not in argv, (
            "evaluate passes --korvid-source-root, which the CLI does not define; "
            "the source root is runtime policy read from KORVID_SOURCE_ROOT"
        )
        assert "--model" not in argv, (
            "evaluate passes --model, which the CLI does not define; the model "
            "comes from the campaign"
        )
        parsed = parser.parse_args(argv)  # SystemExit(2) if argv-incompatible
        assert parsed.command == "evaluate"
        assert str(parsed.campaign) == CAMPAIGN_PATH


def test_round_script_evaluate_argv_carries_campaign_case_sets(tmp_path: Path) -> None:
    result, calls = run_script(tmp_path, original_count=0, evaluation_exit=0)

    assert result.returncode == 0, result.stderr
    argv = recorded_argv(calls, "evaluate")

    assert option_value(argv, "--campaign") == CAMPAIGN_PATH
    assert option_value(argv, "--candidate") == CANDIDATE_PATH
    assert option_value(argv, "--train-case-id") == TRAIN_CASE_ID
    assert option_value(argv, "--validation-case-id") == VALIDATION_CASE_ID
    assert option_values(argv, "--milestone-case-id") == MILESTONE_CASE_IDS.split(",")
    assert option_value(argv, "--artifact-root") == str(
        tmp_path / "artifacts" / "evaluate"
    )


def test_round_script_optimize_argv_carries_budget_seed_and_splits(
    tmp_path: Path,
) -> None:
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        round_type="optimize-evaluate",
    )

    assert result.returncode == 0, result.stderr
    argv = recorded_argv(calls, "optimize")

    assert option_value(argv, "--campaign") == CAMPAIGN_PATH
    assert option_value(argv, "--max-metric-calls") == MAX_METRIC_CALLS
    assert option_value(argv, "--seed") == SEED
    assert (
        option_value(argv, "--reflection-model")
        == _BASE_ENV["GROUNDING_REFLECTION_MODEL"]
    )
    assert option_value(argv, "--train-case-id") == TRAIN_CASE_ID
    assert option_value(argv, "--validation-case-id") == VALIDATION_CASE_ID


def test_round_script_aks_check_argv_is_campaign_only(tmp_path: Path) -> None:
    result, calls = run_script(tmp_path, original_count=0, evaluation_exit=0)

    assert result.returncode == 0, result.stderr
    argv = recorded_argv(calls, "aks-check")
    assert option_value(argv, "--campaign") == CAMPAIGN_PATH
    assert option_value(argv, "--artifact-root") == str(
        tmp_path / "artifacts" / "aks-check"
    )


# ---------------------------------------------------------------------------
# Task 3 finding 1: a usage/config failure must abort, not spin for 15 minutes
# ---------------------------------------------------------------------------


def test_round_script_aborts_immediately_when_aks_check_reports_a_config_error(
    tmp_path: Path,
) -> None:
    """exit 2 from aks-check is systemic configuration, never 'not ready yet'."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        preflight_exit=2,
        extra_env={"_AKS_CHECK_DEADLINE_SECONDS": "900"},
    )

    assert result.returncode == 2, result.stderr
    assert calls.count("aks-check") == 1, (
        f"a configuration failure must not be retried; got {calls.count('aks-check')} attempts"
    )
    assert "timed out" not in result.stderr
    assert calls[-1] == "scale:0", "the node the round scaled up must still be released"


def test_round_script_aborts_immediately_when_aks_check_reports_permanent_failure(
    tmp_path: Path,
) -> None:
    """exit 1 from aks-check is a permanent preflight failure, never retried."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        preflight_exit=1,
        extra_env={"_AKS_CHECK_DEADLINE_SECONDS": "900"},
    )

    assert result.returncode == 1, result.stderr
    assert calls.count("aks-check") == 1, (
        f"a permanent failure must not be retried; got {calls.count('aks-check')} attempts"
    )
    assert "timed out" not in result.stderr
    assert calls[-1] == "scale:0"


def test_round_script_retries_aks_check_while_the_pool_warms_up(tmp_path: Path) -> None:
    """exit 75 (EX_TEMPFAIL) from aks-check stays retryable until the deadline."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        preflight_exit=75,
        preflight_success_after=2,
        evaluation_exit=0,
        extra_env={"_AKS_CHECK_DEADLINE_SECONDS": "60"},
    )

    assert result.returncode == 0, result.stderr
    assert calls.count("aks-check") == 3
    assert "evaluate" in calls


# ---------------------------------------------------------------------------
# Task 3 finding 2: campaign, case, and serving configuration are required
# ---------------------------------------------------------------------------


def test_round_script_requires_campaign_before_any_cloud_call(tmp_path: Path) -> None:
    result, calls = run_script(
        tmp_path, original_count=0, extra_env={"GROUNDING_CAMPAIGN": ""}
    )

    assert result.returncode != 0
    assert "GROUNDING_CAMPAIGN" in result.stderr
    assert calls == []


def test_round_script_requires_campaign_serving_environment(tmp_path: Path) -> None:
    """The campaign resolves models/namespace/service through env: references."""
    for name in ("KORVID_AKS_MODEL", "KORVID_AKS_NAMESPACE", "KORVID_AKS_SERVICE"):
        result, calls = run_script(
            tmp_path / name, original_count=0, extra_env={name: ""}
        )

        assert result.returncode != 0, f"{name} must be required"
        assert name in result.stderr
        assert calls == [], f"a missing {name} must not cost cluster time"


def test_round_script_requires_the_served_model_to_match_the_allowlisted_model(
    tmp_path: Path,
) -> None:
    result, calls = run_script(
        tmp_path,
        original_count=0,
        extra_env={"KORVID_AKS_MODEL": "qwen3:14b"},
    )

    assert result.returncode == 2
    assert "KORVID_AKS_MODEL" in result.stderr
    assert calls == []


def test_round_script_requires_case_identifiers(tmp_path: Path) -> None:
    for name in (
        "GROUNDING_TRAIN_CASE_ID",
        "GROUNDING_VALIDATION_CASE_ID",
        "GROUNDING_MILESTONE_CASE_IDS",
    ):
        result, calls = run_script(
            tmp_path / name, original_count=0, extra_env={name: ""}
        )

        assert result.returncode != 0, f"{name} must be required"
        assert name in result.stderr
        assert calls == []


def test_round_script_requires_disjoint_train_and_validation_cases(
    tmp_path: Path,
) -> None:
    result, calls = run_script(
        tmp_path,
        original_count=0,
        extra_env={"GROUNDING_VALIDATION_CASE_ID": TRAIN_CASE_ID},
    )

    assert result.returncode == 2
    assert "disjoint" in result.stderr
    assert calls == []


def test_round_script_requires_a_positive_metric_call_budget(tmp_path: Path) -> None:
    for index, value in enumerate(("0", "-1", "1e3", "12; rm -rf /", "", "1 2")):
        result, calls = run_script(
            tmp_path / f"budget-{index}",
            original_count=0,
            round_type="optimize-evaluate",
            extra_env={"GROUNDING_MAX_METRIC_CALLS": value},
        )

        assert result.returncode != 0, f"metric budget {value!r} must be rejected"
        assert calls == []


def test_round_script_requires_a_non_negative_seed(tmp_path: Path) -> None:
    for index, value in enumerate(("-1", "abc", "0x10", "")):
        result, calls = run_script(
            tmp_path / f"seed-{index}",
            original_count=0,
            round_type="optimize-evaluate",
            extra_env={"GROUNDING_SEED": value},
        )

        assert result.returncode != 0, f"seed {value!r} must be rejected"
        assert calls == []


# ---------------------------------------------------------------------------
# Task 3 finding 2: milestone ids split into argv without word injection
# ---------------------------------------------------------------------------


def test_round_script_splits_a_single_milestone_id_into_one_argument(
    tmp_path: Path,
) -> None:
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        extra_env={"GROUNDING_MILESTONE_CASE_IDS": VALIDATION_CASE_ID},
    )

    assert result.returncode == 0, result.stderr
    argv = recorded_argv(calls, "evaluate")
    assert option_values(argv, "--milestone-case-id") == [VALIDATION_CASE_ID]


def test_round_script_rejects_hostile_milestone_case_ids(tmp_path: Path) -> None:
    """Splitting must never word-split, glob, or evaluate a hostile value."""
    hostile = (
        "aks-restart-denied extra-case",  # whitespace word splitting
        "*",  # pathname expansion
        "$(touch pwned)",  # command substitution
        "`touch pwned`",
        "a;b",
        "--milestone-case-id",  # option smuggling
        "aks-restart-denied,",  # empty trailing element
        ",aks-restart-denied",
        "aks-restart-denied,,aks-scale-deployment-up",
        "../../etc/passwd",
    )
    for index, value in enumerate(hostile):
        result, calls = run_script(
            tmp_path / f"hostile-{index}",
            original_count=0,
            evaluation_exit=0,
            extra_env={"GROUNDING_MILESTONE_CASE_IDS": value},
        )

        assert result.returncode == 2, (
            f"{value!r} must be rejected (got {result.returncode})"
        )
        assert calls == [], f"{value!r} must be rejected before any cloud call"
        assert not (tmp_path / f"hostile-{index}" / "pwned").exists()
        assert not Path("pwned").exists()


def test_round_script_rejects_hostile_train_and_validation_case_ids(
    tmp_path: Path,
) -> None:
    hostile = ("a b", "$(id)", "a,b", "-x", "")
    for index, value in enumerate(hostile):
        for name in ("GROUNDING_TRAIN_CASE_ID", "GROUNDING_VALIDATION_CASE_ID"):
            result, calls = run_script(
                tmp_path / f"{name}-{index}",
                original_count=0,
                extra_env={name: value},
            )

            assert result.returncode != 0, f"{name}={value!r} must be rejected"
            assert calls == []


def test_round_script_rejects_a_campaign_outside_the_checkout(tmp_path: Path) -> None:
    for index, value in enumerate(
        ("/etc/passwd", "../secrets.yaml", "campaign.yaml; touch pwned")
    ):
        result, calls = run_script(
            tmp_path / f"campaign-{index}",
            original_count=0,
            extra_env={"GROUNDING_CAMPAIGN": value},
        )

        assert result.returncode == 2, f"campaign {value!r} must be rejected"
        assert calls == []


# ---------------------------------------------------------------------------
# Task 3: Missing tools prerequisite, wrong service abort without retry
# ---------------------------------------------------------------------------


def test_round_script_exits_1_when_required_tool_missing(tmp_path: Path) -> None:
    """Missing az/kubectl/kubelogin/uv must exit 1 before any cloud call."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    calls_file = tmp_path / "calls.txt"
    calls_file.touch()

    # Create a fake PATH with only some tools — omit kubectl
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for tool in (
        "az",
        "kubelogin",
        "uv",
        "korvid-prompt-lab",
        "korvid-grounding-report",
    ):
        p = fake_bin / tool
        p.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        p.chmod(0o755)

    env = dict(os.environ)
    env.update(_BASE_ENV)
    env["GROUNDING_ARTIFACT_ROOT"] = str(artifact_root)
    env["PATH"] = f"{fake_bin}:/bin:/usr/bin"  # include system bash
    env["_AKS_CHECK_POLL_INTERVAL"] = "0"
    env["_AKS_CHECK_DEADLINE_SECONDS"] = "1"

    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=15,
    )
    calls = [
        line
        for line in calls_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result.returncode == 1
    assert "required tool not found" in result.stderr
    # No cloud calls should have been made
    assert "nodepool-show" not in calls


def test_round_script_does_not_retry_permanent_preflight_exit_1(tmp_path: Path) -> None:
    """Exit 1 from aks-check (permanent: wrong service, missing tool) aborts without retry."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        preflight_exit=1,
        extra_env={"_AKS_CHECK_DEADLINE_SECONDS": "60"},
    )

    assert result.returncode == 1, result.stderr
    assert calls.count("aks-check") == 1
    assert "not retrying" in result.stderr


# ---------------------------------------------------------------------------
# Task 3 Important finding: evaluate exit 1 must be distinguished by evidence
# ---------------------------------------------------------------------------


def test_round_script_evaluate_exit1_with_summary_is_safety_result(
    tmp_path: Path,
) -> None:
    """exit 1 + evaluation-summary.json present → safety result → runs report."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=1,
        emit_evaluation_summary=True,
    )

    assert result.returncode == 1
    assert "report" in calls, "safety exit 1 with summary must still run the report"
    assert calls[-1] == "scale:0"


def test_round_script_evaluate_exit1_without_summary_is_systemic_error(
    tmp_path: Path,
) -> None:
    """exit 1 + no evaluation-summary.json → systemic error → skip report, exit 1."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=1,
        emit_evaluation_summary=False,
    )

    assert result.returncode == 1
    assert "report" not in calls, (
        "systemic exit 1 (no summary) must skip report generation"
    )
    assert "systemic" in result.stderr.lower(), (
        "a concise systemic error message must be emitted"
    )
    # Must not dump a traceback
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout
    # Cleanup must still occur
    assert calls[-1] == "scale:0"


def test_round_script_evaluate_exit0_without_summary_is_systemic_error(
    tmp_path: Path,
) -> None:
    """exit 0 + no evaluation-summary.json → systemic error → exit non-zero."""
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        emit_evaluation_summary=False,
    )

    assert result.returncode != 0, (
        "exit 0 without evaluation-summary.json is systemic and must fail"
    )
    assert "report" not in calls
    assert "systemic" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Task 3: paired seed and best-candidate evaluation under an identical contract
# ---------------------------------------------------------------------------


def test_optimize_round_compares_seed_and_changed_best_with_identical_contract(
    tmp_path: Path,
) -> None:
    """optimize-evaluate must evaluate the seed before optimizing, then the
    changed best candidate, and pass both roots to the report."""
    result, calls = run_script(
        tmp_path,
        round_type="optimize-evaluate",
        optimize_changed=True,
    )

    assert result.returncode == 0, result.stderr
    evaluations = [call for call in calls if call.startswith("evaluate candidate=")]
    assert len(evaluations) == 2
    assert CANDIDATE_PATH in evaluations[0]
    assert "/evaluate-before" in evaluations[0]
    assert "best-candidate.yaml" in evaluations[1]
    assert "/evaluate" in evaluations[1]
    assert_report_arg(calls, "--before-artifact-root")


def test_unchanged_best_reuses_seed_evidence_without_second_evaluation(
    tmp_path: Path,
) -> None:
    """When optimize leaves the candidate unchanged, the seed evaluation is
    reused as the best-candidate evidence instead of evaluating twice."""
    result, calls = run_script(
        tmp_path,
        round_type="optimize-evaluate",
        optimize_changed=False,
    )

    assert result.returncode == 0, result.stderr
    evaluations = [call for call in calls if call.startswith("evaluate candidate=")]
    assert len(evaluations) == 1
    assert "/evaluate-before" in evaluations[0]
    assert_report_value(
        calls, "--artifact-root", artifact_path(tmp_path, "evaluate-before")
    )
    assert_report_value(
        calls, "--before-artifact-root", artifact_path(tmp_path, "evaluate-before")
    )


def test_before_safety_gate_continues_but_systemic_summary_aborts(
    tmp_path: Path,
) -> None:
    """A validated safety exit 1 on the seed evaluation must not abort the
    round, but a systemic (malformed/inconsistent) summary must abort before
    optimize ever runs."""
    safety_result, safety_calls = run_script(
        tmp_path / "safety",
        round_type="optimize-evaluate",
        evaluation_exits=[1],
        evaluation_systemic_failures=[0],
    )
    assert "optimize" in safety_calls
    assert safety_result.returncode in (0, 1)

    systemic_result, systemic_calls = run_script(
        tmp_path / "systemic",
        round_type="optimize-evaluate",
        evaluation_exits=[1],
        evaluation_systemic_failures=[1],
    )
    assert systemic_result.returncode != 0
    assert "optimize" not in systemic_calls
    assert "systemic" in systemic_result.stderr.lower()


def test_evaluate_only_still_runs_once_without_before_argument(tmp_path: Path) -> None:
    """Plain evaluate rounds have no seed/best comparison: exactly one
    evaluation, and no --before-artifact-root passed to the report."""
    result, calls = run_script(tmp_path, round_type="evaluate")
    assert result.returncode == 0, result.stderr
    assert sum(call.startswith("evaluate candidate=") for call in calls) == 1
    assert "--before-artifact-root" not in calls
