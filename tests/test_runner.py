from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from korvid_prompt_lab import runner
from korvid_prompt_lab.contracts import (
    Campaign,
    Candidate,
    EvalCase,
    KorvidUpstreamServing,
)
from korvid_prompt_lab.runner import (
    BridgeExecutionModeError,
    BridgeInvocationError,
    BridgeSystemError,
    KorvidRunner,
    _require_loopback_endpoint,
    _terminate_process_group,
)
from korvid_prompt_lab.scoring import EvaluationResult

ROOT = Path(__file__).resolve().parents[1]
FAKE_PROCESS_TREE = ROOT / "tests" / "fixtures" / "fake_process_tree.py"


def _campaign() -> Campaign:
    model = "ollama/qwen3:0.6b"
    return Campaign(
        campaign_id="source",
        repetitions=1,
        models=(model,),
        cases=(
            EvalCase(
                case_id="image-pull-typo",
                template_id="korvid-scenario",
                prompt="Why is the pod failing?",
                models=(model,),
            ),
        ),
        serving=KorvidUpstreamServing(
            backend="korvid_upstream",
            source_root="/reviewed/korvid",
            base_url="http://127.0.0.1:11434",
            korvid_revision="33c483e041006eb20259a024ed85a9323e52c8f0",
            timeout_seconds=240,
        ),
    )


class _Runner:
    campaign = _campaign()

    def run(
        self,
        candidate: Candidate,
        case: EvalCase,
        run_dir: Path | str,
        *,
        repetition: int = 1,
        seed: int = 0,
    ) -> EvaluationResult:
        return EvaluationResult(
            success=True,
            execution_mode="live",
            candidate_fingerprint=candidate.fingerprint,
            feedback={"case_id": case.case_id},
            usage={"seed": seed, "repetition": repetition},
        )


def test_runner_protocol_returns_only_current_evaluation_results() -> None:
    implementation = _Runner()
    assert isinstance(implementation, KorvidRunner)
    result = implementation.run(
        Candidate(1, "candidate", (("tier_pack", "prompt"),)),
        implementation.campaign.cases[0],
        "run",
    )
    assert isinstance(result, EvaluationResult)
    assert result.success is True


def test_only_current_runner_errors_and_helpers_remain() -> None:
    assert issubclass(BridgeInvocationError, BridgeSystemError)
    assert issubclass(BridgeExecutionModeError, BridgeSystemError)
    for name in (
        "KorvidProcessRunner",
        "BridgeTimeoutError",
        "BridgeArtifactError",
        "BridgeProcessExitError",
        "BridgeMissingOutputError",
        "BridgeMalformedOutputError",
        "BridgeProtocolMismatchError",
        "BridgeFingerprintMismatchError",
        "BridgeIdentityMismatchError",
        "BridgeStatusError",
        "launcher_timeout_seconds",
        "BRIDGE_TIMEOUT_ENV",
    ):
        assert not hasattr(runner, name)


@pytest.mark.parametrize(
    "endpoint",
    (
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://[::1]:11434",
    ),
)
def test_loopback_endpoint_validation_accepts_current_source_endpoints(
    endpoint: str,
) -> None:
    assert _require_loopback_endpoint(endpoint) == endpoint


@pytest.mark.parametrize(
    "endpoint",
    (
        "https://127.0.0.1:11434",
        "http://example.com:11434",
        "http://127.0.0.1",
        "http://127.0.0.1:11434/v1",
        "http://127.0.0.1:11434?query=yes",
    ),
)
def test_loopback_endpoint_validation_rejects_remote_or_non_base_urls(
    endpoint: str,
) -> None:
    with pytest.raises(ValueError, match="loopback|base URL"):
        _require_loopback_endpoint(endpoint)


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(os.name != "posix", reason="process groups are a POSIX guarantee")
def test_owned_process_group_termination_leaves_no_source_worker_descendant(
    tmp_path: Path,
) -> None:
    response = tmp_path / "response.json"
    pid_file = tmp_path / "pids.txt"
    pid_file.touch()
    process = subprocess.Popen(
        [
            sys.executable,
            str(FAKE_PROCESS_TREE),
            "--request",
            str(tmp_path / "request.json"),
            "--response",
            str(response),
        ],
        env={
            **os.environ,
            "FAKE_TREE_PID_FILE": str(pid_file),
            "FAKE_TREE_DEPTH": "2",
            "FAKE_TREE_SLEEP": "2",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while len(pid_file.read_text(encoding="utf-8").splitlines()) < 4:
        if time.monotonic() >= deadline:
            pytest.fail("fake source worker tree did not start")
        time.sleep(0.05)

    _terminate_process_group(process)

    pids = {
        label: int(pid)
        for label, pid in (
            line.split(":", 1)
            for line in pid_file.read_text(encoding="utf-8").splitlines()
        )
    }
    descendants = {label: pid for label, pid in pids.items() if label != "parent"}
    assert {label: _process_is_alive(pid) for label, pid in descendants.items()} == {
        label: False for label in descendants
    }
    time.sleep(2.2)
    assert not response.exists()
