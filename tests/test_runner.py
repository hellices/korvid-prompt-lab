from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.artifacts import write_json_artifact
from korvid_prompt_lab.contracts import (
    AKSPortForwardServing,
    Campaign,
    Candidate,
    EvalCase,
    ProcessServing,
)
from korvid_prompt_lab.runner import (
    BridgeArtifactError,
    BridgeFingerprintMismatchError,
    BridgeIdentityMismatchError,
    BridgeInvocationError,
    BridgeMalformedOutputError,
    BridgeMissingOutputError,
    BridgeProcessExitError,
    BridgeProtocolMismatchError,
    BridgeStatusError,
    BridgeTimeoutError,
    KorvidProcessRunner,
)

ROOT = Path(__file__).resolve().parents[1]


def _candidate() -> Candidate:
    return Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "candidate-1",
            "components": {
                "system": "Stay safe.",
                "append": "Verify before you finish.",
            },
        }
    )


def _case(case_id: str) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        template_id="template-1",
        prompt="Confirm the postcondition.",
        models=("mock-small",),
    )


def _runner(case: EvalCase, *, timeout_seconds: float = 1.0, campaign_repetitions: int = 1) -> KorvidProcessRunner:
    command = (
        sys.executable,
        str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
        "--request",
        "{request}",
        "--response",
        "{response}",
    )
    campaign = Campaign(
        schema_version=1,
        campaign_id="campaign-1",
        repetitions=campaign_repetitions,
        models=("mock-small",),
        cases=(case,),
        serving=ProcessServing(backend="process", command=command),
    )
    return KorvidProcessRunner(campaign, timeout_seconds=timeout_seconds)


def _runner_with_command(case: EvalCase, command: tuple[str, ...], *, timeout_seconds: float = 1.0) -> KorvidProcessRunner:
    campaign = Campaign(
        schema_version=1,
        campaign_id="campaign-1",
        repetitions=1,
        models=("mock-small",),
        cases=(case,),
        serving=ProcessServing(backend="process", command=command),
    )
    return KorvidProcessRunner(campaign, timeout_seconds=timeout_seconds)


def test_write_json_artifact_replaces_files_atomically_without_temp_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "artifact.json"
    observed: dict[str, bool] = {}

    real_replace = os.replace

    def recording_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        observed["temp_exists_before_replace"] = Path(src).exists()
        real_replace(src, dst)

    monkeypatch.setattr("korvid_prompt_lab.artifacts.os.replace", recording_replace)

    write_json_artifact(artifact_path, {"step": 1})
    write_json_artifact(artifact_path, {"step": 2})

    assert json.loads(artifact_path.read_text(encoding="utf-8")) == {"step": 2}
    assert observed["temp_exists_before_replace"] is True
    assert list(tmp_path.glob("*.tmp")) == []


def test_runner_returns_completed_result_and_persists_json_artifacts(tmp_path: Path) -> None:
    case = _case("case[completed]")
    runner = _runner(case, campaign_repetitions=3)

    result = runner.run(_candidate(), case, tmp_path / "run", repetition=3, seed=17)

    assert result.status == "completed"
    assert result.candidate_fingerprint == _candidate().fingerprint
    assert result.grade is not None
    assert result.grade.hard_failures == ()

    request_path = tmp_path / "run" / "request.json"
    response_path = tmp_path / "run" / "response.json"
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))

    assert request_payload["candidate"]["candidate_id"] == "candidate-1"
    assert request_payload["candidate_fingerprint"] == _candidate().fingerprint
    assert request_payload["case"]["template_id"] == "template-1"
    assert request_payload["case"]["model"] == "mock-small"
    assert request_payload["case"]["repetition"] == 3
    assert request_payload["case"]["seed"] == 17
    assert request_payload["runtime"]["artifact_dir"] == str(tmp_path / "run")
    assert response_path.exists()
    assert list((tmp_path / "run").glob("*.tmp")) == []


def test_runner_accepts_model_failure_status(tmp_path: Path) -> None:
    case = _case("case[model-failure]")

    result = _runner(case).run(_candidate(), case, tmp_path / "run")

    assert result.status == "model_failure"
    assert result.grade is None
    assert result.error is not None
    assert "model" in result.error


def test_runner_rejects_completed_grade_without_hard_failures(tmp_path: Path) -> None:
    case = _case("case[missing-hard-failures]")
    bridge_path = tmp_path / "bridge.py"
    bridge_path.write_text(
        """
import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[sys.argv.index("--request") + 1]).read_text(encoding="utf-8"))
response_path = Path(sys.argv[sys.argv.index("--response") + 1])
request_identity = {
    "case_id": request["case"]["case_id"],
    "template_id": request["case"]["template_id"],
    "model": request["case"]["model"],
    "repetition": request["case"]["repetition"],
    "seed": request["case"]["seed"],
}
response_path.write_text(
    json.dumps(
        {
            "protocol_version": 1,
            "status": "completed",
            "candidate_fingerprint": request["candidate_fingerprint"],
            "request_identity": request_identity,
            "grade": {
                "completion": 0.9,
                "verification": 0.8,
                "efficiency": 0.7,
            },
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    ),
    encoding="utf-8",
)
""".strip(),
        encoding="utf-8",
    )
    bridge_path.chmod(0o755)
    command = (sys.executable, str(bridge_path), "--request", "{request}", "--response", "{response}")

    with pytest.raises(BridgeMalformedOutputError):
        _runner_with_command(case, command).run(_candidate(), case, tmp_path / "run")


def test_runner_does_not_reuse_stale_response_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    candidate = _candidate()

    _runner(_case("case[completed]")).run(candidate, _case("case[completed]"), run_dir)

    with pytest.raises(BridgeMissingOutputError):
        _runner(_case("case[missing-output]")).run(candidate, _case("case[missing-output]"), run_dir)


def test_runner_rejects_repetition_values_outside_campaign_budget(tmp_path: Path) -> None:
    case = _case("case[completed]")

    with pytest.raises(ValueError, match="campaign.repetitions"):
        _runner(case, campaign_repetitions=1).run(_candidate(), case, tmp_path / "run", repetition=2)


def test_runner_rejects_non_integer_repetition_values(tmp_path: Path) -> None:
    case = _case("case[completed]")

    with pytest.raises(ValueError, match="repetition"):
        _runner(case, campaign_repetitions=2).run(
            _candidate(), case, tmp_path / "run", repetition=cast(Any, 1.5)
        )


@pytest.mark.parametrize("seed", [-1, True])
def test_runner_rejects_invalid_seed_values(tmp_path: Path, seed: int) -> None:
    case = _case("case[completed]")

    with pytest.raises(ValueError, match="seed"):
        _runner(case, campaign_repetitions=2).run(_candidate(), case, tmp_path / "run", seed=seed)


def test_runner_wraps_response_path_cleanup_failures(tmp_path: Path) -> None:
    case = _case("case[completed]")
    run_dir = tmp_path / "run"
    response_dir = run_dir / "response.json"
    response_dir.mkdir(parents=True)

    with pytest.raises(BridgeArtifactError):
        _runner(case).run(_candidate(), case, run_dir)


def test_runner_wraps_unreadable_response_artifacts(tmp_path: Path) -> None:
    case = _case("case[response-directory]")

    with pytest.raises(BridgeArtifactError):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_raises_typed_error_when_bridge_command_cannot_launch(tmp_path: Path) -> None:
    case = _case("case[completed]")
    command = ("definitely-not-a-real-bridge-executable", "{request}", "{response}")

    with pytest.raises(BridgeInvocationError):
        _runner_with_command(case, command).run(_candidate(), case, tmp_path / "run")


def test_runner_raises_typed_error_when_bridge_command_is_not_executable(tmp_path: Path) -> None:
    case = _case("case[completed]")
    bridge_path = tmp_path / "bridge-script"
    bridge_path.write_text("#!/bin/sh\necho nope\n", encoding="utf-8")
    bridge_path.chmod(0o644)

    with pytest.raises(BridgeInvocationError):
        _runner_with_command(case, (str(bridge_path), "{request}", "{response}")).run(
            _candidate(),
            case,
            tmp_path / "run",
        )


def test_runner_rejects_empty_bridge_command(tmp_path: Path) -> None:
    case = _case("case[completed]")

    with pytest.raises(BridgeInvocationError):
        _runner_with_command(case, ()).run(_candidate(), case, tmp_path / "run")


def test_runner_wraps_non_utf8_process_failure_output(tmp_path: Path) -> None:
    case = _case("case[completed]")
    script_path = tmp_path / "bridge.py"
    script_path.write_text(
        "import sys\nsys.stderr.buffer.write(b'\\x80\\x81\\x82')\nraise SystemExit(3)\n",
        encoding="utf-8",
    )
    command = (sys.executable, str(script_path), "{request}", "{response}")

    with pytest.raises(BridgeProcessExitError):
        _runner_with_command(case, command).run(_candidate(), case, tmp_path / "run")


@pytest.mark.parametrize(
    ("case_id", "error_type", "timeout_seconds"),
    [
        ("case[timeout]", BridgeTimeoutError, 0.1),
        ("case[non-zero-exit]", BridgeProcessExitError, 1.0),
        ("case[missing-output]", BridgeMissingOutputError, 1.0),
        ("case[malformed-json]", BridgeMalformedOutputError, 1.0),
        ("case[wrong-shape]", BridgeMalformedOutputError, 1.0),
        ("case[invalid-utf8]", BridgeMalformedOutputError, 1.0),
        ("case[protocol-mismatch]", BridgeProtocolMismatchError, 1.0),
        ("case[fingerprint-mismatch]", BridgeFingerprintMismatchError, 1.0),
        ("case[identity-mismatch]", BridgeIdentityMismatchError, 1.0),
        ("case[systemic-status,identity-mismatch]", BridgeIdentityMismatchError, 1.0),
        ("case[systemic-status]", BridgeStatusError, 1.0),
        ("case[model-failure-with-grade]", BridgeMalformedOutputError, 1.0),
        ("case[model-failure-missing-grade]", BridgeMalformedOutputError, 1.0),
        ("case[bad-journal]", BridgeMalformedOutputError, 1.0),
        ("case[bad-usage]", BridgeMalformedOutputError, 1.0),
        ("case[bad-error]", BridgeMalformedOutputError, 1.0),
        ("case[missing-grade]", BridgeMalformedOutputError, 1.0),
        ("case[bad-grade]", BridgeMalformedOutputError, 1.0),
        ("case[bool-grade]", BridgeMalformedOutputError, 1.0),
        ("case[bad-protocol-type]", BridgeMalformedOutputError, 1.0),
        ("case[bad-status-type]", BridgeMalformedOutputError, 1.0),
        ("case[bad-fingerprint-type]", BridgeMalformedOutputError, 1.0),
        ("case[missing-answer]", BridgeMalformedOutputError, 1.0),
        ("case[missing-journal]", BridgeMalformedOutputError, 1.0),
        ("case[missing-usage]", BridgeMalformedOutputError, 1.0),
        ("case[extra-response-field]", BridgeMalformedOutputError, 1.0),
        ("case[extra-grade-field]", BridgeMalformedOutputError, 1.0),
    ],
)
def test_runner_raises_typed_errors_for_systemic_failures(
    tmp_path: Path, case_id: str, error_type: type[Exception], timeout_seconds: float
) -> None:
    case = _case(case_id)

    with pytest.raises(error_type):
        _runner(case, timeout_seconds=timeout_seconds).run(_candidate(), case, tmp_path / "run")


def test_runner_defaults_its_timeout_to_the_campaign_bridge_timeout() -> None:
    case = _case("case[completed]")
    campaign = Campaign(
        schema_version=1,
        campaign_id="campaign-1",
        repetitions=1,
        models=("mock-small",),
        cases=(case,),
        serving=ProcessServing(backend="process", command=("bridge", "{request}", "{response}")),
        bridge_timeout_seconds=123.5,
    )

    assert KorvidProcessRunner(campaign).timeout_seconds == pytest.approx(123.5)
    assert KorvidProcessRunner(campaign, timeout_seconds=2.0).timeout_seconds == pytest.approx(2.0)


def test_runner_enforces_the_campaign_bridge_timeout_on_the_bridge_process(tmp_path: Path) -> None:
    case = _case("case[timeout]")
    campaign = Campaign(
        schema_version=1,
        campaign_id="campaign-1",
        repetitions=1,
        models=("mock-small",),
        cases=(case,),
        serving=ProcessServing(
            backend="process",
            command=(
                sys.executable,
                str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
                "--request",
                "{request}",
                "--response",
                "{response}",
            ),
        ),
        bridge_timeout_seconds=0.1,
    )

    with pytest.raises(BridgeTimeoutError, match="0.1 seconds"):
        KorvidProcessRunner(campaign).run(_candidate(), case, tmp_path / "run")


def _aks_runner(
    case: EvalCase,
    *,
    model_endpoint: str | None,
    timeout_seconds: float = 1.0,
) -> KorvidProcessRunner:
    campaign = Campaign(
        schema_version=1,
        campaign_id="campaign-aks",
        repetitions=1,
        models=("mock-small",),
        cases=(case,),
        serving=AKSPortForwardServing(
            backend="aks_port_forward",
            resource_group="rg-pension-guard",
            cluster_name="aks-shared-runners",
            namespace="korvid",
            service="korvid-api",
            model="mock-small",
            command=(
                sys.executable,
                str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
                "--request",
                "{request}",
                "--response",
                "{response}",
            ),
        ),
    )
    return KorvidProcessRunner(campaign, timeout_seconds=timeout_seconds, model_endpoint=model_endpoint)


def test_runner_supplies_the_loopback_endpoint_to_aks_bridge_requests(tmp_path: Path) -> None:
    case = _case("case[completed]")
    runner = _aks_runner(case, model_endpoint="http://127.0.0.1:41001")

    result = runner.run(_candidate(), case, tmp_path / "run")

    request_payload = json.loads((tmp_path / "run" / "request.json").read_text(encoding="utf-8"))
    assert request_payload["runtime"]["model_endpoint"] == "http://127.0.0.1:41001"
    assert result.journal["model_endpoint"] == "http://127.0.0.1:41001"


def test_process_bridge_requests_record_an_absent_model_endpoint(tmp_path: Path) -> None:
    case = _case("case[completed]")

    result = _runner(case).run(_candidate(), case, tmp_path / "run")

    request_payload = json.loads((tmp_path / "run" / "request.json").read_text(encoding="utf-8"))
    assert request_payload["runtime"]["model_endpoint"] is None
    assert result.journal["model_endpoint"] is None


def test_runner_rejects_aks_serving_without_a_model_endpoint() -> None:
    with pytest.raises(ValueError, match="model_endpoint"):
        _aks_runner(_case("case[completed]"), model_endpoint=None)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://10.0.0.5:41001",
        "http://example.com:41001",
        "http://127.0.0.1",
        "ftp://127.0.0.1:41001",
        "127.0.0.1:41001",
    ],
)
def test_runner_rejects_endpoints_that_are_not_loopback_http(endpoint: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        _aks_runner(_case("case[completed]"), model_endpoint=endpoint)


def test_runner_rejects_a_model_endpoint_for_process_serving() -> None:
    case = _case("case[completed]")
    campaign = Campaign(
        schema_version=1,
        campaign_id="campaign-1",
        repetitions=1,
        models=("mock-small",),
        cases=(case,),
        serving=ProcessServing(
            backend="process",
            command=(sys.executable, "bridge.py", "--request", "{request}", "--response", "{response}"),
        ),
    )

    with pytest.raises(ValueError, match="model_endpoint"):
        KorvidProcessRunner(campaign, model_endpoint="http://127.0.0.1:41001")
