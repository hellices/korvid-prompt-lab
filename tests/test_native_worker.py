from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

LAB_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = LAB_ROOT / "src" / "korvid_prompt_lab" / "native_worker.py"
NATIVE_ROOT = Path(os.environ.get("KORVID_NATIVE_SOURCE_ROOT", ""))
NATIVE_PYTHON = NATIVE_ROOT / ".venv" / "bin" / "python"
pytestmark = pytest.mark.skipif(
    not os.environ.get("KORVID_NATIVE_SOURCE_ROOT"),
    reason="set KORVID_NATIVE_SOURCE_ROOT to the Korvid v0.4.1 source checkout",
)


def _load_worker() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "native_worker_under_test", WORKER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(WORKER_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def worker() -> ModuleType:
    import korvid

    assert Path(sys.prefix).resolve() == (NATIVE_ROOT / ".venv").resolve(), (
        "run worker tests with the native source .venv/bin/python, not the legacy interpreter"
    )
    assert (
        Path(korvid.__file__).resolve().is_relative_to((NATIVE_ROOT / "src").resolve())
    )
    return _load_worker()


def _tool_script(name: str, arguments: dict[str, object]) -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "type": "tool_call",
                "id": "call-1",
                "name": name,
                "arguments": json.dumps(arguments),
            },
            {"type": "done"},
        ],
        [{"type": "text_delta", "text": "done"}, {"type": "done"}],
    ]


def _payload(
    case_id: str,
    script: list[list[dict[str, Any]]],
    *,
    rules: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "operation": "evaluate",
        "rules": rules
        or ["Use read-only tools to put the requested target on screen."],
        "case_id": case_id,
        "model": {
            "reference": "ollama/qwen3:0.6b",
            "endpoint": "http://127.0.0.1:11434",
            "options": {"temperature": 0.0, "seed": 0},
        },
        "script": script,
    }


def _export_config(
    rules: list[str], *, endpoint: str = "http://127.0.0.1:11434"
) -> str:
    rendered_rules = "\n".join(f"    - {rule}" for rule in rules)
    return (
        "readonly: true\n"
        "agent:\n"
        "  active: prompt-lab\n"
        "  model_tier: low\n"
        "  follow: true\n"
        "  rules:\n"
        f"{rendered_rules}\n"
        "  profiles:\n"
        "    prompt-lab:\n"
        "      model: ollama/qwen3:0.6b\n"
        f"      endpoint: {endpoint}\n"
        "      auth:\n"
        "        method: none\n"
        "      options:\n"
        "        temperature: 0.0\n"
        "        seed: 0\n"
    )


@pytest.mark.parametrize(
    ("case_id", "tool", "arguments"),
    [
        ("train-pods", "list_resources", {"kind": "pods", "namespace": "shop"}),
        ("validation-helm", "helm_list_releases", {"namespace": "monitoring"}),
        ("holdout-all-pods", "list_resources", {"kind": "pods"}),
        (
            "train-logs",
            "open_logs",
            {"pod": "checkout-1", "namespace": "shop", "container": "main"},
        ),
        (
            "validation-describe",
            "open_describe",
            {"kind": "pods", "name": "metrics-2", "namespace": "monitoring"},
        ),
    ],
)
def test_scripted_native_low_tools_drive_actual_ui(
    worker: ModuleType,
    case_id: str,
    tool: str,
    arguments: dict[str, object],
) -> None:
    result = worker.run_request(_payload(case_id, _tool_script(tool, arguments)))
    from korvid_prompt_lab.navigation_cases import find_navigation_case

    assert result["execution_mode"] == "scripted"
    assert result["korvid_version"] == "0.4.1"
    assert result["missing_postconditions"] == []
    expected = {
        "logs": "",
        "describe": "",
        **dict(find_navigation_case(case_id).expected),
    }
    assert result["expected"] == expected
    assert set(result["expected"]) == {"kind", "scope", "filter", "logs", "describe"}
    assert all(
        result["observed"][key] == value for key, value in result["expected"].items()
    )
    assert [call["name"] for call in result["calls"]] == [tool]
    assert result["calls"][0]["ok"] is True
    assert result["calls"][0]["result"]
    assert result["policy"]["tier"] == "low"
    assert "navigate" not in result["policy"]["tools"]
    assert result["rules_applied"] is True
    assert result["ui_follow"] is True
    assert len(result["prompt_fingerprint"]) == 64
    assert result["runtime"]["lock_parity"] == "not-asserted"
    assert len(result["runtime"]["fingerprint"]) == 64
    assert math.isfinite(result["wall_time_seconds"])
    assert result["wall_time_seconds"] > 0
    assert result["iterations"] == (
        1 if tool in {"open_logs", "open_describe"} else 2
    )


def test_text_claim_without_ui_action_does_not_satisfy_postcondition(
    worker: ModuleType,
) -> None:
    script = [[{"type": "text_delta", "text": "The logs are open."}, {"type": "done"}]]

    result = worker.run_request(_payload("train-logs", script))

    assert result["calls"] == []
    assert result["observed"]["logs"] == ""
    assert any(item.startswith("logs:") for item in result["missing_postconditions"])
    assert result["iterations"] == 1


def test_multiple_tool_calls_do_not_count_as_provider_iterations(
    worker: ModuleType,
) -> None:
    calls = [
        {
            "type": "tool_call",
            "id": f"call-{index}",
            "name": "list_resources",
            "arguments": json.dumps({"kind": "pods", "namespace": "shop"}),
        }
        for index in range(3)
    ]
    script = [
        [*calls, {"type": "done"}],
        [{"type": "text_delta", "text": "done"}, {"type": "done"}],
    ]

    result = worker.run_request(_payload("train-pods", script))

    assert len(result["calls"]) == 3
    assert result["iterations"] == 2


def test_iteration_limit_is_a_completed_failed_evaluation(
    worker: ModuleType,
) -> None:
    script = [
        [
            {
                "type": "tool_call",
                "id": f"call-{index}",
                "name": "list_resources",
                "arguments": json.dumps({"kind": "pods", "namespace": "shop"}),
            },
            {"type": "done"},
        ]
        for index in range(6)
    ]

    result = worker.run_request(_payload("train-pods", script))

    assert result["iterations"] == 6
    assert result["errors"] == ["agent:iteration-limit"]


@pytest.mark.parametrize(
    ("message", "label"),
    [
        (
            "iteration limit reached (6) — refine the question",
            "agent:iteration-limit",
        ),
        (
            "history budget exceeded mid-turn (24000 chars) — turn ended early",
            "agent:history-budget-exceeded",
        ),
    ],
)
def test_bounded_model_behavior_errors_remain_scoreable(
    worker: ModuleType,
    message: str,
    label: str,
) -> None:
    calls, errors = worker._calls(
        [
            worker.AgentError(message=message),
            worker.TurnComplete(input_tokens=1, output_tokens=2, estimated=False),
        ]
    )

    assert calls == []
    assert errors == [label]


def test_unknown_agent_error_is_systemic_and_withholds_message(
    worker: ModuleType,
) -> None:
    secret = "provider-response-secret"

    with pytest.raises(worker.NativeWorkerError) as exc_info:
        worker._calls([worker.AgentError(message=secret)])

    assert str(exc_info.value) == "native agent turn failed"
    assert secret not in str(exc_info.value)


def test_wrong_target_is_observed_and_failed(worker: ModuleType) -> None:
    result = worker.run_request(
        _payload(
            "holdout-pods",
            _tool_script("list_resources", {"kind": "pods", "namespace": "other"}),
        )
    )

    assert result["observed"]["scope"] == "other"
    assert any(item.startswith("scope:") for item in result["missing_postconditions"])


def test_direct_navigate_is_refused_by_native_low_policy(worker: ModuleType) -> None:
    result = worker.run_request(
        _payload(
            "train-pods",
            _tool_script("navigate", {"view": "pods", "namespace": "shop"}),
        )
    )

    assert result["calls"][0]["name"] == "navigate"
    assert result["calls"][0]["ok"] is False
    assert "not armed" in result["calls"][0]["result"]
    assert result["observed"] == result["initial"]
    assert result["missing_postconditions"]


@contextmanager
def _project_config(contents: str) -> Generator[Path, None, None]:
    path = LAB_ROOT / "tests" / f".native-worker-{uuid.uuid4().hex}.yaml"
    path.write_text(contents, encoding="utf-8")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def test_verify_config_round_trips_rules_through_native_loader(
    worker: ModuleType,
) -> None:
    rules = [
        "Prefer the exact namespace named by the user.",
        "Do not diagnose display-only asks.",
    ]
    with _project_config(_export_config(rules)) as path:
        result = worker.run_request(
            {
                **_payload("train-pods", [[{"type": "done"}]], rules=rules),
                "operation": "verify-config",
                "config_path": str(path),
            }
        )

    assert result["rules"] == rules
    assert result["rules_applied"] is True
    assert result["ui_follow"] is True
    assert result["model"]["options"] == {"temperature": 0.0, "seed": 0}
    assert result["policy"]["tier"] == "low"
    assert len(result["prompt_fingerprint"]) == 64
    assert result["runtime"]["lock_parity"] == "not-asserted"
    assert result["runtime"]["dependencies"]["korvid"] == "0.4.1"
    assert result["runtime"]["dependencies"]["litellm"] == version("litellm")
    assert result["runtime"]["dependencies"]["boto3"] == version("boto3")
    assert len(result["runtime"]["fingerprint"]) == 64


def test_verify_config_round_trips_duplicate_rules_as_one_combined_layer(
    worker: ModuleType,
) -> None:
    rules = [
        "Prefer the exact namespace named by the user.",
        "Prefer the exact namespace named by the user.",
    ]
    with _project_config(_export_config(rules)) as path:
        result = worker.run_request(
            {
                **_payload("train-pods", [[{"type": "done"}]], rules=rules),
                "operation": "verify-config",
                "config_path": str(path),
            }
        )

    assert result["rules"] == rules
    assert result["rules_applied"] is True


def test_runtime_identity_includes_actual_transport_dependency(worker: ModuleType) -> None:
    metadata = worker._runtime_metadata()
    assert metadata["dependencies"]["httpx"] == version("httpx")


def test_rules_with_shared_prefix_are_verified_as_distinct_entries(worker: ModuleType) -> None:
    rules = ["Use list_resources", "Use list_resources for the requested Pod view."]
    result = worker.run_request({
        **_payload("train-pods", [[{"type": "done"}]], rules=rules),
        "operation": "verify-config",
    })
    assert result["rules_applied"] is True
    assert result["rules"] == rules


def test_config_verification_never_writes_into_the_source_working_directory(
    worker: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    original = worker.load_config
    observed: list[Path] = []

    def load(path: Path):
        observed.append(path)
        assert not path.is_relative_to(tmp_path)
        return original(path)

    monkeypatch.setattr(worker, "load_config", load)
    result = worker.run_request({
        **_payload("train-pods", [[{"type": "done"}]]),
        "operation": "verify-config",
    })
    assert result["rules_applied"]
    assert observed and all(not path.exists() for path in observed)


def test_verify_config_without_path_uses_native_loader_and_cleans_private_yaml(
    worker: ModuleType,
) -> None:
    rules = ["Use the exact namespace named by the user."]
    before = set(LAB_ROOT.glob(".korvid-native-verify-*.yaml"))

    result = worker.run_request(
        {
            **_payload("train-pods", [[{"type": "done"}]], rules=rules),
            "operation": "verify-config",
        }
    )

    assert result["rules"] == rules
    assert result["rules_applied"] is True
    assert set(LAB_ROOT.glob(".korvid-native-verify-*.yaml")) == before


def test_verify_config_rejects_model_profile_mismatch(worker: ModuleType) -> None:
    rules = ["Use the exact namespace named by the user."]
    with (
        _project_config(
            _export_config(rules, endpoint="http://127.0.0.1:9999")
        ) as path,
        pytest.raises(ValueError, match="model profile"),
    ):
        worker.run_request(
            {
                **_payload("train-pods", [[{"type": "done"}]], rules=rules),
                "operation": "verify-config",
                "config_path": str(path),
            }
        )


def test_verify_config_rejects_rules_native_loader_would_drop(
    worker: ModuleType,
) -> None:
    with (
        _project_config("agent:\n  rules:\n    - keep me\n    - ''\n") as path,
        pytest.raises(ValueError, match="agent.rules"),
    ):
        worker.run_request(
            {
                **_payload("train-pods", [[{"type": "done"}]], rules=["keep me"]),
                "operation": "verify-config",
                "config_path": str(path),
            }
        )


def test_evaluate_reloads_and_validates_exported_config(worker: ModuleType) -> None:
    rules = ["Use the exact namespace named by the user."]
    with (
        _project_config(
            _export_config(rules, endpoint="http://127.0.0.1:9999")
        ) as path,
        pytest.raises(ValueError, match="model profile"),
    ):
        worker.run_request(
            {
                **_payload(
                    "train-pods",
                    _tool_script(
                        "list_resources", {"kind": "pods", "namespace": "shop"}
                    ),
                    rules=rules,
                ),
                "config_path": str(path),
            }
        )


def test_evaluate_uses_reloaded_export_config_with_actual_ui(
    worker: ModuleType,
) -> None:
    rules = ["Use the exact namespace named by the user."]
    with _project_config(_export_config(rules)) as path:
        result = worker.run_request(
            {
                **_payload(
                    "train-pods",
                    _tool_script(
                        "list_resources", {"kind": "pods", "namespace": "shop"}
                    ),
                    rules=rules,
                ),
                "config_path": str(path),
            }
        )

    assert result["rules_applied"] is True
    assert result["ui_follow"] is True
    assert result["missing_postconditions"] == []
    assert result["observed"]["kind"] == "pods"
    assert result["observed"]["scope"] == "shop"


def test_worker_cli_writes_response_file(worker: ModuleType) -> None:
    del worker
    token = uuid.uuid4().hex
    request_path = LAB_ROOT / "tests" / f".native-request-{token}.json"
    response_path = LAB_ROOT / "tests" / f".native-response-{token}.json"
    request_path.write_text(
        json.dumps(
            _payload(
                "validation-pods",
                _tool_script(
                    "list_resources", {"kind": "pods", "namespace": "monitoring"}
                ),
            )
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (str(LAB_ROOT / "src"), str(NATIVE_ROOT / "src"))
        ),
    }
    try:
        completed = subprocess.run(
            [
                str(NATIVE_PYTHON),
                str(WORKER_PATH),
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ],
            cwd=LAB_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        result = json.loads(response_path.read_text(encoding="utf-8"))
        assert result["missing_postconditions"] == []
        assert all(
            result["observed"][key] == value
            for key, value in result["expected"].items()
        )
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)


def test_worker_cli_failure_is_prefixed_and_leaves_no_response(
    worker: ModuleType,
) -> None:
    del worker
    token = uuid.uuid4().hex
    request_path = LAB_ROOT / "tests" / f".native-failure-request-{token}.json"
    response_path = LAB_ROOT / "tests" / f".native-failure-response-{token}.json"
    invalid_payload = _payload("train-pods", [])
    del invalid_payload["case_id"]
    request_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
    response_path.write_text("stale", encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (str(LAB_ROOT / "src"), str(NATIVE_ROOT / "src"))
        ),
    }
    try:
        completed = subprocess.run(
            [
                str(NATIVE_PYTHON),
                str(WORKER_PATH),
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ],
            cwd=LAB_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode != 0
        assert completed.stderr.startswith("native-worker:")
        assert 0 < len(completed.stderr) <= 300
        assert not response_path.exists()
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)


def test_worker_cli_provider_failure_withholds_event_message(
    worker: ModuleType,
) -> None:
    del worker
    token = uuid.uuid4().hex
    request_path = LAB_ROOT / "tests" / f".native-provider-request-{token}.json"
    response_path = LAB_ROOT / "tests" / f".native-provider-response-{token}.json"
    request_path.write_text(
        json.dumps(_payload("train-pods", [])),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (str(LAB_ROOT / "src"), str(NATIVE_ROOT / "src"))
        ),
    }
    try:
        completed = subprocess.run(
            [
                str(NATIVE_PYTHON),
                str(WORKER_PATH),
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ],
            cwd=LAB_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode != 0
        assert completed.stderr.endswith(
            "native-worker: native agent produced no outbound payload\n"
        )
        assert "scripted provider exhausted" not in completed.stderr
        assert len(completed.stderr) <= 300
        assert not response_path.exists()
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
