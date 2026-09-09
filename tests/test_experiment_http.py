"""Original Korvid source worker + DSPy + GEPA over a synthetic Ollama service.

The server is a deterministic fixture, not an LLM. Its scores are never
real-model improvement evidence.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]
from test_experiment import experiment_mapping

from korvid_prompt_lab.cli import main


@contextmanager
def ollama_fixture(scenarios: tuple[dict[str, Any], ...] = ()) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []
    marker = "upstream-http-candidate"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def respond(self, payload: dict[str, Any], *, ndjson: bool = False) -> None:
            body = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson" if ndjson else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/api/version":
                self.respond({"version": "synthetic-http-fixture"})
            elif self.path == "/api/tags":
                self.respond({"models": [
                    {"name": "qwen3:0.6b", "digest": "1" * 64},
                    {"name": "qwen3:14b", "digest": "2" * 64},
                ]})
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, **body})
            if self.path == "/api/show":
                self.respond({
                    "capabilities": ["completion", "tools", "thinking"],
                    "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 16384},
                })
                return
            if self.path != "/api/chat":
                self.send_error(404)
                return
            message: dict[str, Any] = {"role": "assistant", "content": "Opened."}
            if body["model"] == "qwen3:14b":
                message["content"] = (
                    '[[ ## revised_component_text ## ]]\n'
                    f'Use the required evidence and answer accurately. {marker}\n'
                    '[[ ## completed ## ]]'
                )
            else:
                text = "\n".join(
                    item.get("content", "") for item in body["messages"]
                    if isinstance(item.get("content", ""), str)
                )
                case = next(case for case in scenarios if case["question"] in text)
                evidence = [
                    group[0] if isinstance(group, list) else group
                    for group in case["grading"]["expected_evidence"]
                ]
                observed = sum(item["role"] == "tool" for item in body["messages"])
                if observed < len(evidence):
                    item = evidence[observed]
                    message = {"role": "assistant", "content": "", "tool_calls": [
                        {"function": {"name": item["tool"], "arguments": item["args"]}},
                    ]}
                else:
                    answer = " ".join(group[0] if isinstance(group, list) else group for group in case["grading"]["must_mention"])
                    message["content"] = answer if marker in text else "I cannot assess this."
            self.respond({
                "model": body["model"], "created_at": "2026-01-01T00:00:00Z",
                "message": message, "done": True, "done_reason": "stop",
                "total_duration": 100, "load_duration": 10,
                "prompt_eval_count": 1, "prompt_eval_duration": 10,
                "eval_count": 1, "eval_duration": 10,
            }, ndjson=body.get("stream", False))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_real_dspy_teacher_uses_one_chat_generation_and_allows_metadata_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from korvid_prompt_lab.experiment import build_reflection_lm
    from korvid_prompt_lab.experiment_budget import ExperimentBudget
    from korvid_prompt_lab.experiment_config import load_experiment
    from korvid_prompt_lab.reflection import DSPyInstructionProposer

    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "unused-native-source"))
    with ollama_fixture() as (endpoint, requests):
        monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", endpoint)
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(experiment_mapping(tmp_path)), encoding="utf-8")
        spec = load_experiment(path)
        budget = ExperimentBudget(0, 1, 30)
        lm = build_reflection_lm(spec, endpoint, budget, 0)
        proposed = DSPyInstructionProposer(lm, budget=budget)(
            {"tier_pack": "Teacher transport test only"}, {"tier_pack": []}, ["tier_pack"],
        )
    assert "upstream-http-candidate" in proposed["tier_pack"]
    inferences = [request for request in requests if request["path"] == "/api/chat"]
    assert len(inferences) == 1
    assert inferences[0]["model"] == "qwen3:14b"
    assert inferences[0]["options"]["num_ctx"] == 4096
    assert all(request["path"] in {"/api/chat", "/api/show"} for request in requests)


@pytest.mark.skipif(
    not os.environ.get("KORVID_NATIVE_SOURCE_ROOT"),
    reason="requires the unchanged pinned native source environment",
)
def test_official_command_crosses_real_native_and_dspy_http_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = experiment_mapping(tmp_path)
    refs = {
        "train": ["scenarios/healthy-deployment"],
        "validation": ["scenarios/healthy-service-endpoints"],
        "holdout": ["scenarios/healthy-restart-history"],
    }
    mapping["evaluation"].update(refs)
    mapping["search"]["stages"] = [{"name": "source-fixture", "metric_calls": 8, "seeds": [0]}]
    source = Path(os.environ["KORVID_NATIVE_SOURCE_ROOT"])
    scenarios = tuple(
        yaml.safe_load((source / "src/korvid/evals" / f"{ref}.yaml").read_text())
        for values in refs.values() for ref in values
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    root = tmp_path / "run"
    with ollama_fixture(scenarios) as (endpoint, requests):
        monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", endpoint)
        exit_code = main(["run", "--experiment", str(path), "--artifact-root", str(root)])
    summary = json.loads((root / "experiment-summary.json").read_text())
    assert exit_code == 0, {
        key: summary.get(key) for key in ("status", "phase", "stop_reason", "error_type")
    }
    assert summary["pipeline_completed"]
    assert summary["prompt_override_verified"]
    assert summary["product_application_verified"] is False
    assert summary["proposal_attempts"] >= 1
    assert summary["distinct_proposals"] >= 1
    assert summary["upstream"]["runtime"]["dependencies"]["korvid"] == "0.4.1"
    assert summary["upstream"]["prompt"]["baseline_equivalent"] is True
    assert (root / "baseline-prompt.txt").read_text() == summary["upstream"]["prompt"]["text"]
    assert set(summary["baseline_validation"]["case_ids"]) == {"healthy-service-endpoints"}
    assert set(summary["holdout_candidate"]["case_ids"]) == {"healthy-restart-history"}
    inferences = [request for request in requests if request["path"] == "/api/chat"]
    students = [request for request in inferences if request["model"] == "qwen3:0.6b"]
    teachers = [request for request in inferences if request["model"] == "qwen3:14b"]
    assert students and teachers
    assert all(request["path"] in {"/api/chat", "/api/show"} for request in requests)
    assert all(request["think"] is False and request["stream"] is True for request in students)
    assert all(request["options"]["num_ctx"] == 16384 for request in students)
    assert all(len(request["tools"]) == 12 for request in students)
    assert {request["options"]["seed"] for request in students} >= set(range(10))
    assert all(request.get("stream") is False for request in teachers)
    assert all(request["options"]["num_ctx"] == 4096 for request in teachers)
    assert "upstream-http-candidate" in (root / "candidate-for-review" / "optimized-prompt.txt").read_text()
    assert not (root / "candidate-for-review" / "agent-rules.yaml").exists()
