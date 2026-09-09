from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

LAB_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = LAB_ROOT / "src" / "korvid_prompt_lab" / "upstream_worker.py"
SOURCE_ROOT = Path(os.environ.get("KORVID_NATIVE_SOURCE_ROOT", ""))
pytestmark = pytest.mark.skipif(
    not os.environ.get("KORVID_NATIVE_SOURCE_ROOT"),
    reason="set KORVID_NATIVE_SOURCE_ROOT to the Korvid v0.4.1 source checkout",
)


def _load_worker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("upstream_worker_under_test", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def worker() -> ModuleType:
    import korvid

    assert Path(sys.prefix).resolve() == (SOURCE_ROOT / ".venv").resolve()
    assert Path(korvid.__file__).resolve().is_relative_to((SOURCE_ROOT / "src").resolve())
    return _load_worker()


def _call(name: str, arguments: dict[str, object], call_id: str) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


def _text(text: str) -> list[dict[str, Any]]:
    return [{"type": "text_delta", "text": text}, {"type": "done"}]


def _model() -> dict[str, Any]:
    return {
        "reference": "ollama/qwen3:0.6b",
        "endpoint": "http://127.0.0.1:11434",
        "options": {"temperature": 0.0, "seed": 7},
    }


def _request(
    operation: str,
    reference: str,
    tier_pack: str | None = None,
    script: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    source_path = SOURCE_ROOT / "src" / "korvid" / "evals" / f"{reference}.yaml"
    import hashlib

    payload: dict[str, Any] = {
        "protocol_version": 1,
        "operation": operation,
        "reference": reference,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "model": _model(),
    }
    if tier_pack is not None:
        payload["tier_pack"] = tier_pack
    if script is not None:
        payload["script"] = script
    return payload


def test_inspection_copies_actual_resolved_pack_and_proves_no_grind_equivalence(
    worker: ModuleType,
) -> None:
    from korvid.agent.prompt_harness import PromptHarness, PromptInputs
    from korvid.agent.prompt_packs import PROMPT_PACKS
    from korvid.evals.harness import EVAL_CLUSTER, NO_GRIND, static_prompt

    result = worker.run_request(
        {
            "protocol_version": 1,
            "operation": "inspect",
            "references": ["scenarios/image-pull-typo", "journeys/tui-follow"],
            "model": _model(),
            "script": [[{"type": "done"}]],
        }
    )

    prompt = result["prompt"]
    assert prompt["pack_id"] == "low-korvid-operator"
    assert prompt["text"] == PROMPT_PACKS[prompt["pack_id"]]
    assert prompt["text"] != "[]"
    assert prompt["baseline_equivalent"] is True
    _model_payload, profile = worker._model(_model())
    policy = asyncio.run(
        worker._resolved_policy(profile, [[{"type": "done"}]])
    )
    assert prompt["original_composed"]["text"] == static_prompt(policy, NO_GRIND)
    assert prompt["original_composed"]["rendering"] == (
        "canonical inspection rendering (source static_prompt)"
    )
    assert prompt["original_composed"]["baseline_sha256"] == (
        prompt["original_composed"]["copied_pack_sha256"]
    )
    for item in prompt["authored_contexts"]:
        kind, source, _path, _digest = worker._load_reference(item["reference"])
        if kind == "scenario":
            question = source.question
            interaction = source.interaction
        else:
            source_turn = source.turns[item["turn"] - 1]
            question = source_turn.user
            interaction = (
                source.interaction
                if item["turn"] == 1
                else source_turn.interaction
            )
        composed = PromptHarness().compose(
            question,
            PromptInputs(
                policy=policy,
                interaction=interaction,
                cluster=EVAL_CLUSTER,
                previous_interaction=None,
            ),
        )
        assert item["original_system_message"] == composed.system_message
        assert item["original_user_message"] == composed.user_message
        assert item["baseline_sha256"] == hashlib.sha256(
            f"{composed.system_message}\0{composed.user_message}".encode()
        ).hexdigest()
        assert item["equivalent"] is True
    assert prompt["assets"]["safety_contract"]["text"]
    assert prompt["assets"]["common_role"]["text"]
    assert prompt["assets"]["provider_overlays"]["value"] == {}
    assert prompt["assets"]["model_overlays"]["value"] == {}
    assert len(result["runtime"]["fingerprint"]) == 64
    assert result["runtime"]["lock_parity"] == "not-asserted"
    assert len(result["runtime"]["dependencies"]) > 20
    assert {
        "boto3",
        "httpx",
        "korvid",
        "litellm",
        "pyyaml",
        "textual",
    } <= result["runtime"]["dependencies"].keys()


def test_runtime_fingerprint_covers_every_installed_distribution(
    worker: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "PyYAML": "6.0",
        "boto3": "1",
        "httpx": "2",
        "korvid": "0.4.1",
        "litellm": "3",
        "textual": "4",
        "Demo_Package": "first",
    }

    def installed() -> list[SimpleNamespace]:
        return [
            SimpleNamespace(metadata={"Name": name}, version=version)
            for name, version in versions.items()
        ]

    monkeypatch.setattr(worker, "distributions", installed)
    first = worker._runtime_metadata()
    versions["Demo_Package"] = "second"
    second = worker._runtime_metadata()

    assert first["dependencies"]["demo-package"] == "first"
    assert second["dependencies"]["demo-package"] == "second"
    assert first["fingerprint"] != second["fingerprint"]


def test_inspection_validates_candidate_pack_through_actual_prompt_grind(
    worker: ModuleType,
) -> None:
    candidate = "Use the unchanged Korvid harness and keep each diagnosis evidence-bound."
    result = worker.run_request(
        {
            "protocol_version": 1,
            "operation": "inspect",
            "references": ["scenarios/image-pull-typo"],
            "model": _model(),
            "script": [[{"type": "done"}]],
            "tier_pack": candidate,
        }
    )

    assert result["candidate_validation"] == {
        "valid": True,
        "tier_pack_sha256": __import__("hashlib").sha256(candidate.encode()).hexdigest(),
        "authored_contexts": 1,
    }


@pytest.mark.parametrize(
    ("text", "label"),
    [("Use source evidence.", None), ("A" * 4000, "static_prompt_too_large")],
    ids=["valid", "oversized"],
)
def test_candidate_validation_uses_original_composition_without_inference(
    worker: ModuleType, monkeypatch: pytest.MonkeyPatch, text: str, label: str | None,
) -> None:
    from korvid.agent.model_policy import ResolvedAgentPolicy
    from korvid.agent.prompt_harness import PromptHarness
    from korvid.evals.scripted import ScriptedProvider

    validations: list[ResolvedAgentPolicy] = []
    validate = PromptHarness.validate

    def record_validation(
        self: PromptHarness, policy: ResolvedAgentPolicy,
        user_rules: tuple[str, ...] = (),
    ) -> None:
        validations.append(policy)
        validate(self, policy, user_rules)

    monkeypatch.setattr(PromptHarness, "validate", record_validation)
    monkeypatch.setattr(
        ScriptedProvider, "complete",
        lambda *_args, **_kwargs: pytest.fail("validation must not invoke a model"),
    )
    monkeypatch.setattr(
        worker, "_load_reference",
        lambda *_args: pytest.fail("static validation must not read any split"),
    )
    result = worker.run_request({
        "protocol_version": 1, "operation": "validate_candidate",
        "model": _model(), "script": [[{"type": "done"}]], "tier_pack": text,
    })

    assert len(validations) == 2  # Baseline before candidate-specific classification.
    assert validations[0] is validations[1]
    assert result == {
        "protocol_version": 1, "operation": "validate_candidate",
        "model": _model(), "tier_pack_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "valid": label is None, "error_label": label,
    }


@pytest.mark.parametrize(
    ("failing_call", "exception_name"),
    [
        (1, "StaticPromptTooLargeError"),
        (2, "UnknownPromptPackError"),
        (2, "UnknownPromptOverlayError"),
        (2, "ValueError"),
        (2, "TypeError"),
        (2, "RuntimeError"),
    ],
)
def test_candidate_validation_does_not_reject_source_or_programming_errors(
    worker: ModuleType, monkeypatch: pytest.MonkeyPatch,
    failing_call: int, exception_name: str,
) -> None:
    import builtins

    from korvid.agent import prompt_harness

    exception_type = getattr(prompt_harness, exception_name, None) or getattr(builtins, exception_name)
    failure = exception_type("source failure")
    calls = 0
    validate = prompt_harness.PromptHarness.validate

    def fail_validation(
        self: Any, policy: Any, user_rules: tuple[str, ...] = (),
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == failing_call:
            raise failure
        validate(self, policy, user_rules)

    monkeypatch.setattr(prompt_harness.PromptHarness, "validate", fail_validation)
    with pytest.raises(exception_type) as error:
        worker.run_request({
            "protocol_version": 1, "operation": "validate_candidate",
            "model": _model(), "script": [[{"type": "done"}]], "tier_pack": "Revised.",
        })
    assert error.value is failure


def test_candidate_validation_preserves_provider_factory_failures(
    worker: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = worker.UpstreamWorkerError("provider configuration failed")

    def fail(*_args: Any) -> None:
        raise failure

    monkeypatch.setattr(worker, "_live_provider", fail)
    with pytest.raises(worker.UpstreamWorkerError) as error:
        worker.run_request({
            "protocol_version": 1, "operation": "validate_candidate",
            "model": _model(), "tier_pack": "Revised.",
        })
    assert error.value is failure


def test_scenario_evaluation_calls_original_loader_runner_and_grader(
    worker: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from korvid.agent.prompt_packs import PROMPT_PACKS

    loaded: list[Path] = []
    run: list[str] = []
    reports: list[Any] = []
    original_load = worker.load_scenario
    original_run = worker.run_scenario

    def recording_load(path: Path) -> Any:
        loaded.append(path)
        return original_load(path)

    async def recording_run(scenario: Any, **kwargs: Any) -> Any:
        run.append(scenario.id)
        report = await original_run(scenario, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(worker, "load_scenario", recording_load)
    monkeypatch.setattr(worker, "run_scenario", recording_run)
    script = [
        [_call("diagnose_pod", {"pod": "web-1", "namespace": "front"}, "c1")],
        _text("ImagePullBackOff: tag v99 does not exist; manifest unknown."),
    ]

    result = worker.run_request(
        _request(
            "evaluate",
            "scenarios/image-pull-typo",
            PROMPT_PACKS["low-korvid-operator"],
            script,
        )
    )

    assert loaded == [
        SOURCE_ROOT / "src/korvid/evals/scenarios/image-pull-typo.yaml"
    ]
    assert run == ["image-pull-typo"]
    assert result["source_path"] == "src/korvid/evals/scenarios/image-pull-typo.yaml"
    assert result["success"] is True
    assert result["upstream"]["outcome"] == "success"
    assert result["upstream"]["grade"]["diagnosis_success"] is True
    assert result["upstream"]["grade"]["evidence_fetched"] is True
    assert result["calls"][0]["name"] == "diagnose_pod"
    assert result["calls"][0]["arguments"] == {"pod": "web-1", "namespace": "front"}
    assert result["calls"][0]["arguments_valid"] is True
    assert result["usage"]["diagnostic_calls"] == 1

    from korvid.evals.__main__ import report_payload
    from korvid.evals.fake_kube import FakeKubeClient, builtin_aliases
    from korvid.evals.harness import PromptGrind
    from korvid.evals.scenario import load_scenario
    from korvid.evals.scripted import ScriptedProvider
    from korvid.tools.executor import ToolExecutor

    direct_source = load_scenario(
        SOURCE_ROOT / "src/korvid/evals/scenarios/image-pull-typo.yaml"
    )
    direct = asyncio.run(
        original_run(
            direct_source,
            provider_factory=lambda: ScriptedProvider(
                [
                    [_call("diagnose_pod", {"pod": "web-1", "namespace": "front"}, "c1")],
                    _text("ImagePullBackOff: tag v99 does not exist; manifest unknown."),
                ]
            ),
            executor_factory=lambda: ToolExecutor(
                FakeKubeClient(direct_source), builtin_aliases()
            ),
            repetitions=1,
            model_tier="low",
            grind=PromptGrind(tier_pack=PROMPT_PACKS["low-korvid-operator"]),
        )
    )
    assert result["success"] is (direct.runs[0].outcome == "success")
    assert result["source_report"] == report_payload(reports)[0]


def test_evaluation_reloads_the_exported_prompt_file(
    worker: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from korvid.agent.prompt_packs import PROMPT_PACKS

    pack = PROMPT_PACKS["low-korvid-operator"]
    prompt_path = tmp_path / "optimized-prompt.txt"
    prompt_path.write_text(pack, encoding="utf-8")
    payload = _request(
        "evaluate",
        "scenarios/image-pull-typo",
        pack,
        [
            [_call("diagnose_pod", {"pod": "web-1", "namespace": "front"}, "c1")],
            _text("ImagePullBackOff: tag v99 does not exist; manifest unknown."),
        ],
    )
    payload["prompt_path"] = str(prompt_path)
    loaded: list[str] = []
    constructed: list[str] = []
    read_text = Path.read_text
    prompt_grind = worker.PromptGrind

    def record_read(self: Path, *args: Any, **kwargs: Any) -> str:
        text = read_text(self, *args, **kwargs)
        if self == prompt_path:
            loaded.append(text)
        return text

    def record_grind(**kwargs: Any) -> Any:
        constructed.append(kwargs["tier_pack"])
        return prompt_grind(**kwargs)

    monkeypatch.setattr(Path, "read_text", record_read)
    monkeypatch.setattr(worker, "PromptGrind", record_grind)

    result = worker.run_request(payload)

    assert result["prompt_path_verified"] is True
    assert len(loaded) == 1
    assert constructed[0] is loaded[0]
    assert constructed[0] is not payload["tier_pack"]


def test_evaluation_rejects_a_prompt_file_different_from_the_candidate(
    worker: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _request(
        "evaluate",
        "scenarios/image-pull-typo",
        "candidate prompt",
        [[{"type": "done"}]],
    )
    prompt_path = tmp_path / "optimized-prompt.txt"
    prompt_path.write_text("different prompt", encoding="utf-8")
    payload["prompt_path"] = str(prompt_path)
    monkeypatch.setattr(
        worker, "_provider_factory",
        lambda *_args: pytest.fail("mismatched reload must fail before evaluation"),
    )

    with pytest.raises(ValueError, match="differs"):
        worker.run_request(payload)


def test_evaluation_rejects_a_missing_prompt_file(
    worker: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _request(
        "evaluate", "scenarios/image-pull-typo", "candidate prompt",
        [[{"type": "done"}]],
    )
    payload["prompt_path"] = str(tmp_path / "missing.txt")
    monkeypatch.setattr(
        worker, "_provider_factory",
        lambda *_args: pytest.fail("missing reload must fail before evaluation"),
    )
    with pytest.raises(FileNotFoundError):
        worker.run_request(payload)


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("{not-json", "malformed_arguments"),
        ("[]", "non_object_arguments"),
    ],
)
def test_observer_preserves_malformed_event_and_marks_trace_invalid(
    worker: ModuleType,
    raw: str,
    label: str,
) -> None:
    from korvid.evals.scripted import ScriptedProvider

    event = _call("diagnose_pod", {}, "bad")
    event["arguments"] = raw
    provider = worker._ObservedProvider(
        ScriptedProvider([[event, {"type": "done"}]])
    )

    async def collect() -> list[dict[str, Any]]:
        return [
            item
            async for item in provider.complete([], [], stream=True)
        ]

    emitted = asyncio.run(collect())

    assert emitted == [event, {"type": "done"}]
    assert provider.calls == [
        {
            "name": "diagnose_pod",
            "arguments_valid": False,
            "arguments_raw": raw,
            "error_label": label,
        }
    ]


def test_observer_bounds_invalid_argument_diagnostic_without_changing_event(
    worker: ModuleType,
) -> None:
    from korvid.evals.scripted import ScriptedProvider

    raw = "x" * 600
    event = _call("diagnose_pod", {}, "bad")
    event["arguments"] = raw
    provider = worker._ObservedProvider(
        ScriptedProvider([[event, {"type": "done"}]])
    )

    async def collect() -> list[dict[str, Any]]:
        return [item async for item in provider.complete([], [], stream=True)]

    emitted = asyncio.run(collect())

    assert emitted[0] == event
    assert provider.calls[0]["arguments_raw"] == "x" * 512
    assert provider.calls[0]["arguments_valid"] is False


def test_model_iteration_limit_keeps_original_scoreable_failure(
    worker: ModuleType,
) -> None:
    from korvid.agent.prompt_packs import PROMPT_PACKS

    loop = [
        [
            _call(
                "diagnose_pod",
                {"pod": "web-1", "namespace": "front"},
                f"c{index}",
            )
        ]
        for index in range(6)
    ]
    result = worker.run_request(
        _request(
            "evaluate",
            "scenarios/image-pull-typo",
            PROMPT_PACKS["low-korvid-operator"],
            loop,
        )
    )

    assert result["success"] is False
    assert result["upstream"]["outcome"] == "error"
    assert result["upstream"]["failure_class"] == "provider_error"
    assert result["upstream"]["error_label"] == "model_bound"


def test_unknown_provider_runtime_failure_is_not_converted_to_quality_zero(
    worker: ModuleType,
) -> None:
    from korvid.agent.prompt_packs import PROMPT_PACKS

    with pytest.raises(worker.UpstreamWorkerError, match="provider/runtime"):
        worker.run_request(
            _request(
                "evaluate",
                "scenarios/image-pull-typo",
                PROMPT_PACKS["low-korvid-operator"],
                [],
            )
        )


def _tui_follow_script(*, required_phrase: bool = True) -> list[list[dict[str, Any]]]:
    display = "Opened on screen." if required_phrase else "Done."
    return [
        [_call("diagnose_pod", {"pod": "web-1", "namespace": "front"}, "c1")],
        _text("ImagePullBackOff from nonexistent tag v99: manifest unknown."),
        [
            _call(
                "open_describe",
                {
                    "kind": "pods",
                    "name": "web-1",
                    "namespace": "front",
                    "continue_analysis": True,
                },
                "c2",
            )
        ],
        _text(display),
        [
            _call(
                "open_logs",
                {
                    "pod": "web-1",
                    "namespace": "front",
                    "continue_analysis": True,
                },
                "c3",
            )
        ],
        _text(display),
    ]


def test_whole_journey_runs_all_turns_and_matches_direct_korvid_verdict(
    worker: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from korvid.agent.prompt_packs import PROMPT_PACKS
    from korvid.evals.fake_kube import FakeKubeClient, builtin_aliases
    from korvid.evals.harness import PromptGrind
    from korvid.evals.journey import load_journey
    from korvid.evals.journey_runner import report_payload, run_journey
    from korvid.evals.scripted import ScriptedProvider
    from korvid.tools.executor import ToolExecutor

    called: list[tuple[str, int]] = []
    reports: list[Any] = []
    original_run = worker.run_journey

    async def recording_run(journey: Any, **kwargs: Any) -> Any:
        called.append((journey.id, len(journey.turns)))
        report = await original_run(journey, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(worker, "run_journey", recording_run)
    pack = PROMPT_PACKS["low-korvid-operator"]
    script = _tui_follow_script()
    result = worker.run_request(
        _request("evaluate", "journeys/tui-follow", pack, script)
    )

    journey = load_journey(
        SOURCE_ROOT / "src/korvid/evals/journeys/tui-follow.yaml"
    )
    direct = asyncio.run(
        run_journey(
            journey,
            provider_factory=lambda: ScriptedProvider(_tui_follow_script()),
            executor_factory=lambda fixture: ToolExecutor(
                FakeKubeClient(fixture), builtin_aliases()
            ),
            repetitions=1,
            model_tier="low",
            grind=PromptGrind(tier_pack=pack),
        )
    )

    assert called == [("tui-follow", 3)]
    assert len(result["upstream"]["turns"]) == 3
    assert result["usage"]["tool_calls"] == 3
    assert result["usage"]["diagnostic_calls"] == 1
    assert result["success"] is direct.runs[0].success is True
    assert [turn["outcome"] for turn in result["upstream"]["turns"]] == [
        turn.outcome for turn in direct.runs[0].turns
    ]
    assert result["source_report"] == report_payload(reports)[0]
    assert [turn["answer"] for turn in result["source_report"]["runs"][0]["turns"]]
    assert all(
        "interaction" in turn and "final_interaction" in turn
        for turn in result["source_report"]["runs"][0]["turns"]
    )


def test_ui_action_without_required_answer_phrase_keeps_original_failure(
    worker: ModuleType,
) -> None:
    from korvid.agent.prompt_packs import PROMPT_PACKS
    from korvid.evals.fake_kube import FakeKubeClient, builtin_aliases
    from korvid.evals.harness import PromptGrind
    from korvid.evals.journey import load_journey
    from korvid.evals.journey_runner import run_journey
    from korvid.evals.scripted import ScriptedProvider
    from korvid.tools.executor import ToolExecutor

    result = worker.run_request(
        _request(
            "evaluate",
            "journeys/tui-follow",
            PROMPT_PACKS["low-korvid-operator"],
            _tui_follow_script(required_phrase=False),
        )
    )

    assert result["calls"][1]["name"] == "open_describe"
    assert result["upstream"]["turns"][1]["grade"]["evidence_fetched"] is True
    assert result["upstream"]["turns"][1]["grade"]["diagnosis_success"] is False
    assert result["upstream"]["turns"][1]["failure_class"] == "misdiagnosis"
    assert result["success"] is False
    journey = load_journey(
        SOURCE_ROOT / "src/korvid/evals/journeys/tui-follow.yaml"
    )
    direct = asyncio.run(
        run_journey(
            journey,
            provider_factory=lambda: ScriptedProvider(
                _tui_follow_script(required_phrase=False)
            ),
            executor_factory=lambda fixture: ToolExecutor(
                FakeKubeClient(fixture), builtin_aliases()
            ),
            repetitions=1,
            model_tier="low",
            grind=PromptGrind(
                tier_pack=PROMPT_PACKS["low-korvid-operator"]
            ),
        )
    )
    assert result["success"] is direct.runs[0].success
