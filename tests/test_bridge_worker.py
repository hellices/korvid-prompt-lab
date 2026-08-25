from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab import bridge_worker
from korvid_prompt_lab.bridge_worker import (
    EXECUTION_MODE_LIVE,
    EXECUTION_MODE_SCRIPTED,
    EXIT_SYSTEMIC_FAILURE,
    LIFECYCLE_FALLBACK,
    MAX_ANSWER_CHARS,
    PROTOCOL_VERSION,
    TurnPhase,
    WorkerConfigurationError,
    WorkerModelFailure,
    build_completed_response,
    build_model_failure_response,
    classify_run_failure,
    install_prompt_overrides,
    load_request,
    map_components_to_overrides,
    observe_model_turns,
    parse_request,
    project_journal,
    require_prompt_matches_journey,
    resolve_execution_mode,
    run_bridge,
    sanitize_error,
    select_journey,
    write_response,
)
from korvid_prompt_lab.bridge_worker import main as worker_main
from korvid_prompt_lab.contracts import Candidate, EvalCase
from korvid_prompt_lab.runner import KorvidProcessRunner

CANDIDATE: dict[str, Any] = {
    "schema_version": 1,
    "candidate_id": "shipped-small",
    "components": {
        "system": "You are korvid's bounded Kubernetes operator.",
        "append": "Verify the postcondition before reporting completion.",
        "tool.scale_resource": "Request an approval-gated replica-count change.",
    },
    "metadata": {"source": "shipped"},
}


def _fingerprint(candidate: dict[str, Any]) -> str:
    payload = {
        "schema_version": candidate["schema_version"],
        "candidate_id": candidate["candidate_id"],
        "components": candidate["components"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_fingerprint": _fingerprint(CANDIDATE),
        "candidate": json.loads(json.dumps(CANDIDATE)),
        "case": {
            "case_id": "aks-scale-up",
            "template_id": "scale-deployment-up",
            "prompt": "Scale checkout-a in shop-a from 2 to 3 replicas.",
            "model": "qwen3:4b",
            "repetition": 1,
            "seed": 0,
        },
        "runtime": {
            "campaign_id": "aks-shared-runners",
            "repetitions": 5,
            "artifact_dir": "artifacts/evaluate/aks/runs/1",
            "model_endpoint": "http://127.0.0.1:41001",
        },
    }
    payload.update(overrides)
    return payload


def _fake_grade(**overrides: Any) -> SimpleNamespace:
    grade = {
        "completion": True,
        "verification": True,
        "efficiency": 1.0,
        "hard_failures": (),
        "checkpoints": ("goal_received", "target_resolved", "outcome_reported"),
        "missing_checkpoints": ("mutation_started",),
        "tool_calls": 3,
        "iterations": 4,
    }
    grade.update(overrides)
    return SimpleNamespace(**grade)


def _fake_run(**overrides: Any) -> SimpleNamespace:
    run = {
        "journey_id": "scale-deployment-up",
        "answer": "Scaled checkout-a in shop-a; a fresh read confirms 3 replicas.",
        "grade": _fake_grade(),
        "journal": (
            {
                "sequence": 1,
                "event": "goal_received",
                "actor": "user",
                "detail": "kubeconfig token AKIA-SECRET-VALUE",
            },
            {
                "sequence": 2,
                "event": "target_resolved",
                "target": {"namespace": "shop-a", "name": "checkout-a"},
            },
            {"sequence": 3, "event": "goal_received", "actor": "user"},
            {"sequence": 4, "event": "tool_call", "detail": "apiVersion: apps/v1 kind: Deployment"},
            {"sequence": 5, "event": "outcome_reported", "result": "captured"},
        ),
        "audit": (
            {"intent": "scale", "manifest": "apiVersion: apps/v1", "authorization": "Bearer secret-token"},
        ),
        "wall_time_s": 12.3456,
    }
    run.update(overrides)
    return SimpleNamespace(**run)


# --- strict request validation -------------------------------------------------


def test_parse_request_accepts_a_well_formed_request() -> None:
    request = parse_request(_payload())

    assert request.template_id == "scale-deployment-up"
    assert request.model == "qwen3:4b"
    assert request.model_endpoint == "http://127.0.0.1:41001"
    assert request.components == {
        "system": CANDIDATE["components"]["system"],
        "append": CANDIDATE["components"]["append"],
        "tool.scale_resource": CANDIDATE["components"]["tool.scale_resource"],
    }
    assert request.request_identity == {
        "case_id": "aks-scale-up",
        "template_id": "scale-deployment-up",
        "model": "qwen3:4b",
        "repetition": 1,
        "seed": 0,
    }


def test_parse_request_rejects_an_unsupported_protocol_version() -> None:
    with pytest.raises(WorkerConfigurationError, match="protocol_version"):
        parse_request(_payload(protocol_version=PROTOCOL_VERSION + 1))


def test_parse_request_refuses_the_superseded_protocol_version_one() -> None:
    # Protocol 1 had no execution_mode, so a version-1 peer can never prove that its
    # evidence came from a model. It must be refused, never migrated.
    with pytest.raises(WorkerConfigurationError, match="protocol_version"):
        parse_request(_payload(protocol_version=1))


def test_parse_request_rejects_unknown_fields() -> None:
    with pytest.raises(WorkerConfigurationError, match="unknown field"):
        parse_request(_payload(extra="nope"))

    payload = _payload()
    payload["case"]["unexpected"] = True
    with pytest.raises(WorkerConfigurationError, match="unknown field"):
        parse_request(payload)


def test_parse_request_rejects_a_fingerprint_that_does_not_match_the_candidate() -> None:
    with pytest.raises(WorkerConfigurationError, match="fingerprint"):
        parse_request(_payload(candidate_fingerprint="0" * 64))


def test_parse_request_rejects_unknown_candidate_component_keys() -> None:
    payload = _payload()
    payload["candidate"]["components"]["prefix"] = "nope"
    payload["candidate_fingerprint"] = _fingerprint(payload["candidate"])

    with pytest.raises(WorkerConfigurationError, match="component"):
        parse_request(payload)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:41001",
        "http://10.0.0.4:41001",
        "http://127.0.0.1",
        "http://127.0.0.1:41001/v1",
        "http://127.0.0.1:41001?x=1",
        "not-a-url",
        42,
    ],
)
def test_parse_request_rejects_a_non_loopback_model_endpoint(endpoint: Any) -> None:
    payload = _payload()
    payload["runtime"]["model_endpoint"] = endpoint

    with pytest.raises(WorkerConfigurationError, match="model_endpoint"):
        parse_request(payload)


def test_parse_request_accepts_a_null_model_endpoint_for_process_serving() -> None:
    payload = _payload()
    payload["runtime"]["model_endpoint"] = None

    assert parse_request(payload).model_endpoint is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", ""),
        ("template_id", "  "),
        ("prompt", ""),
        ("model", 7),
        ("repetition", 0),
        ("repetition", True),
        ("seed", -1),
        ("seed", "0"),
    ],
)
def test_parse_request_rejects_malformed_case_identity(field: str, value: Any) -> None:
    payload = _payload()
    payload["case"][field] = value

    with pytest.raises(WorkerConfigurationError):
        parse_request(payload)


def test_load_request_rejects_a_file_that_is_not_protocol_json(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(WorkerConfigurationError, match="mapping"):
        load_request(path)

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(WorkerConfigurationError, match="JSON"):
        load_request(path)


# --- execution mode --------------------------------------------------------------


def test_completed_response_declares_the_live_execution_mode() -> None:
    response = build_completed_response(
        parse_request(_payload()), _fake_run(), LIFECYCLE_FALLBACK, execution_mode=EXECUTION_MODE_LIVE
    )

    assert response["execution_mode"] == "live"


def test_completed_response_declares_the_scripted_execution_mode() -> None:
    payload = _payload()
    payload["runtime"]["model_endpoint"] = None

    response = build_completed_response(
        parse_request(payload), _fake_run(), LIFECYCLE_FALLBACK, execution_mode=EXECUTION_MODE_SCRIPTED
    )

    assert response["execution_mode"] == "scripted"


def test_model_failure_response_declares_the_execution_mode() -> None:
    response = build_model_failure_response(
        parse_request(_payload()),
        RuntimeError("the model never answered"),
        execution_mode=EXECUTION_MODE_LIVE,
        env={},
    )

    assert response["execution_mode"] == "live"


def test_responses_refuse_an_execution_mode_outside_the_closed_vocabulary() -> None:
    with pytest.raises(WorkerConfigurationError, match="execution_mode"):
        build_completed_response(
            parse_request(_payload()), _fake_run(), LIFECYCLE_FALLBACK, execution_mode="simulated"
        )


def test_resolve_execution_mode_refuses_scripted_evidence_for_a_live_endpoint() -> None:
    # A campaign that carries a model endpoint is a live campaign; grading it from
    # Korvid's canned scripts would publish a perfect score no model ever earned.
    with pytest.raises(WorkerConfigurationError, match="scripted"):
        resolve_execution_mode(parse_request(_payload()), scripted=True)


def test_resolve_execution_mode_reports_live_for_a_live_endpoint() -> None:
    assert resolve_execution_mode(parse_request(_payload()), scripted=False) == EXECUTION_MODE_LIVE


def test_resolve_execution_mode_allows_scripted_without_a_model_endpoint() -> None:
    payload = _payload()
    payload["runtime"]["model_endpoint"] = None

    assert resolve_execution_mode(parse_request(payload), scripted=True) == EXECUTION_MODE_SCRIPTED


# --- candidate override mapping ------------------------------------------------


def test_map_components_to_overrides_maps_system_append_and_tool_components() -> None:
    mapped = map_components_to_overrides(
        {
            "system": "Stay bounded.",
            "append": "Verify first.",
            "tool.scale_resource": "Approval-gated scale.",
            "tool.rollout_restart": "Approval-gated restart.",
        }
    )

    assert mapped == {
        "system": "Stay bounded.",
        "append": "Verify first.",
        "tool_descriptions": {
            "scale_resource": "Approval-gated scale.",
            "rollout_restart": "Approval-gated restart.",
        },
    }


def test_map_components_to_overrides_leaves_absent_slots_as_korvid_defaults() -> None:
    assert map_components_to_overrides({"tool.scale_resource": "Approval-gated scale."}) == {
        "system": None,
        "append": None,
        "tool_descriptions": {"scale_resource": "Approval-gated scale."},
    }


@pytest.mark.parametrize("key", ["prefix", "tool.", "System", "tool", ""])
def test_map_components_to_overrides_rejects_unsupported_component_keys(key: str) -> None:
    with pytest.raises(WorkerConfigurationError, match="component"):
        map_components_to_overrides({key: "text"})


def test_map_components_to_overrides_rejects_blank_component_text() -> None:
    with pytest.raises(WorkerConfigurationError, match="component"):
        map_components_to_overrides({"system": "   "})


def test_install_prompt_overrides_binds_overrides_only_in_the_worker_process() -> None:
    calls: list[dict[str, Any]] = []

    def build_profile(name: str, **kwargs: Any) -> str:
        calls.append({"name": name, **kwargs})
        return "profile"

    module = SimpleNamespace(build_profile=build_profile)
    sentinel = object()

    install_prompt_overrides(module, sentinel)

    assert module.build_profile is not build_profile
    assert module.build_profile("small", readonly=False, resize_supported=False) == "profile"
    assert calls == [
        {"name": "small", "readonly": False, "resize_supported": False, "overrides": sentinel}
    ]


def test_install_prompt_overrides_requires_a_patchable_module() -> None:
    with pytest.raises(WorkerConfigurationError, match="build_profile"):
        install_prompt_overrides(SimpleNamespace(), object())


# --- journey selection and prompt parity ---------------------------------------


def test_select_journey_loads_exactly_the_requested_template_id() -> None:
    journeys = [SimpleNamespace(id="restart-denied"), SimpleNamespace(id="scale-deployment-up")]

    assert select_journey(journeys, "restart-denied").id == "restart-denied"


def test_select_journey_rejects_an_unknown_template_id() -> None:
    journeys = [SimpleNamespace(id="restart-denied")]

    with pytest.raises(WorkerConfigurationError, match="operation journey"):
        select_journey(journeys, "smoke-template")


def test_select_journey_rejects_a_duplicated_template_id() -> None:
    journeys = [SimpleNamespace(id="restart-denied"), SimpleNamespace(id="restart-denied")]

    with pytest.raises(WorkerConfigurationError, match="operation journey"):
        select_journey(journeys, "restart-denied")


def test_require_prompt_matches_journey_first_turn() -> None:
    journey = SimpleNamespace(id="restart-denied", turns=("Restart the api deployment in shop-a.",))

    require_prompt_matches_journey("Restart the api deployment in shop-a.", journey)

    with pytest.raises(WorkerConfigurationError, match="first turn"):
        require_prompt_matches_journey("Restart everything.", journey)


def test_require_prompt_matches_journey_rejects_a_journey_without_turns() -> None:
    with pytest.raises(WorkerConfigurationError, match="first turn"):
        require_prompt_matches_journey("anything", SimpleNamespace(id="empty", turns=()))


# --- safe response projection ---------------------------------------------------


def test_project_journal_reports_checkpoint_names_and_counts_only() -> None:
    projection = project_journal(_fake_run(), LIFECYCLE_FALLBACK)

    assert set(projection) == {
        "journey_id",
        "checkpoints",
        "missing_checkpoints",
        "checkpoint_counts",
        "journal_event_count",
        "audit_record_count",
        "hard_failure_count",
    }
    assert projection["checkpoints"] == ["goal_received", "target_resolved", "outcome_reported"]
    assert projection["missing_checkpoints"] == ["mutation_started"]
    assert projection["checkpoint_counts"] == {
        "goal_received": 2,
        "outcome_reported": 1,
        "target_resolved": 1,
    }
    assert projection["journal_event_count"] == 5
    assert projection["audit_record_count"] == 1
    assert projection["hard_failure_count"] == 0


def test_project_journal_ignores_events_outside_the_lifecycle_vocabulary() -> None:
    projection = project_journal(_fake_run(), LIFECYCLE_FALLBACK)

    assert "tool_call" not in projection["checkpoint_counts"]


def test_completed_response_never_carries_audit_manifests_or_raw_journal_payload() -> None:
    response = build_completed_response(parse_request(_payload()), _fake_run(), LIFECYCLE_FALLBACK, execution_mode=EXECUTION_MODE_LIVE)

    encoded = json.dumps(response)
    for leaked in (
        "AKIA-SECRET-VALUE",  # credential inside a journal detail
        "Bearer secret-token",  # credential inside an audit record
        "apiVersion",  # manifest fragment
        "kubeconfig",
        "intent",  # audit record field
    ):
        assert leaked not in encoded, f"response leaked {leaked!r}"

    # The reflection journal itself carries checkpoint names and counts, nothing else.
    journal = json.dumps(response["journal"])
    for leaked in ('"sequence"', '"actor"', '"detail"', '"target"', "shop-a", "checkout-a", "manifest"):
        assert leaked not in journal, f"reflection journal leaked {leaked!r}"


def test_completed_response_matches_the_runner_protocol_shape() -> None:
    response = build_completed_response(parse_request(_payload()), _fake_run(), LIFECYCLE_FALLBACK, execution_mode=EXECUTION_MODE_LIVE)

    assert set(response) == {
        "protocol_version",
        "status",
        "execution_mode",
        "candidate_fingerprint",
        "request_identity",
        "grade",
        "answer",
        "journal",
        "usage",
        "error",
    }
    assert response["protocol_version"] == PROTOCOL_VERSION
    assert response["status"] == "completed"
    assert response["execution_mode"] == "live"
    assert response["candidate_fingerprint"] == _fingerprint(CANDIDATE)
    assert response["request_identity"] == {
        "case_id": "aks-scale-up",
        "template_id": "scale-deployment-up",
        "model": "qwen3:4b",
        "repetition": 1,
        "seed": 0,
    }
    assert response["error"] is None
    assert response["usage"] == {"tool_calls": 3, "iterations": 4, "wall_time_seconds": 12.346}


def test_completed_response_maps_korvid_boolean_grade_signals_to_metrics() -> None:
    run = _fake_run(
        grade=_fake_grade(
            completion=False,
            verification=True,
            efficiency=0.25,
            hard_failures=("write_without_approval",),
        )
    )

    response = build_completed_response(parse_request(_payload()), run, LIFECYCLE_FALLBACK, execution_mode=EXECUTION_MODE_LIVE)

    assert response["grade"] == {
        "completion": 0.0,
        "verification": 1.0,
        "efficiency": 0.25,
        "hard_failures": ["write_without_approval"],
    }


def test_a_graded_incomplete_run_stays_completed_with_a_low_grade() -> None:
    run = _fake_run(grade=_fake_grade(completion=False, verification=False, efficiency=0.0), answer="")

    response = build_completed_response(parse_request(_payload()), run, LIFECYCLE_FALLBACK, execution_mode=EXECUTION_MODE_LIVE)

    assert response["status"] == "completed"
    assert response["grade"]["completion"] == 0.0


def test_completed_response_bounds_the_answer() -> None:
    run = _fake_run(answer="x" * (MAX_ANSWER_CHARS + 500))

    response = build_completed_response(parse_request(_payload()), run, LIFECYCLE_FALLBACK, execution_mode=EXECUTION_MODE_LIVE)

    assert len(response["answer"]) <= MAX_ANSWER_CHARS + 32
    assert response["answer"].endswith("[truncated]")


def test_completed_response_clamps_efficiency_into_the_scoreable_range() -> None:
    run = _fake_run(grade=_fake_grade(efficiency=1.5))

    response = build_completed_response(parse_request(_payload()), run, LIFECYCLE_FALLBACK, execution_mode=EXECUTION_MODE_LIVE)

    assert response["grade"]["efficiency"] == 1.0


def test_model_failure_response_has_no_grade_and_a_sanitized_error() -> None:
    response = build_model_failure_response(
        parse_request(_payload()),
        RuntimeError("upstream rejected key hunter2-secret"),
        execution_mode=EXECUTION_MODE_LIVE,
        env={"KORVID_EVAL_API_KEY": "hunter2-secret"},
    )

    assert response["status"] == "model_failure"
    assert response["grade"] is None
    assert response["answer"] == ""
    assert response["journal"] == {"checkpoints": [], "checkpoint_counts": {}}
    assert response["usage"] == {}
    assert "hunter2-secret" not in response["error"]
    assert "RuntimeError" in response["error"]


# --- error sanitization ---------------------------------------------------------


def test_sanitize_error_redacts_configured_secret_values() -> None:
    text = sanitize_error(
        ValueError("Authorization: Bearer abc123 rejected for key top-secret"),
        env={"KORVID_EVAL_API_KEY": "top-secret", "KORVID_BRIDGE_API_KEY": "  "},
    )

    assert "top-secret" not in text
    assert "abc123" not in text
    assert "[redacted-credential]" in text
    assert "***" in text
    assert text.startswith("ValueError: ")


def test_sanitize_error_is_bounded() -> None:
    text = sanitize_error(ValueError("y" * 5000), env={})

    assert len(text) <= 320


# --- atomic response write ------------------------------------------------------


def test_write_response_is_atomic_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "response.json"

    write_response(path, {"protocol_version": 1})

    assert json.loads(path.read_text(encoding="utf-8")) == {"protocol_version": 1}
    assert sorted(item.name for item in path.parent.iterdir()) == ["response.json"]


def test_write_response_replaces_a_previous_response(tmp_path: Path) -> None:
    path = tmp_path / "response.json"
    path.write_text("stale", encoding="utf-8")

    write_response(path, {"protocol_version": 1, "status": "completed"})

    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "completed"


def test_write_response_reports_an_unserializable_payload_as_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(WorkerConfigurationError):
        write_response(tmp_path / "response.json", {"bad": object()})

    assert list(tmp_path.iterdir()) == []


# --- runner compatibility -------------------------------------------------------


def test_worker_response_is_accepted_by_the_korvid_process_runner(tmp_path: Path) -> None:
    from korvid_prompt_lab.contracts import Campaign, ProcessServing

    payload_path = tmp_path / "worker-response.json"
    payload_path.write_text(
        json.dumps(build_completed_response(parse_request(_payload()), _fake_run(), LIFECYCLE_FALLBACK, execution_mode=EXECUTION_MODE_LIVE)),
        encoding="utf-8",
    )

    candidate = Candidate.from_mapping(CANDIDATE)
    case = EvalCase(
        case_id="aks-scale-up",
        template_id="scale-deployment-up",
        prompt="Scale checkout-a in shop-a from 2 to 3 replicas.",
        models=("qwen3:4b",),
    )
    command = (
        sys.executable,
        "-c",
        "import shutil, sys; shutil.copyfile(sys.argv[3], sys.argv[2])",
        "{request}",
        "{response}",
        str(payload_path),
    )
    campaign = Campaign(
        schema_version=1,
        campaign_id="aks-shared-runners",
        repetitions=5,
        models=("qwen3:4b",),
        cases=(case,),
        serving=ProcessServing(backend="process", command=command),
    )

    result = KorvidProcessRunner(campaign, timeout_seconds=60.0).run(candidate, case, tmp_path / "run")

    assert result.status == "completed"
    assert result.candidate_fingerprint == candidate.fingerprint
    assert result.grade is not None
    assert result.grade.completion == 1.0
    assert result.journal["checkpoint_counts"]["goal_received"] == 2


# --- reflection feedback -----------------------------------------------------------


def _bridge_result(response: dict[str, Any]) -> Any:
    from korvid_prompt_lab.scoring import BridgeResult, OperationGrade

    grade_payload = response["grade"]
    return BridgeResult(
        protocol_version=response["protocol_version"],
        status=response["status"],
        execution_mode=response["execution_mode"],
        candidate_fingerprint=response["candidate_fingerprint"],
        grade=None
        if grade_payload is None
        else OperationGrade(
            completion=grade_payload["completion"],
            verification=grade_payload["verification"],
            efficiency=grade_payload["efficiency"],
            hard_failures=tuple(grade_payload["hard_failures"]),
        ),
        answer=response["answer"],
        journal=response["journal"],
        usage=response["usage"],
        error=response["error"],
    )


def test_reflection_records_report_the_bridge_reported_gaps_and_tool_calls(tmp_path: Path) -> None:
    """The optimizer must never be told a run had no gaps when the grader found some."""
    from korvid_prompt_lab.adapter import KorvidGEPAAdapter
    from korvid_prompt_lab.contracts import Campaign, ProcessServing
    from korvid_prompt_lab.scoring import score_result

    run = _fake_run(grade=_fake_grade(completion=False, verification=False, efficiency=0.2, tool_calls=7))
    response = build_completed_response(parse_request(_payload()), run, LIFECYCLE_FALLBACK, execution_mode=EXECUTION_MODE_LIVE)

    case = EvalCase(
        case_id="aks-scale-up",
        template_id="scale-deployment-up",
        prompt="Scale checkout-a in shop-a from 2 to 3 replicas.",
        models=("qwen3:4b",),
    )
    campaign = Campaign(
        schema_version=1,
        campaign_id="aks-shared-runners",
        repetitions=1,
        models=("qwen3:4b",),
        cases=(case,),
        serving=ProcessServing(backend="process", command=("true", "{request}", "{response}")),
    )
    adapter = KorvidGEPAAdapter(
        runner=KorvidProcessRunner(campaign, timeout_seconds=60.0),
        artifact_root=tmp_path / "runs",
    )
    result = _bridge_result(response)
    scored = score_result(result)

    trace = adapter._build_trace(case, result, score=scored.score, unsafe=scored.unsafe)

    assert trace.missing_checkpoints == ("mutation_started",)
    assert trace.tool_call_count == 7

    record = adapter._trace_to_record(trace)
    assert "Missing checkpoints: mutation_started." in record["Feedback"]
    assert "No missing checkpoints" not in record["Feedback"]


def test_reflection_records_trust_an_empty_bridge_reported_gap_list(tmp_path: Path) -> None:
    from korvid_prompt_lab.adapter import _missing_checkpoints

    assert _missing_checkpoints({"missing_checkpoints": []}, ("dispatch",)) == ()
    # A bridge that reports no gap list at all keeps the legacy inference.
    assert _missing_checkpoints({"checkpoints": ["dispatch"]}, ("dispatch",)) == ("verify",)


# --- failure taxonomy: only a real model turn can produce a model_failure ----------


class _StubProvider:
    """The smallest provider surface Korvid's runtime consults."""

    def __init__(self) -> None:
        self.completions: list[tuple[Any, ...]] = []
        self.closed = False

    def complete(self, messages: Any, tools: Any, *, stream: bool = True) -> str:
        self.completions.append((messages, tools, stream))
        return "stream"

    async def aclose(self) -> None:
        self.closed = True

    @property
    def name(self) -> str:
        return "stub-model"


def test_turn_phase_starts_only_when_the_provider_is_asked_for_a_completion() -> None:
    phase = TurnPhase()
    provider = _StubProvider()
    observed = observe_model_turns(provider, phase)

    assert phase.model_turn_started is False
    # Building and inspecting the provider is pre-turn work; only `complete` is a turn.
    assert observed.name == "stub-model"
    assert phase.model_turn_started is False

    assert observed.complete(["message"], ["tool"], stream=False) == "stream"

    assert phase.model_turn_started is True
    assert provider.completions == [(["message"], ["tool"], False)]


def test_observed_provider_still_exposes_the_close_hook_korvid_calls() -> None:
    phase = TurnPhase()
    provider = _StubProvider()
    observed = observe_model_turns(provider, phase)

    # Korvid looks the hook up exactly this way on whatever the factory returned.
    aclose = getattr(observed, "aclose", None)
    assert callable(aclose)
    asyncio.run(aclose())

    assert provider.closed is True


def test_a_harness_timeout_before_any_model_turn_is_systemic_not_a_model_failure() -> None:
    # Korvid's pre-turn Textual work (navigate, select the fixture row) can time out
    # with the same WaitTimeout the turn loop raises. Blaming the model for a failure
    # that happened before the model was ever asked would score a broken harness 0.0
    # and let optimization keep running against nothing.
    phase = TurnPhase()

    with pytest.raises(WorkerConfigurationError, match="before the model was asked"):
        classify_run_failure(TimeoutError("fixture target row selected not met within 5.0s"), phase)


def test_a_timeout_after_a_model_turn_started_stays_a_model_failure() -> None:
    phase = TurnPhase()
    phase.mark_model_turn_started()

    with pytest.raises(WorkerModelFailure):
        classify_run_failure(TimeoutError("turn did not finish within 120.0s"), phase)


def test_a_provider_failure_is_a_model_failure_even_before_a_turn_completes() -> None:
    class ProviderError(RuntimeError):
        pass

    phase = TurnPhase()

    with pytest.raises(WorkerModelFailure):
        classify_run_failure(
            ProviderError("connection refused"), phase, provider_errors=(ProviderError,)
        )


class _FakeWaitTimeout(Exception):
    """Stands in for Korvid's `tests.ui.waits.WaitTimeout`."""


class _FakeProviderError(RuntimeError):
    """Stands in for `korvid.providers.openai_compat.ProviderError`."""


def _fake_korvid(run_journey: Any) -> Any:
    """Build the Korvid boundary `run_bridge` resolves out of KORVID_SOURCE_ROOT.

    Only that boundary is stubbed: it lives in another checkout, so it cannot be
    imported here. Everything `run_bridge` itself does — phase tracking, provider
    wrapping, failure classification — is the real code under test.
    """
    from korvid_prompt_lab.bridge_worker import _Korvid

    operation_app = ModuleType("fake_operation_app")
    operation_app.build_profile = lambda name, **kwargs: SimpleNamespace(name=name)  # type: ignore[attr-defined]
    journey = SimpleNamespace(
        id="scale-deployment-up",
        turns=("Scale checkout-a in shop-a from 2 to 3 replicas.",),
    )
    return _Korvid(
        operation_app=operation_app,
        run_operation_journey=run_journey,
        approval_timeout_for=lambda _journey, timeout: timeout,
        load_operation_journeys=lambda _directory: [journey],
        bundled_operations_dir=lambda: Path("operations"),
        lifecycle_checkpoints=LIFECYCLE_FALLBACK,
        prompt_overrides=lambda **kwargs: SimpleNamespace(**kwargs),
        scripted_provider=lambda script: _StubProvider(),
        operation_scripts={"scale-deployment-up": ("script",)},
        live_provider=lambda *args, **kwargs: _StubProvider(),
        static_credentials=lambda key: SimpleNamespace(key=key),
        provider_errors=(_FakeProviderError,),
        turn_timeout_errors=(_FakeWaitTimeout, TimeoutError),
    )


def test_run_bridge_reports_a_pre_turn_harness_timeout_as_systemic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def never_reaches_the_model(journey: Any, **kwargs: Any) -> Any:
        kwargs["provider_factory"]()  # Korvid builds the provider before it selects a row.
        raise _FakeWaitTimeout("fixture target row selected not met within 5.0s")

    monkeypatch.setattr(
        "korvid_prompt_lab.bridge_worker._import_korvid",
        lambda: _fake_korvid(never_reaches_the_model),
    )
    payload = _payload()
    payload["runtime"]["artifact_dir"] = str(tmp_path / "run")

    with pytest.raises(WorkerConfigurationError, match="before the model was asked"):
        run_bridge(parse_request(payload), execution_mode=EXECUTION_MODE_LIVE, env={})


def test_run_bridge_reports_a_timeout_after_a_model_turn_as_a_model_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def stalls_after_the_first_turn(journey: Any, **kwargs: Any) -> Any:
        provider = kwargs["provider_factory"]()
        provider.complete([{"role": "user"}], [])
        raise _FakeWaitTimeout("turn did not finish within 120.0s")

    monkeypatch.setattr(
        "korvid_prompt_lab.bridge_worker._import_korvid",
        lambda: _fake_korvid(stalls_after_the_first_turn),
    )
    payload = _payload()
    payload["runtime"]["artifact_dir"] = str(tmp_path / "run")

    with pytest.raises(WorkerModelFailure):
        run_bridge(parse_request(payload), execution_mode=EXECUTION_MODE_LIVE, env={})


def test_run_bridge_reports_a_provider_error_as_a_model_failure_before_any_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def provider_refuses(journey: Any, **kwargs: Any) -> Any:
        kwargs["provider_factory"]()
        raise _FakeProviderError("connection refused by the model endpoint")

    monkeypatch.setattr(
        "korvid_prompt_lab.bridge_worker._import_korvid",
        lambda: _fake_korvid(provider_refuses),
    )
    payload = _payload()
    payload["runtime"]["artifact_dir"] = str(tmp_path / "run")

    with pytest.raises(WorkerModelFailure):
        run_bridge(parse_request(payload), execution_mode=EXECUTION_MODE_LIVE, env={})


def test_worker_exits_nonzero_without_a_response_when_the_harness_fails_pre_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def never_reaches_the_model(journey: Any, **kwargs: Any) -> Any:
        kwargs["provider_factory"]()
        raise _FakeWaitTimeout("fixture target row selected not met within 5.0s")

    monkeypatch.setattr(
        "korvid_prompt_lab.bridge_worker._import_korvid",
        lambda: _fake_korvid(never_reaches_the_model),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = _payload()
    payload["runtime"]["artifact_dir"] = str(run_dir)
    request_path = run_dir / "request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    response_path = run_dir / "response.json"

    exit_code = worker_main(["--request", str(request_path), "--response", str(response_path)])

    assert exit_code == EXIT_SYSTEMIC_FAILURE
    assert not response_path.exists()
    assert "before the model was asked" in capsys.readouterr().err


def test_worker_writes_a_model_failure_response_when_a_started_turn_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def stalls_after_the_first_turn(journey: Any, **kwargs: Any) -> Any:
        provider = kwargs["provider_factory"]()
        provider.complete([{"role": "user"}], [])
        raise _FakeWaitTimeout("turn did not finish within 120.0s")

    monkeypatch.setattr(
        "korvid_prompt_lab.bridge_worker._import_korvid",
        lambda: _fake_korvid(stalls_after_the_first_turn),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = _payload()
    payload["runtime"]["artifact_dir"] = str(run_dir)
    request_path = run_dir / "request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    response_path = run_dir / "response.json"

    exit_code = worker_main(["--request", str(request_path), "--response", str(response_path)])

    assert exit_code == 0
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "model_failure"
    assert response["execution_mode"] == "live"
    assert response["grade"] is None


def test_worker_check_imports_reports_missing_name_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        bridge_worker,
        "_import_korvid",
        lambda: (_ for _ in ()).throw(
            ImportError("cannot import name 'LIFECYCLE_CHECKPOINTS' from 'korvid.evals.operation'")
        ),
    )

    assert bridge_worker.main(["--check-imports"]) == bridge_worker.EXIT_SYSTEMIC_FAILURE
    captured = capsys.readouterr()
    assert "korvid.evals.operation" in captured.err
    assert "LIFECYCLE_CHECKPOINTS" in captured.err
    assert "Traceback" not in captured.err
