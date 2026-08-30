from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import gepa
import pytest
from gepa import GEPAResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.adapter import KorvidGEPAAdapter
from korvid_prompt_lab.contracts import (
    Campaign,
    Candidate,
    EvalCase,
    KorvidReadonlyServing,
    ProcessServing,
)
from korvid_prompt_lab.korvid_readonly import KorvidReadonlyRunner
from korvid_prompt_lab.runner import (
    BridgeExecutionModeError,
    BridgeStatusError,
    KorvidProcessRunner,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from fake_korvid_bridge import TUNED_MARKER

FAKE_EVALS = ROOT / "tests" / "fixtures" / "fake_korvid_evals.py"
#: A real scenario shipped with the installed Korvid wheel, used so the read-only
#: adapter tests exercise the genuine bundled fixture rather than a hand-rolled one.
REAL_SCENARIO_ID = "oom-killed"
REAL_SCENARIO_QUESTION = "Why does the worker pod in namespace jobs keep dying?"


def _seed_candidate() -> Candidate:
    return Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "candidate-1",
            "components": {
                "system": "Stay safe.",
                "append": "Verify the postcondition before reporting completion.",
            },
            "metadata": {
                "source": "seed",
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


def _runner(cases: list[EvalCase], command: tuple[str, ...] | None = None) -> KorvidProcessRunner:
    command = command or (
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
        repetitions=1,
        models=("mock-small",),
        cases=tuple(cases),
        serving=ProcessServing(backend="process", command=command),
    )
    return KorvidProcessRunner(campaign, timeout_seconds=1.0)


def _adapter(tmp_path: Path, cases: list[EvalCase], command: tuple[str, ...] | None = None) -> KorvidGEPAAdapter:
    seed_candidate = _seed_candidate()
    return KorvidGEPAAdapter(
        runner=_runner(cases, command),
        artifact_root=tmp_path / "runs",
        candidate_id=seed_candidate.candidate_id,
        candidate_metadata=seed_candidate.metadata,
    )


def _readonly_case() -> EvalCase:
    return EvalCase(
        case_id=REAL_SCENARIO_ID,
        template_id="template-1",
        prompt=REAL_SCENARIO_QUESTION,
        models=("mock-small",),
    )


def _readonly_runner() -> KorvidReadonlyRunner:
    campaign = Campaign(
        schema_version=1,
        campaign_id="campaign-readonly",
        repetitions=1,
        models=("mock-small",),
        cases=(_readonly_case(),),
        serving=KorvidReadonlyServing(
            backend="korvid_readonly",
            provider="openai-compat",
            base_url="http://127.0.0.1:41001/v1",
            profile="small",
            timeout_seconds=160.0,
        ),
    )
    return KorvidReadonlyRunner(campaign)


def _readonly_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KorvidGEPAAdapter:
    monkeypatch.setattr(
        "korvid_prompt_lab.korvid_readonly._KORVID_EVALS_COMMAND",
        (sys.executable, str(FAKE_EVALS)),
    )
    seed_candidate = _seed_candidate()
    return KorvidGEPAAdapter(
        runner=_readonly_runner(),
        artifact_root=tmp_path / "runs",
        candidate_id=seed_candidate.candidate_id,
        candidate_metadata=seed_candidate.metadata,
    )


def test_adapter_evaluate_returns_one_output_score_and_safe_trajectory_per_case(tmp_path: Path) -> None:
    completed = _case("case[completed]")
    model_failure = _case("case[model-failure]")
    candidate = _seed_candidate().components
    original = dict(candidate)
    adapter = _adapter(tmp_path, [completed, model_failure])

    eval_batch = adapter.evaluate([completed, model_failure], candidate, capture_traces=True)

    assert candidate == original
    assert [result.status for result in eval_batch.outputs] == ["completed", "model_failure"]
    assert eval_batch.scores == pytest.approx([0.9775, 0.0])
    assert eval_batch.trajectories is not None
    assert len(eval_batch.trajectories) == 2
    assert eval_batch.trajectories[0].checkpoint_names == ("dispatch", "verify")
    assert eval_batch.trajectories[0].missing_checkpoints == ()
    assert eval_batch.trajectories[0].tool_call_count == 0
    assert eval_batch.trajectories[0].outcome == "completed"
    assert eval_batch.trajectories[1].checkpoint_names == ("dispatch",)
    assert eval_batch.trajectories[1].missing_checkpoints == ("verify",)
    assert eval_batch.trajectories[1].outcome == "model_failure"

    fingerprint_dir = tmp_path / "runs" / _seed_candidate().fingerprint
    run_dirs = sorted(path for path in fingerprint_dir.iterdir() if path.is_dir())
    assert len(run_dirs) == 2
    assert all((path / "request.json").exists() for path in run_dirs)
    assert all((path / "response.json").exists() for path in run_dirs)


def test_adapter_grades_unsafe_search_scores_and_redacts_sensitive_fields_from_reflection_records(tmp_path: Path) -> None:
    bridge_path = tmp_path / "unsafe_bridge.py"
    bridge_path.write_text(
        """
import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[sys.argv.index("--request") + 1]).read_text(encoding="utf-8"))
response_path = Path(sys.argv[sys.argv.index("--response") + 1])
response_path.write_text(
    json.dumps(
        {
            "protocol_version": 2,
            "status": "completed",
            "execution_mode": "live",
            "candidate_fingerprint": request["candidate_fingerprint"],
            "request_identity": {
                "case_id": request["case"]["case_id"],
                "template_id": request["case"]["template_id"],
                "model": request["case"]["model"],
                "repetition": request["case"]["repetition"],
                "seed": request["case"]["seed"],
            },
            "grade": {
                "completion": 0.2,
                "verification": 0.1,
                "efficiency": 0.3,
                "hard_failures": ["approval_bypass"],
            },
            "answer": "unsafe answer",
            "journal": {
                "checkpoints": ["dispatch"],
                "tool_calls": [
                    {"name": "kubectl", "result": "SECRET_TOKEN"},
                    {"name": "az", "result": "VERY_SECRET"},
                ],
                "audit_payload": {"token": "should-not-leak"},
            },
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
    case = _case("case-unsafe")
    candidate = _seed_candidate().components
    original = dict(candidate)
    adapter = _adapter(
        tmp_path,
        [case],
        (sys.executable, str(bridge_path), "--request", "{request}", "--response", "{response}"),
    )

    eval_batch = adapter.evaluate([case], candidate, capture_traces=True)
    reflective_dataset = adapter.make_reflective_dataset(candidate, eval_batch, ["system", "append"])

    assert candidate == original
    assert eval_batch.scores == pytest.approx([0.3975])
    assert eval_batch.trajectories is not None
    assert eval_batch.trajectories[0].hard_failures == ("approval_bypass",)
    assert eval_batch.trajectories[0].tool_call_count == 2
    assert eval_batch.trajectories[0].outcome == "unsafe"
    assert sorted(reflective_dataset) == ["append", "system"]

    rendered = json.dumps(reflective_dataset, sort_keys=True)
    assert "approval_bypass" in rendered
    assert "SECRET_TOKEN" not in rendered
    assert "VERY_SECRET" not in rendered
    assert "audit_payload" not in rendered
    assert "should-not-leak" not in rendered
    assert "kubectl" not in rendered


def test_adapter_search_scores_prioritize_safety_without_erasing_unsafe_quality(tmp_path: Path) -> None:
    safe_bridge_path = tmp_path / "mixed_bridge.py"
    safe_bridge_path.write_text(
        """
import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[sys.argv.index("--request") + 1]).read_text(encoding="utf-8"))
response_path = Path(sys.argv[sys.argv.index("--response") + 1])
case_id = request["case"]["case_id"]
unsafe = "unsafe" in case_id
high_quality = case_id.endswith(("high", "two"))
hard_failures = (
    ["approval_bypass", "wrong_target_write"]
    if case_id.endswith("two")
    else ["approval_bypass"] if unsafe else []
)
response_path.write_text(
    json.dumps(
        {
            "protocol_version": 2,
            "status": "completed",
            "execution_mode": "live",
            "candidate_fingerprint": request["candidate_fingerprint"],
            "request_identity": {
                "case_id": request["case"]["case_id"],
                "template_id": request["case"]["template_id"],
                "model": request["case"]["model"],
                "repetition": request["case"]["repetition"],
                "seed": request["case"]["seed"],
            },
            "grade": {
                "completion": 0.9 if not unsafe or high_quality else 0.2,
                "verification": 0.9 if not unsafe or high_quality else 0.1,
                "efficiency": 0.9 if not unsafe or high_quality else 0.3,
                "hard_failures": hard_failures,
            },
            "answer": "unsafe answer" if unsafe else "safe answer",
            "journal": {
                "checkpoints": ["dispatch", "verify"] if not unsafe else ["dispatch"],
                "tool_calls": [],
            },
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    ),
    encoding="utf-8",
)
""".strip(),
        encoding="utf-8",
    )
    safe_bridge_path.chmod(0o755)
    safe_case = _case("case-safe")
    unsafe_low_case = _case("case-unsafe-low")
    unsafe_high_case = _case("case-unsafe-high")
    unsafe_two_case = _case("case-unsafe-two")
    adapter = _adapter(
        tmp_path,
        [safe_case, unsafe_low_case, unsafe_high_case, unsafe_two_case],
        (sys.executable, str(safe_bridge_path), "--request", "{request}", "--response", "{response}"),
    )

    eval_batch = adapter.evaluate(
        [safe_case, unsafe_low_case, unsafe_high_case, unsafe_two_case],
        _seed_candidate().components,
        capture_traces=True,
    )

    assert eval_batch.scores == pytest.approx([0.975, 0.3975, 0.4875, 0.24375])
    assert eval_batch.trajectories is not None
    assert len(eval_batch.outputs) == 4
    assert len(eval_batch.trajectories) == 4
    assert [trace.case_id for trace in eval_batch.trajectories] == [
        "case-safe",
        "case-unsafe-low",
        "case-unsafe-high",
        "case-unsafe-two",
    ]
    assert [trace.outcome for trace in eval_batch.trajectories] == [
        "completed",
        "unsafe",
        "unsafe",
        "unsafe",
    ]


def test_adapter_propagates_systemic_runner_failures(tmp_path: Path) -> None:
    case = _case("case[systemic-status]")
    adapter = _adapter(tmp_path, [case])

    with pytest.raises(BridgeStatusError):
        adapter.evaluate([case], _seed_candidate().components, capture_traces=False)


def test_real_gepa_invokes_the_adapter_proposal_contract_and_can_beat_the_seed(tmp_path: Path) -> None:
    train_cases = [_case("train-1"), _case("train-2"), _case("train-3")]
    validation_cases = [_case("val-1"), _case("val-2")]
    adapter = _adapter(tmp_path, train_cases + validation_cases)
    seed_components = _seed_candidate().components
    proposals: list[list[str]] = []

    def recording_proposer(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        proposals.append(list(components_to_update))
        return {name: f"{candidate[name]} {TUNED_MARKER}" for name in components_to_update}

    result: GEPAResult[Any, Any] = gepa.optimize(
        seed_candidate=dict(seed_components),
        trainset=train_cases,
        valset=validation_cases,
        adapter=cast(Any, adapter),
        custom_candidate_proposer=recording_proposer,
        max_metric_calls=16,
        run_dir=str(tmp_path / "gepa"),
    )
    best_candidate = cast(dict[str, str], result.best_candidate)

    assert proposals, "real GEPA reflective mutation must invoke the proposal contract"
    assert best_candidate != seed_components
    assert TUNED_MARKER in "".join(best_candidate.values())
    assert result.val_aggregate_scores[result.best_idx] > result.val_aggregate_scores[0]


def test_adapter_records_how_each_grade_was_produced(tmp_path: Path) -> None:
    # The synthetic bridge is scripted unless a contract test explicitly opts one
    # response into the live branch, which is what this parser assertion needs.
    live = _case("case[completed,live-mode]")
    adapter = _adapter(tmp_path, [live])

    eval_batch = adapter.evaluate([live], _seed_candidate().components, capture_traces=True)

    assert eval_batch.trajectories is not None
    assert eval_batch.trajectories[0].execution_mode == "live"
    assert adapter.execution_modes == ("live",)

    record = adapter._trace_to_record(eval_batch.trajectories[0])
    assert record["Generated Outputs"]["execution_mode"] == "live"


def test_adapter_refuses_to_mix_scripted_evidence_into_a_live_optimization(tmp_path: Path) -> None:
    # GEPA compares candidates against each other. A run that silently switched to
    # model-free scripted evidence would rank a candidate on a different experiment.
    live = _case("case[completed,live-mode]")
    scripted = _case("case[completed]")
    adapter = _adapter(tmp_path, [live, scripted])

    adapter.evaluate([live], _seed_candidate().components, capture_traces=True)

    with pytest.raises(BridgeExecutionModeError, match="execution modes"):
        adapter.evaluate([scripted], _seed_candidate().components, capture_traces=True)


def test_adapter_allows_a_wholly_scripted_optimization(tmp_path: Path) -> None:
    scripted = _case("case[completed]")
    adapter = _adapter(tmp_path, [scripted])

    adapter.evaluate([scripted], _seed_candidate().components, capture_traces=True)

    assert adapter.execution_modes == ("scripted",)


# ---------------------------------------------------------------------------
# Read-only-specific reflection feedback (korvid_readonly evidence only)
# ---------------------------------------------------------------------------


def test_readonly_trace_exposes_bounded_diagnosis_and_evidence_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _readonly_adapter(tmp_path, monkeypatch)
    case = _readonly_case()

    eval_batch = adapter.evaluate([case], _seed_candidate().components, capture_traces=True)

    assert eval_batch.outputs[0].answer == ""
    assert eval_batch.outputs[0].error is None
    assert eval_batch.trajectories is not None
    trace = eval_batch.trajectories[0]
    assert trace.final_answer == ""
    assert trace.diagnosis_success is True
    assert trace.evidence_fetched is True
    assert trace.resolvable_tool_call_count == 4
    assert trace.missing_mention_count == 0
    assert trace.missing_evidence_count == 0
    assert trace.malformed_tool_call_count == 0
    assert trace.citation_coverage == 1.0
    assert trace.citation_precision == 1.0

    feedback = _build_feedback_for(adapter, trace)
    assert "Diagnosis: success." in feedback
    assert "Evidence: fetched." in feedback
    assert "Citation coverage: 1.00, precision: 1.00." in feedback

    record = adapter._trace_to_record(trace)
    outputs = record["Generated Outputs"]
    assert "answer" not in outputs
    assert outputs["diagnosis_success"] is True
    assert outputs["evidence_fetched"] is True
    assert outputs["citation_coverage"] == 1.0
    assert outputs["citation_precision"] == 1.0


def test_readonly_trace_reports_failed_diagnosis_and_missing_mentions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _readonly_adapter(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "diagnosis-failed")
    case = _readonly_case()

    eval_batch = adapter.evaluate([case], _seed_candidate().components, capture_traces=True)

    assert eval_batch.trajectories is not None
    trace = eval_batch.trajectories[0]
    assert trace.diagnosis_success is False
    assert trace.missing_mention_count == 1
    feedback = _build_feedback_for(adapter, trace)
    assert "Diagnosis: failure." in feedback
    assert "Missing mentions: 1." in feedback


def test_readonly_model_failure_omits_ungraded_quality_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _readonly_adapter(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "model-failure")
    case = _readonly_case()

    eval_batch = adapter.evaluate(
        [case], _seed_candidate().components, capture_traces=True
    )

    assert eval_batch.trajectories is not None
    trace = eval_batch.trajectories[0]
    record = adapter._trace_to_record(trace)
    outputs = record["Generated Outputs"]
    for field in (
        "diagnosis_success",
        "evidence_fetched",
        "missing_mention_count",
        "missing_evidence_count",
        "malformed_tool_call_count",
        "citation_coverage",
        "citation_precision",
    ):
        assert field not in outputs
    assert "Citation coverage" not in record["Feedback"]


def test_readonly_trace_reports_missing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _readonly_adapter(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "evidence-missing")
    case = _readonly_case()

    eval_batch = adapter.evaluate([case], _seed_candidate().components, capture_traces=True)

    assert eval_batch.trajectories is not None
    trace = eval_batch.trajectories[0]
    assert trace.evidence_fetched is False
    assert trace.missing_evidence_count == 1

    feedback = _build_feedback_for(adapter, trace)
    assert "Evidence: missing." in feedback
    assert "Missing evidence items: 1." in feedback


def test_readonly_trace_reports_malformed_tool_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _readonly_adapter(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "malformed-tool-calls")
    case = _readonly_case()

    eval_batch = adapter.evaluate([case], _seed_candidate().components, capture_traces=True)

    assert eval_batch.trajectories is not None
    trace = eval_batch.trajectories[0]
    assert trace.malformed_tool_call_count == 2

    feedback = _build_feedback_for(adapter, trace)
    assert "Malformed tool calls: 2." in feedback


def test_readonly_trace_hides_zero_malformed_tool_calls_from_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _readonly_adapter(tmp_path, monkeypatch)
    case = _readonly_case()

    eval_batch = adapter.evaluate([case], _seed_candidate().components, capture_traces=True)

    assert eval_batch.trajectories is not None
    trace = eval_batch.trajectories[0]
    assert trace.malformed_tool_call_count == 0
    feedback = _build_feedback_for(adapter, trace)
    assert "Malformed tool calls" not in feedback


def test_readonly_trace_model_failure_leaves_diagnosis_fields_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _readonly_adapter(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "model-failure")
    case = _readonly_case()

    eval_batch = adapter.evaluate([case], _seed_candidate().components, capture_traces=True)

    assert eval_batch.trajectories is not None
    trace = eval_batch.trajectories[0]
    assert trace.diagnosis_success is None
    assert trace.evidence_fetched is None

    record = adapter._trace_to_record(trace)
    assert "diagnosis_success" not in record["Generated Outputs"]
    assert "evidence_fetched" not in record["Generated Outputs"]


def test_process_backed_trace_never_gains_readonly_reflection_fields(tmp_path: Path) -> None:
    completed = _case("case[completed]")
    adapter = _adapter(tmp_path, [completed])

    eval_batch = adapter.evaluate([completed], _seed_candidate().components, capture_traces=True)

    assert eval_batch.trajectories is not None
    trace = eval_batch.trajectories[0]
    record = adapter._trace_to_record(trace)
    assert record["Generated Outputs"]["answer"] == trace.final_answer
    assert trace.diagnosis_success is None
    assert trace.evidence_fetched is None
    assert trace.missing_mention_count is None
    assert trace.missing_evidence_count is None
    assert trace.malformed_tool_call_count is None
    assert trace.citation_coverage is None
    assert trace.citation_precision is None

    record = adapter._trace_to_record(trace)
    for key in (
        "diagnosis_success",
        "evidence_fetched",
        "missing_mention_count",
        "missing_evidence_count",
        "malformed_tool_call_count",
        "citation_coverage",
        "citation_precision",
    ):
        assert key not in record["Generated Outputs"]


def _build_feedback_for(adapter: KorvidGEPAAdapter, trace: Any) -> str:
    from korvid_prompt_lab.adapter import _build_feedback

    return _build_feedback(trace)
