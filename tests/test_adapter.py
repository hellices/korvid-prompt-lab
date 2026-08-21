from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.contracts import Candidate, Campaign, EvalCase, ProcessServing
from korvid_prompt_lab.runner import BridgeStatusError, KorvidProcessRunner

from korvid_prompt_lab.adapter import KorvidGEPAAdapter


ROOT = Path(__file__).resolve().parents[1]


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


def test_adapter_evaluate_returns_one_output_score_and_safe_trajectory_per_case(tmp_path: Path) -> None:
    completed = _case("case[completed]")
    model_failure = _case("case[model-failure]")
    candidate = _seed_candidate().components
    original = dict(candidate)
    adapter = _adapter(tmp_path, [completed, model_failure])

    eval_batch = adapter.evaluate([completed, model_failure], candidate, capture_traces=True)

    assert candidate == original
    assert [result.status for result in eval_batch.outputs] == ["completed", "model_failure"]
    assert eval_batch.scores == pytest.approx([0.85, 0.0])
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


def test_adapter_zeroes_unsafe_scores_and_redacts_sensitive_fields_from_reflection_records(tmp_path: Path) -> None:
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
            "protocol_version": 1,
            "status": "completed",
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
    assert eval_batch.scores == [0.0]
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


def test_adapter_zeroes_all_scores_when_any_case_is_unsafe(tmp_path: Path) -> None:
    safe_bridge_path = tmp_path / "mixed_bridge.py"
    safe_bridge_path.write_text(
        """
import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[sys.argv.index("--request") + 1]).read_text(encoding="utf-8"))
response_path = Path(sys.argv[sys.argv.index("--response") + 1])
case_id = request["case"]["case_id"]
unsafe = case_id.endswith("unsafe")
response_path.write_text(
    json.dumps(
        {
            "protocol_version": 1,
            "status": "completed",
            "candidate_fingerprint": request["candidate_fingerprint"],
            "request_identity": {
                "case_id": request["case"]["case_id"],
                "template_id": request["case"]["template_id"],
                "model": request["case"]["model"],
                "repetition": request["case"]["repetition"],
                "seed": request["case"]["seed"],
            },
            "grade": {
                "completion": 0.9 if not unsafe else 0.2,
                "verification": 0.9 if not unsafe else 0.1,
                "efficiency": 0.9 if not unsafe else 0.3,
                "hard_failures": ["approval_bypass"] if unsafe else [],
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
    unsafe_case = _case("case-unsafe")
    adapter = _adapter(
        tmp_path,
        [safe_case, unsafe_case],
        (sys.executable, str(safe_bridge_path), "--request", "{request}", "--response", "{response}"),
    )

    eval_batch = adapter.evaluate([safe_case, unsafe_case], _seed_candidate().components, capture_traces=True)

    assert eval_batch.scores == [0.0, 0.0]
    assert eval_batch.trajectories is not None
    assert len(eval_batch.outputs) == 2
    assert len(eval_batch.trajectories) == 2
    assert [trace.case_id for trace in eval_batch.trajectories] == ["case-safe", "case-unsafe"]
    assert [trace.outcome for trace in eval_batch.trajectories] == ["completed", "unsafe"]


def test_adapter_propagates_systemic_runner_failures(tmp_path: Path) -> None:
    case = _case("case[systemic-status]")
    adapter = _adapter(tmp_path, [case])

    with pytest.raises(BridgeStatusError):
        adapter.evaluate([case], _seed_candidate().components, capture_traces=False)
