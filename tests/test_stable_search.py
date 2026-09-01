from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.contracts import Campaign, Candidate, EvalCase, ProcessServing
from korvid_prompt_lab.scoring import BridgeResult, OperationGrade
from korvid_prompt_lab.stable_candidates import (
    StructuredCandidate,
    build_structured_candidates,
)
from korvid_prompt_lab.stable_scenarios import (
    ScenarioAssignment,
    ScenarioClass,
    ScenarioManifest,
    ScenarioSplitSummary,
)
from korvid_prompt_lab.stable_search import StableSearchConfig, run_stable_search


@dataclass(frozen=True, slots=True)
class _ScriptedRun:
    status: str = "completed"
    score: float = 0.0
    verification: float = 1.0
    hard_failures: tuple[str, ...] = ()
    tool_calls: int = 4
    resolvable_tool_calls: int = 4
    malformed_tool_calls: int = 0
    execution_mode: str = "scripted"
    error: str | None = None


@dataclass(slots=True)
class _FakeRunner:
    campaign: Campaign
    case_splits: Mapping[str, str]
    scripts: Mapping[tuple[str, str], Sequence[_ScriptedRun]]
    calls: list[tuple[str, str, int, Path]] = field(default_factory=list)

    def run(
        self,
        candidate: Candidate,
        case: EvalCase,
        run_dir: Path | str,
        *,
        repetition: int = 1,
        seed: int = 0,
    ) -> BridgeResult:
        split = self.case_splits[case.case_id]
        script = self.scripts[(candidate.candidate_id, split)][repetition - 1]
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        self.calls.append((candidate.candidate_id, case.case_id, repetition, run_path))
        (run_path / "request.json").write_text(
            json.dumps({"candidate_id": candidate.candidate_id, "case_id": case.case_id}, indent=2) + "\n",
            encoding="utf-8",
        )

        payload = {
            "protocol_version": 2,
            "status": script.status,
            "execution_mode": script.execution_mode,
            "candidate_fingerprint": candidate.fingerprint,
            "request_identity": {
                "case_id": case.case_id,
                "template_id": case.template_id,
                "model": case.models[0],
                "repetition": repetition,
                "seed": seed,
            },
            "grade": None,
            "answer": "",
            "journal": {
                "checkpoints": ["dispatch", "verify"] if script.status == "completed" else ["dispatch"],
                "tool_calls": script.tool_calls,
                "resolvable_tool_calls": script.resolvable_tool_calls,
                "malformed_tool_calls": script.malformed_tool_calls,
            },
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "error": script.error,
        }
        grade: OperationGrade | None = None
        if script.status == "completed":
            grade = OperationGrade(
                completion=script.score,
                verification=script.verification,
                efficiency=1.0,
                hard_failures=script.hard_failures,
            )
            payload["grade"] = {
                "completion": script.score,
                "verification": script.verification,
                "efficiency": 1.0,
                "hard_failures": list(script.hard_failures),
            }

        (run_path / "response.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return BridgeResult(
            protocol_version=2,
            status=script.status,
            execution_mode=script.execution_mode,
            candidate_fingerprint=candidate.fingerprint,
            grade=grade,
            answer="",
            journal=cast(Mapping[str, object], payload["journal"]),
            usage=cast(Mapping[str, object], payload["usage"]),
            error=script.error,
        )


_SPLITS_BY_CASE = {
    "train-a": "train",
    "train-b": "train",
    "validation-a": "validation",
    "validation-b": "validation",
    "milestone-a": "milestone",
    "milestone-b": "milestone",
}


def _baseline() -> Candidate:
    return Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "baseline",
            "components": {"system": "Stay safe."},
            "metadata": {"profile": "small", "korvid_version": "0.3.0"},
        }
    )



def _candidates() -> tuple[StructuredCandidate, ...]:
    selected = {
        "evidence-first",
        "one-tool-at-a-time",
        "cite-before-conclusion",
        "stop-with-uncertainty",
        "evidence-first+one-tool-at-a-time",
    }
    return tuple(
        candidate for candidate in build_structured_candidates(_baseline()) if candidate.candidate.candidate_id in selected
    )



def _cases() -> tuple[EvalCase, ...]:
    return tuple(
        EvalCase(
            case_id=case_id,
            template_id="readonly-template",
            prompt=f"Question for {case_id}",
            models=("mock-small",),
        )
        for case_id in _SPLITS_BY_CASE
    )



def _manifest() -> ScenarioManifest:
    assignments = tuple(
        ScenarioAssignment(
            scenario_id=case_id,
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            split=cast(Literal["train", "validation", "milestone"], split),
            question_sha256=f"question-{case_id}",
            fixture_sha256=f"fixture-{case_id}",
            korvid_version="0.3.0",
        )
        for case_id, split in _SPLITS_BY_CASE.items()
    )
    return ScenarioManifest(
        korvid_version="0.3.0",
        assignments=assignments,
        train=("train-a", "train-b"),
        validation=("validation-a", "validation-b"),
        milestone=("milestone-a", "milestone-b"),
        split_summaries=(
            ScenarioSplitSummary(
                split_name="train",
                classes=(ScenarioClass.WORKLOAD_HEALTH,),
                scenario_ids=("train-a", "train-b"),
            ),
            ScenarioSplitSummary(
                split_name="validation",
                classes=(ScenarioClass.WORKLOAD_HEALTH,),
                scenario_ids=("validation-a", "validation-b"),
            ),
            ScenarioSplitSummary(
                split_name="milestone",
                classes=(ScenarioClass.WORKLOAD_HEALTH,),
                scenario_ids=("milestone-a", "milestone-b"),
            ),
        ),
    )



def _runner_scripts(*, promote_winner: bool) -> dict[tuple[str, str], Sequence[_ScriptedRun]]:
    return {
        ("baseline", "train"): (_ScriptedRun(score=0.40, verification=0.50),),
        ("baseline", "validation"): tuple(_ScriptedRun(score=0.40, verification=0.50) for _ in range(5)),
        ("baseline", "milestone"): tuple(_ScriptedRun(score=0.50, verification=0.50) for _ in range(5)),
        ("evidence-first", "train"): (_ScriptedRun(score=0.70, verification=0.90),),
        ("evidence-first", "validation"): tuple(
            _ScriptedRun(score=score, verification=0.50) for score in ([0.57] * 5 if promote_winner else [0.55] * 5)
        ),
        ("evidence-first", "milestone"): tuple(
            _ScriptedRun(score=score, verification=0.50) for score in ([0.67] * 5 if promote_winner else [0.65] * 5)
        ),
        ("one-tool-at-a-time", "train"): (_ScriptedRun(score=0.60, verification=0.80),),
        ("one-tool-at-a-time", "validation"): tuple(
            _ScriptedRun(score=score, verification=0.50) for score in ([0.52] * 3 + [0.54] * 2)
        ),
        ("one-tool-at-a-time", "milestone"): tuple(_ScriptedRun(score=0.62, verification=0.50) for _ in range(5)),
        ("cite-before-conclusion", "train"): (_ScriptedRun(score=0.40, verification=0.70),),
        ("cite-before-conclusion", "validation"): tuple(_ScriptedRun(score=0.40, verification=0.50) for _ in range(5)),
        ("cite-before-conclusion", "milestone"): tuple(_ScriptedRun(score=0.40, verification=0.50) for _ in range(5)),
        (
            "stop-with-uncertainty",
            "train",
        ): (_ScriptedRun(score=0.0, verification=0.20, hard_failures=("safety_violation",)),),
        (
            "stop-with-uncertainty",
            "validation",
        ): tuple(_ScriptedRun(score=0.0, verification=0.20, hard_failures=("safety_violation",)) for _ in range(5)),
        ("stop-with-uncertainty", "milestone"): tuple(
            _ScriptedRun(score=0.0, verification=0.20, hard_failures=("safety_violation",)) for _ in range(5)
        ),
        (
            "evidence-first+one-tool-at-a-time",
            "train",
        ): (_ScriptedRun(status="system_failure", error="bridge crashed"),),
        ("evidence-first+one-tool-at-a-time", "validation"): tuple(
            _ScriptedRun(status="system_failure", error="bridge crashed") for _ in range(5)
        ),
        ("evidence-first+one-tool-at-a-time", "milestone"): tuple(
            _ScriptedRun(status="system_failure", error="bridge crashed") for _ in range(5)
        ),
    }



def _runner(*, promote_winner: bool) -> _FakeRunner:
    campaign = Campaign(
        schema_version=1,
        campaign_id="stable-search-campaign",
        repetitions=5,
        models=("mock-small",),
        cases=_cases(),
        serving=ProcessServing(backend="process", command=(sys.executable, "-c", "print('unused')")),
    )
    return _FakeRunner(campaign=campaign, case_splits=_SPLITS_BY_CASE, scripts=_runner_scripts(promote_winner=promote_winner))



def _calls_for_stage(
    calls: Sequence[tuple[str, str, int, Path]], stage_name: str
) -> list[tuple[str, str, int, Path]]:
    return [call for call in calls if stage_name in call[3].parts]



def test_stable_search_promotes_known_winner(tmp_path: Path) -> None:
    runner = _runner(promote_winner=True)
    artifacts = run_stable_search(
        runner=runner,
        baseline=_baseline(),
        candidates=_candidates(),
        manifest=_manifest(),
        artifact_root=tmp_path / "campaign",
        config=StableSearchConfig(),
    )

    assert artifacts.decision.status == "promote"
    assert artifacts.decision.candidate_id == "evidence-first"
    assert [item.candidate_id for item in artifacts.screening.survivors] == [
        "evidence-first",
        "one-tool-at-a-time",
        "cite-before-conclusion",
    ]
    assert [item.candidate_id for item in artifacts.validation.survivors] == [
        "evidence-first",
        "one-tool-at-a-time",
    ]

    stage_a_calls = _calls_for_stage(runner.calls, "stage-a")
    stage_b_calls = _calls_for_stage(runner.calls, "stage-b")
    stage_c_calls = _calls_for_stage(runner.calls, "stage-c")

    assert {call[2] for call in stage_a_calls} == {1}
    assert {call[2] for call in stage_b_calls} == {1, 2, 3}
    assert {call[2] for call in stage_c_calls} == {1, 2, 3, 4, 5}

    assert sum(call[0] == "stop-with-uncertainty" for call in stage_a_calls) == 1
    assert sum(call[0] == "evidence-first+one-tool-at-a-time" for call in stage_a_calls) == 1
    assert all(call[0] not in {"stop-with-uncertainty", "evidence-first+one-tool-at-a-time"} for call in stage_b_calls)
    assert all(call[0] not in {"stop-with-uncertainty", "evidence-first+one-tool-at-a-time", "cite-before-conclusion"} for call in stage_c_calls)

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["decision"]["status"] == "promote"
    assert summary["decision"]["candidate_id"] == "evidence-first"
    assert sorted(path.name for path in (tmp_path / "campaign").rglob("request.json")) == []
    assert all(path.suffix == ".json" for path in (tmp_path / "campaign").rglob("*") if path.is_file())



def test_stable_search_records_no_winner(tmp_path: Path) -> None:
    artifacts = run_stable_search(
        runner=_runner(promote_winner=False),
        baseline=_baseline(),
        candidates=_candidates(),
        manifest=_manifest(),
        artifact_root=tmp_path / "campaign",
    )

    assert artifacts.decision.status == "no_stable_winner"
    assert artifacts.decision.candidate_id is None
    assert artifacts.decision.reasons == (
        "evidence-first:validation_delta_below_0_10",
        "evidence-first:milestone_delta_below_0_10",
        "one-tool-at-a-time:validation_delta_below_0_10",
        "one-tool-at-a-time:milestone_delta_below_0_10",
    )



def test_stable_search_refuses_to_reuse_an_existing_artifact_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "existing"
    artifact_root.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        run_stable_search(
            runner=_runner(promote_winner=True),
            baseline=_baseline(),
            candidates=_candidates(),
            manifest=_manifest(),
            artifact_root=artifact_root,
        )
