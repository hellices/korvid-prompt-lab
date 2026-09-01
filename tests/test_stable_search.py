from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import pytest
from litellm.exceptions import APIError, AuthenticationError, BadRequestError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.contracts import Campaign, Candidate, EvalCase, ProcessServing
from korvid_prompt_lab.runner import BridgeProcessExitError
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
from korvid_prompt_lab.stable_search import (
    StableSearchConfig,
    StableSearchExtension,
    run_stable_search,
)


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
    answer: str = "RAW_ANSWER_SHOULD_NOT_PERSIST"
    request_body: str = "SECRET_REQUEST_PROMPT"
    raised_error: BaseException | None = None


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
            json.dumps(
                {
                    "candidate_id": candidate.candidate_id,
                    "case_id": case.case_id,
                    "prompt": script.request_body,
                },
                indent=2,
            )
            + "\n",
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
            "answer": script.answer,
            "journal": {
                "checkpoints": ["dispatch", "verify"] if script.status == "completed" else ["dispatch"],
                "tool_calls": script.tool_calls,
                "resolvable_tool_calls": script.resolvable_tool_calls,
                "malformed_tool_calls": script.malformed_tool_calls,
                "sensitive_tool_output": "TOP_SECRET_TOOL_OUTPUT",
            },
            "usage": {"input_tokens": 11, "output_tokens": 7, "wall_time_seconds": 3.25},
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
        if script.raised_error is not None:
            raise script.raised_error

        return BridgeResult(
            protocol_version=2,
            status=script.status,
            execution_mode=script.execution_mode,
            candidate_fingerprint=candidate.fingerprint,
            grade=grade,
            answer=script.answer,
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

_PATHOLOGICAL_SPLITS_BY_CASE = {
    "../../../../train:?*case": "train",
    "../../../validation<>case": "validation",
    "milestone/../../../../escape|case": "milestone",
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



def _structured_candidate(candidate_id: str, append: str = "Inspect evidence first.") -> StructuredCandidate:
    return StructuredCandidate(
        axes=(),
        candidate=Candidate.from_mapping(
            {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "components": {
                    "system": _baseline().components["system"],
                    "append": append,
                },
                "metadata": {"source": "test"},
            }
        ),
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
        candidate
        for candidate in build_structured_candidates(_baseline())
        if candidate.candidate.candidate_id in selected
    )



def _cases(case_splits: Mapping[str, str] = _SPLITS_BY_CASE) -> tuple[EvalCase, ...]:
    return tuple(
        EvalCase(
            case_id=case_id,
            template_id="readonly-template",
            prompt=f"Question for {case_id}",
            models=("mock-small",),
        )
        for case_id in case_splits
    )



def _manifest(case_splits: Mapping[str, str] = _SPLITS_BY_CASE) -> ScenarioManifest:
    assignments = tuple(
        ScenarioAssignment(
            scenario_id=case_id,
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            split=cast(Literal["train", "validation", "milestone"], split),
            question_sha256=f"question-{case_id}",
            fixture_sha256=f"fixture-{case_id}",
            korvid_version="0.3.0",
        )
        for case_id, split in case_splits.items()
    )
    split_summaries = tuple(
        ScenarioSplitSummary(
            split_name=cast(Literal["train", "validation", "milestone"], split_name),
            classes=(ScenarioClass.WORKLOAD_HEALTH,),
            scenario_ids=tuple(case_id for case_id, split in case_splits.items() if split == split_name),
        )
        for split_name in ("train", "validation", "milestone")
    )
    return ScenarioManifest(
        korvid_version="0.3.0",
        assignments=assignments,
        train=tuple(case_id for case_id, split in case_splits.items() if split == "train"),
        validation=tuple(case_id for case_id, split in case_splits.items() if split == "validation"),
        milestone=tuple(case_id for case_id, split in case_splits.items() if split == "milestone"),
        split_summaries=split_summaries,
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
        ("stop-with-uncertainty", "train"): (_ScriptedRun(score=0.0, verification=0.20, hard_failures=("safety_violation",)),),
        ("stop-with-uncertainty", "validation"): tuple(
            _ScriptedRun(score=0.0, verification=0.20, hard_failures=("safety_violation",)) for _ in range(5)
        ),
        ("stop-with-uncertainty", "milestone"): tuple(
            _ScriptedRun(score=0.0, verification=0.20, hard_failures=("safety_violation",)) for _ in range(5)
        ),
        ("evidence-first+one-tool-at-a-time", "train"): (_ScriptedRun(score=0.30, verification=0.45),),
        ("evidence-first+one-tool-at-a-time", "validation"): tuple(
            _ScriptedRun(score=0.30, verification=0.45) for _ in range(5)
        ),
        ("evidence-first+one-tool-at-a-time", "milestone"): tuple(
            _ScriptedRun(score=0.30, verification=0.45) for _ in range(5)
        ),
    }



def _runner_for_scripts(
    *,
    case_splits: Mapping[str, str],
    scripts: Mapping[tuple[str, str], Sequence[_ScriptedRun]],
) -> _FakeRunner:
    campaign = Campaign(
        schema_version=1,
        campaign_id="stable-search-campaign",
        repetitions=5,
        models=("mock-small",),
        cases=_cases(case_splits),
        serving=ProcessServing(backend="process", command=(sys.executable, "-c", "print('unused')")),
    )
    return _FakeRunner(campaign=campaign, case_splits=case_splits, scripts=scripts)



def _runner(*, promote_winner: bool) -> _FakeRunner:
    return _runner_for_scripts(case_splits=_SPLITS_BY_CASE, scripts=_runner_scripts(promote_winner=promote_winner))



def _calls_for_stage(calls: Sequence[tuple[str, str, int, Path]], stage_name: str) -> list[tuple[str, str, int, Path]]:
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
    assert all(call[0] != "stop-with-uncertainty" for call in stage_b_calls)
    assert all(call[0] not in {"stop-with-uncertainty", "cite-before-conclusion"} for call in stage_c_calls)

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



def test_stable_search_aborts_on_bridge_system_error_without_writing_a_success_decision(tmp_path: Path) -> None:
    scripts = dict(_runner_scripts(promote_winner=True))
    scripts[("evidence-first+one-tool-at-a-time", "train")] = (
        _ScriptedRun(
            raised_error=BridgeProcessExitError("backend exploded with TOP_SECRET_BACKTRACE"),
            error="TOP_SECRET_BACKTRACE",
            answer="LEAKED_RAW_ANSWER",
        ),
    )
    artifact_root = tmp_path / "campaign"

    with pytest.raises(BridgeProcessExitError, match="backend exploded"):
        run_stable_search(
            runner=_runner_for_scripts(case_splits=_SPLITS_BY_CASE, scripts=scripts),
            baseline=_baseline(),
            candidates=_candidates(),
            manifest=_manifest(),
            artifact_root=artifact_root,
        )

    assert not (artifact_root / "stage-a" / "screening-summary.json").exists()
    assert not (artifact_root / "stable-search-summary.json").exists()



def test_stable_search_persists_only_normalized_run_artifacts_and_sanitizes_paths(tmp_path: Path) -> None:
    weird_candidate = _structured_candidate("../../../../winner:?*[]")
    weird_case_splits = _PATHOLOGICAL_SPLITS_BY_CASE
    scripts = {
        ("baseline", "train"): (_ScriptedRun(score=0.40, verification=0.50),),
        ("baseline", "validation"): tuple(_ScriptedRun(score=0.40, verification=0.50) for _ in range(3)),
        ("baseline", "milestone"): tuple(_ScriptedRun(score=0.50, verification=0.50) for _ in range(5)),
        ("../../../../winner:?*[]", "train"): (
            _ScriptedRun(status="model_failure", error="RAW_MODEL_FAILURE_WITH_SECRET", answer="LEAKED_RAW_ANSWER"),
        ),
        ("../../../../winner:?*[]", "validation"): tuple(
            _ScriptedRun(status="model_failure", error="RAW_MODEL_FAILURE_WITH_SECRET", answer="LEAKED_RAW_ANSWER")
            for _ in range(3)
        ),
        ("../../../../winner:?*[]", "milestone"): tuple(
            _ScriptedRun(status="model_failure", error="RAW_MODEL_FAILURE_WITH_SECRET", answer="LEAKED_RAW_ANSWER")
            for _ in range(5)
        ),
    }
    runner = _runner_for_scripts(case_splits=weird_case_splits, scripts=scripts)
    artifact_root = tmp_path / "campaign"

    artifacts = run_stable_search(
        runner=runner,
        baseline=_baseline(),
        candidates=(weird_candidate,),
        manifest=_manifest(weird_case_splits),
        artifact_root=artifact_root,
        config=StableSearchConfig(screening_survivors=1, finalists=1),
    )

    assert artifacts.decision.status == "no_stable_winner"
    assert all(call[3].resolve().is_relative_to(artifact_root.resolve()) for call in runner.calls)

    written = "\n".join(path.read_text(encoding="utf-8") for path in artifact_root.rglob("*") if path.is_file())
    for forbidden in (
        "LEAKED_RAW_ANSWER",
        "RAW_MODEL_FAILURE_WITH_SECRET",
        "SECRET_REQUEST_PROMPT",
        "TOP_SECRET_TOOL_OUTPUT",
        "request.json",
    ):
        assert forbidden not in written

    response_paths = sorted(artifact_root.rglob("response.json"))
    assert response_paths
    assert all(path.resolve().is_relative_to(artifact_root.resolve()) for path in response_paths)
    assert all(".." not in part for path in response_paths for part in path.relative_to(artifact_root).parts)

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in response_paths]
    model_failure_payload = next(payload for payload in payloads if payload["status"] == "model_failure")
    assert model_failure_payload["answer"] == ""
    assert model_failure_payload["error"] == "model_failure"
    assert model_failure_payload["case_id"] in weird_case_splits
    assert model_failure_payload["candidate_id"] == "../../../../winner:?*[]"



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


def _proposal_candidate_id(candidate_id: str, proposed_append: str) -> str:
    digest = hashlib.sha256(proposed_append.encode("utf-8")).hexdigest()[:8]
    return f"{candidate_id}+proposal-{digest}"


def test_stable_search_promotes_a_proposer_candidate_after_stage_b_replay(
    tmp_path: Path,
) -> None:
    requests: list[object] = []
    proposed_append = "inspect runtime evidence before stating a diagnosis.\nkorvid-tuned"
    proposed_candidate_id = _proposal_candidate_id("bounded-finalist", proposed_append)

    class FakeProposer:
        def __init__(self) -> None:
            self.reflection_lm = {"model": "ollama_chat/qwen3:4b"}

        def safe_propose(self, request_or_context: object, **kwargs: object) -> str | None:
            del kwargs
            requests.append(request_or_context)
            return proposed_append

    scripts = {
        ("baseline", "train"): (_ScriptedRun(score=0.40, verification=0.60),),
        ("baseline", "validation"): tuple(_ScriptedRun(score=0.40, verification=0.60) for _ in range(5)),
        ("baseline", "milestone"): tuple(_ScriptedRun(score=0.40, verification=0.60) for _ in range(5)),
        ("bounded-finalist", "train"): (_ScriptedRun(score=0.60, verification=0.60, tool_calls=4, resolvable_tool_calls=2),),
        ("bounded-finalist", "validation"): tuple(
            _ScriptedRun(score=0.55, verification=0.60, tool_calls=4, resolvable_tool_calls=2) for _ in range(5)
        ),
        ("bounded-finalist", "milestone"): tuple(
            _ScriptedRun(score=0.45, verification=0.60, tool_calls=4, resolvable_tool_calls=2) for _ in range(5)
        ),
        (proposed_candidate_id, "validation"): tuple(
            _ScriptedRun(score=0.56, verification=0.70, tool_calls=2, resolvable_tool_calls=2) for _ in range(5)
        ),
        (proposed_candidate_id, "milestone"): tuple(
            _ScriptedRun(score=0.55, verification=0.70, tool_calls=2, resolvable_tool_calls=2) for _ in range(5)
        ),
    }
    candidate = _structured_candidate(
        "bounded-finalist",
        append="inspect runtime evidence before stating a diagnosis.",
    )

    artifacts = run_stable_search(
        runner=_runner_for_scripts(case_splits=_SPLITS_BY_CASE, scripts=scripts),
        baseline=_baseline(),
        candidates=(candidate,),
        manifest=_manifest(),
        artifact_root=tmp_path / "campaign",
        config=StableSearchConfig(screening_survivors=1, finalists=1),
        extension=StableSearchExtension(bounded_append_proposer=FakeProposer()),
    )

    assert artifacts.decision.status == "promote"
    assert artifacts.decision.candidate_id == proposed_candidate_id
    assert artifacts.extension is not None
    assert len(artifacts.extension.bounded_proposals) == 1
    proposal = artifacts.extension.bounded_proposals[0]
    assert proposal.finalist_candidate_id == "bounded-finalist"
    assert proposal.failure_axis == "one-tool-at-a-time"
    assert proposal.status == "promote"
    assert proposal.proposed_append == proposed_append
    assert proposal.proposed_candidate_id == proposed_candidate_id
    assert proposal.validation_measurement is not None
    assert proposal.qualification is not None
    assert requests


def test_stable_search_records_proposer_validation_rejection_without_stage_c_replay(
    tmp_path: Path,
) -> None:
    proposed_append = "inspect runtime evidence before stating a diagnosis.\nkorvid-rejected"
    proposed_candidate_id = _proposal_candidate_id("bounded-finalist", proposed_append)

    class FakeProposer:
        def __init__(self) -> None:
            self.reflection_lm = {"model": "ollama_chat/qwen3:4b"}

        def safe_propose(self, request_or_context: object, **kwargs: object) -> str | None:
            del request_or_context, kwargs
            return proposed_append

    scripts = {
        ("baseline", "train"): (_ScriptedRun(score=0.40, verification=0.60),),
        ("baseline", "validation"): tuple(_ScriptedRun(score=0.40, verification=0.60) for _ in range(5)),
        ("baseline", "milestone"): tuple(_ScriptedRun(score=0.40, verification=0.60) for _ in range(5)),
        ("bounded-finalist", "train"): (_ScriptedRun(score=0.60, verification=0.60, tool_calls=4, resolvable_tool_calls=2),),
        ("bounded-finalist", "validation"): tuple(
            _ScriptedRun(score=0.55, verification=0.60, tool_calls=4, resolvable_tool_calls=2) for _ in range(5)
        ),
        ("bounded-finalist", "milestone"): tuple(
            _ScriptedRun(score=0.45, verification=0.60, tool_calls=4, resolvable_tool_calls=2) for _ in range(5)
        ),
        (proposed_candidate_id, "validation"): tuple(
            _ScriptedRun(score=0.35, verification=0.70, tool_calls=2, resolvable_tool_calls=2) for _ in range(5)
        ),
    }

    runner = _runner_for_scripts(case_splits=_SPLITS_BY_CASE, scripts=scripts)
    artifacts = run_stable_search(
        runner=runner,
        baseline=_baseline(),
        candidates=(
            _structured_candidate(
                "bounded-finalist",
                append="inspect runtime evidence before stating a diagnosis.",
            ),
        ),
        manifest=_manifest(),
        artifact_root=tmp_path / "campaign",
        config=StableSearchConfig(screening_survivors=1, finalists=1),
        extension=StableSearchExtension(bounded_append_proposer=FakeProposer()),
    )

    assert artifacts.decision.status == "no_stable_winner"
    assert artifacts.extension is not None
    proposal = artifacts.extension.bounded_proposals[0]
    assert proposal.proposed_candidate_id == proposed_candidate_id
    assert proposal.status == "validation_rejected"
    assert proposal.validation_measurement is not None
    assert "mean_delta_not_positive" in proposal.validation_rejection_reasons
    assert proposal.qualification is None
    assert all(call[0] != proposed_candidate_id for call in _calls_for_stage(runner.calls, "stage-c"))


@pytest.mark.parametrize(
    ("error", "label"),
    [
        (
            AuthenticationError("TOP_SECRET_AUTH", "openai", "model"),
            "authentication_error",
        ),
        (
            BadRequestError("TOP_SECRET_BAD_REQUEST", "model", "openai"),
            "bad_request_error",
        ),
        (
            APIError(500, "TOP_SECRET_API", "openai", "model"),
            "api_error",
        ),
    ],
)
def test_stable_search_catches_known_proposer_errors_without_invalidating_structured_results(
    tmp_path: Path,
    error: BaseException,
    label: str,
) -> None:
    class RaisingProposer:
        def __init__(self) -> None:
            self.reflection_lm = {"model": "ollama_chat/qwen3:4b"}

        def safe_propose(self, request_or_context: object, **kwargs: object) -> str | None:
            del request_or_context, kwargs
            raise error

    scripts = {
        ("baseline", "train"): (_ScriptedRun(score=0.40, verification=0.60),),
        ("baseline", "validation"): tuple(_ScriptedRun(score=0.40, verification=0.60) for _ in range(5)),
        ("baseline", "milestone"): tuple(_ScriptedRun(score=0.40, verification=0.60) for _ in range(5)),
        ("bounded-finalist", "train"): (_ScriptedRun(score=0.60, verification=0.60, tool_calls=4, resolvable_tool_calls=2),),
        ("bounded-finalist", "validation"): tuple(
            _ScriptedRun(score=0.55, verification=0.60, tool_calls=4, resolvable_tool_calls=2) for _ in range(5)
        ),
        ("bounded-finalist", "milestone"): tuple(
            _ScriptedRun(score=0.45, verification=0.60, tool_calls=4, resolvable_tool_calls=2) for _ in range(5)
        ),
    }

    artifacts = run_stable_search(
        runner=_runner_for_scripts(case_splits=_SPLITS_BY_CASE, scripts=scripts),
        baseline=_baseline(),
        candidates=(
            _structured_candidate(
                "bounded-finalist",
                append="inspect runtime evidence before stating a diagnosis.",
            ),
        ),
        manifest=_manifest(),
        artifact_root=tmp_path / "campaign",
        config=StableSearchConfig(screening_survivors=1, finalists=1),
        extension=StableSearchExtension(bounded_append_proposer=RaisingProposer()),
    )

    assert artifacts.decision.status == "no_stable_winner"
    assert artifacts.decision.candidate_id is None
    assert artifacts.extension is not None
    proposal = artifacts.extension.bounded_proposals[0]
    assert proposal.status == "proposal_error"
    assert proposal.error_label == label
    written = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "campaign").rglob("*") if path.is_file())
    assert "TOP_SECRET_" not in written
