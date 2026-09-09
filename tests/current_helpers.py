from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gepa import GEPAResult

from korvid_prompt_lab.contracts import (
    Campaign,
    Candidate,
    EvalCase,
    KorvidUpstreamServing,
)
from korvid_prompt_lab.scoring import EvaluationResult
from korvid_prompt_lab.upstream_contract import KORVID_REVISION

MODEL = "ollama/qwen3:0.6b"
SEED_TEXT = "Original operating prompt.\n"
IMPROVED_TEXT = "Use decisive source evidence before answering.\n"


def candidate(text: str = SEED_TEXT) -> Candidate:
    return Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "source-prompt",
            "components": {"tier_pack": text},
            "metadata": {"source": "test"},
        }
    )


def case(case_id: str, *, journey: bool = False) -> EvalCase:
    questions = (
        f'["Diagnose {case_id}.", "Show the evidence for {case_id}."]'
        if journey
        else f"Diagnose {case_id}."
    )
    return EvalCase(
        case_id=case_id,
        template_id="korvid-journey" if journey else "korvid-scenario",
        prompt=questions,
        models=(MODEL,),
    )


def serving() -> KorvidUpstreamServing:
    return KorvidUpstreamServing(
        backend="korvid_upstream",
        source_root="/reviewed/korvid",
        base_url="http://127.0.0.1:11434",
        korvid_revision=KORVID_REVISION,
        timeout_seconds=30.0,
    )


def campaign(
    cases: Sequence[EvalCase],
    *,
    splits: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
) -> Campaign:
    if splits is None:
        splits = ()
    return Campaign(
        campaign_id="source-search",
        repetitions=1,
        models=(MODEL,),
        cases=tuple(cases),
        serving=serving(),
        evaluation_splits=splits,
    )


def feedback(
    case_value: EvalCase,
    *,
    success: bool,
    calls: list[dict[str, Any]] | None = None,
    turns: list[dict[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "success": success,
        "reference": (
            f"journeys/{case_value.case_id}"
            if case_value.template_id == "korvid-journey"
            else f"scenarios/{case_value.case_id}"
        ),
        "source_path": (
            "src/korvid/evals/journeys/"
            if case_value.template_id == "korvid-journey"
            else "src/korvid/evals/scenarios/"
        )
        + f"{case_value.case_id}.yaml",
        "source_sha256": "a" * 64,
        "questions": (
            json.loads(case_value.prompt)
            if case_value.template_id == "korvid-journey"
            else [case_value.prompt]
        ),
        "turns": turns
        or [
            {
                "outcome": "success" if success else "failure",
                "failure_class": None if success else "misdiagnosis",
                "grade": {"diagnosis_success": success},
            }
        ],
        "calls": calls if calls is not None else [],
        "policy": {
            "source": "low-korvid-operator",
            "tier": "low",
            "prompt_pack": "low-korvid-operator",
            "tools": ["diagnose_pod"],
        },
    }
    value.update(extra or {})
    return value


class FakeRunner:
    def __init__(
        self,
        cases: Sequence[EvalCase],
        *,
        success: bool | Callable[[Candidate, EvalCase], bool] = False,
        execution_modes: Sequence[str] = ("scripted",),
        result_factory: Callable[[Candidate, EvalCase], EvaluationResult] | None = None,
        splits: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
    ) -> None:
        self.campaign = campaign(cases, splits=splits)
        self.success = success
        self.execution_modes = tuple(execution_modes)
        self.result_factory = result_factory
        self.calls: list[tuple[Candidate, EvalCase, Path]] = []

    def run(
        self,
        candidate_value: Candidate,
        case_value: EvalCase,
        run_dir: Path | str,
        *,
        repetition: int = 1,
        seed: int = 0,
    ) -> EvaluationResult:
        del repetition, seed
        target = Path(run_dir)
        self.calls.append((candidate_value, case_value, target))
        if self.result_factory is not None:
            return self.result_factory(candidate_value, case_value)
        passed = (
            self.success(candidate_value, case_value)
            if callable(self.success)
            else self.success
        )
        mode = self.execution_modes[
            min(len(self.calls) - 1, len(self.execution_modes) - 1)
        ]
        return EvaluationResult(
            success=passed,
            execution_mode=mode,
            candidate_fingerprint=candidate_value.fingerprint,
            feedback=feedback(case_value, success=passed),
            usage={"iterations": 1, "tool_calls": 0, "wall_time_seconds": 0.01},
        )


def fake_gepa_result(
    run_dir: str,
    candidates: list[dict[str, str]],
    *,
    scores: list[float] | None = None,
) -> GEPAResult[Any, Any]:
    values = scores or [0.5 + index / 10 for index in range(len(candidates))]
    parents: list[list[int | None]] = [[None]]
    parents.extend([[0] for _ in candidates[1:]])
    return GEPAResult(
        candidates=candidates,
        parents=parents,
        val_aggregate_scores=values,
        val_subscores=[{"validation": score} for score in values],
        per_val_instance_best_candidates={"validation": {len(candidates) - 1}},
        discovery_eval_counts=list(range(1, len(candidates) + 1)),
        total_metric_calls=max(1, len(candidates)),
        num_full_val_evals=len(candidates),
        run_dir=run_dir,
        seed=0,
    )
