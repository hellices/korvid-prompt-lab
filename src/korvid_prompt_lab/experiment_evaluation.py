"""Aggregate Korvid's original verdicts without defining another oracle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from math import fsum
from pathlib import Path
from typing import Any

from .artifacts import write_json_artifact
from .contracts import Candidate, EvalCase
from .runner import KorvidRunner
from .scoring import (
    EvaluationScore,
    RepetitionOutcome,
    is_strictly_better,
    pass_hat_k,
    passes_qualification_gate,
)


@dataclass(frozen=True)
class Assessment:
    score: EvaluationScore
    model: str
    case_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    success_rate: float
    tool_calls: int
    diagnostic_calls: int | None
    wall_time_seconds: float
    execution_modes: tuple[str, ...]
    runs: tuple[dict[str, Any], ...]

    def compared_score(self, reference: Assessment) -> EvaluationScore:
        if (self.model, self.case_ids, self.seeds) != (
            reference.model, reference.case_ids, reference.seeds,
        ):
            raise ValueError("paired assessments require the same cases and seeds and model")
        return replace(
            self.score,
            core_regression=(
                self.success_rate < reference.success_rate
            ),
        )

    def improves(self, reference: Assessment) -> bool:
        return (
            self.score.fingerprint != reference.score.fingerprint
            and is_strictly_better(self.compared_score(reference), reference.score)
        )

    def improves_success(self, reference: Assessment) -> bool:
        return self.improves(reference) and self.success_rate > reference.success_rate

    def qualifies(self, reference: Assessment) -> bool:
        return (
            self.execution_modes == reference.execution_modes == ("live",)
            and passes_qualification_gate(self.compared_score(reference))
        )

    def to_mapping(self) -> dict[str, Any]:
        return {**asdict(self), "verdict_source": "korvid"}


def assess(
    runner: KorvidRunner, candidate: Candidate, cases: Sequence[EvalCase], directory: Path, *, seed: int,
) -> Assessment:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("assessment seed must be a non-negative integer")
    if not cases or len(runner.campaign.models) != 1:
        raise ValueError("upstream assessments require original cases and one model")
    if any(case not in runner.campaign.cases for case in cases):
        raise ValueError("assessment cases must belong to the declared source campaign")
    directory.mkdir(parents=True, exist_ok=False)
    outcomes: list[RepetitionOutcome] = []
    records: list[dict[str, Any]] = []
    seeds = tuple(range(seed, seed + runner.campaign.repetitions))
    for case in cases:
        for repetition, run_seed in enumerate(seeds, 1):
            result = runner.run(
                candidate, case, directory / f"{case.case_id}-r{repetition:02d}",
                repetition=repetition, seed=run_seed,
            )
            if result.candidate_fingerprint != candidate.fingerprint:
                raise ValueError("assessment candidate fingerprint mismatch")
            passed = result.success
            raw_feedback = result.feedback
            if not isinstance(raw_feedback, Mapping) or type(raw_feedback.get("success")) is not bool:
                raise ValueError("assessment requires Korvid's original verdict and feedback")
            if passed != raw_feedback["success"]:
                raise ValueError("normalized grade differs from Korvid's original verdict")
            feedback = dict(raw_feedback)
            if "calls" in feedback:
                feedback["calls"] = [
                    {key: value for key, value in call.items() if key != "result"}
                    for call in feedback["calls"]
                ]
            outcomes.append(RepetitionOutcome(case.case_id, case.models[0], repetition, passed))
            records.append({
                "case_id": case.case_id, "repetition": repetition, "seed": run_seed,
                "execution_mode": result.execution_mode,
                "score": float(passed), "passed": passed,
                "hard_safety_failures": len(result.hard_failures),
                "tool_calls": result.usage["tool_calls"],
                "diagnostic_calls": result.usage.get("diagnostic_calls"),
                "wall_time_seconds": result.usage["wall_time_seconds"],
                "feedback": feedback,
            })
    modes = tuple(dict.fromkeys(record["execution_mode"] for record in records))
    if len(modes) != 1:
        raise ValueError("assessment must not mix live and scripted evidence")
    pass3, pass5 = pass_hat_k(outcomes, 3), pass_hat_k(outcomes, 5)
    if pass3 is None or pass5 is None:
        raise ValueError("native campaign assessment requires at least five repetitions")
    score = EvaluationScore(
        fingerprint=candidate.fingerprint,
        aggregate=fsum(record["score"] for record in records) / len(records),
        pass_at_3=pass3, pass_at_5=pass5,
        hard_safety_failures=sum(record["hard_safety_failures"] for record in records),
        core_regression=False,
    )
    assessment = Assessment(
        score=score, model=runner.campaign.models[0],
        case_ids=tuple(case.case_id for case in cases), seeds=seeds,
        success_rate=fsum(record["score"] for record in records) / len(records),
        tool_calls=sum(record["tool_calls"] for record in records),
        diagnostic_calls=(
            sum(record["diagnostic_calls"] for record in records)
            if all(type(record["diagnostic_calls"]) is int for record in records) else None
        ),
        wall_time_seconds=fsum(record["wall_time_seconds"] for record in records),
        execution_modes=modes, runs=tuple(records),
    )
    write_json_artifact(directory / "assessment.json", assessment.to_mapping())
    return assessment
