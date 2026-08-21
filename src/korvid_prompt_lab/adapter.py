from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch

from .contracts import Candidate, EvalCase
from .runner import KorvidProcessRunner
from .scoring import BridgeResult, score_result


@dataclass(frozen=True, slots=True)
class SafeExecutionTrace:
    case_id: str
    template_id: str
    model: str
    final_answer: str
    checkpoint_names: tuple[str, ...]
    tool_call_count: int
    outcome: str
    missing_checkpoints: tuple[str, ...]
    hard_failures: tuple[str, ...]
    score: float


class KorvidGEPAAdapter:
    def __init__(
        self,
        runner: KorvidProcessRunner,
        artifact_root: Path | str,
        *,
        candidate_id: str = "gepa-candidate",
        candidate_metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.runner = runner
        self.artifact_root = Path(artifact_root)
        self.candidate_id = candidate_id
        self.candidate_metadata = dict(candidate_metadata or {})
        self._execution_index = 0

    def evaluate(
        self,
        batch: list[EvalCase],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[SafeExecutionTrace, BridgeResult]:
        resolved_candidate = self._materialize_candidate(candidate)
        outputs: list[BridgeResult] = []
        scores: list[float] = []
        traces: list[SafeExecutionTrace] = []
        any_unsafe = False

        for case in batch:
            run_dir = self._next_run_dir(resolved_candidate.fingerprint, case)
            result = self.runner.run(resolved_candidate, case, run_dir)
            scored = score_result(result)
            outputs.append(result)
            scores.append(scored.score)
            any_unsafe = any_unsafe or scored.unsafe
            if capture_traces:
                traces.append(self._build_trace(case, result, score=scored.score, unsafe=scored.unsafe))

        if any_unsafe:
            scores = [0.0 for _ in scores]
            if capture_traces:
                traces = [replace(trace, score=0.0) for trace in traces]

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=traces if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[SafeExecutionTrace, BridgeResult],
        components_to_update: list[str],
    ) -> dict[str, Sequence[Mapping[str, Any]]]:
        validated_candidate = self._materialize_candidate(candidate).components
        if eval_batch.trajectories is None:
            raise ValueError("evaluate(..., capture_traces=True) is required to build reflection records")
        if len(eval_batch.trajectories) != len(eval_batch.outputs) or len(eval_batch.trajectories) != len(eval_batch.scores):
            raise ValueError("evaluation batch trajectories must align with outputs and scores")

        reflective_dataset: dict[str, Sequence[Mapping[str, Any]]] = {}
        for component_name in components_to_update:
            if component_name not in validated_candidate:
                raise ValueError(f"unknown candidate component: {component_name}")
            reflective_dataset[component_name] = [self._trace_to_record(trace) for trace in eval_batch.trajectories]
        return reflective_dataset

    def _materialize_candidate(self, candidate: Mapping[str, str]) -> Candidate:
        return Candidate.from_mapping(
            {
                "schema_version": 1,
                "candidate_id": self.candidate_id,
                "components": dict(candidate),
                "metadata": dict(self.candidate_metadata),
            }
        )

    def _next_run_dir(self, fingerprint: str, case: EvalCase) -> Path:
        run_dir = self.artifact_root / fingerprint / f"{self._execution_index:06d}-{_slugify(case.case_id)}"
        self._execution_index += 1
        return run_dir

    def _build_trace(self, case: EvalCase, result: BridgeResult, *, score: float, unsafe: bool) -> SafeExecutionTrace:
        checkpoint_names = _coerce_string_sequence(result.journal.get("checkpoints"))
        return SafeExecutionTrace(
            case_id=case.case_id,
            template_id=case.template_id,
            model=case.models[0],
            final_answer=result.answer,
            checkpoint_names=checkpoint_names,
            tool_call_count=_count_tool_calls(result.journal.get("tool_calls")),
            outcome="unsafe" if unsafe else result.status,
            missing_checkpoints=_missing_checkpoints(checkpoint_names),
            hard_failures=result.grade.hard_failures if result.grade is not None else (),
            score=score,
        )

    def _trace_to_record(self, trace: SafeExecutionTrace) -> Mapping[str, Any]:
        return {
            "Inputs": {
                "case_id": trace.case_id,
                "template_id": trace.template_id,
                "model": trace.model,
            },
            "Generated Outputs": {
                "answer": trace.final_answer,
                "checkpoint_names": list(trace.checkpoint_names),
                "tool_call_count": trace.tool_call_count,
                "outcome": trace.outcome,
            },
            "Feedback": _build_feedback(trace),
            "score": trace.score,
        }


def _build_feedback(trace: SafeExecutionTrace) -> str:
    parts = [f"Outcome: {trace.outcome}."]
    if trace.missing_checkpoints:
        parts.append(f"Missing checkpoints: {', '.join(trace.missing_checkpoints)}.")
    if trace.hard_failures:
        parts.append(f"Hard failures: {', '.join(trace.hard_failures)}.")
    if not trace.missing_checkpoints and not trace.hard_failures:
        parts.append("No missing checkpoints or hard failures.")
    return " ".join(parts)


def _coerce_string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item.strip())


def _count_tool_calls(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, Mapping):
        return sum(item for item in value.values() if isinstance(item, int) and item >= 0)
    return 0


def _missing_checkpoints(checkpoint_names: Sequence[str]) -> tuple[str, ...]:
    observed = set(checkpoint_names)
    if "dispatch" in observed and "verify" not in observed:
        return ("verify",)
    return ()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "case"
