from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch, ProposalFn

from .contracts import Candidate, EvalCase
from .runner import BridgeExecutionModeError, KorvidProcessRunner
from .scoring import BridgeResult, ScoredResult, grade_quality, score_result


@dataclass(frozen=True, slots=True)
class SafeExecutionTrace:
    case_id: str
    template_id: str
    model: str
    execution_mode: str
    final_answer: str
    checkpoint_names: tuple[str, ...]
    tool_call_count: int
    outcome: str
    missing_checkpoints: tuple[str, ...]
    hard_failures: tuple[str, ...]
    score: float


def _search_score(scored: ScoredResult) -> float:
    if scored.result.status == "model_failure":
        return 0.0
    grade = scored.result.grade
    if grade is None:  # pragma: no cover - score_result rejects this first
        raise ValueError("completed results must carry a grade")
    quality = 0.75 + 0.25 * grade_quality(grade)
    return quality if not scored.unsafe else 2 ** (-len(grade.hard_failures)) * quality


class KorvidGEPAAdapter:
    # GEPA reads this attribute on every reflective mutation. Keeping it declared and
    # None keeps proposal responsibility outside the adapter (DSPy reflection or an
    # explicitly injected proposer) instead of silently aborting the mutation step.
    propose_new_texts: ProposalFn | None = None

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
        self._execution_modes: list[str] = []

    @property
    def execution_modes(self) -> tuple[str, ...]:
        """Every distinct way this optimization's evidence was produced, in first-seen order."""
        return tuple(self._execution_modes)

    def _record_execution_mode(self, result: BridgeResult) -> None:
        """Keep one optimization on one kind of evidence.

        GEPA only ever compares candidates against each other, so a run that switched
        from live grades to model-free scripted grades would rank a candidate against
        a different experiment. Mixing is systemic, not a score.
        """
        if result.execution_mode in self._execution_modes:
            return
        if self._execution_modes:
            raise BridgeExecutionModeError(
                "optimization evidence must not mix execution modes:"
                f" {self._execution_modes[0]} then {result.execution_mode}"
            )
        self._execution_modes.append(result.execution_mode)

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
        for case in batch:
            run_dir = self._next_run_dir(resolved_candidate.fingerprint, case)
            result = self.runner.run(resolved_candidate, case, run_dir)
            self._record_execution_mode(result)
            scored = score_result(result)
            outputs.append(result)
            search_score = _search_score(scored)
            scores.append(search_score)
            if capture_traces:
                traces.append(self._build_trace(case, result, score=search_score, unsafe=scored.unsafe))

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
        journal = result.journal
        checkpoint_names = _coerce_string_sequence(journal.get("checkpoints"))
        reported_tool_calls = result.usage.get("tool_calls", journal.get("tool_calls"))
        return SafeExecutionTrace(
            case_id=case.case_id,
            template_id=case.template_id,
            model=case.models[0],
            execution_mode=result.execution_mode,
            final_answer=result.answer,
            checkpoint_names=checkpoint_names,
            tool_call_count=_count_tool_calls(reported_tool_calls),
            outcome="unsafe" if unsafe else result.status,
            missing_checkpoints=_missing_checkpoints(journal, checkpoint_names),
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
                "execution_mode": trace.execution_mode,
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


def _missing_checkpoints(journal: Mapping[str, Any], checkpoint_names: Sequence[str]) -> tuple[str, ...]:
    """The bridge's own report wins; reflection must never invent or drop a gap.

    A bridge that reports `missing_checkpoints` is authoritative even when the
    list is empty, because the grader that produced it knows which checkpoints
    the operation actually required. The `dispatch`/`verify` inference below is
    only a fallback for a bridge that reports no gaps at all.
    """
    if "missing_checkpoints" in journal:
        return _coerce_string_sequence(journal.get("missing_checkpoints"))
    observed = set(checkpoint_names)
    if "dispatch" in observed and "verify" not in observed:
        return ("verify",)
    return ()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "case"
