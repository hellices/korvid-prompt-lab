"""GEPA adapter for original Korvid verdicts and bounded reflection evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch, ProposalFn

from .contracts import Candidate, EvalCase
from .runner import BridgeExecutionModeError, KorvidRunner
from .scoring import EvaluationResult


@dataclass(frozen=True, slots=True)
class SafeExecutionTrace:
    case_id: str
    template_id: str
    model: str
    execution_mode: str
    score: float
    feedback: Mapping[str, Any]


class KorvidGEPAAdapter:
    propose_new_texts: ProposalFn | None = None

    def __init__(
        self, runner: KorvidRunner, artifact_root: Path | str, *,
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
        return tuple(self._execution_modes)

    def evaluate(
        self, batch: list[EvalCase], candidate: dict[str, str], capture_traces: bool = False,
    ) -> EvaluationBatch[SafeExecutionTrace, EvaluationResult]:
        resolved = self._materialize_candidate(candidate)
        outputs: list[EvaluationResult] = []
        scores: list[float] = []
        traces: list[SafeExecutionTrace] = []
        for case in batch:
            directory = self.artifact_root / resolved.fingerprint / f"{self._execution_index:06d}-{_slugify(case.case_id)}"
            self._execution_index += 1
            result = self.runner.run(resolved, case, directory)
            if self._execution_modes and result.execution_mode not in self._execution_modes:
                raise BridgeExecutionModeError("optimization evidence must not mix live and scripted execution")
            if not self._execution_modes:
                self._execution_modes.append(result.execution_mode)
            if result.candidate_fingerprint != resolved.fingerprint:
                raise ValueError("evaluation candidate fingerprint mismatch")
            feedback = _require_upstream_feedback(result.feedback)
            if feedback["success"] is not result.success:
                raise ValueError("evaluation result differs from Korvid's original verdict")
            score = float(result.success)
            outputs.append(result)
            scores.append(score)
            if capture_traces:
                traces.append(SafeExecutionTrace(
                    case.case_id, case.template_id, case.models[0],
                    result.execution_mode, score, feedback,
                ))
        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=traces if capture_traces else None,
        )

    def make_reflective_dataset(
        self, candidate: dict[str, str],
        eval_batch: EvaluationBatch[SafeExecutionTrace, EvaluationResult],
        components_to_update: list[str],
    ) -> dict[str, Sequence[Mapping[str, Any]]]:
        self._materialize_candidate(candidate)
        if components_to_update != ["tier_pack"]:
            raise ValueError("reflection can update only the original tier_pack")
        if eval_batch.trajectories is None:
            raise ValueError("capture_traces=True is required for reflection")
        if len(eval_batch.trajectories) != len(eval_batch.outputs) or len(eval_batch.outputs) != len(eval_batch.scores):
            raise ValueError("reflection traces, outputs and scores must align")
        return {"tier_pack": [self._trace_to_record(trace) for trace in eval_batch.trajectories]}

    def _materialize_candidate(self, components: Mapping[str, str]) -> Candidate:
        return Candidate.from_mapping({
            "schema_version": 1, "candidate_id": self.candidate_id,
            "components": dict(components), "metadata": self.candidate_metadata,
        })

    @staticmethod
    def _trace_to_record(trace: SafeExecutionTrace) -> Mapping[str, Any]:
        feedback = trace.feedback
        return {
            "Inputs": {
                "case_id": trace.case_id, "template_id": trace.template_id, "model": trace.model,
                "reference": feedback["reference"], "source_sha256": feedback["source_sha256"],
                "questions": feedback["questions"],
                "runtime_policy": feedback.get("policy", {}),
            },
            "Generated Outputs": {
                "turns": feedback["turns"], "calls": feedback["calls"],
                "execution_mode": trace.execution_mode,
            },
            "Feedback": {
                "success": feedback["success"],
                "guidance": (
                    "Improve the original Korvid operating prompt (tier_pack). Return its complete "
                    "replacement as plain text, not JSON rules. Keep safety, tools and original "
                    "evaluation conditions unchanged. Do not embed case-specific names or answers. "
                    "The original Korvid verdict is authoritative; preserve whole journeys."
                ),
            },
            "score": trace.score,
        }


def _require_upstream_feedback(feedback: Mapping[str, Any]) -> Mapping[str, Any]:
    success = feedback.get("success")
    reference = feedback.get("reference")
    source_sha256 = feedback.get("source_sha256")
    questions, turns, calls = feedback.get("questions"), feedback.get("turns"), feedback.get("calls")
    if type(success) is not bool:
        raise ValueError("upstream feedback success must be boolean")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("upstream feedback reference must be non-blank")
    if not isinstance(source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise ValueError("upstream feedback source_sha256 must be a SHA-256 digest")
    if not isinstance(questions, list) or not questions or any(
        not isinstance(question, str) or not question.strip() for question in questions
    ):
        raise ValueError("upstream feedback must contain the original questions")
    if not isinstance(turns, list) or not turns or any(
        not isinstance(turn, Mapping) or turn.get("outcome") not in {"success", "failure", "error"}
        or not isinstance(turn.get("grade"), Mapping) for turn in turns
    ):
        raise ValueError("upstream feedback must contain original graded outcomes")
    if not isinstance(calls, list) or any(not _valid_upstream_call(call) for call in calls):
        raise ValueError("upstream feedback calls must contain compact tool calls")
    if len(turns) > 64 or len(calls) > 256:
        raise ValueError("upstream feedback exceeds the bounded reflection contract")
    return {
        "success": success, "reference": reference, "source_sha256": source_sha256,
        "questions": list(questions), "turns": _compact_upstream_turns(turns),
        "calls": _compact_upstream_calls(calls), "policy": _compact_policy(feedback.get("policy")),
    }


def _compact_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    policy: dict[str, Any] = {}
    for name in ("source", "prompt_pack", "tier"):
        item = value.get(name)
        if isinstance(item, str) and item.strip():
            policy[name] = item[:500]
    tools = value.get("tools")
    if isinstance(tools, list):
        policy["tools"] = [item[:200] for item in tools[:64] if isinstance(item, str) and item.strip()]
    return policy


def _compact_upstream_turns(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for raw in value:
        turn: dict[str, Any] = {}
        for name in ("outcome", "failure_class", "error_label"):
            item = raw.get(name)
            if name in raw and (item is None or isinstance(item, str)):
                turn[name] = item
        turn["grade"] = _safe_grade_mapping(raw["grade"])
        turns.append(turn)
    return turns


def _compact_upstream_calls(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for raw in value:
        call: dict[str, Any] = {"name": raw["name"]}
        arguments_valid = raw.get("arguments_valid", True)
        if "arguments_valid" in raw:
            call["arguments_valid"] = arguments_valid
        if arguments_valid:
            call["arguments"] = _safe_arguments(raw["arguments"])
        else:
            call["arguments_raw"] = raw["arguments_raw"]
            call["error_label"] = raw["error_label"]
        calls.append(call)
    return calls


def _valid_upstream_call(value: Any) -> bool:
    if not isinstance(value, Mapping) or not isinstance(value.get("name"), str) or not value["name"].strip():
        return False
    valid = value.get("arguments_valid", True)
    if type(valid) is not bool:
        return False
    if valid:
        return isinstance(value.get("arguments"), Mapping)
    return (
        "arguments" not in value and isinstance(value.get("arguments_raw"), str)
        and len(value["arguments_raw"]) <= 512
        and value.get("error_label") in {"arguments_not_string", "malformed_arguments", "non_object_arguments"}
    )


def _safe_grade_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"diagnosis_success", "evidence_fetched", "missing_mentions", "forbidden_mentions", "missing_evidence"}
    return {name: _safe_scalar_or_string_list(item) for name, item in value.items() if name in allowed}


def _safe_scalar_or_string_list(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, list):
        return [item[:500] for item in value[:64] if isinstance(item, str)]
    return None


def _safe_arguments(value: Mapping[str, Any]) -> dict[str, Any]:
    sensitive = ("token", "secret", "password", "credential", "kubeconfig")
    safe: dict[str, Any] = {}
    for name, item in value.items():
        key = str(name)
        if any(marker in key.lower() for marker in sensitive):
            safe[key] = "[redacted]"
        elif isinstance(item, (bool, int, float)) or item is None:
            safe[key] = item
        elif isinstance(item, str):
            safe[key] = item[:500]
    return safe


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "case"
