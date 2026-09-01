from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from .artifacts import write_json_artifact
from .contracts import Candidate, EvalCase
from .runner import BridgeStatusError, KorvidRunner
from .scoring import BridgeResult, result_passed, score_result
from .stable_candidates import CandidateAxis, StructuredCandidate
from .stable_proposer import build_proposal_request
from .stable_ranking import (
    NormalizedRunRecord,
    QualificationCandidate,
    QualificationDecision,
    RankedCandidate,
    StageDecision,
    measure_candidate,
    qualify_winner,
    rank_screening,
    select_finalists,
)
from .stable_scenarios import ScenarioManifest

__all__ = [
    "BoundedProposalArtifact",
    "StableSearchArtifacts",
    "StableSearchConfig",
    "StableSearchExtension",
    "StableSearchExtensionArtifacts",
    "run_stable_search",
]

_SEARCH_SCHEMA_VERSION = 1
_SCOREABLE_STATUSES = frozenset({"completed", "model_failure"})
_JOURNAL_INT_FIELDS = frozenset(
    {
        "audit_record_count",
        "forbidden_mentions",
        "hard_failure_count",
        "journal_event_count",
        "malformed_tool_calls",
        "missing_evidence",
        "missing_mentions",
        "on_target_tool_calls",
        "resolvable_tool_calls",
        "tool_calls",
    }
)
_JOURNAL_BOOL_FIELDS = frozenset({"diagnosis_success", "evidence_fetched"})
_JOURNAL_FLOAT_FIELDS = frozenset({"citation_coverage", "citation_precision"})
_JOURNAL_STRING_LIST_FIELDS = frozenset(
    {
        "checkpoints",
        "hard_failure_labels",
        "missing_checkpoints",
    }
)
_JOURNAL_COUNT_MAP_FIELDS = frozenset({"checkpoint_counts"})
_USAGE_INT_FIELDS = frozenset(
    {
        "completion_tokens",
        "input_tokens",
        "iterations",
        "output_tokens",
        "prompt_tokens",
        "tool_calls",
    }
)
_USAGE_BOOL_FIELDS = frozenset({"tokens_estimated"})
_USAGE_FLOAT_FIELDS = frozenset({"wall_time_s", "wall_time_seconds"})


class SupportsBoundedAppendProposer(Protocol):
    reflection_lm: Any

    def safe_propose(self, request_or_context: Any, **kwargs: Any) -> str | None: ...


@dataclass(frozen=True, slots=True)
class StableSearchConfig:
    screening_repetitions: int = 1
    validation_repetitions: int = 3
    qualification_repetitions: int = 5
    screening_survivors: int = 3
    finalists: int = 2
    minimum_mean_delta: float = 0.10

    def __post_init__(self) -> None:
        screening_repetitions = _require_positive_int(self.screening_repetitions, "screening_repetitions")
        validation_repetitions = _require_positive_int(self.validation_repetitions, "validation_repetitions")
        qualification_repetitions = _require_positive_int(
            self.qualification_repetitions, "qualification_repetitions"
        )
        screening_survivors = _require_positive_int(self.screening_survivors, "screening_survivors")
        finalists = _require_positive_int(self.finalists, "finalists")
        minimum_mean_delta = _require_non_negative_float(self.minimum_mean_delta, "minimum_mean_delta")
        if screening_repetitions > validation_repetitions:
            raise ValueError("validation_repetitions must be at least screening_repetitions")
        if validation_repetitions > qualification_repetitions:
            raise ValueError("qualification_repetitions must be at least validation_repetitions")
        if finalists > screening_survivors:
            raise ValueError("finalists must not exceed screening_survivors")
        object.__setattr__(self, "minimum_mean_delta", minimum_mean_delta)


@dataclass(frozen=True, slots=True)
class StableSearchExtension:
    bounded_append_proposer: SupportsBoundedAppendProposer | None = None


@dataclass(frozen=True, slots=True)
class BoundedProposalArtifact:
    finalist_candidate_id: str
    failure_axis: str | None
    status: Literal["no_signal", "proposal_failed", "unchanged", "proposed"]
    proposed_candidate_id: str | None = None
    proposed_append: str | None = None


@dataclass(frozen=True, slots=True)
class StableSearchExtensionArtifacts:
    bounded_proposals: tuple[BoundedProposalArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class StableSearchArtifacts:
    artifact_root: Path
    config: StableSearchConfig
    screening: StageDecision
    validation: StageDecision
    qualification: tuple[QualificationCandidate, ...]
    decision: QualificationDecision
    extension: StableSearchExtensionArtifacts | None
    baseline_manifest_path: Path
    candidate_manifest_path: Path
    scenario_manifest_path: Path
    screening_summary_path: Path
    validation_summary_path: Path
    qualification_summary_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class _PairedStageArtifacts:
    decision: StageDecision
    summary_path: Path


@dataclass(frozen=True, slots=True)
class _QualificationStageArtifacts:
    candidates: tuple[QualificationCandidate, ...]
    summary_path: Path


class _ExecutionModeTracker:
    def __init__(self) -> None:
        self._modes: list[str] = []

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(self._modes)

    def observe(self, mode: str) -> None:
        if mode in self._modes:
            return
        if self._modes:
            raise ValueError(f"stable search evidence must not mix execution modes: {self._modes[0]} then {mode}")
        self._modes.append(mode)


StageName = Literal["stage-a", "stage-b"]


def run_stable_search(
    *,
    runner: KorvidRunner,
    baseline: Candidate,
    candidates: Sequence[StructuredCandidate],
    manifest: ScenarioManifest,
    artifact_root: Path | str,
    config: StableSearchConfig | None = None,
    extension: StableSearchExtension | None = None,
) -> StableSearchArtifacts:
    if config is None:
        config = StableSearchConfig()
    if extension is None:
        extension = StableSearchExtension()
    artifact_root_path = Path(artifact_root)
    _prepare_artifact_root(artifact_root_path)
    _validate_candidates(baseline, candidates)
    _require_runner_repetitions(runner, config.qualification_repetitions)

    case_index = _case_index(runner.campaign.cases)
    _validate_manifest_coverage(case_index, manifest)
    execution_modes = _ExecutionModeTracker()
    candidate_index = {candidate.candidate.candidate_id: candidate for candidate in candidates}

    baseline_manifest_path = write_json_artifact(
        artifact_root_path / "baseline-candidate.json", _candidate_payload(baseline)
    )
    candidate_manifest_path = write_json_artifact(
        artifact_root_path / "candidate-manifest.json",
        {
            "schema_version": _SEARCH_SCHEMA_VERSION,
            "candidates": [_structured_candidate_payload(candidate) for candidate in candidates],
        },
    )
    scenario_manifest_path = write_json_artifact(
        artifact_root_path / "scenario-manifest.json", asdict(manifest)
    )

    screening = _run_paired_stage(
        runner=runner,
        baseline=baseline,
        candidates=candidates,
        cases=_select_cases(case_index, manifest.train),
        split="train",
        repetitions=config.screening_repetitions,
        stage_dir=artifact_root_path / "stage-a",
        stage_name="stage-a",
        summary_name="screening-summary.json",
        ranker=rank_screening,
        limit=config.screening_survivors,
        execution_modes=execution_modes,
    )
    stage_b_candidates = tuple(candidate_index[item.candidate_id] for item in screening.decision.survivors)
    validation = _run_paired_stage(
        runner=runner,
        baseline=baseline,
        candidates=stage_b_candidates,
        cases=_select_cases(case_index, manifest.validation),
        split="validation",
        repetitions=config.validation_repetitions,
        stage_dir=artifact_root_path / "stage-b",
        stage_name="stage-b",
        summary_name="validation-summary.json",
        ranker=select_finalists,
        limit=config.finalists,
        execution_modes=execution_modes,
    )
    finalists = tuple(candidate_index[item.candidate_id] for item in validation.decision.survivors)
    extension_artifacts = _run_extension(
        extension=extension,
        validation=validation.decision,
        candidate_index=candidate_index,
    )
    qualification = _run_qualification_stage(
        runner=runner,
        baseline=baseline,
        finalists=finalists,
        validation_cases=_select_cases(case_index, manifest.validation),
        milestone_cases=_select_cases(case_index, manifest.milestone),
        repetitions=config.qualification_repetitions,
        stage_dir=artifact_root_path / "stage-c",
        execution_modes=execution_modes,
        minimum_mean_delta=config.minimum_mean_delta,
    )
    decision = qualify_winner(
        qualification.candidates,
        minimum_mean_delta=config.minimum_mean_delta,
        required_repetitions=config.qualification_repetitions,
    )
    qualification_summary_path = write_json_artifact(
        qualification.summary_path,
        {
            "schema_version": _SEARCH_SCHEMA_VERSION,
            "stage": "qualification",
            "repetitions": config.qualification_repetitions,
            "candidates": [asdict(candidate) for candidate in qualification.candidates],
            "decision": asdict(decision),
            "extension": asdict(extension_artifacts) if extension_artifacts is not None else None,
            "execution_modes": list(execution_modes.modes),
        },
    )

    summary_path = write_json_artifact(
        artifact_root_path / "stable-search-summary.json",
        {
            "schema_version": _SEARCH_SCHEMA_VERSION,
            "campaign_id": runner.campaign.campaign_id,
            "config": asdict(config),
            "decision": asdict(decision),
            "winner_append": _winner_append(candidate_index, decision),
            "extension": asdict(extension_artifacts) if extension_artifacts is not None else None,
            "execution_modes": list(execution_modes.modes),
            "artifacts": {
                "baseline_candidate": str(baseline_manifest_path.relative_to(artifact_root_path)),
                "candidate_manifest": str(candidate_manifest_path.relative_to(artifact_root_path)),
                "scenario_manifest": str(scenario_manifest_path.relative_to(artifact_root_path)),
                "screening_summary": str(screening.summary_path.relative_to(artifact_root_path)),
                "validation_summary": str(validation.summary_path.relative_to(artifact_root_path)),
                "qualification_summary": str(qualification_summary_path.relative_to(artifact_root_path)),
            },
        },
    )
    return StableSearchArtifacts(
        artifact_root=artifact_root_path,
        config=config,
        screening=screening.decision,
        validation=validation.decision,
        qualification=qualification.candidates,
        decision=decision,
        extension=extension_artifacts,
        baseline_manifest_path=baseline_manifest_path,
        candidate_manifest_path=candidate_manifest_path,
        scenario_manifest_path=scenario_manifest_path,
        screening_summary_path=screening.summary_path,
        validation_summary_path=validation.summary_path,
        qualification_summary_path=qualification_summary_path,
        summary_path=summary_path,
    )

def _run_paired_stage(
    *,
    runner: KorvidRunner,
    baseline: Candidate,
    candidates: Sequence[StructuredCandidate],
    cases: Sequence[EvalCase],
    split: str,
    repetitions: int,
    stage_dir: Path,
    stage_name: StageName,
    summary_name: str,
    ranker: Any,
    limit: int,
    execution_modes: _ExecutionModeTracker,
) -> _PairedStageArtifacts:
    stage_dir.mkdir(parents=True, exist_ok=False)
    baseline_measurement = measure_candidate(
        _run_candidate(
            runner=runner,
            candidate=baseline,
            cases=cases,
            split=split,
            repetitions=repetitions,
            stage_dir=stage_dir,
            execution_modes=execution_modes,
        )
    )
    measurements = tuple(
        measure_candidate(
            _run_candidate(
                runner=runner,
                candidate=structured.candidate,
                cases=cases,
                split=split,
                repetitions=repetitions,
                stage_dir=stage_dir,
                execution_modes=execution_modes,
            )
        )
        for structured in candidates
    )
    decision = ranker(
        baseline_measurement,
        measurements,
        limit=limit,
        stage="screening" if stage_name == "stage-a" else "validation",
    )
    summary_path = write_json_artifact(
        stage_dir / summary_name,
        {
            "schema_version": _SEARCH_SCHEMA_VERSION,
            "stage": decision.stage,
            "split": split,
            "repetitions": repetitions,
            "baseline": asdict(baseline_measurement),
            "candidates": [asdict(measurement) for measurement in measurements],
            "decision": asdict(decision),
            "execution_modes": list(execution_modes.modes),
        },
    )
    return _PairedStageArtifacts(decision=decision, summary_path=summary_path)



def _run_qualification_stage(
    *,
    runner: KorvidRunner,
    baseline: Candidate,
    finalists: Sequence[StructuredCandidate],
    validation_cases: Sequence[EvalCase],
    milestone_cases: Sequence[EvalCase],
    repetitions: int,
    stage_dir: Path,
    execution_modes: _ExecutionModeTracker,
    minimum_mean_delta: float,
) -> _QualificationStageArtifacts:
    stage_dir.mkdir(parents=True, exist_ok=False)
    if not finalists:
        return _QualificationStageArtifacts(candidates=(), summary_path=stage_dir / "qualification-summary.json")

    baseline_validation = measure_candidate(
        _run_candidate(
            runner=runner,
            candidate=baseline,
            cases=validation_cases,
            split="validation",
            repetitions=repetitions,
            stage_dir=stage_dir,
            execution_modes=execution_modes,
        )
    )
    baseline_milestone = measure_candidate(
        _run_candidate(
            runner=runner,
            candidate=baseline,
            cases=milestone_cases,
            split="milestone",
            repetitions=repetitions,
            stage_dir=stage_dir,
            execution_modes=execution_modes,
        )
    )
    candidates = tuple(
        QualificationCandidate(
            candidate_id=structured.candidate.candidate_id,
            baseline_validation=baseline_validation,
            candidate_validation=measure_candidate(
                _run_candidate(
                    runner=runner,
                    candidate=structured.candidate,
                    cases=validation_cases,
                    split="validation",
                    repetitions=repetitions,
                    stage_dir=stage_dir,
                    execution_modes=execution_modes,
                )
            ),
            baseline_milestone=baseline_milestone,
            candidate_milestone=measure_candidate(
                _run_candidate(
                    runner=runner,
                    candidate=structured.candidate,
                    cases=milestone_cases,
                    split="milestone",
                    repetitions=repetitions,
                    stage_dir=stage_dir,
                    execution_modes=execution_modes,
                )
            ),
        )
        for structured in finalists
    )
    # Ensure the stage payload is written with the exact qualification decision the
    # caller will return; the temporary call here only validates finalist evidence.
    qualify_winner(
        candidates,
        minimum_mean_delta=minimum_mean_delta,
        required_repetitions=repetitions,
    )
    return _QualificationStageArtifacts(candidates=candidates, summary_path=stage_dir / "qualification-summary.json")



def _run_extension(
    *,
    extension: StableSearchExtension,
    validation: StageDecision,
    candidate_index: Mapping[str, StructuredCandidate],
) -> StableSearchExtensionArtifacts | None:
    proposer = extension.bounded_append_proposer
    if proposer is None:
        return None

    proposals: list[BoundedProposalArtifact] = []
    for ranked in validation.survivors:
        proposals.append(_bounded_proposal_artifact(proposer, ranked, candidate_index[ranked.candidate_id]))
    return StableSearchExtensionArtifacts(bounded_proposals=tuple(proposals))


def _bounded_proposal_artifact(
    proposer: SupportsBoundedAppendProposer,
    ranked: RankedCandidate,
    structured: StructuredCandidate,
) -> BoundedProposalArtifact:
    failure_axis = _structured_failure_axis(ranked)
    if failure_axis is None:
        return BoundedProposalArtifact(
            finalist_candidate_id=ranked.candidate_id,
            failure_axis=None,
            status="no_signal",
        )

    finalist_append = structured.candidate.components.get("append")
    if finalist_append is None:
        return BoundedProposalArtifact(
            finalist_candidate_id=ranked.candidate_id,
            failure_axis=failure_axis.value,
            status="proposal_failed",
        )
    request = build_proposal_request(
        ranked.candidate,
        finalist_append=finalist_append,
        failure_axis=failure_axis,
    )
    proposed_append = proposer.safe_propose(request)
    if proposed_append is None:
        return BoundedProposalArtifact(
            finalist_candidate_id=ranked.candidate_id,
            failure_axis=failure_axis.value,
            status="proposal_failed",
        )
    if proposed_append == finalist_append:
        return BoundedProposalArtifact(
            finalist_candidate_id=ranked.candidate_id,
            failure_axis=failure_axis.value,
            status="unchanged",
            proposed_append=proposed_append,
        )
    return BoundedProposalArtifact(
        finalist_candidate_id=ranked.candidate_id,
        failure_axis=failure_axis.value,
        status="proposed",
        proposed_candidate_id=_proposal_candidate_id(ranked.candidate_id, proposed_append),
        proposed_append=proposed_append,
    )


def _structured_failure_axis(ranked: RankedCandidate) -> CandidateAxis | None:
    if ranked.verification_delta < 0.0:
        return CandidateAxis.EVIDENCE_FIRST
    if (
        ranked.candidate.unresolvable_tool_calls > ranked.baseline.unresolvable_tool_calls
        or ranked.candidate.malformed_tool_calls > ranked.baseline.malformed_tool_calls
    ):
        return CandidateAxis.ONE_TOOL_AT_A_TIME
    return None


def _proposal_candidate_id(candidate_id: str, proposed_append: str) -> str:
    digest = hashlib.sha256(proposed_append.encode("utf-8")).hexdigest()[:8]
    return f"{candidate_id}+proposal-{digest}"


def _run_candidate(
    *,
    runner: KorvidRunner,
    candidate: Candidate,
    cases: Sequence[EvalCase],
    split: str,
    repetitions: int,
    stage_dir: Path,
    execution_modes: _ExecutionModeTracker,
) -> tuple[NormalizedRunRecord, ...]:
    records: list[NormalizedRunRecord] = []
    for case in cases:
        for repetition in range(1, repetitions + 1):
            raw_run_dir = _run_artifact_dir(
                stage_dir=stage_dir,
                namespace="_runner",
                candidate_id=candidate.candidate_id,
                split=split,
                case_id=case.case_id,
                repetition=repetition,
            )
            normalized_run_dir = _run_artifact_dir(
                stage_dir=stage_dir,
                namespace="runs",
                candidate_id=candidate.candidate_id,
                split=split,
                case_id=case.case_id,
                repetition=repetition,
            )
            try:
                result = runner.run(
                    candidate,
                    case,
                    raw_run_dir,
                    repetition=repetition,
                    seed=repetition - 1,
                )
            finally:
                _remove_runner_artifacts(raw_run_dir)
            execution_modes.observe(result.execution_mode)
            record = _normalize_result(candidate, split, case.case_id, repetition, result)
            write_json_artifact(
                normalized_run_dir / "response.json",
                _normalized_run_artifact(
                    candidate=candidate,
                    case=case,
                    split=split,
                    repetition=repetition,
                    result=result,
                    record=record,
                ),
            )
            records.append(record)
            if _should_stop_early(record):
                return tuple(records)
    return tuple(records)



def _normalize_result(
    candidate: Candidate,
    split: str,
    case_id: str,
    repetition: int,
    result: BridgeResult,
) -> NormalizedRunRecord:
    if result.candidate_fingerprint != candidate.fingerprint:
        raise ValueError("runner result fingerprint does not match the requested candidate")
    if result.status == "completed":
        scored = score_result(result)
        grade = result.grade
        if grade is None:  # pragma: no cover - score_result rejects this first
            raise ValueError("completed results must carry a grade")
        score = scored.score
        verification = grade.verification
        passed = result_passed(scored)
        hard_safety_failures = len(grade.hard_failures)
    elif result.status == "model_failure":
        score = 0.0
        verification = 0.0
        passed = False
        hard_safety_failures = 0
    else:
        raise BridgeStatusError(f"runner returned systemic status: {result.status}")

    tool_calls = _journal_count(result.journal, "tool_calls")
    resolvable_tool_calls = _journal_count(result.journal, "resolvable_tool_calls")
    malformed_tool_calls = _journal_count(result.journal, "malformed_tool_calls")
    if resolvable_tool_calls > tool_calls:
        raise ValueError("journal resolvable_tool_calls must not exceed tool_calls")
    if malformed_tool_calls > tool_calls:
        raise ValueError("journal malformed_tool_calls must not exceed tool_calls")

    return NormalizedRunRecord(
        candidate_id=candidate.candidate_id,
        split=split,
        case_id=case_id,
        repetition=repetition,
        status=result.status,
        score=score,
        verification=verification,
        passed=passed,
        hard_safety_failures=hard_safety_failures,
        malformed_tool_calls=malformed_tool_calls,
        unresolvable_tool_calls=tool_calls - resolvable_tool_calls,
    )



def _should_stop_early(record: NormalizedRunRecord) -> bool:
    return record.hard_safety_failures > 0 or record.status not in _SCOREABLE_STATUSES



def _prepare_artifact_root(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"stable search artifact root already exists: {path}")
    path.mkdir(parents=True)



def _require_runner_repetitions(runner: KorvidRunner, repetitions: int) -> None:
    if runner.campaign.repetitions < repetitions:
        raise ValueError(
            "runner campaign repetitions must cover the qualification stage: "
            f"{runner.campaign.repetitions} < {repetitions}"
        )



def _validate_candidates(baseline: Candidate, candidates: Sequence[StructuredCandidate]) -> None:
    baseline_components = set(baseline.components)
    if baseline_components != {"system"}:
        raise ValueError("baseline components must be exactly {'system'}")
    if not candidates:
        raise ValueError("candidates must not be empty")
    candidate_ids: set[str] = set()
    baseline_system = baseline.components["system"]
    for structured in candidates:
        candidate = structured.candidate
        if candidate.candidate_id == baseline.candidate_id:
            raise ValueError("candidate_id must differ from the baseline candidate_id")
        if candidate.candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_id: {candidate.candidate_id}")
        candidate_ids.add(candidate.candidate_id)
        components = candidate.components
        if components.get("system") != baseline_system:
            raise ValueError("candidate system component must match the exact baseline system prompt")



def _case_index(cases: Sequence[EvalCase]) -> dict[str, EvalCase]:
    index: dict[str, EvalCase] = {}
    for case in cases:
        if case.case_id in index:
            raise ValueError(f"runner campaign declares duplicate case_id: {case.case_id}")
        index[case.case_id] = case
    return index



def _validate_manifest_coverage(case_index: Mapping[str, EvalCase], manifest: ScenarioManifest) -> None:
    requested = (*manifest.train, *manifest.validation, *manifest.milestone)
    if not requested:
        raise ValueError("scenario manifest must contain at least one case")
    missing = sorted(case_id for case_id in requested if case_id not in case_index)
    if missing:
        raise ValueError(f"runner campaign is missing manifest case(s): {', '.join(missing)}")



def _select_cases(case_index: Mapping[str, EvalCase], case_ids: Sequence[str]) -> tuple[EvalCase, ...]:
    if not case_ids:
        raise ValueError("stage case set must not be empty")
    return tuple(case_index[case_id] for case_id in case_ids)



def _journal_count(journal: Mapping[str, Any], field_name: str) -> int:
    value = journal.get(field_name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"journal {field_name} must be a non-negative integer")
    return value


def _remove_runner_artifacts(run_dir: Path) -> None:
    if run_dir.is_symlink():
        raise ValueError(f"stable search runner artifact directory must not be a symlink: {run_dir}")
    if not run_dir.exists():
        return
    shutil.rmtree(run_dir)
    _prune_empty_parents(run_dir)


def _run_artifact_dir(
    *,
    stage_dir: Path,
    namespace: str,
    candidate_id: str,
    split: str,
    case_id: str,
    repetition: int,
) -> Path:
    return (
        stage_dir
        / namespace
        / _safe_path_component(candidate_id, fallback="candidate")
        / _safe_path_component(split, fallback="split")
        / _safe_path_component(case_id, fallback="case")
        / f"r{repetition:02d}"
    )


def _safe_path_component(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{slug or fallback}-{digest}"


def _normalized_run_artifact(
    *,
    candidate: Candidate,
    case: EvalCase,
    split: str,
    repetition: int,
    result: BridgeResult,
    record: NormalizedRunRecord,
) -> dict[str, Any]:
    return {
        "schema_version": _SEARCH_SCHEMA_VERSION,
        "candidate_id": candidate.candidate_id,
        "candidate_fingerprint": candidate.fingerprint,
        "case_id": case.case_id,
        "template_id": case.template_id,
        "model": case.models[0],
        "split": split,
        "repetition": repetition,
        "status": result.status,
        "execution_mode": result.execution_mode,
        "answer": "",
        "error": None if result.status == "completed" else "model_failure",
        "normalized_record": asdict(record),
        "grade": _safe_grade_payload(result),
        "journal": _safe_journal_payload(result.journal),
        "usage": _safe_usage_payload(result.usage),
    }


def _safe_grade_payload(result: BridgeResult) -> dict[str, Any] | None:
    grade = result.grade
    if grade is None:
        return None
    return {
        "completion": float(grade.completion),
        "verification": float(grade.verification),
        "efficiency": float(grade.efficiency),
        "hard_failures": list(grade.hard_failures),
    }


def _safe_journal_payload(journal: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in _JOURNAL_INT_FIELDS:
        value = journal.get(field_name)
        if value is not None:
            payload[field_name] = _non_negative_int(value, f"journal {field_name}")
    for field_name in _JOURNAL_BOOL_FIELDS:
        value = journal.get(field_name)
        if value is not None:
            payload[field_name] = _require_bool(value, f"journal {field_name}")
    for field_name in _JOURNAL_FLOAT_FIELDS:
        value = journal.get(field_name)
        if value is not None:
            payload[field_name] = _unit_interval_float(value, f"journal {field_name}")
    for field_name in _JOURNAL_STRING_LIST_FIELDS:
        value = journal.get(field_name)
        if value is not None:
            payload[field_name] = _string_list(value, f"journal {field_name}")
    for field_name in _JOURNAL_COUNT_MAP_FIELDS:
        value = journal.get(field_name)
        if value is not None:
            payload[field_name] = _count_mapping(value, f"journal {field_name}")
    return payload


def _safe_usage_payload(usage: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in _USAGE_INT_FIELDS:
        value = usage.get(field_name)
        if value is not None:
            payload[field_name] = _non_negative_int(value, f"usage {field_name}")
    for field_name in _USAGE_BOOL_FIELDS:
        value = usage.get(field_name)
        if value is not None:
            payload[field_name] = _require_bool(value, f"usage {field_name}")
    for field_name in _USAGE_FLOAT_FIELDS:
        value = usage.get(field_name)
        if value is not None:
            payload[field_name] = _require_non_negative_float(value, f"usage {field_name}")
    return payload


def _count_mapping(value: Any, field_name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    payload: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        payload[key] = _non_negative_int(item, f"{field_name}[{key}]")
    return payload


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list of strings")
    payload: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field_name} must contain non-empty strings")
        payload.append(item)
    return payload


def _unit_interval_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a unit-interval number")
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be a unit-interval number")
    return normalized


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _prune_empty_parents(path: Path) -> None:
    for parent in path.parents:
        if parent.name in {"stage-a", "stage-b", "stage-c"}:
            return
        if not parent.exists():
            continue
        if parent.is_symlink():
            raise ValueError(f"stable search runner artifact parent must not be a symlink: {parent}")
        try:
            parent.rmdir()
        except OSError:
            return


def _require_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value



def _require_non_negative_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a non-negative number")
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be a non-negative number")
    return normalized



def _candidate_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "schema_version": candidate.schema_version,
        "candidate_id": candidate.candidate_id,
        "candidate_fingerprint": candidate.fingerprint,
        "components": candidate.components,
        "metadata": candidate.metadata,
    }



def _structured_candidate_payload(candidate: StructuredCandidate) -> dict[str, Any]:
    return {
        "axes": [axis.value for axis in candidate.axes],
        **_candidate_payload(candidate.candidate),
    }



def _winner_append(
    candidate_index: Mapping[str, StructuredCandidate], decision: QualificationDecision
) -> str | None:
    if decision.candidate_id is None:
        return None
    return candidate_index[decision.candidate_id].candidate.components.get("append")
