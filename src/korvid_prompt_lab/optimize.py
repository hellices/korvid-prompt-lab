from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import gepa
import yaml  # type: ignore[import-untyped]
from gepa import GEPAResult
from gepa.core.adapter import ProposalFn

from .adapter import KorvidGEPAAdapter
from .artifacts import write_json_artifact
from .contracts import (
    GEPA_REFLECTION_MINIBATCH_SIZE,
    Campaign,
    Candidate,
    EvalCase,
)
from .experiment_budget import BudgetExhausted, ExperimentBudget
from .reflection import (
    AuditedProposalSource,
    DSPyInstructionProposer,
    ProposalProviderError,
    ProposalRejected,
)
from .runner import KorvidRunner
from .scoring import EvaluationResult
from .upstream_contract import tier_pack_from_candidate

__all__ = [
    "OptimizationArtifacts",
    "ProposalProviderError",
    "ProposalRejected",
    "optimize_campaign",
]

DEFAULT_OPTIMIZATION_SEED = 0
RUN_IDENTITY_SCHEMA_VERSION = 1
_RUN_ID_LENGTH = 16
_GEPA_STATE_FILENAME = "gepa_state.bin"


@dataclass(frozen=True, slots=True)
class OptimizationArtifacts:
    result: GEPAResult[Any, Any]
    best_candidate: Candidate
    best_candidate_path: Path
    summary_path: Path
    run_id: str
    invocation_dir: Path


def optimize_campaign(
    *,
    runner: KorvidRunner,
    seed_candidate: Candidate,
    train_cases: Sequence[EvalCase],
    validation_cases: Sequence[EvalCase],
    artifact_root: Path | str,
    max_metric_calls: int,
    seed: int = DEFAULT_OPTIMIZATION_SEED,
    reflection_lm: object | None = None,
    candidate_proposer: ProposalFn | None = None,
    budget: ExperimentBudget,
    run_context: Mapping[str, Any] | None = None,
) -> OptimizationArtifacts:
    if isinstance(max_metric_calls, bool) or not isinstance(max_metric_calls, int) or max_metric_calls <= 0:
        raise ValueError("max_metric_calls must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (reflection_lm is None) == (candidate_proposer is None):
        raise ValueError("provide exactly one reflection_lm or candidate_proposer")

    train_case_ids, validation_case_ids = _validate_case_splits(train_cases, validation_cases)
    tier_pack_from_candidate(seed_candidate)
    _validate_upstream_case_splits(runner.campaign, train_cases=train_cases, validation_cases=validation_cases)

    custom_candidate_proposer: ProposalFn | None = candidate_proposer
    proposal_source = "candidate_proposer" if candidate_proposer is not None else "none"
    if custom_candidate_proposer is None and reflection_lm is not None:
        custom_candidate_proposer = DSPyInstructionProposer(reflection_lm, budget=budget)
        proposal_source = "reflection_lm"

    identity = _run_identity(
        runner=runner,
        seed_candidate=seed_candidate,
        train_case_ids=train_case_ids,
        validation_case_ids=validation_case_ids,
        max_metric_calls=max_metric_calls,
        seed=seed,
        proposal_source=proposal_source,
        run_context=run_context,
    )
    run_id = _run_id(identity)
    artifact_root_path = Path(artifact_root)
    invocation_dir = artifact_root_path / "invocations" / run_id
    if invocation_dir.exists():
        raise ValueError(
            f"optimization invocation directory already exists: {invocation_dir}; "
            "korvid-prompt-lab never resumes a GEPA run, so change the run identity "
            "(seed, case splits, budget, or seed candidate) or use a fresh artifact root"
        )
    invocation_dir.mkdir(parents=True)
    write_json_artifact(invocation_dir / "run-identity.json", {**identity, "run_id": run_id})

    if custom_candidate_proposer is None:
        raise ValueError("optimization requires a candidate proposal source")
    audited_proposer = AuditedProposalSource(
            proposer=custom_candidate_proposer,
            source=proposal_source,
            invocation_dir=invocation_dir,
            seed_candidate=seed_candidate,
            budget=budget,
        )
    runner_for_adapter = _runner_with_budget(runner, budget)
    adapter = KorvidGEPAAdapter(
        runner=runner_for_adapter,
        artifact_root=invocation_dir / "runs",
        candidate_id=seed_candidate.candidate_id,
        candidate_metadata=seed_candidate.metadata,
    )
    run_dir = invocation_dir / "gepa"
    state_path = run_dir / _GEPA_STATE_FILENAME
    if state_path.exists():
        raise ValueError(
            f"refusing to resume incompatible GEPA state: {state_path}; "
            "korvid-prompt-lab has no resume feature, so remove the directory or use a fresh artifact root"
        )

    stop_callback = _OptimizationStopper(
        budget=budget,
        audited_proposer=audited_proposer,
        runner=runner_for_adapter,
        next_iteration_evaluations=(
            2 * GEPA_REFLECTION_MINIBATCH_SIZE + len(validation_cases)
        ),
    )
    result: GEPAResult[Any, Any] = gepa.optimize(  # type: ignore[assignment]
        seed_candidate=seed_candidate.components,
        trainset=list(train_cases),
        valset=list(validation_cases),
        adapter=cast(Any, adapter),
        custom_candidate_proposer=audited_proposer,
        reflection_minibatch_size=GEPA_REFLECTION_MINIBATCH_SIZE,
        max_metric_calls=max_metric_calls,
        stop_callbacks=stop_callback,
        run_dir=str(run_dir),
        seed=seed,
        raise_on_exception=True,
    )
    pending_exception = stop_callback.pending_exception
    if pending_exception is not None and not isinstance(pending_exception, ProposalRejected):
        raise pending_exception
    budget.check()
    best_candidate_components = cast(dict[str, str], result.best_candidate)

    best_candidate = Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": seed_candidate.candidate_id,
            "components": dict(best_candidate_components),
            "metadata": seed_candidate.metadata,
        }
    )
    best_candidate_path = invocation_dir / "best-candidate.yaml"
    summary_path = invocation_dir / "optimization-summary.json"

    _write_candidate_yaml(best_candidate_path, best_candidate)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "run_identity": identity,
        "invocation_dir": str(invocation_dir),
        "best_idx": result.best_idx,
        "best_validation_score": result.val_aggregate_scores[result.best_idx],
        "best_candidate_fingerprint": best_candidate.fingerprint,
        "seed_candidate_fingerprint": seed_candidate.fingerprint,
        "best_candidate_differs_from_seed": best_candidate.fingerprint != seed_candidate.fingerprint,
        "train_case_ids": train_case_ids,
        "validation_case_ids": validation_case_ids,
        # How this search's evidence was produced. A candidate tuned against
        # model-free scripted grades is not comparable to one tuned live.
        "execution_modes": list(adapter.execution_modes),
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "run_dir": result.run_dir,
        "stop_reason": "proposal_rejected" if isinstance(pending_exception, ProposalRejected) else "search_complete",
    }
    summary.update(audited_proposer.summary())
    write_json_artifact(summary_path, summary)

    return OptimizationArtifacts(
        result=result,
        best_candidate=best_candidate,
        best_candidate_path=best_candidate_path,
        summary_path=summary_path,
        run_id=run_id,
        invocation_dir=invocation_dir,
    )


def _run_identity(
    *,
    runner: KorvidRunner,
    seed_candidate: Candidate,
    train_case_ids: Sequence[str],
    validation_case_ids: Sequence[str],
    max_metric_calls: int,
    seed: int,
    proposal_source: str,
    run_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Describe everything that makes one optimization invocation reproducible.

    Two invocations that share this identity would search the same space, so they must
    never share a directory: GEPA resumes any state it finds in ``run_dir``.
    """
    identity = {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "campaign_id": runner.campaign.campaign_id,
        "candidate_id": seed_candidate.candidate_id,
        "seed_candidate_fingerprint": seed_candidate.fingerprint,
        "train_case_ids": list(train_case_ids),
        "validation_case_ids": list(validation_case_ids),
        "max_metric_calls": max_metric_calls,
        "seed": seed,
        "proposal_source": proposal_source,
    }
    normalized_context = _normalize_run_context(run_context)
    if normalized_context:
        identity["run_context"] = normalized_context
    return identity


def _run_id(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:_RUN_ID_LENGTH]


def _validate_case_splits(
    train_cases: Sequence[EvalCase], validation_cases: Sequence[EvalCase]
) -> tuple[list[str], list[str]]:
    train_case_ids = list(dict.fromkeys(case.case_id for case in train_cases))
    validation_case_ids = list(dict.fromkeys(case.case_id for case in validation_cases))
    if not train_case_ids:
        raise ValueError("train_cases must not be empty")
    if not validation_case_ids:
        raise ValueError("validation_cases must not be empty")
    overlap = sorted(set(train_case_ids) & set(validation_case_ids))
    if overlap:
        raise ValueError(f"train and validation case sets must be disjoint: {', '.join(overlap)}")
    return train_case_ids, validation_case_ids


def _validate_upstream_case_splits(
    campaign: Campaign,
    *,
    train_cases: Sequence[EvalCase],
    validation_cases: Sequence[EvalCase],
) -> None:
    declared = dict(campaign.evaluation_splits)
    if not declared.get("train") or not declared.get("validation"):
        raise ValueError(
            "upstream optimization requires declared train and validation source splits"
        )
    source_catalog = {
        _upstream_case_reference(case): case for case in campaign.cases
    }
    for case in (*train_cases, *validation_cases):
        if source_catalog.get(_upstream_case_reference(case)) != case:
            raise ValueError(
                "upstream optimization case differs from its campaign source identity"
            )
    selected = {
        "train": {case.case_id for case in train_cases},
        "validation": {case.case_id for case in validation_cases},
    }
    for label in ("train", "validation"):
        unexpected = sorted(selected[label] - set(declared[label]))
        if unexpected:
            raise ValueError(
                f"upstream optimization requires cases from the declared {label} "
                f"split, never holdout: {', '.join(unexpected)}"
            )


def _upstream_case_reference(case: EvalCase) -> str:
    if case.template_id == "korvid-scenario":
        return f"scenarios/{case.case_id}"
    if case.template_id == "korvid-journey":
        return f"journeys/{case.case_id}"
    raise ValueError(
        "upstream optimization cases must be original Korvid scenarios or journeys"
    )


def _write_candidate_yaml(path: Path, candidate: Candidate) -> Path:
    payload = {
        "schema_version": candidate.schema_version,
        "candidate_id": candidate.candidate_id,
        "components": candidate.components,
        "metadata": candidate.metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(temp_path, path)
    return path


@runtime_checkable
class _EvaluationBudgetOwner(Protocol):
    @property
    def evaluation_budget(self) -> ExperimentBudget: ...


@runtime_checkable
class _SearchCallOwner(Protocol):
    @property
    def remaining_search_calls(self) -> int: ...


class _BudgetedRunner:
    def __init__(self, runner: KorvidRunner, budget: ExperimentBudget) -> None:
        self._runner = runner
        self.evaluation_budget = budget

    @property
    def campaign(self) -> Campaign:
        return self._runner.campaign

    def run(
        self,
        candidate: Candidate,
        case: EvalCase,
        run_dir: Path | str,
        *,
        repetition: int = 1,
        seed: int = 0,
    ) -> EvaluationResult:
        self.evaluation_budget.consume_evaluation()
        return self._runner.run(
            candidate,
            case,
            run_dir,
            repetition=repetition,
            seed=seed,
        )


def _runner_with_budget(
    runner: KorvidRunner,
    budget: ExperimentBudget,
) -> KorvidRunner:
    if isinstance(runner, _EvaluationBudgetOwner):
        if runner.evaluation_budget is not budget:
            raise ValueError("runner evaluation_budget must be the optimize_campaign budget")
        return runner
    return _BudgetedRunner(runner, budget)


class _OptimizationStopper:
    def __init__(
        self,
        *,
        budget: ExperimentBudget,
        audited_proposer: AuditedProposalSource,
        runner: KorvidRunner,
        next_iteration_evaluations: int,
    ) -> None:
        self.budget = budget
        self.audited_proposer = audited_proposer
        self.search_call_owner = (
            runner if isinstance(runner, _SearchCallOwner) else None
        )
        self.next_iteration_evaluations = next_iteration_evaluations
        self._budget_exception: BudgetExhausted | None = None

    @property
    def pending_exception(self) -> Exception | None:
        if self.audited_proposer.pending_exception is not None:
            return self.audited_proposer.pending_exception
        return self._budget_exception

    def __call__(self, gepa_state: object) -> bool:
        if self.pending_exception is not None:
            return True
        try:
            self.budget.check()
        except BudgetExhausted as exc:
            self._budget_exception = exc
            return True
        if self.budget.proposals >= self.budget.max_proposals:
            return True
        return (
            self.search_call_owner is not None
            and self.search_call_owner.remaining_search_calls
            < self.next_iteration_evaluations
        )


def _normalize_run_context(run_context: Mapping[str, Any] | None) -> dict[str, Any]:
    if run_context is None:
        return {}
    if not isinstance(run_context, Mapping):
        raise ValueError("run_context must be a mapping")  # noqa: TRY004 - preserve validation API
    unknown = sorted(set(run_context) - {"profile", "source", "manifest_fingerprint"})
    if unknown:
        raise ValueError(f"run_context has unknown field(s): {', '.join(unknown)}")
    selected = {
        name: run_context[name]
        for name in ("profile", "source", "manifest_fingerprint")
        if name in run_context
    }
    try:
        return cast(
            dict[str, Any],
            json.loads(json.dumps(selected, sort_keys=True, ensure_ascii=False)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("run_context reproducibility fields must be JSON-serializable") from exc
