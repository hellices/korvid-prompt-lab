from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import gepa
import yaml  # type: ignore[import-untyped]
from gepa import GEPAResult
from gepa.core.adapter import ProposalFn

from .adapter import KorvidGEPAAdapter
from .artifacts import write_json_artifact
from .contracts import Candidate, EvalCase
from .reflection import DSPyInstructionProposer
from .runner import KorvidProcessRunner


@dataclass(frozen=True, slots=True)
class OptimizationArtifacts:
    result: GEPAResult[Any, Any]
    best_candidate: Candidate
    best_candidate_path: Path
    summary_path: Path


def optimize_campaign(
    *,
    runner: KorvidProcessRunner,
    seed_candidate: Candidate,
    train_cases: Sequence[EvalCase],
    validation_cases: Sequence[EvalCase],
    artifact_root: Path | str,
    max_metric_calls: int,
    reflection_lm: object | None = None,
    candidate_proposer: ProposalFn | None = None,
) -> OptimizationArtifacts:
    if isinstance(max_metric_calls, bool) or not isinstance(max_metric_calls, int) or max_metric_calls <= 0:
        raise ValueError("max_metric_calls must be a positive integer")
    if reflection_lm is not None and candidate_proposer is not None:
        raise ValueError("reflection_lm and candidate_proposer are mutually exclusive proposal sources")

    train_case_ids, validation_case_ids = _validate_case_splits(train_cases, validation_cases)

    artifact_root_path = Path(artifact_root)
    adapter = KorvidGEPAAdapter(
        runner=runner,
        artifact_root=artifact_root_path / "runs",
        candidate_id=seed_candidate.candidate_id,
        candidate_metadata=seed_candidate.metadata,
    )
    custom_candidate_proposer: ProposalFn | None = candidate_proposer
    if custom_candidate_proposer is None and reflection_lm is not None:
        custom_candidate_proposer = DSPyInstructionProposer(reflection_lm)
    run_dir = artifact_root_path / "gepa"

    result: GEPAResult[Any, Any] = gepa.optimize(  # type: ignore[assignment]
        seed_candidate=seed_candidate.components,
        trainset=list(train_cases),
        valset=list(validation_cases),
        adapter=cast(Any, adapter),
        custom_candidate_proposer=custom_candidate_proposer,
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir),
    )
    best_candidate_components = cast(dict[str, str], result.best_candidate)

    best_candidate = Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": seed_candidate.candidate_id,
            "components": dict(best_candidate_components),
            "metadata": seed_candidate.metadata,
        }
    )
    best_candidate_path = artifact_root_path / "best-candidate.yaml"
    summary_path = artifact_root_path / "optimization-summary.json"

    _write_candidate_yaml(best_candidate_path, best_candidate)
    write_json_artifact(
        summary_path,
        {
            "best_idx": result.best_idx,
            "best_validation_score": result.val_aggregate_scores[result.best_idx],
            "best_candidate_fingerprint": best_candidate.fingerprint,
            "seed_candidate_fingerprint": seed_candidate.fingerprint,
            "best_candidate_differs_from_seed": best_candidate.fingerprint != seed_candidate.fingerprint,
            "train_case_ids": train_case_ids,
            "validation_case_ids": validation_case_ids,
            "num_candidates": result.num_candidates,
            "total_metric_calls": result.total_metric_calls,
            "num_full_val_evals": result.num_full_val_evals,
            "run_dir": result.run_dir,
        },
    )

    return OptimizationArtifacts(
        result=result,
        best_candidate=best_candidate,
        best_candidate_path=best_candidate_path,
        summary_path=summary_path,
    )


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
