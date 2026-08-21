from __future__ import annotations

import hashlib
import json
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
    runner: KorvidProcessRunner,
    seed_candidate: Candidate,
    train_cases: Sequence[EvalCase],
    validation_cases: Sequence[EvalCase],
    artifact_root: Path | str,
    max_metric_calls: int,
    seed: int = DEFAULT_OPTIMIZATION_SEED,
    reflection_lm: object | None = None,
    candidate_proposer: ProposalFn | None = None,
) -> OptimizationArtifacts:
    if isinstance(max_metric_calls, bool) or not isinstance(max_metric_calls, int) or max_metric_calls <= 0:
        raise ValueError("max_metric_calls must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if reflection_lm is not None and candidate_proposer is not None:
        raise ValueError("reflection_lm and candidate_proposer are mutually exclusive proposal sources")

    train_case_ids, validation_case_ids = _validate_case_splits(train_cases, validation_cases)

    custom_candidate_proposer: ProposalFn | None = candidate_proposer
    proposal_source = "candidate_proposer" if candidate_proposer is not None else "none"
    if custom_candidate_proposer is None and reflection_lm is not None:
        custom_candidate_proposer = DSPyInstructionProposer(reflection_lm)
        proposal_source = "reflection_lm"

    identity = _run_identity(
        runner=runner,
        seed_candidate=seed_candidate,
        train_case_ids=train_case_ids,
        validation_case_ids=validation_case_ids,
        max_metric_calls=max_metric_calls,
        seed=seed,
        proposal_source=proposal_source,
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

    adapter = KorvidGEPAAdapter(
        runner=runner,
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

    result: GEPAResult[Any, Any] = gepa.optimize(  # type: ignore[assignment]
        seed_candidate=seed_candidate.components,
        trainset=list(train_cases),
        valset=list(validation_cases),
        adapter=cast(Any, adapter),
        custom_candidate_proposer=custom_candidate_proposer,
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir),
        seed=seed,
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
    best_candidate_path = invocation_dir / "best-candidate.yaml"
    summary_path = invocation_dir / "optimization-summary.json"

    _write_candidate_yaml(best_candidate_path, best_candidate)
    write_json_artifact(
        summary_path,
        {
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
        run_id=run_id,
        invocation_dir=invocation_dir,
    )


def _run_identity(
    *,
    runner: KorvidProcessRunner,
    seed_candidate: Candidate,
    train_case_ids: Sequence[str],
    validation_case_ids: Sequence[str],
    max_metric_calls: int,
    seed: int,
    proposal_source: str,
) -> dict[str, Any]:
    """Describe everything that makes one optimization invocation reproducible.

    Two invocations that share this identity would search the same space, so they must
    never share a directory: GEPA resumes any state it finds in ``run_dir``.
    """
    return {
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
