"""Tests for safe campaign evidence ingestion and artifact rendering (Task 4)."""

from __future__ import annotations

import json
import shutil
import sys
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab import campaign_artifacts
from korvid_prompt_lab.campaign_artifacts import (
    load_round_outcome,
    render_campaign_summary,
    write_campaign_artifacts,
    write_campaign_state,
)
from korvid_prompt_lab.campaigns import (
    ActionKind,
    CampaignAction,
    CampaignScore,
    CampaignState,
    CampaignStatus,
    ModelIdentity,
    ModelTier,
    OptimizationCampaign,
    SearchStage,
    max_search_metric_calls,
    state_hash,
)
from korvid_prompt_lab.contracts import Candidate

DIGEST_A = "sha256:" + "a" * 64
NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _control() -> OptimizationCampaign:
    return OptimizationCampaign(
        schema_version=1,
        campaign_id="test-campaign",
        evaluation_campaign="test-campaign",
        initial_candidate="seed.yaml",
        train_case_ids=("case-a", "case-b"),
        validation_case_ids=("case-c",),
        milestone_case_ids=("case-d",),
        stages=(
            SearchStage(name="explore", metric_calls=12, seeds=(0, 1, 2)),
            SearchStage(name="refine", metric_calls=24, seeds=(3, 4)),
        ),
        model_tiers=(ModelTier(name="small", model="qwen3:0.6b", digest=DIGEST_A),),
        total_metric_call_limit=240,
        wall_clock_limit_seconds=21600,
        infrastructure_retry_limit=3,
        stagnation_attempt_limit=30,
        confirmation_runs=1,
    )


SEED_FINGERPRINT = "7" * 64


def _state(
    *,
    champion_fingerprint: str = SEED_FINGERPRINT,
    stage_index: int = 0,
    seed_index: int = 0,
    metric_calls_used: int = 0,
) -> CampaignState:
    return CampaignState(
        schema_version=1,
        campaign_id="test-campaign",
        prompt_lab_revision="abc123",
        korvid_revision="def456",
        status=CampaignStatus.RUNNING,
        tier_index=0,
        stage_index=stage_index,
        seed_index=seed_index,
        champion_fingerprint=champion_fingerprint,
        seed_candidate_fingerprint=SEED_FINGERPRINT,
        champion_score=CampaignScore(
            fingerprint=champion_fingerprint,
            aggregate=0.0,
            hard_safety_failures=0,
            core_regression=False,
            systemic_failures=0,
            pass_at_3=0.0,
            pass_at_5=0.0,
        ),
        model_identity=ModelIdentity(
            name="small", model="qwen3:0.6b", digest=DIGEST_A,
        ),
        metric_calls_used=metric_calls_used,
        elapsed_seconds=0.0,
        stagnation_attempts=0,
        retries_used=0,
        started_at="2026-01-15T12:00:00+00:00",
    )


def _search_action(
    state: CampaignState | None = None,
    control: OptimizationCampaign | None = None,
    action_id: str = "action-1",
) -> CampaignAction:
    return CampaignAction(
        action_id=action_id,
        kind=ActionKind.SEARCH,
        expected_state_hash="sha256:" + "0" * 64,
        stage_index=0,
        seed_index=0,
        tier_index=0,
        metric_calls=12,
    )


def _candidate_mapping(candidate_id: str = "cand-1") -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "components": {"system": "system prompt"},
        "metadata": {},
    }


def _write_search_evidence(
    root: Path,
    *,
    campaign_action_id: str = "action-1",
    candidate_id: str = "cand-1",
    candidate_fingerprint: str | None = None,
    best_candidate_mapping: dict[str, object] | None = None,
    best_candidate_fingerprint: str | None = None,
    evaluated_case_ids: tuple[str, ...] = ("case-c",),
    models: tuple[str, ...] = ("qwen3:0.6b",),
    seed: int = 0,
    seed_candidate_fingerprint: str = SEED_FINGERPRINT,
    total_metric_calls: int = 10,
    max_metric_calls: int = 12,
    train_case_ids: tuple[str, ...] = ("case-a", "case-b"),
    validation_case_ids: tuple[str, ...] = ("case-c",),
    prompt_lab_revision: str = "abc123",
    korvid_revision: str = "def456",
    comparison_outcome: str = "improved",
    comparison_metrics: list[dict[str, object]] | None = None,
    before_passed: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    best_candidate_mapping = best_candidate_mapping or _candidate_mapping(candidate_id)
    best_candidate = Candidate.from_mapping(best_candidate_mapping)
    resolved_candidate_fingerprint = (
        candidate_fingerprint
        if candidate_fingerprint is not None
        else best_candidate.fingerprint
    )
    resolved_best_candidate_fingerprint = (
        best_candidate_fingerprint
        if best_candidate_fingerprint is not None
        else resolved_candidate_fingerprint
    )

    eval_summary = {
        "bundle_kind": "validation",
        "candidate_id": candidate_id,
        "candidate_fingerprint": resolved_candidate_fingerprint,
        "campaign_id": "test-campaign",
        "campaign_case_ids": list(evaluated_case_ids),
        "evaluated_case_ids": list(evaluated_case_ids),
        "evaluated_models": list(models),
        "campaign_case_model_pairs": [
            f"{c}:{m}" for c in evaluated_case_ids for m in models
        ],
        "evaluated_case_model_pairs": [
            f"{c}:{m}" for c in evaluated_case_ids for m in models
        ],
        "aggregate_score": 0.6,
        "model_scores": {m: 0.6 for m in models},
        "execution_modes": ["live"],
        "run_execution_modes": {
            f"{c}:{m}": "live" for c in evaluated_case_ids for m in models
        },
        "repetitions_per_case": 5,
        "pass_at_3": 1.0,
        "pass_at_5": 1.0,
        "hard_safety_failures": 0,
        "systemic_failures": 0,
        "milestone_passed": False,
        "case_sets": {
            "train": list(train_case_ids),
            "validation": list(validation_case_ids),
            "milestone": [],
        },
        "artifact_refs": ["evaluation-summary.json"],
        "reproduction_command": ["echo", "test"],
    }
    (root / "evaluation-summary.json").write_text(json.dumps(eval_summary))
    before_eval_summary = {
        **eval_summary,
        "candidate_id": "seed-candidate",
        "candidate_fingerprint": seed_candidate_fingerprint,
        "aggregate_score": 1.0 if before_passed else 0.4,
        "model_scores": {
            model: 1.0 if before_passed else 0.4 for model in models
        },
        "pass_at_3": 1.0 if before_passed else 0.0,
        "pass_at_5": 1.0 if before_passed else 0.0,
        "artifact_refs": [
            "before-evaluation-summary.json",
            *[
                f"before-responses/case-c-r{repetition:02d}.json"
                for repetition in range(1, 6)
            ],
        ],
    }
    (root / "before-evaluation-summary.json").write_text(
        json.dumps(before_eval_summary)
    )
    before_responses = root / "before-responses"
    before_responses.mkdir()
    for repetition in range(1, 6):
        (before_responses / f"case-c-r{repetition:02d}.json").write_text(
            json.dumps(
                {
                    "protocol_version": 1,
                    "status": "completed",
                    "execution_mode": "live",
                    "candidate_fingerprint": seed_candidate_fingerprint,
                    "request_identity": {
                        "case_id": "case-c",
                        "template_id": "case-c-template",
                        "model": "qwen3:0.6b",
                        "repetition": repetition,
                        "seed": repetition - 1,
                    },
                    "grade": {
                        "completion": 1.0 if before_passed else 0.0,
                        "verification": 1.0,
                        "efficiency": 1.0,
                        "hard_failures": [],
                    },
                    "answer": "",
                    "journal": {
                        "journey_id": "",
                        "checkpoints": [],
                        "missing_checkpoints": [],
                        "checkpoint_counts": {},
                        "journal_event_count": 0,
                        "audit_record_count": 0,
                        "hard_failure_count": 0,
                    },
                    "usage": {
                        "tool_calls": 0,
                        "iterations": 1,
                        "wall_time_seconds": 1.0,
                    },
                    "error": None,
                }
            )
        )

    round_summary = {
        "schema_version": 1,
        "campaign_id": "test-campaign",
        "candidate_id": candidate_id,
        "candidate_fingerprint": resolved_candidate_fingerprint,
        "models": list(models),
        "aggregate_score": 0.6,
        "model_scores": {m: 0.6 for m in models},
        "pass_at_3": 1.0,
        "pass_at_5": 1.0,
        "systemic_failures": 0,
        "promotion_eligible": True,
        "promotion_blockers": [],
        "status_counts": {"completed": len(evaluated_case_ids)},
        "hard_failure_counts": {},
        "runs": [],
        "artifact_refs": [
            "round-summary.json",
            "evaluation-summary.json",
            "comparison-summary.json",
            "optimization-summary.json",
            "best-candidate.yaml",
        ],
        "evaluation_artifact_refs": ["evaluation-summary.json"],
        "prompt_lab_revision": prompt_lab_revision,
        "korvid_revision": korvid_revision,
        "workflow_run_url": "https://github.com/example/actions/runs/1",
        "reproduction_command": ["echo", "test"],
        "campaign_action_id": campaign_action_id,
    }
    (root / "round-summary.json").write_text(json.dumps(round_summary))

    resolved_comparison_metrics = (
        comparison_metrics
        if comparison_metrics is not None
        else [
            {
                "key": "aggregate_score",
                "label": "Aggregate score",
                "before": 0.4,
                "after": 0.6,
                "delta": 0.2,
                "result": "improved",
                "integer": False,
                "core": True,
            },
            {
                "key": "systemic_failures",
                "label": "Systemic failures",
                "before": 0,
                "after": 0,
                "delta": 0,
                "result": "unchanged",
                "integer": True,
                "core": True,
            },
        ]
    )
    for metric in resolved_comparison_metrics:
        key = metric["key"]
        after = metric["after"]
        if key == "aggregate_score":
            eval_summary["aggregate_score"] = after
            eval_summary["model_scores"] = {model: after for model in models}
            round_summary["aggregate_score"] = after
            round_summary["model_scores"] = {model: after for model in models}
        elif key in {
            "pass_at_3",
            "pass_at_5",
            "systemic_failures",
            "hard_safety_failures",
        }:
            eval_summary[key] = after
            if key != "hard_safety_failures":
                round_summary[key] = after
    (root / "evaluation-summary.json").write_text(json.dumps(eval_summary))
    (root / "round-summary.json").write_text(json.dumps(round_summary))
    comparison_summary = {
        "schema_version": 1,
        "status": "changed",
        "outcome": comparison_outcome,
        "seed_candidate_fingerprint": seed_candidate_fingerprint,
        "best_candidate_fingerprint": resolved_candidate_fingerprint,
        "contract": {
            "campaign_id": "test-campaign",
            "models": list(models),
            "case_repetitions": sorted(
                [case_id, models[0], rep]
                for case_id in evaluated_case_ids
                for rep in range(1, 6)  # repetitions_per_case=5
            ),
            "execution_modes": ["live"],
        },
        "metrics": resolved_comparison_metrics,
        **{
            f"{result}_count": sum(
                metric["result"] == result for metric in resolved_comparison_metrics
            )
            for result in (
                "improved",
                "unchanged",
                "regressed",
                "not_comparable",
            )
        },
    }
    (root / "comparison-summary.json").write_text(json.dumps(comparison_summary))

    opt_summary = {
        "run_id": "run-001",
        "seed": seed,
        "run_identity": {
            "schema_version": 1,
            "campaign_id": "test-campaign",
            "candidate_id": candidate_id,
            "seed_candidate_fingerprint": seed_candidate_fingerprint,
            "train_case_ids": list(train_case_ids),
            "validation_case_ids": list(validation_case_ids),
            "max_metric_calls": max_metric_calls,
            "seed": seed,
            "proposal_source": "dspy",
        },
        "invocation_dir": "artifacts/run-001",
        "best_idx": 1,
        "best_validation_score": 0.6,
        "best_candidate_fingerprint": resolved_best_candidate_fingerprint,
        "seed_candidate_fingerprint": seed_candidate_fingerprint,
        "best_candidate_differs_from_seed": (
            resolved_best_candidate_fingerprint != seed_candidate_fingerprint
        ),
        "train_case_ids": list(train_case_ids),
        "validation_case_ids": list(validation_case_ids),
        "execution_modes": ["live"],
        "num_candidates": 5,
        "total_metric_calls": total_metric_calls,
        "num_full_val_evals": 2,
        "run_dir": "artifacts/run-001",
    }
    (root / "optimization-summary.json").write_text(json.dumps(opt_summary))
    (root / "best-candidate.yaml").write_text(yaml.dump(best_candidate_mapping))


def _upgrade_round_summary_to_readonly(root: Path) -> None:
    path = root / "round-summary.json"
    summary = json.loads(path.read_text())
    response_refs: list[str] = []
    responses_dir = root / "responses"
    responses_dir.mkdir()
    for repetition in range(1, 6):
        response_ref = f"responses/case-c-r{repetition:02d}.json"
        response_refs.append(response_ref)
        (root / response_ref).write_text(
            json.dumps(
                {
                    "protocol_version": 1,
                    "status": "completed",
                    "execution_mode": "live",
                    "request_identity": {
                        "case_id": "case-c",
                        "template_id": "case-c-template",
                        "model": "qwen3:0.6b",
                        "repetition": repetition,
                        "seed": repetition - 1,
                        "seed_applied": False,
                    },
                    "candidate_fingerprint": summary["candidate_fingerprint"],
                    "evidence_source": {
                        "kind": "korvid_readonly",
                        "korvid_version": "0.3.0",
                        "scenario_sha256": "a" * 64,
                    },
                    "grade": {
                        "completion": 1.0,
                        "verification": 1.0,
                        "efficiency": 1.0,
                        "hard_failures": [],
                    },
                    "answer": "",
                    "journal": {
                        "journey_id": "",
                        "checkpoints": [],
                        "missing_checkpoints": [],
                        "checkpoint_counts": {},
                        "journal_event_count": 0,
                        "audit_record_count": 0,
                        "hard_failure_count": 0,
                    },
                    "usage": {
                        "tool_calls": 0,
                        "iterations": 1,
                        "wall_time_seconds": 1.0,
                    },
                    "error": None,
                }
            )
        )
    summary["schema_version"] = 2
    summary["aggregate_score"] = 1.0
    summary["model_scores"] = {"qwen3:0.6b": 1.0}
    summary["evaluation_backend"] = "korvid_readonly"
    summary["evidence_sources"] = [
        ["case-c", "qwen3:0.6b", repetition, "korvid_readonly", "0.3.0", "a" * 64]
        for repetition in range(1, 6)
    ]
    summary["runs"] = [
        {
            "run_id": f"case-c-r{repetition:02d}",
            "case_id": "case-c",
            "model": "qwen3:0.6b",
            "repetition": repetition,
            "status": "completed",
            "completion": 1.0,
            "verification": 1.0,
            "efficiency": 1.0,
            "elapsed_seconds": 1.0,
            "hard_failures": [],
            "execution_mode": "live",
        }
        for repetition in range(1, 6)
    ]
    summary["status_counts"] = {"completed": 5}
    summary["promotion_eligible"] = False
    summary["promotion_blockers"] = ["milestone_failed"]
    summary["evaluation_artifact_refs"] = [
        "evaluation-summary.json",
        *response_refs,
    ]
    path.write_text(json.dumps(summary))
    evaluation_path = root / "evaluation-summary.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["aggregate_score"] = 1.0
    evaluation["model_scores"] = {"qwen3:0.6b": 1.0}
    evaluation_path.write_text(json.dumps(evaluation))
    for response_path in sorted((root / "before-responses").glob("*.json")):
        response = json.loads(response_path.read_text())
        response["request_identity"]["seed_applied"] = False
        response["evidence_source"] = {
            "kind": "korvid_readonly",
            "korvid_version": "0.3.0",
            "scenario_sha256": "a" * 64,
        }
        response_path.write_text(json.dumps(response))


def _upgrade_comparison_to_readonly(root: Path, *, version: str = "0.3.0") -> None:
    path = root / "comparison-summary.json"
    summary = json.loads(path.read_text())
    summary["schema_version"] = 2
    summary["contract"]["evidence_sources"] = [
        ["case-c", "qwen3:0.6b", repetition, "korvid_readonly", version, "a" * 64]
        for repetition in range(1, 6)
    ]
    summary["metrics"] = [
        {
            "key": "aggregate_score",
            "label": "Aggregate score",
            "before": 0.4,
            "after": 1.0,
            "delta": 0.6,
            "result": "improved",
            "integer": False,
            "core": True,
        },
        {
            "key": "pass_at_3",
            "label": "pass@3",
            "before": 0.0,
            "after": 1.0,
            "delta": 1.0,
            "result": "improved",
            "integer": False,
            "core": True,
        },
        {
            "key": "pass_at_5",
            "label": "pass@5",
            "before": 0.0,
            "after": 1.0,
            "delta": 1.0,
            "result": "improved",
            "integer": False,
            "core": True,
        },
        {
            "key": "hard_safety_failures",
            "label": "Hard safety failures",
            "before": 0,
            "after": 0,
            "delta": 0,
            "result": "unchanged",
            "integer": True,
            "core": True,
        },
        {
            "key": "systemic_failures",
            "label": "Systemic failures",
            "before": 0,
            "after": 0,
            "delta": 0,
            "result": "unchanged",
            "integer": True,
            "core": True,
        },
    ]
    summary["improved_count"] = 3
    summary["unchanged_count"] = 2
    summary["regressed_count"] = 0
    summary["not_comparable_count"] = 0
    path.write_text(json.dumps(summary))


# ---------------------------------------------------------------------------
# Safe Ingestion Tests
# ---------------------------------------------------------------------------


class TestLoadRoundOutcome:
    def test_loads_valid_search_evidence(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        action = _search_action()
        ctrl = _control()
        st = _state()
        expected_fingerprint = Candidate.from_mapping(_candidate_mapping()).fingerprint
        outcome = load_round_outcome(root, action, control=ctrl, state=st)
        assert outcome.candidate_fingerprint == expected_fingerprint
        assert outcome.aggregate_score == 0.6
        assert outcome.search_improved is True

    def test_rejects_malformed_comparison_evidence_source(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        path = root / "comparison-summary.json"
        comparison = json.loads(path.read_text())
        comparison["schema_version"] = 2
        comparison["contract"]["evidence_sources"] = [
            [
                "case-c",
                "qwen3:0.6b",
                1,
                "korvid_readonly",
                "secret\nversion",
                "a" * 64,
            ]
        ]
        path.write_text(json.dumps(comparison))
        _upgrade_round_summary_to_readonly(root)
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="korvid_version"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_rejects_deeply_nested_response(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        response_path = root / "responses" / "case-c-r01.json"
        response_path.write_text("[" * 2_000 + "0" + "]" * 2_000)
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="response"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_rejects_schema_v2_comparison_with_empty_evidence_sources(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        path = root / "comparison-summary.json"
        comparison = json.loads(path.read_text())
        comparison["schema_version"] = 2
        comparison["contract"]["evidence_sources"] = []
        path.write_text(json.dumps(comparison))
        _upgrade_round_summary_to_readonly(root)
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="evidence_sources"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_accepts_legacy_comparison_without_evidence_sources(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        path = root / "comparison-summary.json"
        comparison = json.loads(path.read_text())
        path.write_text(json.dumps(comparison))

        outcome = load_round_outcome(
            root, _search_action(), control=_control(), state=_state()
        )

        assert outcome.search_improved is True

    def test_readonly_campaign_rejects_provenance_schema_downgrade(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="schema_version must be 2"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_search_rejects_comparison_provenance_mismatch(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root, version="0.4.0")
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="provenance.*round-summary"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_rejects_response_provenance_mismatch(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        response_path = root / "responses" / "case-c-r01.json"
        response = json.loads(response_path.read_text())
        response["evidence_source"]["korvid_version"] = "0.4.0"
        response_path.write_text(json.dumps(response))
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="response.*provenance"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_rejects_response_candidate_mismatch(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        response_path = root / "responses" / "case-c-r01.json"
        response = json.loads(response_path.read_text())
        response["candidate_fingerprint"] = "f" * 64
        response_path.write_text(json.dumps(response))
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="candidate_fingerprint"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_rejects_unknown_response_fields(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        response_path = root / "responses" / "case-c-r01.json"
        response = json.loads(response_path.read_text())
        response["raw_payload"] = {"token": "must-not-pass"}
        response_path.write_text(json.dumps(response))
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="unknown key"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_rejects_malformed_projected_values(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        response_path = root / "responses" / "case-c-r01.json"
        response = json.loads(response_path.read_text())
        response["grade"]["completion"] = {"raw": "payload"}
        response_path.write_text(json.dumps(response))
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="completion"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_rejects_response_grade_mismatch(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        response_path = root / "responses" / "case-c-r01.json"
        response = json.loads(response_path.read_text())
        response["grade"]["completion"] = 0.0
        response_path.write_text(json.dumps(response))
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="response.*round-summary run"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_recomputes_pass_metrics_from_runs(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        round_path = root / "round-summary.json"
        round_summary = json.loads(round_path.read_text())
        round_summary["pass_at_3"] = 0.0
        round_summary["pass_at_5"] = 0.0
        round_path.write_text(json.dumps(round_summary))
        eval_path = root / "evaluation-summary.json"
        eval_summary = json.loads(eval_path.read_text())
        eval_summary["pass_at_3"] = 0.0
        eval_summary["pass_at_5"] = 0.0
        eval_path.write_text(json.dumps(eval_summary))
        comparison_path = root / "comparison-summary.json"
        comparison = json.loads(comparison_path.read_text())
        for metric in comparison["metrics"]:
            if metric["key"] == "pass_at_3":
                metric.update(after=0.0, delta=-0.8, result="regressed")
            elif metric["key"] == "pass_at_5":
                metric.update(after=0.0, delta=-1.0, result="regressed")
        comparison["outcome"] = "regressed"
        comparison["improved_count"] = 1
        comparison["unchanged_count"] = 2
        comparison["regressed_count"] = 2
        comparison_path.write_text(json.dumps(comparison))
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="pass_at_3.*runs"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_recomputes_hard_failure_total(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        eval_path = root / "evaluation-summary.json"
        eval_summary = json.loads(eval_path.read_text())
        eval_summary["hard_safety_failures"] = 1
        eval_path.write_text(json.dumps(eval_summary))
        comparison_path = root / "comparison-summary.json"
        comparison = json.loads(comparison_path.read_text())
        hard_metric = next(
            metric
            for metric in comparison["metrics"]
            if metric["key"] == "hard_safety_failures"
        )
        hard_metric.update(after=1, delta=1, result="regressed")
        comparison["outcome"] = "regressed"
        comparison["unchanged_count"] = 2
        comparison["regressed_count"] = 1
        comparison_path.write_text(json.dumps(comparison))
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="hard_safety_failures.*runs"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_recomputes_evaluation_aggregate(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        eval_path = root / "evaluation-summary.json"
        eval_summary = json.loads(eval_path.read_text())
        eval_summary["aggregate_score"] = 0.5
        eval_summary["model_scores"] = {"qwen3:0.6b": 0.5}
        eval_path.write_text(json.dumps(eval_summary))
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="evaluation-summary.aggregate_score.*runs"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_recomputes_model_scores(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        for filename in ("round-summary.json", "evaluation-summary.json"):
            path = root / filename
            summary = json.loads(path.read_text())
            summary["model_scores"] = {"qwen3:0.6b": 0.5}
            path.write_text(json.dumps(summary))
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="model_scores.*runs"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_rejects_response_symlink_swap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        response_path = root / "responses" / "case-c-r01.json"
        outside = tmp_path / "outside.json"
        original_response = response_path.read_bytes()
        outside.write_bytes(original_response)
        real_resolve = campaign_artifacts._resolve_safe_path
        real_read_text = Path.read_text
        swapped = False

        def swapping_resolve(safe_root: Path, filename: str) -> Path:
            nonlocal swapped
            path = real_resolve(safe_root, filename)
            if filename.startswith("responses/") and not swapped:
                swapped = True
                path.unlink()
                path.symlink_to(outside)
            return path

        def restoring_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
            text = real_read_text(path, *args, **kwargs)
            if path == response_path and path.is_symlink():
                path.unlink()
                path.write_bytes(original_response)
            return text

        monkeypatch.setattr(
            campaign_artifacts, "_resolve_safe_path", swapping_resolve
        )
        monkeypatch.setattr(Path, "read_text", restoring_read_text)
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="response|symlink"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_rejects_package_root_swap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        original_root = tmp_path / "original-evidence"
        real_resolve = campaign_artifacts._resolve_safe_path
        swapped = False

        def swapping_root(safe_root: Path, filename: str) -> Path:
            nonlocal swapped
            path = real_resolve(safe_root, filename)
            if filename.startswith("responses/") and not swapped:
                swapped = True
                safe_root.rename(original_root)
                shutil.copytree(original_root, safe_root)
            return path

        monkeypatch.setattr(
            campaign_artifacts, "_resolve_safe_path", swapping_root
        )
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="root.*changed"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_readonly_campaign_rejects_oversized_response(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        response_path = root / "responses" / "case-c-r01.json"
        response_path.write_bytes(response_path.read_bytes() + b" " * 100_000)
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="too large"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_rejects_comparison_metric_count_mismatch(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        path = root / "comparison-summary.json"
        comparison = json.loads(path.read_text())
        comparison["improved_count"] = 999
        path.write_text(json.dumps(comparison))

        with pytest.raises(ValueError, match="improved_count"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_rejects_json_parser_recursion(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        monkeypatch.setattr(
            campaign_artifacts.json,
            "loads",
            lambda _text: (_ for _ in ()).throw(RecursionError("too deep")),
        )

        with pytest.raises(ValueError, match="malformed JSON"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_rejects_comparison_metric_not_bound_to_round_summary(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        path = root / "comparison-summary.json"
        comparison = json.loads(path.read_text())
        aggregate = next(
            metric
            for metric in comparison["metrics"]
            if metric["key"] == "aggregate_score"
        )
        aggregate["after"] = 0.9
        aggregate["delta"] = 0.5
        path.write_text(json.dumps(comparison))

        with pytest.raises(ValueError, match="aggregate_score.*evidence"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_unmeasured_search_binds_comparison_to_before_summary(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        path = root / "comparison-summary.json"
        comparison = json.loads(path.read_text())
        aggregate = next(
            metric
            for metric in comparison["metrics"]
            if metric["key"] == "aggregate_score"
        )
        aggregate["before"] = 0.9
        aggregate["delta"] = -0.3
        aggregate["result"] = "regressed"
        comparison["outcome"] = "regressed"
        comparison["improved_count"] = 0
        comparison["regressed_count"] = 1
        path.write_text(json.dumps(comparison))

        with pytest.raises(ValueError, match="aggregate_score.*before evidence"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_comparison_status_must_match_candidate_fingerprints(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        path = root / "comparison-summary.json"
        comparison = json.loads(path.read_text())
        aggregate = next(
            metric
            for metric in comparison["metrics"]
            if metric["key"] == "aggregate_score"
        )
        aggregate.update(before=0.6, delta=0.0, result="unchanged")
        comparison["status"] = "unchanged"
        comparison["outcome"] = "unchanged"
        comparison["improved_count"] = 0
        comparison["unchanged_count"] = 2
        path.write_text(json.dumps(comparison))

        with pytest.raises(ValueError, match="status.*fingerprint"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_measured_unchanged_search_uses_current_evidence_for_both_sides(
        self, tmp_path: Path
    ) -> None:
        candidate = Candidate.from_mapping(_candidate_mapping())
        state = replace(
            _state(champion_fingerprint=candidate.fingerprint),
            champion_score=CampaignScore(
                fingerprint=candidate.fingerprint,
                aggregate=0.4,
                hard_safety_failures=0,
                core_regression=False,
                systemic_failures=0,
                pass_at_3=0.0,
                pass_at_5=0.0,
            ),
        )
        root = tmp_path / "evidence"
        _write_search_evidence(
            root,
            seed_candidate_fingerprint=candidate.fingerprint,
        )
        path = root / "comparison-summary.json"
        comparison = json.loads(path.read_text())
        aggregate = next(
            metric
            for metric in comparison["metrics"]
            if metric["key"] == "aggregate_score"
        )
        aggregate.update(before=0.6, delta=0.0, result="unchanged")
        comparison["status"] = "unchanged"
        comparison["outcome"] = "unchanged"
        comparison["improved_count"] = 0
        comparison["unchanged_count"] = 2
        path.write_text(json.dumps(comparison))

        outcome = load_round_outcome(
            root, _search_action(), control=_control(), state=state
        )

        assert outcome.search_improved is False

    def test_unmeasured_search_recomputes_before_summary_from_responses(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        before_path = root / "before-evaluation-summary.json"
        before = json.loads(before_path.read_text())
        before["aggregate_score"] = 0.9
        before["model_scores"] = {"qwen3:0.6b": 0.9}
        before_path.write_text(json.dumps(before))
        comparison_path = root / "comparison-summary.json"
        comparison = json.loads(comparison_path.read_text())
        aggregate = next(
            metric
            for metric in comparison["metrics"]
            if metric["key"] == "aggregate_score"
        )
        aggregate.update(before=0.9, delta=-0.3, result="regressed")
        comparison["outcome"] = "regressed"
        comparison["improved_count"] = 0
        comparison["regressed_count"] = 1
        comparison_path.write_text(json.dumps(comparison))

        with pytest.raises(ValueError, match="before.*aggregate_score.*responses"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_readonly_unmeasured_search_rejects_before_response_leakage(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        _upgrade_round_summary_to_readonly(root)
        _upgrade_comparison_to_readonly(root)
        response_path = root / "before-responses" / "case-c-r01.json"
        response = json.loads(response_path.read_text())
        response["answer"] = "raw secret answer"
        response["evidence_source"]["korvid_version"] = "0.4.0"
        response_path.write_text(json.dumps(response))
        control = replace(_control(), evaluation_backend="korvid_readonly")

        with pytest.raises(ValueError, match="answer|provenance"):
            load_round_outcome(
                root, _search_action(control=control), control=control, state=_state()
            )

    def test_measured_search_still_validates_before_response_leakage(
        self, tmp_path: Path
    ) -> None:
        champion = Candidate.from_mapping(_candidate_mapping("champion")).fingerprint
        state = replace(
            _state(champion_fingerprint=champion),
            champion_score=CampaignScore(
                fingerprint=champion,
                aggregate=0.4,
                hard_safety_failures=0,
                core_regression=False,
                systemic_failures=0,
                pass_at_3=0.0,
                pass_at_5=0.0,
            ),
        )
        root = tmp_path / "evidence"
        _write_search_evidence(root, seed_candidate_fingerprint=champion)
        response_path = root / "before-responses" / "case-c-r01.json"
        response = json.loads(response_path.read_text())
        response["answer"] = "raw secret answer"
        response_path.write_text(json.dumps(response))

        with pytest.raises(ValueError, match="answer"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=state
            )

    def test_measured_changed_search_accepts_fresh_stochastic_before_score(
        self, tmp_path: Path
    ) -> None:
        champion = Candidate.from_mapping(_candidate_mapping("champion")).fingerprint
        state = replace(
            _state(champion_fingerprint=champion),
            champion_score=CampaignScore(
                fingerprint=champion,
                aggregate=0.4,
                hard_safety_failures=0,
                core_regression=False,
                systemic_failures=0,
                pass_at_3=0.0,
                pass_at_5=0.0,
            ),
        )
        root = tmp_path / "evidence"
        _write_search_evidence(root, seed_candidate_fingerprint=champion)
        before_path = root / "before-evaluation-summary.json"
        before = json.loads(before_path.read_text())
        before["aggregate_score"] = 0.5
        before["model_scores"] = {"qwen3:0.6b": 0.5}
        before_path.write_text(json.dumps(before))
        for response_path in (root / "before-responses").glob("*.json"):
            response = json.loads(response_path.read_text())
            response["grade"]["completion"] = 1 / 6
            response_path.write_text(json.dumps(response))
        comparison_path = root / "comparison-summary.json"
        comparison = json.loads(comparison_path.read_text())
        aggregate = next(
            metric
            for metric in comparison["metrics"]
            if metric["key"] == "aggregate_score"
        )
        aggregate.update(before=0.5, delta=0.1)
        comparison_path.write_text(json.dumps(comparison))

        outcome = load_round_outcome(
            root, _search_action(), control=_control(), state=state
        )

        assert outcome.search_improved is True

    def test_search_recomputes_milestone_passed(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        path = root / "evaluation-summary.json"
        summary = json.loads(path.read_text())
        summary["milestone_passed"] = True
        path.write_text(json.dumps(summary))

        with pytest.raises(ValueError, match="milestone_passed"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_search_rejects_underreported_metric_calls(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        path = root / "optimization-summary.json"
        summary = json.loads(path.read_text())
        summary["total_metric_calls"] = 1
        path.write_text(json.dumps(summary))

        with pytest.raises(ValueError, match="total_metric_calls.*minimum"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_search_charges_trusted_bounded_metric_maximum(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        path = root / "optimization-summary.json"
        summary = json.loads(path.read_text())
        summary["num_candidates"] = 1
        summary["num_full_val_evals"] = 0
        summary["total_metric_calls"] = 1
        path.write_text(json.dumps(summary))
        action = _search_action()
        control = _control()

        outcome = load_round_outcome(
            root, action, control=control, state=_state()
        )

        assert outcome.metric_calls_used == max_search_metric_calls(
            control, action.metric_calls
        )

    def test_rejects_deeply_nested_best_candidate_yaml(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        (root / "best-candidate.yaml").write_text(
            "[" * 2_000 + "0" + "]" * 2_000
        )

        with pytest.raises(ValueError, match="malformed YAML"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_rejects_oversized_top_level_summary(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        path = root / "round-summary.json"
        path.write_bytes(path.read_bytes() + b" " * 1_100_000)

        with pytest.raises(ValueError, match="too large"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_rejects_top_level_summary_root_swap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        original_root = tmp_path / "original-evidence"
        real_resolve = campaign_artifacts._resolve_safe_path
        swapped = False

        def swapping_root(safe_root: Path, filename: str) -> Path:
            nonlocal swapped
            path = real_resolve(safe_root, filename)
            if filename == "round-summary.json" and not swapped:
                swapped = True
                safe_root.rename(original_root)
                shutil.copytree(original_root, safe_root)
            return path

        monkeypatch.setattr(
            campaign_artifacts, "_resolve_safe_path", swapping_root
        )

        with pytest.raises(ValueError, match="root.*changed"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_rejects_oversized_best_candidate_yaml(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        path = root / "best-candidate.yaml"
        path.write_bytes(path.read_bytes() + b" " * 1_100_000)

        with pytest.raises(ValueError, match="too large"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_rejects_best_candidate_root_swap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        original_root = tmp_path / "original-evidence"
        real_resolve = campaign_artifacts._resolve_safe_path
        swapped = False

        def swapping_root(safe_root: Path, filename: str) -> Path:
            nonlocal swapped
            path = real_resolve(safe_root, filename)
            if filename == "best-candidate.yaml" and not swapped:
                swapped = True
                safe_root.rename(original_root)
                shutil.copytree(original_root, safe_root)
            return path

        monkeypatch.setattr(
            campaign_artifacts, "_resolve_safe_path", swapping_root
        )

        with pytest.raises(ValueError, match="root.*changed"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state()
            )

    def test_evidence_campaign_id_binds_to_evaluation_campaign(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        control = replace(_control(), campaign_id="optimization-controller")
        state = replace(_state(), campaign_id="optimization-controller")

        outcome = load_round_outcome(
            root,
            _search_action(),
            control=control,
            state=state,
        )

        assert outcome.evaluated_case_ids == ("case-c",)

    def test_rejects_wrong_action_id(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, campaign_action_id="action-1")
        action = _search_action(action_id="action-WRONG")
        with pytest.raises(ValueError, match="campaign_action_id mismatch"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_accepts_three_element_case_repetitions_entries(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)

        outcome = load_round_outcome(root, _search_action(), control=_control(), state=_state())

        assert outcome.aggregate_score == 0.6

    def test_rejects_two_element_case_repetitions_entries(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        comparison_summary = json.loads((root / "comparison-summary.json").read_text())
        comparison_summary["contract"]["case_repetitions"] = [["case-c", 5]]
        (root / "comparison-summary.json").write_text(json.dumps(comparison_summary))

        with pytest.raises(ValueError, match=r"case_repetitions\[0\] must be \[case_id, model, repetition\]"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_rejects_run_identity_max_metric_calls_lower_than_action_budget(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, max_metric_calls=11)

        with pytest.raises(ValueError, match=r"run_identity\.max_metric_calls \(11\) != action\.metric_calls \(12\)"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_rejects_run_identity_max_metric_calls_higher_than_action_budget(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, max_metric_calls=13)

        with pytest.raises(ValueError, match=r"run_identity\.max_metric_calls \(13\) != action\.metric_calls \(12\)"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_accepts_run_identity_max_metric_calls_equal_to_action_budget(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, max_metric_calls=12)

        outcome = load_round_outcome(root, _search_action(), control=_control(), state=_state())

        assert outcome.aggregate_score == 0.6

    def test_rejects_wrong_case_set(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        # Evidence has case-c but action is SEARCH expecting validation_case_ids
        _write_search_evidence(root, evaluated_case_ids=("wrong-case",))
        action = _search_action()
        with pytest.raises(ValueError, match="evaluated case set"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_rejects_wrong_model(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, models=("wrong-model",))
        action = _search_action()
        with pytest.raises(ValueError, match="model mismatch"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_rejects_wrong_revision(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, prompt_lab_revision="wrong-rev")
        action = _search_action()
        with pytest.raises(ValueError, match="prompt_lab_revision mismatch"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_rejects_unknown_round_summary_key(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        round_summary = json.loads((root / "round-summary.json").read_text())
        round_summary["unexpected"] = True
        (root / "round-summary.json").write_text(json.dumps(round_summary))
        with pytest.raises(ValueError, match="round-summary unknown key"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_rejects_missing_prompt_lab_revision(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        round_summary = json.loads((root / "round-summary.json").read_text())
        round_summary.pop("prompt_lab_revision")
        (root / "round-summary.json").write_text(json.dumps(round_summary))
        with pytest.raises(ValueError, match=r"round-summary missing key\(s\): prompt_lab_revision"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_rejects_extra_evaluated_model(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        eval_summary = json.loads((root / "evaluation-summary.json").read_text())
        eval_summary["evaluated_models"] = ["qwen3:0.6b", "extra-model"]
        (root / "evaluation-summary.json").write_text(json.dumps(eval_summary))
        with pytest.raises(ValueError, match="evaluated_models"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_rejects_campaign_id_mismatch(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        round_summary = json.loads((root / "round-summary.json").read_text())
        round_summary["campaign_id"] = "other-campaign"
        (root / "round-summary.json").write_text(json.dumps(round_summary))
        eval_summary = json.loads((root / "evaluation-summary.json").read_text())
        eval_summary["campaign_id"] = "other-campaign"
        (root / "evaluation-summary.json").write_text(json.dumps(eval_summary))
        comparison_summary = json.loads((root / "comparison-summary.json").read_text())
        comparison_summary["contract"]["campaign_id"] = "other-campaign"
        (root / "comparison-summary.json").write_text(json.dumps(comparison_summary))
        with pytest.raises(ValueError, match="campaign_id mismatch"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_rejects_symlinked_file(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        target = tmp_path / "evil.json"
        target.write_text("{}")
        real = root / "evaluation-summary.json"
        real.unlink()
        real.symlink_to(target)
        action = _search_action()
        with pytest.raises(ValueError, match="symlink"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_rejects_malformed_json(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        root.mkdir()
        (root / "round-summary.json").write_text("not json{{{")
        (root / "evaluation-summary.json").write_text("{}")
        action = _search_action()
        with pytest.raises(ValueError, match="malformed"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_rejects_bool_as_int(self, tmp_path: Path) -> None:
        """Strict types: bool must not be accepted as int."""
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        # Corrupt hard_safety_failures to bool
        es = json.loads((root / "evaluation-summary.json").read_text())
        es["hard_safety_failures"] = True
        (root / "evaluation-summary.json").write_text(json.dumps(es))
        action = _search_action()
        with pytest.raises(ValueError, match="integer"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_rejects_non_finite_float(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        rs = json.loads((root / "round-summary.json").read_text())
        rs["aggregate_score"] = float("inf")
        (root / "round-summary.json").write_text(json.dumps(rs))
        action = _search_action()
        with pytest.raises(ValueError, match="finite"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_search_rejects_missing_optimization_summary(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        (root / "optimization-summary.json").unlink()
        action = _search_action()
        with pytest.raises(ValueError, match="required file missing"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_search_requires_comparison_summary(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        (root / "comparison-summary.json").unlink()
        with pytest.raises(ValueError, match="comparison-summary.json"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_search_rejects_wrong_seed(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, seed=99)
        action = _search_action()
        with pytest.raises(ValueError, match="seed mismatch"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_search_accepts_and_reports_bounded_gepa_overshoot(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, total_metric_calls=15)

        outcome = load_round_outcome(
            root,
            _search_action(),
            control=_control(),
            state=_state(),
        )

        assert outcome.metric_calls_used == 15

    def test_search_rejects_metric_budget_exceeded(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, total_metric_calls=16)
        action = _search_action()
        with pytest.raises(ValueError, match="exceeds bounded GEPA maximum"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_search_rejects_wrong_seed_fingerprint(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, seed_candidate_fingerprint="wrong")
        action = _search_action()
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            load_round_outcome(root, action, control=_control(), state=_state())

    @pytest.mark.parametrize(
        ("bad_ref", "error_match"),
        [
            ("/absolute/path.json", "relative"),
            ("../escape.json", "travers"),
            ("missing.json", "regular file"),
        ],
    )
    def test_rejects_unsafe_artifact_refs(
        self, tmp_path: Path, bad_ref: str, error_match: str,
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        round_summary = json.loads((root / "round-summary.json").read_text())
        round_summary["artifact_refs"] = [bad_ref]
        (root / "round-summary.json").write_text(json.dumps(round_summary))
        with pytest.raises(ValueError, match=error_match):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_rejects_best_candidate_fingerprint_mismatch(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, candidate_fingerprint="wrong-fingerprint")
        with pytest.raises(ValueError, match="best-candidate fingerprint"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_milestone_rejects_optimization_files(self, tmp_path: Path) -> None:
        """MILESTONE must not have optimization-summary.json."""
        root = tmp_path / "evidence"
        root.mkdir(parents=True)
        ctrl = _control()
        st = _state()
        # Write milestone-style evidence
        action = CampaignAction(
            action_id="ms-1",
            kind=ActionKind.MILESTONE,
            expected_state_hash="sha256:" + "0" * 64,
            tier_index=0,
            metric_calls=0,
        )
        eval_summary = {
            "bundle_kind": "milestone",
            "candidate_id": "cand-1",
            "candidate_fingerprint": SEED_FINGERPRINT,
            "campaign_id": "test-campaign",
            "campaign_case_ids": ["case-d"],
            "evaluated_case_ids": ["case-d"],
            "evaluated_models": ["qwen3:0.6b"],
            "campaign_case_model_pairs": ["case-d:qwen3:0.6b"],
            "evaluated_case_model_pairs": ["case-d:qwen3:0.6b"],
            "aggregate_score": 1.0,
            "model_scores": {"qwen3:0.6b": 1.0},
            "execution_modes": ["live"],
            "run_execution_modes": {"case-d:qwen3:0.6b": "live"},
            "repetitions_per_case": 5,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "milestone_passed": True,
            "case_sets": {"train": [], "validation": [], "milestone": ["case-d"]},
            "artifact_refs": ["evaluation-summary.json"],
            "reproduction_command": ["echo", "test"],
        }
        (root / "evaluation-summary.json").write_text(json.dumps(eval_summary))
        rs = {
            "schema_version": 1,
            "campaign_id": "test-campaign",
            "candidate_id": "cand-1",
            "candidate_fingerprint": SEED_FINGERPRINT,
            "models": ["qwen3:0.6b"],
            "aggregate_score": 1.0,
            "model_scores": {"qwen3:0.6b": 1.0},
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "systemic_failures": 0,
            "promotion_eligible": True,
            "promotion_blockers": [],
            "status_counts": {"completed": 1},
            "hard_failure_counts": {},
            "runs": [],
            "artifact_refs": ["round-summary.json", "evaluation-summary.json"],
            "evaluation_artifact_refs": ["evaluation-summary.json"],
            "prompt_lab_revision": "abc123",
            "korvid_revision": "def456",
            "workflow_run_url": "",
            "reproduction_command": [],
            "campaign_action_id": "ms-1",
        }
        (root / "round-summary.json").write_text(json.dumps(rs))
        # Add forbidden optimization file
        (root / "optimization-summary.json").write_text("{}")
        with pytest.raises(ValueError, match="must not contain"):
            load_round_outcome(root, action, control=ctrl, state=st)

    def test_milestone_rejects_comparison_summary(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        root.mkdir(parents=True)
        ctrl = _control()
        st = _state()
        action = CampaignAction(
            action_id="ms-1",
            kind=ActionKind.MILESTONE,
            expected_state_hash="sha256:" + "0" * 64,
            tier_index=0,
            metric_calls=0,
        )
        eval_summary = {
            "bundle_kind": "milestone",
            "candidate_id": "cand-1",
            "candidate_fingerprint": SEED_FINGERPRINT,
            "campaign_id": "test-campaign",
            "campaign_case_ids": ["case-d"],
            "evaluated_case_ids": ["case-d"],
            "evaluated_models": ["qwen3:0.6b"],
            "campaign_case_model_pairs": ["case-d:qwen3:0.6b"],
            "evaluated_case_model_pairs": ["case-d:qwen3:0.6b"],
            "aggregate_score": 1.0,
            "model_scores": {"qwen3:0.6b": 1.0},
            "execution_modes": ["live"],
            "run_execution_modes": {"case-d:qwen3:0.6b": "live"},
            "repetitions_per_case": 5,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "milestone_passed": True,
            "case_sets": {"train": [], "validation": [], "milestone": ["case-d"]},
            "artifact_refs": ["evaluation-summary.json"],
            "reproduction_command": ["echo", "test"],
        }
        (root / "evaluation-summary.json").write_text(json.dumps(eval_summary))
        round_summary = {
            "schema_version": 1,
            "campaign_id": "test-campaign",
            "candidate_id": "cand-1",
            "candidate_fingerprint": SEED_FINGERPRINT,
            "models": ["qwen3:0.6b"],
            "aggregate_score": 1.0,
            "model_scores": {"qwen3:0.6b": 1.0},
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "systemic_failures": 0,
            "promotion_eligible": True,
            "promotion_blockers": [],
            "status_counts": {"completed": 1},
            "hard_failure_counts": {},
            "runs": [],
            "artifact_refs": ["round-summary.json", "evaluation-summary.json"],
            "evaluation_artifact_refs": ["evaluation-summary.json"],
            "prompt_lab_revision": "abc123",
            "korvid_revision": "def456",
            "workflow_run_url": "",
            "reproduction_command": [],
            "campaign_action_id": "ms-1",
        }
        (root / "round-summary.json").write_text(json.dumps(round_summary))
        comparison_summary = {
            "schema_version": 1,
            "status": "unchanged",
            "outcome": "unchanged",
            "seed_candidate_fingerprint": SEED_FINGERPRINT,
            "best_candidate_fingerprint": SEED_FINGERPRINT,
            "contract": {
                "campaign_id": "test-campaign",
                "models": ["qwen3:0.6b"],
                "case_repetitions": [["case-d", "qwen3:0.6b", 5]],
                "execution_modes": ["live"],
            },
            "metrics": [],
            "improved_count": 0,
            "unchanged_count": 0,
            "regressed_count": 0,
            "not_comparable_count": 0,
        }
        (root / "comparison-summary.json").write_text(json.dumps(comparison_summary))
        with pytest.raises(ValueError, match="comparison-summary.json"):
            load_round_outcome(root, action, control=ctrl, state=st)


# ---------------------------------------------------------------------------
# Rendering Tests
# ---------------------------------------------------------------------------


class TestRenderCampaignSummary:
    def test_running_decision_surface(self, tmp_path: Path) -> None:
        ctrl = _control()
        st = _state(metric_calls_used=12, stage_index=1, seed_index=1)
        path = write_campaign_artifacts(st, tmp_path / "safe", ctrl)
        md = (path / "campaign-summary.md").read_text()
        assert md.startswith("# Optimization Campaign Outcome\n\n## 🔄 RUNNING — refine stage")
        assert "Budget: 12 / 240 metric calls" in md
        assert "- Next: refine seed 1 with 24 metric calls" in md
        assert "## Failure movement" in md
        assert md.rstrip().endswith("No failure data available")

    def test_render_campaign_summary_has_exact_stage_suffix(self) -> None:
        summary = render_campaign_summary(
            _state(metric_calls_used=12, stage_index=1, seed_index=1),
            _control(),
        )
        assert "## 🔄 RUNNING — refine stage" in summary
        assert "## Failure movement" in summary
        assert summary.rstrip().endswith("No failure data available")

    def test_render_campaign_summary_handles_post_search_running_state(self) -> None:
        ctrl = _control()
        summary = render_campaign_summary(
            _state(stage_index=len(ctrl.stages), seed_index=0),
            ctrl,
        )
        assert summary.startswith("# Optimization Campaign Outcome")
        assert "## 🔄 RUNNING" in summary
        assert "milestone evaluation" in summary

    def test_qualified_status(self, tmp_path: Path) -> None:
        ctrl = _control()
        st = CampaignState(
            schema_version=1,
            campaign_id="test-campaign",
            prompt_lab_revision="abc123",
            korvid_revision="def456",
            status=CampaignStatus.QUALIFIED,
            tier_index=0,
            stage_index=2,
            seed_index=3,
            champion_fingerprint="fp_q",
            seed_candidate_fingerprint=SEED_FINGERPRINT,
            champion_score=CampaignScore(
                fingerprint="fp_q", aggregate=0.9,
                hard_safety_failures=0, core_regression=False,
                systemic_failures=0, pass_at_3=1.0, pass_at_5=1.0,
            ),
            model_identity=ModelIdentity(
                name="small", model="qwen3:0.6b", digest=DIGEST_A,
            ),
            metric_calls_used=200, elapsed_seconds=3600.0,
            stagnation_attempts=0, retries_used=0,
            started_at="2026-01-15T12:00:00+00:00",
            milestone_passed=True, confirmations_passed=1,
        )
        path = write_campaign_artifacts(st, tmp_path / "safe", ctrl)
        md = (path / "campaign-summary.md").read_text()
        assert "## ✅ QUALIFIED" in md
        assert "## ✅ QUALIFIED —" not in md

    def test_rejects_existing_output(self, tmp_path: Path) -> None:
        out = tmp_path / "safe"
        out.mkdir()
        with pytest.raises(FileExistsError):
            write_campaign_artifacts(_state(), out, _control())


# ---------------------------------------------------------------------------
# CAS Tests
# ---------------------------------------------------------------------------


class TestCompareAndSwap:
    def test_initial_write(self, tmp_path: Path) -> None:
        st = _state()
        path = tmp_path / "state.json"
        h = state_hash(st)
        # For initial write, prior hash is not checked against file
        write_campaign_state(st, path, expected_prior_hash=h)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["state_hash"] == h

    def test_rejects_stale_hash(self, tmp_path: Path) -> None:
        st = _state()
        path = tmp_path / "state.json"
        h = state_hash(st)
        write_campaign_state(st, path, expected_prior_hash=h)
        with pytest.raises(ValueError, match="stale"):
            write_campaign_state(st, path, expected_prior_hash="sha256:" + "f" * 64)

    def test_concurrent_cas_exactly_one_wins(self, tmp_path: Path) -> None:
        st1 = _state()
        path = tmp_path / "state.json"
        prior_hash = state_hash(st1)
        write_campaign_state(st1, path, expected_prior_hash=prior_hash)

        barrier = threading.Barrier(2)
        cas_barrier = threading.Barrier(2)
        original_read_text = Path.read_text
        results: list[str] = []
        errors: list[Exception] = []

        def synchronized_read_text(
            self: Path, *args: Any, **kwargs: Any,
        ) -> str:
            content = original_read_text(self, *args, **kwargs)
            if self.resolve() == path.resolve():
                try:
                    cas_barrier.wait(timeout=1)
                except threading.BrokenBarrierError:
                    pass
            return content

        def worker(next_state: CampaignState) -> None:
            barrier.wait()
            try:
                write_campaign_state(next_state, path, expected_prior_hash=prior_hash)
            except Exception as exc:  # noqa: BLE001 - asserting exact failure below
                errors.append(exc)
            else:
                results.append(state_hash(next_state))

        threads = [
            threading.Thread(target=worker, args=(_state(metric_calls_used=12),)),
            threading.Thread(target=worker, args=(_state(metric_calls_used=24),)),
        ]
        with patch("pathlib.Path.read_text", new=synchronized_read_text):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert "stale" in str(errors[0])

    def test_preserves_state_on_write_failure(self, tmp_path: Path) -> None:
        """On write error, original state must remain intact."""
        st = _state()
        path = tmp_path / "state.json"
        h = state_hash(st)
        write_campaign_state(st, path, expected_prior_hash=h)
        original_content = path.read_text()

        # Inject failure during atomic replace
        st2 = _state(metric_calls_used=12)
        new_hash = state_hash(st)
        with patch("os.replace", side_effect=OSError("injected")):  # noqa: SIM117
            with pytest.raises(OSError, match="injected"):
                write_campaign_state(st2, path, expected_prior_hash=new_hash)

        # Original state preserved
        assert path.read_text() == original_content
        # No temp file left
        assert not list(path.parent.glob(f"{path.name}.*.tmp"))

    def test_no_temp_leftovers_on_write_failure(self, tmp_path: Path) -> None:
        """Temp files must be cleaned on failure."""
        st = _state()
        path = tmp_path / "new_state.json"
        h = state_hash(st)
        with patch("os.replace", side_effect=OSError("disk full")), pytest.raises(OSError):
            write_campaign_state(st, path, expected_prior_hash=h)
        assert not list(path.parent.glob(f"{path.name}.*.tmp"))


class TestCaseRepetitionsCartesian:
    """Full Cartesian case_repetitions validation with N=5 repetitions."""

    def test_full_cartesian_product_success(self, tmp_path: Path) -> None:
        """2 cases × 5 reps = 10 triplets in sorted order passes."""
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        # _write_search_evidence already produces the full Cartesian product
        outcome = load_round_outcome(root, _search_action(), control=_control(), state=_state())
        assert outcome.aggregate_score == 0.6

    def test_missing_one_triplet_rejected(self, tmp_path: Path) -> None:
        """Omitting one case/repetition triplet is rejected."""
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        comparison = json.loads((root / "comparison-summary.json").read_text())
        # Remove last triplet
        comparison["contract"]["case_repetitions"] = comparison["contract"]["case_repetitions"][:-1]
        (root / "comparison-summary.json").write_text(json.dumps(comparison))
        with pytest.raises(ValueError, match="does not match expected Cartesian set"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_duplicate_triplet_rejected(self, tmp_path: Path) -> None:
        """Duplicate triplets are rejected."""
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        comparison = json.loads((root / "comparison-summary.json").read_text())
        reps = comparison["contract"]["case_repetitions"]
        reps.append(reps[0])  # duplicate first
        (root / "comparison-summary.json").write_text(json.dumps(comparison))
        with pytest.raises(ValueError, match="duplicate triplet"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_wrong_repetition_out_of_range_rejected(self, tmp_path: Path) -> None:
        """Repetition 0 or > N is rejected."""
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        comparison = json.loads((root / "comparison-summary.json").read_text())
        reps = comparison["contract"]["case_repetitions"]
        reps[0][2] = 0  # out of range
        (root / "comparison-summary.json").write_text(json.dumps(comparison))
        with pytest.raises(ValueError, match="must be positive"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_repetition_above_n_rejected(self, tmp_path: Path) -> None:
        """Repetition 6 with repetitions_per_case=5 is out of range."""
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        comparison = json.loads((root / "comparison-summary.json").read_text())
        reps = comparison["contract"]["case_repetitions"]
        reps[0][2] = 6  # above N=5
        (root / "comparison-summary.json").write_text(json.dumps(comparison))
        with pytest.raises(ValueError, match="out of range"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())

    def test_wrong_model_in_triplet_rejected(self, tmp_path: Path) -> None:
        """Wrong model in a triplet is rejected."""
        root = tmp_path / "evidence"
        _write_search_evidence(root)
        comparison = json.loads((root / "comparison-summary.json").read_text())
        reps = comparison["contract"]["case_repetitions"]
        reps[0][1] = "wrong-model"
        (root / "comparison-summary.json").write_text(json.dumps(comparison))
        with pytest.raises(ValueError, match="model mismatch"):
            load_round_outcome(root, _search_action(), control=_control(), state=_state())


# ---------------------------------------------------------------------------
# Core-metric regression derivation (final review finding 2)
# ---------------------------------------------------------------------------


_REGRESSED_CORE_METRICS: list[dict[str, object]] = [
    {
        "key": "aggregate_score",
        "label": "Aggregate score",
        "before": 1.0,
        "after": 0.9,
        "delta": -0.1,
        "result": "regressed",
        "integer": False,
        "core": True,
    },
    {
        "key": "pass_at_3",
        "label": "pass@3",
        "before": 1.0,
        "after": 0.2,
        "delta": -0.8,
        "result": "regressed",
        "integer": False,
        "core": True,
    },
]


class TestCoreRegressionDerivation:
    def test_core_regression_is_derived_from_comparison_metrics(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(
            root,
            comparison_outcome="regressed",
            comparison_metrics=_REGRESSED_CORE_METRICS,
            before_passed=True,
        )
        outcome = load_round_outcome(
            root, _search_action(), control=_control(), state=_state(),
        )
        assert outcome.core_regression is True

    def test_no_core_regression_when_only_non_core_metric_regresses(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(
            root,
            comparison_outcome="improved",
            comparison_metrics=[
                {
                    "key": "aggregate_score",
                    "label": "Aggregate score",
                    "before": 0.4,
                    "after": 0.6,
                    "delta": 0.2,
                    "result": "improved",
                    "integer": False,
                    "core": True,
                },
                {
                    "key": "write_before_fresh_read",
                    "label": "write_before_fresh_read",
                    "before": 0,
                    "after": 2,
                    "delta": 2,
                    "result": "regressed",
                    "integer": True,
                    "core": False,
                },
            ],
        )
        outcome = load_round_outcome(
            root, _search_action(), control=_control(), state=_state(),
        )
        assert outcome.core_regression is False

    def test_summary_outcome_contradicting_core_metrics_is_rejected(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(
            root,
            comparison_outcome="improved",
            comparison_metrics=_REGRESSED_CORE_METRICS,
            before_passed=True,
        )
        with pytest.raises(ValueError, match="status|contradicts"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state(),
            )

    def test_unchanged_status_with_regressed_core_metric_is_rejected(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(
            root,
            comparison_outcome="unchanged",
            comparison_metrics=_REGRESSED_CORE_METRICS,
            before_passed=True,
        )
        comparison = json.loads((root / "comparison-summary.json").read_text())
        comparison["status"] = "unchanged"
        (root / "comparison-summary.json").write_text(json.dumps(comparison))
        with pytest.raises(ValueError, match="status|contradicts"):
            load_round_outcome(
                root, _search_action(), control=_control(), state=_state(),
            )

    def test_core_regression_blocks_promotion_through_real_advance(
        self, tmp_path: Path,
    ) -> None:
        """Ingestion -> advance: a core regression can never promote."""
        from korvid_prompt_lab.campaigns import (
            AttemptOutcome,
            advance_state,
            next_action,
        )

        control = _control()
        champion = Candidate.from_mapping(_candidate_mapping("champion")).fingerprint
        state = replace(
            _state(champion_fingerprint=champion),
            champion_score=CampaignScore(
                fingerprint=champion,
                aggregate=1.0,
                hard_safety_failures=0,
                core_regression=False,
                systemic_failures=0,
                pass_at_3=1.0,
                pass_at_5=1.0,
            ),
        )
        action = next_action(control, state, NOW)
        assert action is not None

        root = tmp_path / "evidence"
        _write_search_evidence(
            root,
            campaign_action_id=action.action_id,
            seed_candidate_fingerprint=champion,
            comparison_outcome="regressed",
            comparison_metrics=_REGRESSED_CORE_METRICS,
            before_passed=True,
        )
        outcome = load_round_outcome(root, action, control=control, state=state)
        assert outcome.core_regression is True

        score = CampaignScore(
            fingerprint=outcome.candidate_fingerprint,
            aggregate=0.9,
            hard_safety_failures=outcome.hard_safety_failures,
            core_regression=outcome.core_regression,
            systemic_failures=outcome.systemic_failures,
            pass_at_3=outcome.pass_at_3,
            pass_at_5=outcome.pass_at_5,
        )
        advanced = advance_state(
            control,
            state,
            action,
            AttemptOutcome(
                kind="evidence",
                score=score,
                search_improved=outcome.search_improved,
            ),
            NOW,
        )
        assert advanced.champion_fingerprint == champion
        assert advanced.champion_score.aggregate == 1.0
