"""Tests for safe campaign evidence ingestion and artifact rendering (Task 4)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.campaign_artifacts import (
    load_round_outcome,
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
    state_hash,
)

DIGEST_A = "sha256:" + "a" * 64
NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _control() -> OptimizationCampaign:
    return OptimizationCampaign(
        schema_version=1,
        campaign_id="test-campaign",
        evaluation_campaign="eval-campaign",
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


def _state(
    *,
    champion_fingerprint: str = "seed.yaml",
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


def _write_search_evidence(
    root: Path,
    *,
    action_id: str = "action-1",
    candidate_fingerprint: str = "new_fp",
    evaluated_case_ids: tuple[str, ...] = ("case-c",),
    models: tuple[str, ...] = ("qwen3:0.6b",),
    seed: int = 0,
    seed_candidate_fingerprint: str = "seed.yaml",
    total_metric_calls: int = 10,
    max_metric_calls: int = 12,
    train_case_ids: tuple[str, ...] = ("case-a", "case-b"),
    validation_case_ids: tuple[str, ...] = ("case-c",),
    prompt_lab_revision: str = "abc123",
    korvid_revision: str = "def456",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    eval_summary = {
        "bundle_kind": "validation",
        "candidate_id": "cand-1",
        "candidate_fingerprint": candidate_fingerprint,
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

    round_summary = {
        "schema_version": 1,
        "campaign_id": "test-campaign",
        "candidate_id": "cand-1",
        "candidate_fingerprint": candidate_fingerprint,
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
        "artifact_refs": ["round-summary.json", "evaluation-summary.json"],
        "evaluation_artifact_refs": ["evaluation-summary.json"],
        "prompt_lab_revision": prompt_lab_revision,
        "korvid_revision": korvid_revision,
        "workflow_run_url": "https://github.com/example/actions/runs/1",
        "reproduction_command": ["echo", "test"],
        "action_id": action_id,
    }
    (root / "round-summary.json").write_text(json.dumps(round_summary))

    opt_summary = {
        "run_id": "run-001",
        "seed": seed,
        "run_identity": {
            "schema_version": 1,
            "campaign_id": "test-campaign",
            "candidate_id": "cand-1",
            "seed_candidate_fingerprint": seed_candidate_fingerprint,
            "train_case_ids": list(train_case_ids),
            "validation_case_ids": list(validation_case_ids),
            "max_metric_calls": max_metric_calls,
            "seed": seed,
            "proposal_source": "dspy",
        },
        "invocation_dir": "/tmp/run",
        "best_idx": 1,
        "best_validation_score": 0.6,
        "best_candidate_fingerprint": candidate_fingerprint,
        "seed_candidate_fingerprint": seed_candidate_fingerprint,
        "best_candidate_differs_from_seed": True,
        "train_case_ids": list(train_case_ids),
        "validation_case_ids": list(validation_case_ids),
        "execution_modes": ["live"],
        "num_candidates": 5,
        "total_metric_calls": total_metric_calls,
        "num_full_val_evals": 2,
        "run_dir": "/tmp/run",
    }
    (root / "optimization-summary.json").write_text(json.dumps(opt_summary))

    bc = {"schema_version": 1, "candidate_id": "cand-1", "components": {}}
    (root / "best-candidate.yaml").write_text(yaml.dump(bc))


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
        outcome = load_round_outcome(root, action, control=ctrl, state=st)
        assert outcome.candidate_fingerprint == "new_fp"
        assert outcome.aggregate_score == 0.6

    def test_rejects_wrong_action_id(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, action_id="action-1")
        action = _search_action(action_id="action-WRONG")
        with pytest.raises(ValueError, match="action_id mismatch"):
            load_round_outcome(root, action, control=_control(), state=_state())

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

    def test_search_rejects_wrong_seed(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, seed=99)
        action = _search_action()
        with pytest.raises(ValueError, match="seed mismatch"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_search_rejects_metric_budget_exceeded(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, total_metric_calls=999)
        action = _search_action()
        with pytest.raises(ValueError, match="exceeds action budget"):
            load_round_outcome(root, action, control=_control(), state=_state())

    def test_search_rejects_wrong_seed_fingerprint(self, tmp_path: Path) -> None:
        root = tmp_path / "evidence"
        _write_search_evidence(root, seed_candidate_fingerprint="wrong")
        action = _search_action()
        with pytest.raises(ValueError, match="seed_candidate_fingerprint mismatch"):
            load_round_outcome(root, action, control=_control(), state=_state())

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
            "candidate_fingerprint": "seed.yaml",
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
            "candidate_fingerprint": "seed.yaml",
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
            "artifact_refs": [],
            "evaluation_artifact_refs": [],
            "prompt_lab_revision": "abc123",
            "korvid_revision": "def456",
            "workflow_run_url": "",
            "reproduction_command": [],
            "action_id": "ms-1",
        }
        (root / "round-summary.json").write_text(json.dumps(rs))
        # Add forbidden optimization file
        (root / "optimization-summary.json").write_text("{}")
        with pytest.raises(ValueError, match="must not contain"):
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
        assert md.startswith("# Optimization Campaign Outcome\n\n## 🔄 RUNNING")
        assert "Budget: 12 / 240 metric calls" in md
        assert "Next:" in md

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

    def test_concurrent_second_write_fails(self, tmp_path: Path) -> None:
        """Two workers with same prior: first succeeds, second sees new hash."""
        st1 = _state()
        path = tmp_path / "state.json"
        h1 = state_hash(st1)
        write_campaign_state(st1, path, expected_prior_hash=h1)

        # Worker A advances
        st2 = _state(metric_calls_used=12)
        h_loaded = state_hash(st1)  # both loaded the same prior
        write_campaign_state(st2, path, expected_prior_hash=h_loaded)

        # Worker B tries same prior — now stale
        st3 = _state(metric_calls_used=24)
        with pytest.raises(ValueError, match="stale"):
            write_campaign_state(st3, path, expected_prior_hash=h_loaded)

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
        with patch("pathlib.Path.replace", side_effect=OSError("injected")):  # noqa: SIM117
            with pytest.raises(OSError, match="injected"):
                write_campaign_state(st2, path, expected_prior_hash=new_hash)

        # Original state preserved
        assert path.read_text() == original_content
        # No temp file left
        assert not path.with_suffix(".cas_tmp").exists()

    def test_no_temp_leftovers_on_write_failure(self, tmp_path: Path) -> None:
        """Temp files must be cleaned on failure."""
        st = _state()
        path = tmp_path / "new_state.json"
        h = state_hash(st)
        with patch(
            "pathlib.Path.write_text", side_effect=OSError("disk full"),
        ), pytest.raises(OSError):
            write_campaign_state(st, path, expected_prior_hash=h)
        assert not path.with_suffix(".cas_tmp").exists()
