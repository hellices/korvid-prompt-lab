"""Tests for safe campaign evidence ingestion and artifact rendering (Task 4)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.campaign_artifacts import (
    load_round_outcome,
    write_campaign_artifacts,
)
from korvid_prompt_lab.campaigns import (
    ActionKind,
    CampaignAction,
    CampaignScore,
    CampaignState,
    CampaignStatus,
    ModelIdentity,
    state_hash,
)

DIGEST_A = "sha256:" + "a" * 64
NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _running_state(
    *,
    metric_calls_used: int = 12,
    elapsed_seconds: float = 2533.0,
    stage_index: int = 1,
    seed_index: int = 1,
    champion_fingerprint: str = "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7",
    aggregate: float = 0.120,
    pass_at_3: float = 0.200,
    pass_at_5: float = 0.0,
    hard_safety_failures: int = 0,
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
            aggregate=aggregate,
            hard_safety_failures=hard_safety_failures,
            core_regression=False,
            systemic_failures=0,
            pass_at_3=pass_at_3,
            pass_at_5=pass_at_5,
        ),
        model_identity=ModelIdentity(name="small", model="qwen3:0.6b", digest=DIGEST_A),
        metric_calls_used=metric_calls_used,
        elapsed_seconds=elapsed_seconds,
        stagnation_attempts=3,
        retries_used=0,
        started_at="2026-01-15T12:00:00+00:00",
    )


def _safe_round(
    tmp_path: Path,
    *,
    case_ids: tuple[str, ...] = ("case-a", "case-b"),
    candidate_fingerprint: str = "fp1234",
    aggregate_score: float = 0.5,
    pass_at_3: float = 1.0,
    pass_at_5: float = 1.0,
    hard_safety_failures: int = 0,
    systemic_failures: int = 0,
    milestone_passed: bool = False,
    models: tuple[str, ...] = ("qwen3:0.6b",),
    action_id: str = "action-1",
) -> Path:
    """Create a minimal safe-evidence directory for ingestion."""
    root = tmp_path / "safe-evidence"
    root.mkdir(parents=True)

    eval_summary = {
        "bundle_kind": "validation",
        "candidate_id": "cand-1",
        "candidate_fingerprint": candidate_fingerprint,
        "campaign_id": "test-campaign",
        "campaign_case_ids": list(case_ids),
        "evaluated_case_ids": list(case_ids),
        "evaluated_models": list(models),
        "campaign_case_model_pairs": [f"{c}:{m}" for c in case_ids for m in models],
        "evaluated_case_model_pairs": [f"{c}:{m}" for c in case_ids for m in models],
        "aggregate_score": aggregate_score,
        "model_scores": {m: aggregate_score for m in models},
        "execution_modes": ["live"],
        "run_execution_modes": {f"{c}:{m}": "live" for c in case_ids for m in models},
        "repetitions_per_case": 5,
        "pass_at_3": pass_at_3,
        "pass_at_5": pass_at_5,
        "hard_safety_failures": hard_safety_failures,
        "systemic_failures": systemic_failures,
        "milestone_passed": milestone_passed,
        "case_sets": {"train": [], "validation": list(case_ids), "milestone": []},
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
        "aggregate_score": aggregate_score,
        "model_scores": {m: aggregate_score for m in models},
        "pass_at_3": pass_at_3,
        "pass_at_5": pass_at_5,
        "systemic_failures": systemic_failures,
        "promotion_eligible": True,
        "promotion_blockers": [],
        "status_counts": {"completed": len(case_ids)},
        "hard_failure_counts": {},
        "runs": [],
        "artifact_refs": ["round-summary.json", "evaluation-summary.json"],
        "evaluation_artifact_refs": ["evaluation-summary.json"],
        "prompt_lab_revision": "abc123",
        "korvid_revision": "def456",
        "workflow_run_url": "https://github.com/example/actions/runs/1",
        "reproduction_command": ["echo", "test"],
        "action_id": action_id,
    }
    (root / "round-summary.json").write_text(json.dumps(round_summary))
    return root


def _search_action(
    *,
    case_ids: tuple[str, ...] = ("case-a", "case-b"),
    action_id: str = "action-1",
    candidate_fingerprint: str = "fp1234",
    model: str = "qwen3:0.6b",
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


def _validation_action(
    *,
    case_ids: tuple[str, ...] = ("validation-a",),
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


# ---------------------------------------------------------------------------
# Safe Ingestion Tests
# ---------------------------------------------------------------------------


class TestLoadRoundOutcome:
    def test_loads_valid_round(self, tmp_path: Path) -> None:
        root = _safe_round(tmp_path, case_ids=("case-a", "case-b"))
        action = _search_action(case_ids=("case-a", "case-b"))
        outcome = load_round_outcome(root, action)
        assert outcome.candidate_fingerprint == "fp1234"
        assert outcome.aggregate_score == 0.5

    def test_rejects_round_with_wrong_action_id(self, tmp_path: Path) -> None:
        root = _safe_round(tmp_path, action_id="action-1")
        action = _search_action(action_id="action-WRONG")
        with pytest.raises(ValueError, match="action_id"):
            load_round_outcome(root, action)

    def test_rejects_round_with_wrong_evaluated_case_set(self, tmp_path: Path) -> None:
        root = _safe_round(tmp_path, case_ids=("milestone-a",))
        action = _validation_action(case_ids=("validation-a",))
        # The action expects certain cases but the evidence has different ones
        with pytest.raises(ValueError, match="evaluated case set"):
            load_round_outcome(root, action, expected_case_ids=("validation-a",))

    def test_rejects_symlinked_file(self, tmp_path: Path) -> None:
        root = _safe_round(tmp_path)
        # Replace evaluation-summary.json with a symlink
        target = tmp_path / "evil.json"
        target.write_text("{}")
        real = root / "evaluation-summary.json"
        real.unlink()
        real.symlink_to(target)
        action = _search_action()
        with pytest.raises(ValueError, match="symlink"):
            load_round_outcome(root, action)

    def test_rejects_malformed_json(self, tmp_path: Path) -> None:
        root = tmp_path / "safe-evidence"
        root.mkdir()
        (root / "round-summary.json").write_text("not json{{{")
        (root / "evaluation-summary.json").write_text("{}")
        action = _search_action()
        with pytest.raises(ValueError, match="malformed|JSON|json"):
            load_round_outcome(root, action)

    def test_rejects_responses_directory(self, tmp_path: Path) -> None:
        """load_round_outcome must never read from responses/."""
        root = _safe_round(tmp_path)
        (root / "responses").mkdir()
        (root / "responses" / "evil.json").write_text("{}")
        action = _search_action()
        # Should still load fine — it just doesn't traverse responses/
        outcome = load_round_outcome(root, action)
        assert outcome.candidate_fingerprint == "fp1234"

    def test_rejects_wrong_model(self, tmp_path: Path) -> None:
        root = _safe_round(tmp_path, models=("wrong-model",))
        action = _search_action(model="qwen3:0.6b")
        with pytest.raises(ValueError, match="model"):
            load_round_outcome(root, action, expected_model="qwen3:0.6b")


# ---------------------------------------------------------------------------
# Campaign Summary Rendering Tests
# ---------------------------------------------------------------------------


class TestWriteCampaignArtifacts:
    def test_campaign_summary_leads_with_decision_surface(self, tmp_path: Path) -> None:
        state = _running_state()
        path = write_campaign_artifacts(state, tmp_path / "safe", total_metric_call_limit=240, wall_clock_limit_seconds=21600, stages_count=3)
        markdown = (path / "campaign-summary.md").read_text()
        assert markdown.startswith("# Optimization Campaign Outcome\n\n## 🔄 RUNNING")
        assert "Budget: 12 / 240 metric calls" in markdown
        assert "Next:" in markdown

    def test_renders_qualified_status(self, tmp_path: Path) -> None:
        state = CampaignState(
            schema_version=1,
            campaign_id="test-campaign",
            prompt_lab_revision="abc123",
            korvid_revision="def456",
            status=CampaignStatus.QUALIFIED,
            tier_index=0,
            stage_index=2,
            seed_index=3,
            champion_fingerprint="fp_qualified",
            champion_score=CampaignScore(
                fingerprint="fp_qualified",
                aggregate=0.9,
                hard_safety_failures=0,
                core_regression=False,
                systemic_failures=0,
                pass_at_3=1.0,
                pass_at_5=1.0,
            ),
            model_identity=ModelIdentity(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            metric_calls_used=200,
            elapsed_seconds=3600.0,
            stagnation_attempts=0,
            retries_used=0,
            started_at="2026-01-15T12:00:00+00:00",
            milestone_passed=True,
            confirmations_passed=2,
        )
        path = write_campaign_artifacts(state, tmp_path / "safe", total_metric_call_limit=240, wall_clock_limit_seconds=21600, stages_count=3)
        markdown = (path / "campaign-summary.md").read_text()
        assert "## ✅ QUALIFIED" in markdown

    def test_renders_not_converged(self, tmp_path: Path) -> None:
        state = CampaignState(
            schema_version=1,
            campaign_id="test-campaign",
            prompt_lab_revision="abc123",
            korvid_revision="def456",
            status=CampaignStatus.NOT_CONVERGED,
            tier_index=0,
            stage_index=2,
            seed_index=3,
            champion_fingerprint="fp_stale",
            champion_score=CampaignScore(
                fingerprint="fp_stale",
                aggregate=0.3,
                hard_safety_failures=2,
                core_regression=False,
                systemic_failures=0,
                pass_at_3=0.5,
                pass_at_5=0.3,
            ),
            model_identity=ModelIdentity(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            metric_calls_used=240,
            elapsed_seconds=21600.0,
            stagnation_attempts=30,
            retries_used=0,
            started_at="2026-01-15T12:00:00+00:00",
            stop_reason="total_metric_call_limit",
        )
        path = write_campaign_artifacts(state, tmp_path / "safe", total_metric_call_limit=240, wall_clock_limit_seconds=21600, stages_count=3)
        markdown = (path / "campaign-summary.md").read_text()
        assert "## ❌ NOT_CONVERGED" in markdown

    def test_renders_system_error(self, tmp_path: Path) -> None:
        state = CampaignState(
            schema_version=1,
            campaign_id="test-campaign",
            prompt_lab_revision="abc123",
            korvid_revision="def456",
            status=CampaignStatus.SYSTEM_ERROR,
            tier_index=0,
            stage_index=0,
            seed_index=0,
            champion_fingerprint="fp_err",
            champion_score=CampaignScore(
                fingerprint="fp_err",
                aggregate=0.0,
                hard_safety_failures=0,
                core_regression=False,
                systemic_failures=0,
                pass_at_3=0.0,
                pass_at_5=0.0,
            ),
            model_identity=ModelIdentity(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            metric_calls_used=0,
            elapsed_seconds=10.0,
            stagnation_attempts=0,
            retries_used=3,
            started_at="2026-01-15T12:00:00+00:00",
            stop_reason="infrastructure_retry_limit_exhausted",
        )
        path = write_campaign_artifacts(state, tmp_path / "safe", total_metric_call_limit=240, wall_clock_limit_seconds=21600, stages_count=3)
        markdown = (path / "campaign-summary.md").read_text()
        assert "## ⚠️ SYSTEM_ERROR" in markdown

    def test_writes_state_json(self, tmp_path: Path) -> None:
        state = _running_state()
        path = write_campaign_artifacts(state, tmp_path / "safe", total_metric_call_limit=240, wall_clock_limit_seconds=21600, stages_count=3)
        state_data = json.loads((path / "campaign-state.json").read_text())
        assert state_data["status"] == "running"
        assert state_data["champion_fingerprint"] == state.champion_fingerprint

    def test_rejects_existing_output(self, tmp_path: Path) -> None:
        out = tmp_path / "safe"
        out.mkdir()
        state = _running_state()
        with pytest.raises(FileExistsError):
            write_campaign_artifacts(state, out, total_metric_call_limit=240, wall_clock_limit_seconds=21600, stages_count=3)


# ---------------------------------------------------------------------------
# CAS (Compare-and-Swap) Tests
# ---------------------------------------------------------------------------


class TestCompareAndSwap:
    def test_write_state_with_expected_hash(self, tmp_path: Path) -> None:
        """Atomic state write must validate expected prior hash."""
        from korvid_prompt_lab.campaign_artifacts import write_campaign_state

        state = _running_state()
        state_path = tmp_path / "state.json"
        # First write with no prior
        write_campaign_state(state, state_path, expected_prior_hash=None)
        assert state_path.exists()

        # Second write with correct prior hash
        current_hash = state_hash(state)
        new_state = CampaignState(
            schema_version=1,
            campaign_id="test-campaign",
            prompt_lab_revision="abc123",
            korvid_revision="def456",
            status=CampaignStatus.RUNNING,
            tier_index=0,
            stage_index=1,
            seed_index=2,
            champion_fingerprint="new_fp",
            champion_score=CampaignScore(
                fingerprint="new_fp",
                aggregate=0.6,
                hard_safety_failures=0,
                core_regression=False,
                systemic_failures=0,
                pass_at_3=1.0,
                pass_at_5=1.0,
            ),
            model_identity=ModelIdentity(name="small", model="qwen3:0.6b", digest=DIGEST_A),
            metric_calls_used=24,
            elapsed_seconds=600.0,
            stagnation_attempts=0,
            retries_used=0,
            started_at="2026-01-15T12:00:00+00:00",
        )
        write_campaign_state(new_state, state_path, expected_prior_hash=current_hash)
        loaded = json.loads(state_path.read_text())
        assert loaded["champion_fingerprint"] == "new_fp"

    def test_rejects_stale_hash(self, tmp_path: Path) -> None:
        from korvid_prompt_lab.campaign_artifacts import write_campaign_state

        state = _running_state()
        state_path = tmp_path / "state.json"
        write_campaign_state(state, state_path, expected_prior_hash=None)

        with pytest.raises(ValueError, match="stale|mismatch|expected"):
            write_campaign_state(state, state_path, expected_prior_hash="sha256:" + "f" * 64)
