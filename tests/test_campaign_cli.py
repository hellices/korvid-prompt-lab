"""Tests for the korvid-campaign CLI (Task 4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.campaign_cli import main


class TestCLIPlan:
    def test_plan_writes_action_json(self, tmp_path: Path) -> None:
        """plan subcommand emits next action as JSON."""
        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        output = tmp_path / "action.json"

        _write_minimal_control(control_path)
        _write_minimal_state(state_path)

        rc = main([
            "plan",
            "--control", str(control_path),
            "--state", str(state_path),
            "--output", str(output),
        ])
        assert rc == 0
        action = json.loads(output.read_text())
        assert "action_id" in action
        assert "kind" in action

    def test_plan_terminal_state(self, tmp_path: Path) -> None:
        """plan on terminal state exits 0 with empty action."""
        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        output = tmp_path / "action.json"

        _write_minimal_control(control_path)
        _write_terminal_state(state_path)

        rc = main([
            "plan",
            "--control", str(control_path),
            "--state", str(state_path),
            "--output", str(output),
        ])
        assert rc == 0
        action = json.loads(output.read_text())
        assert action.get("terminal") is True


class TestCLIRender:
    def test_render_writes_markdown(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        output_dir = tmp_path / "out"

        _write_minimal_state(state_path)

        rc = main([
            "render",
            "--state", str(state_path),
            "--output-dir", str(output_dir),
            "--total-metric-call-limit", "240",
            "--wall-clock-limit-seconds", "21600",
            "--stages-count", "3",
        ])
        assert rc == 0
        assert (output_dir / "campaign-summary.md").exists()


class TestCLIAdvance:
    def test_advance_updates_state(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        evidence_path = tmp_path / "evidence"
        output_state = tmp_path / "new-state.json"

        _write_minimal_control(control_path)
        _write_minimal_state(state_path)

        # Get the planned action first
        action_path = tmp_path / "action.json"
        main([
            "plan",
            "--control", str(control_path),
            "--state", str(state_path),
            "--output", str(action_path),
        ])
        action = json.loads(action_path.read_text())

        # Build evidence matching the action
        _write_evidence_for_action(evidence_path, action)

        rc = main([
            "advance",
            "--control", str(control_path),
            "--state", str(state_path),
            "--action", str(action_path),
            "--evidence", str(evidence_path),
            "--output-state", str(output_state),
        ])
        assert rc == 0
        new_state = json.loads(output_state.read_text())
        assert new_state["status"] == "running"

    def test_advance_rejects_stale_state(self, tmp_path: Path) -> None:
        """advance with wrong expected_state_hash fails."""
        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        evidence_path = tmp_path / "evidence"
        output_state = tmp_path / "new-state.json"

        _write_minimal_control(control_path)
        _write_minimal_state(state_path)

        # Build a fake action with wrong hash
        action_path = tmp_path / "action.json"
        action_path.write_text(json.dumps({
            "action_id": "wrong",
            "kind": "search",
            "expected_state_hash": "sha256:" + "f" * 64,
            "stage_index": 0,
            "seed_index": 0,
            "tier_index": 0,
            "metric_calls": 12,
        }))
        _write_evidence_for_action(evidence_path, json.loads(action_path.read_text()))

        rc = main([
            "advance",
            "--control", str(control_path),
            "--state", str(state_path),
            "--action", str(action_path),
            "--evidence", str(evidence_path),
            "--output-state", str(output_state),
        ])
        assert rc != 0


class TestCLIGithubOutput:
    def test_ignores_env_github_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI must not trust GITHUB_OUTPUT env var."""
        evil = tmp_path / "evil-output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(evil))

        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        output = tmp_path / "action.json"

        _write_minimal_control(control_path)
        _write_minimal_state(state_path)

        rc = main([
            "plan",
            "--control", str(control_path),
            "--state", str(state_path),
            "--output", str(output),
        ])
        assert rc == 0
        assert not evil.exists()

    def test_writes_to_explicit_github_output(self, tmp_path: Path) -> None:
        """--github-output is the only path the CLI will write GH outputs to."""
        gh_out = tmp_path / "gh-output.txt"
        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        output = tmp_path / "action.json"

        _write_minimal_control(control_path)
        _write_minimal_state(state_path)

        rc = main([
            "plan",
            "--control", str(control_path),
            "--state", str(state_path),
            "--output", str(output),
            "--github-output", str(gh_out),
        ])
        assert rc == 0
        assert gh_out.exists()
        content = gh_out.read_text()
        assert "action_id=" in content or "terminal=" in content


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

import yaml  # type: ignore[import-untyped]


def _write_minimal_control(path: Path) -> None:
    """Write a minimal optimization campaign control file."""
    control = {
        "schema_version": 1,
        "campaign_id": "test-campaign",
        "evaluation_campaign": "eval-campaign",
        "initial_candidate": "seed.yaml",
        "train_case_ids": ["case-a", "case-b"],
        "validation_case_ids": ["case-c"],
        "milestone_case_ids": ["case-d"],
        "stages": [
            {"name": "explore", "metric_calls": 12, "seeds": [0, 1, 2]},
            {"name": "refine", "metric_calls": 24, "seeds": [3, 4]},
        ],
        "model_tiers": [
            {"name": "small", "model": "qwen3:0.6b", "digest": "sha256:" + "a" * 64},
        ],
        "total_metric_call_limit": 240,
        "wall_clock_limit_seconds": 21600,
        "infrastructure_retry_limit": 3,
        "stagnation_attempt_limit": 30,
        "confirmation_runs": 1,
    }
    path.write_text(yaml.dump(control))


def _write_minimal_state(path: Path) -> None:
    """Write a minimal running campaign state file."""
    from datetime import UTC, datetime

    from korvid_prompt_lab.campaigns import (
        CampaignScore,
        CampaignState,
        CampaignStatus,
        ModelIdentity,
    )
    # Use a recent started_at so wall clock doesn't expire during test
    started_at = datetime.now(tz=UTC).isoformat()
    state = CampaignState(
        schema_version=1,
        campaign_id="test-campaign",
        prompt_lab_revision="abc123",
        korvid_revision="def456",
        status=CampaignStatus.RUNNING,
        tier_index=0,
        stage_index=0,
        seed_index=0,
        champion_fingerprint="seed.yaml",
        champion_score=CampaignScore(
            fingerprint="seed.yaml",
            aggregate=0.0,
            hard_safety_failures=0,
            core_regression=False,
            systemic_failures=0,
            pass_at_3=0.0,
            pass_at_5=0.0,
        ),
        model_identity=ModelIdentity(name="small", model="qwen3:0.6b", digest="sha256:" + "a" * 64),
        metric_calls_used=0,
        elapsed_seconds=0.0,
        stagnation_attempts=0,
        retries_used=0,
        started_at=started_at,
    )
    _write_state_to_path(state, path)


def _write_terminal_state(path: Path) -> None:
    from korvid_prompt_lab.campaigns import (
        CampaignScore,
        CampaignState,
        CampaignStatus,
        ModelIdentity,
    )
    state = CampaignState(
        schema_version=1,
        campaign_id="test-campaign",
        prompt_lab_revision="abc123",
        korvid_revision="def456",
        status=CampaignStatus.QUALIFIED,
        tier_index=0,
        stage_index=2,
        seed_index=3,
        champion_fingerprint="qualified_fp",
        champion_score=CampaignScore(
            fingerprint="qualified_fp",
            aggregate=0.9,
            hard_safety_failures=0,
            core_regression=False,
            systemic_failures=0,
            pass_at_3=1.0,
            pass_at_5=1.0,
        ),
        model_identity=ModelIdentity(name="small", model="qwen3:0.6b", digest="sha256:" + "a" * 64),
        metric_calls_used=200,
        elapsed_seconds=3600.0,
        stagnation_attempts=0,
        retries_used=0,
        started_at="2026-01-15T12:00:00+00:00",
        milestone_passed=True,
        confirmations_passed=1,
    )
    _write_state_to_path(state, path)


def _write_state_to_path(state, path: Path) -> None:
    from korvid_prompt_lab.campaigns import state_hash
    data = {
        "schema_version": state.schema_version,
        "campaign_id": state.campaign_id,
        "prompt_lab_revision": state.prompt_lab_revision,
        "korvid_revision": state.korvid_revision,
        "status": state.status.value,
        "tier_index": state.tier_index,
        "stage_index": state.stage_index,
        "seed_index": state.seed_index,
        "champion_fingerprint": state.champion_fingerprint,
        "champion_score": {
            "fingerprint": state.champion_score.fingerprint,
            "aggregate": state.champion_score.aggregate,
            "hard_safety_failures": state.champion_score.hard_safety_failures,
            "core_regression": state.champion_score.core_regression,
            "systemic_failures": state.champion_score.systemic_failures,
            "pass_at_3": state.champion_score.pass_at_3,
            "pass_at_5": state.champion_score.pass_at_5,
        },
        "model_identity": {
            "name": state.model_identity.name,
            "model": state.model_identity.model,
            "digest": state.model_identity.digest,
        },
        "metric_calls_used": state.metric_calls_used,
        "elapsed_seconds": state.elapsed_seconds,
        "stagnation_attempts": state.stagnation_attempts,
        "retries_used": state.retries_used,
        "started_at": state.started_at,
        "pending_action_id": state.pending_action_id,
        "milestone_passed": state.milestone_passed,
        "confirmations_passed": state.confirmations_passed,
        "stop_reason": state.stop_reason,
        "tier_results": [],
        "state_hash": state_hash(state),
    }
    path.write_text(json.dumps(data))


def _write_evidence_for_action(evidence_path: Path, action: dict) -> None:
    """Write minimal evidence matching an action."""
    evidence_path.mkdir(parents=True, exist_ok=True)
    eval_summary = {
        "bundle_kind": "validation",
        "candidate_id": "cand-1",
        "candidate_fingerprint": "new_fp",
        "campaign_id": "test-campaign",
        "campaign_case_ids": ["case-c"],
        "evaluated_case_ids": ["case-c"],
        "evaluated_models": ["qwen3:0.6b"],
        "campaign_case_model_pairs": ["case-c:qwen3:0.6b"],
        "evaluated_case_model_pairs": ["case-c:qwen3:0.6b"],
        "aggregate_score": 0.6,
        "model_scores": {"qwen3:0.6b": 0.6},
        "execution_modes": ["live"],
        "run_execution_modes": {"case-c:qwen3:0.6b": "live"},
        "repetitions_per_case": 5,
        "pass_at_3": 1.0,
        "pass_at_5": 1.0,
        "hard_safety_failures": 0,
        "systemic_failures": 0,
        "milestone_passed": False,
        "case_sets": {"train": [], "validation": ["case-c"], "milestone": []},
        "artifact_refs": ["evaluation-summary.json"],
        "reproduction_command": ["echo", "test"],
    }
    (evidence_path / "evaluation-summary.json").write_text(json.dumps(eval_summary))

    round_summary = {
        "schema_version": 1,
        "campaign_id": "test-campaign",
        "candidate_id": "cand-1",
        "candidate_fingerprint": "new_fp",
        "models": ["qwen3:0.6b"],
        "aggregate_score": 0.6,
        "model_scores": {"qwen3:0.6b": 0.6},
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
        "workflow_run_url": "https://github.com/example/actions/runs/1",
        "reproduction_command": ["echo", "test"],
        "action_id": action.get("action_id", "unknown"),
    }
    (evidence_path / "round-summary.json").write_text(json.dumps(round_summary))
