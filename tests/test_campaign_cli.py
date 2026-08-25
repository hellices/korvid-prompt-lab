"""Tests for the korvid-campaign CLI (Task 4)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.campaign_cli import main
from korvid_prompt_lab.campaigns import (
    CampaignScore,
    CampaignState,
    CampaignStatus,
    ModelIdentity,
    state_hash,
)
from korvid_prompt_lab.contracts import Candidate

DIGEST_A = "sha256:" + "a" * 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_control(path: Path) -> None:
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
            {"name": "small", "model": "qwen3:0.6b", "digest": DIGEST_A},
        ],
        "total_metric_call_limit": 240,
        "wall_clock_limit_seconds": 21600,
        "infrastructure_retry_limit": 3,
        "stagnation_attempt_limit": 30,
        "confirmation_runs": 1,
    }
    path.write_text(yaml.dump(control))


def _make_state(started_at: str | None = None) -> CampaignState:
    if started_at is None:
        started_at = datetime.now(tz=UTC).isoformat()
    return CampaignState(
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
        model_identity=ModelIdentity(
            name="small", model="qwen3:0.6b", digest=DIGEST_A,
        ),
        metric_calls_used=0,
        elapsed_seconds=0.0,
        stagnation_attempts=0,
        retries_used=0,
        started_at=started_at,
    )


def _write_state(path: Path, state: CampaignState | None = None) -> str:
    """Write state and return its hash."""
    if state is None:
        state = _make_state()
    from korvid_prompt_lab.campaign_artifacts import _serialize_state
    data = _serialize_state(state)
    path.write_text(json.dumps(data))
    return state_hash(state)


def _write_evidence(
    evidence_path: Path, action: dict[str, object],
) -> None:
    """Write evidence matching an action for SEARCH kind."""
    evidence_path.mkdir(parents=True, exist_ok=True)
    candidate = Candidate.from_mapping({
        "schema_version": 1,
        "candidate_id": "cand-1",
        "components": {"system": "system prompt"},
        "metadata": {},
    })
    candidate_fingerprint = candidate.fingerprint
    eval_summary = {
        "bundle_kind": "validation",
        "candidate_id": "cand-1",
        "candidate_fingerprint": candidate_fingerprint,
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
        "case_sets": {
            "train": ["case-a", "case-b"],
            "validation": ["case-c"],
            "milestone": [],
        },
        "artifact_refs": ["evaluation-summary.json"],
        "reproduction_command": ["echo", "test"],
    }
    (evidence_path / "evaluation-summary.json").write_text(
        json.dumps(eval_summary),
    )

    round_summary = {
        "schema_version": 1,
        "campaign_id": "test-campaign",
        "candidate_id": "cand-1",
        "candidate_fingerprint": candidate_fingerprint,
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
        "artifact_refs": [
            "round-summary.json",
            "evaluation-summary.json",
            "comparison-summary.json",
            "optimization-summary.json",
            "best-candidate.yaml",
        ],
        "evaluation_artifact_refs": ["evaluation-summary.json"],
        "prompt_lab_revision": "abc123",
        "korvid_revision": "def456",
        "workflow_run_url": "https://github.com/example/actions/runs/1",
        "reproduction_command": ["echo", "test"],
        "campaign_action_id": action.get("action_id", "unknown"),
    }
    (evidence_path / "round-summary.json").write_text(json.dumps(round_summary))

    comparison_summary = {
        "schema_version": 1,
        "status": "changed",
        "outcome": "improved",
        "seed_candidate_fingerprint": "seed.yaml",
        "best_candidate_fingerprint": candidate_fingerprint,
        "contract": {
            "campaign_id": "test-campaign",
            "models": ["qwen3:0.6b"],
            "case_repetitions": [["case-c", "qwen3:0.6b", 5]],
            "execution_modes": ["live"],
        },
        "metrics": [
            {
                "key": "systemic_failures",
                "label": "Systemic failures",
                "before": 0,
                "after": 0,
                "delta": 0,
                "result": "unchanged",
                "integer": True,
                "core": True,
            }
        ],
        "improved_count": 0,
        "unchanged_count": 1,
        "regressed_count": 0,
        "not_comparable_count": 0,
    }
    (evidence_path / "comparison-summary.json").write_text(json.dumps(comparison_summary))

    opt_summary = {
        "run_id": "run-001",
        "seed": 0,
        "run_identity": {
            "schema_version": 1,
            "campaign_id": "test-campaign",
            "candidate_id": "cand-1",
            "seed_candidate_fingerprint": "seed.yaml",
            "train_case_ids": ["case-a", "case-b"],
            "validation_case_ids": ["case-c"],
            "max_metric_calls": 12,
            "seed": 0,
            "proposal_source": "dspy",
        },
        "invocation_dir": "artifacts/run-001",
        "best_idx": 1,
        "best_validation_score": 0.6,
        "best_candidate_fingerprint": candidate_fingerprint,
        "seed_candidate_fingerprint": "seed.yaml",
        "best_candidate_differs_from_seed": True,
        "train_case_ids": ["case-a", "case-b"],
        "validation_case_ids": ["case-c"],
        "execution_modes": ["live"],
        "num_candidates": 5,
        "total_metric_calls": 10,
        "num_full_val_evals": 2,
        "run_dir": "artifacts/run-001",
    }
    (evidence_path / "optimization-summary.json").write_text(
        json.dumps(opt_summary),
    )

    best_candidate = {
        "schema_version": 1,
        "candidate_id": "cand-1",
        "components": {"system": "system prompt"},
        "metadata": {},
    }
    (evidence_path / "best-candidate.yaml").write_text(yaml.dump(best_candidate))


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------


class TestCLIPlan:
    def test_plan_writes_action_json(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        output = tmp_path / "action.json"

        _write_control(control_path)
        _write_state(state_path)

        rc = main([
            "plan",
            "--control", str(control_path),
            "--state", str(state_path),
            "--output", str(output),
        ])
        assert rc == 0
        action = json.loads(output.read_text())
        assert "action_id" in action
        assert action["kind"] == "search"

    def test_plan_terminal_state(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        output = tmp_path / "action.json"

        _write_control(control_path)
        terminal = CampaignState(
            schema_version=1,
            campaign_id="test-campaign",
            prompt_lab_revision="abc123",
            korvid_revision="def456",
            status=CampaignStatus.QUALIFIED,
            tier_index=0, stage_index=2, seed_index=3,
            champion_fingerprint="fp",
            champion_score=CampaignScore(
                fingerprint="fp", aggregate=0.9,
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
        _write_state(state_path, terminal)

        rc = main([
            "plan",
            "--control", str(control_path),
            "--state", str(state_path),
            "--output", str(output),
        ])
        assert rc == 0
        result = json.loads(output.read_text())
        assert result["terminal"] is True


class TestCLIAdvance:
    def test_advance_requires_expected_prior_hash(self, tmp_path: Path) -> None:
        """--expected-prior-hash is required."""
        with pytest.raises(SystemExit):
            main([
                "advance",
                "--control", "/dev/null",
                "--state", "/dev/null",
                "--action", "/dev/null",
                "--evidence", "/dev/null",
                "--output-state", "/dev/null",
                # missing --expected-prior-hash
            ])

    def test_advance_updates_state(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        evidence_path = tmp_path / "evidence"
        output_state = tmp_path / "new-state.json"

        _write_control(control_path)
        current_hash = _write_state(state_path)

        # Plan to get valid action
        action_path = tmp_path / "action.json"
        main([
            "plan",
            "--control", str(control_path),
            "--state", str(state_path),
            "--output", str(action_path),
        ])
        action = json.loads(action_path.read_text())
        _write_evidence(evidence_path, action)

        rc = main([
            "advance",
            "--control", str(control_path),
            "--state", str(state_path),
            "--action", str(action_path),
            "--evidence", str(evidence_path),
            "--output-state", str(output_state),
            "--expected-prior-hash", current_hash,
        ])
        assert rc == 0
        new_state = json.loads(output_state.read_text())
        assert new_state["status"] == "running"
        assert new_state["seed_index"] == 1

    def test_advance_rejects_stale_prior_hash(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        evidence_path = tmp_path / "evidence"
        output_state = tmp_path / "new-state.json"

        _write_control(control_path)
        _write_state(state_path)

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
        _write_evidence(evidence_path, {"action_id": "wrong"})

        rc = main([
            "advance",
            "--control", str(control_path),
            "--state", str(state_path),
            "--action", str(action_path),
            "--evidence", str(evidence_path),
            "--output-state", str(output_state),
            "--expected-prior-hash", "sha256:" + "f" * 64,
        ])
        assert rc != 0

    def test_advance_wrong_model_in_evidence(self, tmp_path: Path) -> None:
        """Regression: wrong model in evidence must fail at CLI level."""
        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        evidence_path = tmp_path / "evidence"
        output_state = tmp_path / "new-state.json"

        _write_control(control_path)
        current_hash = _write_state(state_path)

        action_path = tmp_path / "action.json"
        main([
            "plan",
            "--control", str(control_path),
            "--state", str(state_path),
            "--output", str(action_path),
        ])
        action = json.loads(action_path.read_text())
        _write_evidence(evidence_path, action)
        # Corrupt model in evidence
        es = json.loads(
            (evidence_path / "evaluation-summary.json").read_text(),
        )
        es["evaluated_models"] = ["wrong-model"]
        (evidence_path / "evaluation-summary.json").write_text(json.dumps(es))

        rc = main([
            "advance",
            "--control", str(control_path),
            "--state", str(state_path),
            "--action", str(action_path),
            "--evidence", str(evidence_path),
            "--output-state", str(output_state),
            "--expected-prior-hash", current_hash,
        ])
        assert rc != 0


class TestCLIRender:
    def test_render_writes_markdown(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        output_dir = tmp_path / "out"

        _write_control(control_path)
        _write_state(state_path)

        rc = main([
            "render",
            "--control", str(control_path),
            "--state", str(state_path),
            "--output-dir", str(output_dir),
        ])
        assert rc == 0
        assert (output_dir / "campaign-summary.md").exists()
        md = (output_dir / "campaign-summary.md").read_text()
        assert "explore" in md  # uses real stage name from control


class TestCLIGithubOutput:
    def test_ignores_env_github_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evil = tmp_path / "evil-output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(evil))

        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        output = tmp_path / "action.json"

        _write_control(control_path)
        _write_state(state_path)

        main([
            "plan",
            "--control", str(control_path),
            "--state", str(state_path),
            "--output", str(output),
        ])
        assert not evil.exists()

    def test_writes_explicit_github_output(self, tmp_path: Path) -> None:
        gh_out = tmp_path / "gh-output.txt"
        state_path = tmp_path / "state.json"
        control_path = tmp_path / "control.yaml"
        output = tmp_path / "action.json"

        _write_control(control_path)
        _write_state(state_path)

        main([
            "plan",
            "--control", str(control_path),
            "--state", str(state_path),
            "--output", str(output),
            "--github-output", str(gh_out),
        ])
        assert gh_out.exists()
        content = gh_out.read_text()
        assert "action_id=" in content
