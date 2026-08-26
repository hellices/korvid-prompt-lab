"""Multi-tier rollover carries a real seed candidate fingerprint (wave 2, finding 2).

`OptimizationCampaign.initial_candidate` is a repository-relative YAML *path*, a
piece of control configuration. It is not a candidate identity. Persisting it as
`champion_fingerprint` makes the rolled state unpackageable, unresumable and
therefore fatal for every multi-tier campaign.

These tests roll a real two-tier campaign through the pure state machine and
then run the *actual* workflow packaging predicate against the rolled state.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_optimization_campaign_workflow import (
    embedded_python,
    load_workflow,
    step,
)

from korvid_prompt_lab.campaign_artifacts import _serialize_state
from korvid_prompt_lab.campaigns import (
    AttemptOutcome,
    CampaignScore,
    CampaignState,
    CampaignStatus,
    OptimizationCampaign,
    advance_state,
    load_optimization_campaign,
    next_action,
    state_hash,
)
from korvid_prompt_lab.config import load_campaign, load_candidate

MANIFEST = ROOT / "examples/optimization-campaigns/qwen3-small-operator.yaml"
EVALUATION = ROOT / "examples/campaigns/aks-small-operator-qualification.yaml"
INITIAL_CANDIDATE = ROOT / "examples/candidates/shipped-small.yaml"
SECOND_TIER_DIGEST = "sha256:" + "c" * 64


@pytest.fixture()
def scratch() -> Iterator[Path]:
    directory = ROOT / "artifacts" / f"tier-rollover-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _two_tier_manifest(scratch: Path) -> Path:
    """The shipped manifest with a synthetic second model tier."""
    mapping = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    mapping["model_tiers"] = [
        *mapping["model_tiers"],
        {"name": "large", "model": "qwen3:14b", "digest": SECOND_TIER_DIGEST},
    ]
    mapping["stagnation_attempt_limit"] = 2
    path = scratch / "two-tier.yaml"
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return path


def _prepare_initial(scratch: Path, manifest: Path) -> dict[str, str]:
    """Run the real workflow `prepare` step for a fresh campaign."""
    code = embedded_python(step(load_workflow(), "campaign", "prepare"))
    output = scratch / "prepare-output"
    result = subprocess.run(
        [sys.executable, "-", str(output)],
        input=code,
        cwd=ROOT,
        env={
            "PATH": _path(),
            "HOME": str(scratch),
            "MANIFEST": _relative(manifest),
            "CAMPAIGN_ID": "qwen3-small-operator-v5",
            "MANIFEST_SHA256": (
                "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
            ),
            "EVALUATION_CAMPAIGN": _relative(EVALUATION),
            "INITIAL_CANDIDATE": _relative(INITIAL_CANDIDATE),
            "PROMPT_LAB_REF": "a" * 40,
            "KORVID_REF": "b" * 40,
            "PRIOR_RUN_ID": "",
            "EXPECTED_STATE_HASH": "",
            "PRIOR_ROOT": str(scratch / "unused-prior"),
            "CAMPAIGN_ROOT": str(scratch / "campaign"),
            "KORVID_AKS_NAMESPACE": "ollama",
            "KORVID_AKS_SERVICE": "ollama",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return dict(
        line.split("=", 1)
        for line in output.read_text(encoding="utf-8").splitlines()
    )


def _path() -> str:
    import os

    return os.environ["PATH"]


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _load_two_tier_control(manifest: Path) -> OptimizationCampaign:
    import os

    os.environ.setdefault("KORVID_AKS_MODEL", "qwen3:0.6b")
    os.environ.setdefault("KORVID_AKS_NAMESPACE", "ollama")
    os.environ.setdefault("KORVID_AKS_SERVICE", "ollama")
    return load_optimization_campaign(manifest, load_campaign(EVALUATION))


def _improved_candidate(scratch: Path) -> tuple[Path, str]:
    """A genuinely different candidate, promoted as tier 0's champion."""
    mapping = yaml.safe_load(INITIAL_CANDIDATE.read_text(encoding="utf-8"))
    mapping["candidate_id"] = "improved-small"
    mapping["components"]["append"] = "Improved: verify before reporting."
    path = scratch / "improved-small.yaml"
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return path, load_candidate(path).fingerprint


def _roll_to_second_tier(
    control: OptimizationCampaign, state: CampaignState, champion: str,
) -> CampaignState:
    now = datetime.now(tz=UTC)
    action = next_action(control, state, now)
    assert action is not None
    state = advance_state(
        control,
        state,
        action,
        AttemptOutcome(
            kind="evidence",
            score=CampaignScore(
                fingerprint=champion,
                aggregate=0.9,
                hard_safety_failures=0,
                core_regression=False,
                systemic_failures=0,
                pass_at_3=0.5,
                pass_at_5=0.5,
            ),
        ),
        now,
    )
    assert state.champion_fingerprint == champion
    for _ in range(control.stagnation_attempt_limit):
        action = next_action(control, state, now)
        assert action is not None
        state = advance_state(
            control,
            state,
            action,
            AttemptOutcome(
                kind="evidence",
                score=CampaignScore(
                    fingerprint="d" * 64,
                    aggregate=0.0,
                    hard_safety_failures=0,
                    core_regression=False,
                    systemic_failures=0,
                    pass_at_3=0.0,
                    pass_at_5=0.0,
                ),
            ),
            now,
        )
    return state


class TestTierRolloverSeedIdentity:
    def test_rolled_champion_is_the_real_seed_candidate_fingerprint(
        self, scratch: Path,
    ) -> None:
        manifest = _two_tier_manifest(scratch)
        control = _load_two_tier_control(manifest)
        prepared = _prepare_initial(scratch, manifest)
        seed_fingerprint = load_candidate(INITIAL_CANDIDATE).fingerprint
        assert prepared["seed-candidate-fingerprint"] == seed_fingerprint

        from korvid_prompt_lab.campaign_cli import _load_state

        state = _load_state(Path(prepared["state-path"]))
        assert state.seed_candidate_fingerprint == seed_fingerprint

        _improved, champion = _improved_candidate(scratch)
        rolled = _roll_to_second_tier(control, state, champion)

        assert rolled.status is CampaignStatus.RUNNING
        assert rolled.tier_index == 1
        assert rolled.champion_fingerprint == seed_fingerprint
        assert rolled.champion_score.fingerprint == seed_fingerprint
        assert rolled.seed_candidate_fingerprint == seed_fingerprint
        # The manifest path must never appear as a candidate identity.
        assert control.initial_candidate not in (
            rolled.champion_fingerprint,
            rolled.champion_score.fingerprint,
            rolled.seed_candidate_fingerprint,
        )
        # The identity is resolvable back to a real candidate file.
        assert load_candidate(INITIAL_CANDIDATE).fingerprint == (
            rolled.champion_fingerprint
        )

    def test_rolled_state_survives_the_real_package_step(
        self, scratch: Path,
    ) -> None:
        """The reviewer's fatal scenario: package the rolled two-tier state."""
        manifest = _two_tier_manifest(scratch)
        control = _load_two_tier_control(manifest)
        prepared = _prepare_initial(scratch, manifest)
        improved, champion = _improved_candidate(scratch)

        from korvid_prompt_lab.campaign_cli import _load_state

        state = _load_state(Path(prepared["state-path"]))
        rolled = _roll_to_second_tier(control, state, champion)

        next_root = scratch / "next"
        next_root.mkdir()
        payload = _serialize_state(rolled)
        payload["state_hash"] = state_hash(rolled)
        (next_root / "campaign-state.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        (next_root / "campaign-summary.md").write_text("# c\n", encoding="utf-8")
        (next_root / "campaign-action.json").write_text(
            json.dumps({"action_id": "11111111-1111-5111-8111-111111111111"}),
            encoding="utf-8",
        )

        code = embedded_python(step(load_workflow(), "campaign", "package"))
        output = scratch / "package-output"
        result = subprocess.run(
            [sys.executable, "-", str(output)],
            input=code,
            cwd=ROOT,
            env={
                "PATH": _path(),
                "HOME": str(scratch),
                "MANIFEST": _relative(manifest),
                "CAMPAIGN_ID": "qwen3-small-operator-v5",
                "MANIFEST_SHA256": (
                    "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
                ),
                "PROMPT_LAB_REF": "a" * 40,
                "KORVID_REF": "b" * 40,
                # The run's current candidate is tier 0's improved champion,
                # which is deliberately NOT the rolled tier's champion.
                "CURRENT_CANDIDATE": str(improved),
                "SEED_CANDIDATE": _relative(INITIAL_CANDIDATE),
                "SEED_CANDIDATE_FINGERPRINT": (
                    prepared["seed-candidate-fingerprint"]
                ),
                "NEXT_ROOT": str(next_root),
                "SAFE_UPLOAD_ROOT": str(scratch / "safe-upload"),
                "WRAPPER_EXIT": "0",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

        entries = dict(
            line.split("=", 1)
            for line in output.read_text(encoding="utf-8").splitlines()
        )
        assert entries["status"] == "running"
        assert entries["state-hash"] == state_hash(rolled)
        packaged = (
            scratch / "safe-upload" / "safe-campaign" / "champion-candidate.yaml"
        )
        assert packaged.is_file()
        assert load_candidate(packaged).fingerprint == (
            rolled.champion_fingerprint
        )
