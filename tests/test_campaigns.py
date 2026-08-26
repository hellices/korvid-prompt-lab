from __future__ import annotations

import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.campaigns import (
    ActionKind,
    AttemptOutcome,
    CampaignScore,
    CampaignStatus,
    OptimizationCampaign,
    advance_state,
    initial_state,
    load_optimization_campaign,
    next_action,
    validate_model_tier_digests,
)
from korvid_prompt_lab.config import load_campaign
from korvid_prompt_lab.contracts import Campaign, EvalCase, ProcessServing

ROOT = Path(__file__).resolve().parents[1]
EXACT_DIGEST_HEX = "0123456789abcdef" * 4
EXACT_DIGEST = f"sha256:{EXACT_DIGEST_HEX}"


def write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _case(case_id: str) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        template_id=case_id,
        prompt=f"Prompt for {case_id}.",
        models=("qwen3:0.6b",),
    )


def qualification_evaluation_campaign() -> Campaign:
    case_ids = (
        "scale-deployment-up",
        "restart-denied",
        "scale-no-op",
        "scale-deployment-down",
        "restart-deployment",
        "scale-rbac-denied",
        "scale-ambiguous-namespace",
        "restart-approval-expired",
        "restart-daemonset",
        "scale-same-name-replacement",
        "scale-statefulset-down",
        "edit-unsupported",
    )
    return Campaign(
        schema_version=1,
        campaign_id="aks-small-operator-qualification",
        repetitions=5,
        models=("qwen3:0.6b",),
        cases=tuple(_case(case_id) for case_id in case_ids),
        serving=ProcessServing(
            backend="process",
            command=("python3", "bridge.py", "--request", "{request}", "--response", "{response}"),
        ),
        bridge_timeout_seconds=900.0,
    )


def valid_manifest_mapping() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": "qwen3-small-operator",
        "evaluation_campaign": "aks-small-operator-qualification",
        "initial_candidate": "examples/candidates/shipped-small.yaml",
        "train_case_ids": [
            "scale-deployment-up",
            "restart-denied",
            "scale-no-op",
        ],
        "validation_case_ids": [
            "scale-deployment-down",
            "restart-deployment",
            "scale-rbac-denied",
        ],
        "milestone_case_ids": [
            "scale-ambiguous-namespace",
            "restart-approval-expired",
            "restart-daemonset",
            "scale-same-name-replacement",
            "scale-statefulset-down",
            "edit-unsupported",
        ],
        "stages": [
            {"name": "explore", "metric_calls": 12, "seeds": [0, 1, 2]},
            {"name": "refine", "metric_calls": 24, "seeds": [3, 4]},
            {"name": "final", "metric_calls": 48, "seeds": [5]},
        ],
        "model_tiers": [
            {"name": "small", "model": "qwen3:0.6b", "digest": EXACT_DIGEST},
        ],
        "total_metric_call_limit": 240,
        "wall_clock_limit_seconds": 21600,
        "infrastructure_retry_limit": 1,
        "stagnation_attempt_limit": 3,
        "confirmation_runs": 1,
    }


def test_loads_bounded_disjoint_campaign(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KORVID_AKS_NAMESPACE", "ollama")
    monkeypatch.setenv("KORVID_AKS_SERVICE", "ollama")
    monkeypatch.setenv("KORVID_AKS_MODEL", "qwen3:0.6b")
    evaluation = load_campaign(ROOT / "examples/campaigns/aks-small-operator-qualification.yaml")
    control = load_optimization_campaign(
        ROOT / "examples/optimization-campaigns/qwen3-small-operator.yaml",
        evaluation,
    )

    assert isinstance(control, OptimizationCampaign)
    assert control.campaign_id == "qwen3-small-operator-v5"
    assert control.train_case_ids == (
        "scale-deployment-up",
        "restart-denied",
        "scale-no-op",
    )
    assert control.validation_case_ids == (
        "scale-deployment-down",
        "restart-deployment",
        "scale-rbac-denied",
    )
    assert set(control.milestone_case_ids).isdisjoint(control.train_case_ids)
    assert set(control.milestone_case_ids).isdisjoint(control.validation_case_ids)
    assert control.total_metric_call_limit == 240
    assert control.wall_clock_limit_seconds == 21600
    assert control.infrastructure_retry_limit == 1
    assert control.stagnation_attempt_limit == 3
    assert control.confirmation_runs == 1
    assert control.stages[0].name == "explore"
    assert control.stages[0].metric_calls == 12
    assert control.stages[0].seeds == (0, 1, 2)
    assert control.stages[1].name == "refine"
    assert control.stages[1].seeds == (3, 4)
    assert control.stages[2].name == "final"
    assert control.stages[2].seeds == (5,)
    assert control.model_tiers[0].model == "qwen3:0.6b"
    assert control.model_tiers[0].digest == "sha256:7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", control.model_tiers[0].digest)


def test_loads_single_transition_live_canary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KORVID_AKS_NAMESPACE", "ollama")
    monkeypatch.setenv("KORVID_AKS_SERVICE", "ollama")
    monkeypatch.setenv("KORVID_AKS_MODEL", "qwen3:0.6b")
    evaluation = load_campaign(ROOT / "examples/campaigns/aks-small-operator-qualification.yaml")
    control = load_optimization_campaign(
        ROOT / "examples/optimization-campaigns/qwen3-small-operator-canary.yaml",
        evaluation,
    )

    assert control.campaign_id == "qwen3-small-operator-canary"
    assert len(control.stages) == 1
    assert control.stages[0].name == "explore"
    assert control.stages[0].metric_calls == 4
    assert control.stages[0].seeds == (0,)
    assert control.total_metric_call_limit == 9
    assert control.infrastructure_retry_limit == 0
    assert control.train_case_ids == (
        "scale-deployment-up",
        "restart-denied",
        "scale-no-op",
    )
    assert control.validation_case_ids == (
        "scale-deployment-down",
        "restart-deployment",
        "scale-rbac-denied",
    )
    assert control.milestone_case_ids == (
        "scale-ambiguous-namespace",
        "restart-approval-expired",
        "restart-daemonset",
        "scale-same-name-replacement",
        "scale-statefulset-down",
        "edit-unsupported",
    )
    assert control.model_tiers[0].digest == (
        "sha256:7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435"
    )

    state = initial_state(
        control,
        prompt_lab_revision="a" * 40,
        korvid_revision="b" * 40,
        started_at=datetime(2026, 8, 26, tzinfo=UTC),
        seed_candidate_fingerprint="c" * 64,
    )
    action = next_action(control, state, datetime(2026, 8, 26, 0, 0, 1, tzinfo=UTC))
    assert action is not None
    assert action.kind is ActionKind.SEARCH
    assert action.metric_calls == 4
    terminal = advance_state(
        control,
        state,
        action,
        AttemptOutcome(
            kind="evidence",
            metric_calls_used=9,
            score=CampaignScore(
                fingerprint="d" * 64,
                aggregate=0.5,
                hard_safety_failures=0,
                core_regression=False,
                systemic_failures=0,
                pass_at_3=0.0,
                pass_at_5=0.0,
            ),
        ),
        datetime(2026, 8, 26, 0, 5, tzinfo=UTC),
    )
    assert terminal.status is CampaignStatus.NOT_CONVERGED
    assert terminal.stop_reason == "total_metric_call_limit"
    assert terminal.metric_calls_used == 9
    assert next_action(control, terminal, datetime(2026, 8, 26, 0, 5, 1, tzinfo=UTC)) is None


@pytest.mark.parametrize(
    ("field", "duplicate"),
    [
        ("validation_case_ids", "scale-deployment-up"),
        ("milestone_case_ids", "restart-denied"),
    ],
)
def test_rejects_case_set_overlap(tmp_path: Path, field: str, duplicate: str) -> None:
    manifest = valid_manifest_mapping()
    manifest[field][0] = duplicate
    path = write_yaml(tmp_path / "control.yaml", manifest)

    with pytest.raises(ValueError, match="pairwise disjoint"):
        load_optimization_campaign(path, qualification_evaluation_campaign())


@pytest.mark.parametrize(
    "field",
    ["train_case_ids", "validation_case_ids", "milestone_case_ids"],
)
def test_rejects_empty_case_sets(tmp_path: Path, field: str) -> None:
    manifest = valid_manifest_mapping()
    manifest[field] = []
    path = write_yaml(tmp_path / "control.yaml", manifest)

    with pytest.raises(ValueError, match=field):
        load_optimization_campaign(path, qualification_evaluation_campaign())


def test_rejects_unknown_case_ids(tmp_path: Path) -> None:
    manifest = valid_manifest_mapping()
    manifest["milestone_case_ids"][0] = "unknown-case"
    path = write_yaml(tmp_path / "control.yaml", manifest)

    with pytest.raises(ValueError, match="unknown case_id"):
        load_optimization_campaign(path, qualification_evaluation_campaign())


def test_rejects_duplicate_stage_seeds(tmp_path: Path) -> None:
    manifest = valid_manifest_mapping()
    manifest["stages"][1]["seeds"] = [2, 4]
    path = write_yaml(tmp_path / "control.yaml", manifest)

    with pytest.raises(ValueError, match="duplicate seed"):
        load_optimization_campaign(path, qualification_evaluation_campaign())


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda manifest: manifest.__setitem__("total_metric_call_limit", 0), "total_metric_call_limit"),
        (lambda manifest: manifest.__setitem__("wall_clock_limit_seconds", 0), "wall_clock_limit_seconds"),
        (lambda manifest: manifest.__setitem__("stagnation_attempt_limit", 0), "stagnation_attempt_limit"),
        (lambda manifest: manifest.__setitem__("confirmation_runs", 0), "confirmation_runs"),
        (lambda manifest: manifest["stages"][0].__setitem__("metric_calls", 0), "stages\\[0\\]\\.metric_calls"),
    ],
)
def test_rejects_non_positive_budgets(
    tmp_path: Path, mutator: Any, message: str
) -> None:
    manifest = valid_manifest_mapping()
    mutator(manifest)
    path = write_yaml(tmp_path / "control.yaml", manifest)

    with pytest.raises(ValueError, match=message):
        load_optimization_campaign(path, qualification_evaluation_campaign())


def test_rejects_negative_infrastructure_retry_limit(tmp_path: Path) -> None:
    manifest = valid_manifest_mapping()
    manifest["infrastructure_retry_limit"] = -1
    path = write_yaml(tmp_path / "control.yaml", manifest)

    with pytest.raises(ValueError, match="infrastructure_retry_limit"):
        load_optimization_campaign(path, qualification_evaluation_campaign())


@pytest.mark.parametrize(
    "digest",
    [
        "qwen3:0.6b",
        EXACT_DIGEST_HEX,
        "sha256:not-hex",
        "1234",
        "SHA256:" + EXACT_DIGEST_HEX,
    ],
)
def test_rejects_mutable_or_invalid_model_digests(tmp_path: Path, digest: str) -> None:
    manifest = valid_manifest_mapping()
    manifest["model_tiers"][0]["digest"] = digest
    path = write_yaml(tmp_path / "control.yaml", manifest)

    with pytest.raises(ValueError, match="digest"):
        load_optimization_campaign(path, qualification_evaluation_campaign())


@pytest.mark.parametrize(
    "field",
    [
        "total_metric_call_limit",
        "wall_clock_limit_seconds",
        "infrastructure_retry_limit",
        "stagnation_attempt_limit",
        "confirmation_runs",
    ],
)
def test_rejects_missing_required_limits(tmp_path: Path, field: str) -> None:
    manifest = valid_manifest_mapping()
    del manifest[field]
    path = write_yaml(tmp_path / "control.yaml", manifest)

    with pytest.raises(ValueError, match="unknown field|missing"):
        load_optimization_campaign(path, qualification_evaluation_campaign())


@pytest.mark.parametrize(
    ("path_name", "mutator", "message"),
    [
        ("control.yaml", lambda manifest: manifest.__setitem__("unexpected", True), "unknown field"),
        (
            "control.yaml",
            lambda manifest: manifest["stages"][0].__setitem__("unexpected", True),
            "unknown field",
        ),
        (
            "control.yaml",
            lambda manifest: manifest["model_tiers"][0].__setitem__("unexpected", True),
            "unknown field",
        ),
    ],
)
def test_rejects_unknown_keys(tmp_path: Path, path_name: str, mutator: Any, message: str) -> None:
    manifest = valid_manifest_mapping()
    mutator(manifest)
    path = write_yaml(tmp_path / path_name, manifest)

    with pytest.raises(ValueError, match=message):
        load_optimization_campaign(path, qualification_evaluation_campaign())


@pytest.mark.parametrize("live_digest", [EXACT_DIGEST_HEX, EXACT_DIGEST])
def test_validate_model_tier_digests_accepts_matching_live_tags(
    tmp_path: Path, live_digest: str
) -> None:
    path = write_yaml(tmp_path / "control.yaml", valid_manifest_mapping())
    control = load_optimization_campaign(path, qualification_evaluation_campaign())

    validate_model_tier_digests(
        control,
        "http://127.0.0.1:11434",
        http_get_json=lambda url: {
            "models": [
                {"name": "qwen3:0.6b", "digest": live_digest},
                {"name": "qwen3:14b", "digest": "sha256:" + ("f" * 64)},
            ]
        },
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"models": []}, "did not advertise model"),
        (
            {
                "models": [
                    {"name": "qwen3:0.6b", "digest": EXACT_DIGEST},
                    {"name": "qwen3:0.6b", "digest": EXACT_DIGEST_HEX},
                ]
            },
            "duplicate",
        ),
        (
            {"models": [{"name": "qwen3:0.6b", "digest": "sha256:" + ("f" * 64)}]},
            "mismatch",
        ),
        (
            {"models": [{"name": "qwen3:0.6b", "digest": "SHA256:" + EXACT_DIGEST_HEX}]},
            "sha256",
        ),
        (
            {"models": [{"name": "qwen3:0.6b", "digest": "sha256:1234"}]},
            "sha256",
        ),
        (
            {"models": [{"name": "qwen3:0.6b", "digest": "md5:" + EXACT_DIGEST_HEX}]},
            "sha256",
        ),
        (
            {"models": [{"name": "qwen3:0.6b", "digest": 7}]},
            "digest",
        ),
    ],
)
def test_validate_model_tier_digests_rejects_invalid_live_tags(
    tmp_path: Path, payload: dict[str, object], message: str
) -> None:
    path = write_yaml(tmp_path / "control.yaml", valid_manifest_mapping())
    control = load_optimization_campaign(path, qualification_evaluation_campaign())

    with pytest.raises(ValueError, match=message):
        validate_model_tier_digests(
            control,
            "http://127.0.0.1:11434",
            http_get_json=lambda url: payload,
        )


def test_optimization_campaign_dataclasses_are_frozen_and_slotted(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "control.yaml", valid_manifest_mapping())
    control = load_optimization_campaign(path, qualification_evaluation_campaign())

    assert not hasattr(control, "__dict__")
    assert not hasattr(control.stages[0], "__dict__")
    assert not hasattr(control.model_tiers[0], "__dict__")
