from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.contracts import Campaign, Candidate, EvalCase, ProcessServing
from korvid_prompt_lab.publish import (
    DEFAULT_MINIMUM_MODEL_IMPROVEMENT,
    PromotionDecision,
    PromptBundle,
    publish_bundle,
    render_scoreboard,
)

ROOT = Path(__file__).resolve().parents[1]


def _candidate(*, candidate_id: str = "candidate-common") -> Candidate:
    return Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "components": {
                "system": "Stay safe.",
                "append": "Verify the postcondition before reporting completion.",
                "tool.search": "Read-only search only.",
            },
            "metadata": {"source": "optimization", "owner": "korvid"},
        }
    )


def _candidate_reordered(*, candidate_id: str = "candidate-common") -> Candidate:
    return Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "components": {
                "tool.search": "Read-only search only.",
                "append": "Verify the postcondition before reporting completion.",
                "system": "Stay safe.",
            },
            "metadata": {"owner": "korvid", "source": "optimization"},
        }
    )


def _campaign() -> Campaign:
    return _campaign_with_id("campaign-1")


def _campaign_with_id(campaign_id: str) -> Campaign:
    cases = tuple(
        EvalCase(
            case_id=case_id,
            template_id="template-1",
            prompt="Confirm the postcondition.",
            models=("mock-small",),
        )
        for case_id in ("case-train-a", "case-train-b", "case-val-a", "case-val-b")
    )
    return Campaign(
        schema_version=1,
        campaign_id=campaign_id,
        repetitions=5,
        models=("mock-small",),
        cases=cases,
        serving=ProcessServing(
            backend="process",
            command=(
                sys.executable,
                str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
                "--request",
                "{request}",
                "--response",
                "{response}",
            ),
        ),
    )



def _model_metadata(**overrides: object) -> dict[str, object]:
    model = {
        "model_family": "mock-small",
        "model_name": "mock-small@2026-08-21",
        "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "quantization": "fp16",
        "context_length": 8192,
        "serving_engine": "korvid-process",
    }
    model.update(overrides)
    return model



def _evaluation_summary(**overrides: object) -> dict[str, object]:
    summary = {
        "bundle_kind": "common",
        "aggregate_score": 0.91,
        "pass_at_3": 1.0,
        "pass_at_5": 1.0,
        "hard_safety_failures": 0,
        "systemic_failures": 0,
        "milestone_passed": True,
        "case_sets": {
            "train": ["case-train-a", "case-train-b"],
            "validation": ["case-val-a", "case-val-b"],
            "milestone": ["case-train-a", "case-train-b", "case-val-a", "case-val-b"],
        },
        "artifact_refs": ["artifacts/evaluation-summary.json"],
        "reproduction_command": [
            "uv",
            "run",
            "--python",
            "3.12",
            "korvid-prompt-lab",
            "evaluate",
            "--campaign",
            "campaign.yaml",
        ],
    }
    summary.update(overrides)
    summary.setdefault("model_scores", {"mock-small": summary["aggregate_score"]})
    return summary



def test_publish_bundle_writes_deterministic_common_bundle_and_registry_files(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"

    first = publish_bundle(
        candidate=_candidate(),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(
            artifact_refs=[
                "artifacts/evaluation-summary.json",
                "artifacts/request.json",
            ],
            case_sets={
                "train": ["case-train-b", "case-train-a"],
                "validation": ["case-val-b", "case-val-a"],
                "milestone": ["case-val-b", "case-val-a", "case-train-b", "case-train-a"],
            },
        ),
        registry_root=registry_root,
    )
    second = publish_bundle(
        candidate=_candidate_reordered(),
        campaign=_campaign(),
        model_metadata={
            "serving_engine": "korvid-process",
            "context_length": 8192,
            "quantization": "fp16",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "model_name": "mock-small@2026-08-21",
            "model_family": "mock-small",
        },
        evaluation_summary=_evaluation_summary(
            artifact_refs=[
                "artifacts/request.json",
                "artifacts/evaluation-summary.json",
            ],
            reproduction_command=[
                "uv",
                "run",
                "--python",
                "3.12",
                "korvid-prompt-lab",
                "evaluate",
                "--campaign",
                "campaign.yaml",
            ],
            case_sets={
                "milestone": ["case-train-a", "case-train-b", "case-val-a", "case-val-b"],
                "validation": ["case-val-a", "case-val-b"],
                "train": ["case-train-a", "case-train-b"],
            },
        ),
        registry_root=registry_root,
    )

    assert isinstance(first, PromotionDecision)
    assert isinstance(first.bundle, PromptBundle)
    assert first.bundle is not None
    assert second.bundle is not None
    assert first.bundle.version == second.bundle.version
    assert first.published is True
    assert second.published is True
    assert first.bundle.bundle_dir == registry_root / "bundles" / "mock-small" / first.bundle.version
    assert (first.bundle.bundle_dir / "prompt-bundle.yaml").is_file()
    assert (first.bundle.bundle_dir / "evaluation-summary.json").is_file()
    assert (registry_root / "index.json").is_file()
    assert (registry_root / "scoreboard.md").is_file()

    index_payload = json.loads((registry_root / "index.json").read_text(encoding="utf-8"))
    assert [entry["version"] for entry in index_payload["bundles"]] == [first.bundle.version]
    assert ".tmp" not in "\n".join(path.name for path in registry_root.rglob("*"))


def test_publish_bundle_treats_reproduction_command_as_ordered_for_versioning(tmp_path: Path) -> None:
    first = publish_bundle(
        candidate=_candidate(),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(),
        registry_root=tmp_path / "registry-first",
    )
    second = publish_bundle(
        candidate=_candidate_reordered(),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(
            reproduction_command=[
                "uv",
                "run",
                "--python",
                "3.12",
                "korvid-prompt-lab",
                "--campaign",
                "campaign.yaml",
                "evaluate",
            ]
        ),
        registry_root=tmp_path / "registry-second",
    )

    assert first.bundle is not None
    assert second.bundle is not None
    assert first.bundle.version != second.bundle.version


def test_publish_bundle_preserves_evaluation_provenance_fields(tmp_path: Path) -> None:
    decision = publish_bundle(
        candidate=_candidate(),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(
            candidate_id="candidate-common",
            candidate_fingerprint=_candidate().fingerprint,
            campaign_id="campaign-1",
            campaign_case_ids=["case-1"],
            evaluated_case_ids=["case-1"],
            evaluated_models=["mock-small"],
            campaign_case_model_pairs=["case-1::mock-small"],
            evaluated_case_model_pairs=["case-1::mock-small"],
        ),
        registry_root=tmp_path / "registry",
    )

    assert decision.bundle is not None
    persisted_summary = json.loads(decision.bundle.evaluation_summary_path.read_text(encoding="utf-8"))
    assert persisted_summary["candidate_id"] == "candidate-common"
    assert persisted_summary["candidate_fingerprint"] == _candidate().fingerprint
    assert persisted_summary["campaign_id"] == "campaign-1"
    assert persisted_summary["campaign_case_ids"] == ["case-1"]
    assert persisted_summary["evaluated_case_ids"] == ["case-1"]
    assert persisted_summary["evaluated_models"] == ["mock-small"]
    assert persisted_summary["campaign_case_model_pairs"] == ["case-1::mock-small"]
    assert persisted_summary["evaluated_case_model_pairs"] == ["case-1::mock-small"]


def test_publish_bundle_requires_exact_model_digest() -> None:
    with pytest.raises(ValueError, match="model_digest"):
        publish_bundle(
            candidate=_candidate(),
            campaign=_campaign(),
            model_metadata=_model_metadata(model_digest="mock-small:latest"),
            evaluation_summary=_evaluation_summary(),
            registry_root=Path("registry"),
        )



def test_publish_bundle_aborts_on_systemic_bridge_failures(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="systemic bridge failures"):
        publish_bundle(
            candidate=_candidate(),
            campaign=_campaign(),
            model_metadata=_model_metadata(),
            evaluation_summary=_evaluation_summary(systemic_failures=1),
            registry_root=tmp_path / "registry",
        )



def test_publish_bundle_zeroes_unsafe_scores_and_skips_promotion(tmp_path: Path) -> None:
    decision = publish_bundle(
        candidate=_candidate(),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(aggregate_score=0.97, hard_safety_failures=2),
        registry_root=tmp_path / "registry",
    )

    assert decision.published is False
    assert decision.bundle is None
    assert decision.effective_score == 0.0
    assert "hard safety" in decision.reason
    assert not (tmp_path / "registry" / "index.json").exists()



def test_publish_bundle_requires_common_baseline_for_model_specific_promotion(tmp_path: Path) -> None:
    decision = publish_bundle(
        candidate=_candidate(candidate_id="candidate-model"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="model-specific", aggregate_score=0.95),
        registry_root=tmp_path / "registry",
        minimum_model_improvement=0.03,
    )

    assert decision.published is False
    assert decision.bundle is None
    assert "common baseline" in decision.reason


def test_publish_bundle_uses_strongest_common_baseline_for_model_specific_comparison(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    low_common = publish_bundle(
        candidate=_candidate(candidate_id="a-common"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="common", aggregate_score=0.80),
        registry_root=registry_root,
    )
    high_common = publish_bundle(
        candidate=_candidate(candidate_id="z-common"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="common", aggregate_score=0.90),
        registry_root=registry_root,
    )
    candidate_specific = publish_bundle(
        candidate=_candidate(candidate_id="candidate-model"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="model-specific", aggregate_score=0.84),
        registry_root=registry_root,
        minimum_model_improvement=0.03,
    )

    assert low_common.published is True
    assert high_common.published is True
    assert candidate_specific.published is False
    assert candidate_specific.bundle is None
    assert "improvement" in candidate_specific.reason


def test_publish_bundle_scopes_common_baseline_to_matching_campaign(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    same_campaign_common = publish_bundle(
        candidate=_candidate(candidate_id="campaign-a-common"),
        campaign=_campaign_with_id("campaign-a"),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="common", aggregate_score=0.80),
        registry_root=registry_root,
    )
    different_campaign_common = publish_bundle(
        candidate=_candidate(candidate_id="campaign-b-common"),
        campaign=_campaign_with_id("campaign-b"),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="common", aggregate_score=0.95),
        registry_root=registry_root,
    )
    same_campaign_model_specific = publish_bundle(
        candidate=_candidate(candidate_id="campaign-a-model"),
        campaign=_campaign_with_id("campaign-a"),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="model-specific", aggregate_score=0.84),
        registry_root=registry_root,
        minimum_model_improvement=0.03,
    )

    assert same_campaign_common.published is True
    assert different_campaign_common.published is True
    assert same_campaign_model_specific.published is True
    assert same_campaign_model_specific.bundle is not None


def test_publish_bundle_compares_model_specific_scores_against_target_model_common_score(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    common = publish_bundle(
        candidate=_candidate(candidate_id="candidate-common"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(
            bundle_kind="common",
            aggregate_score=0.95,
            model_scores={"mock-small": 0.70, "mock-large": 0.99},
        ),
        registry_root=registry_root,
    )
    model_specific = publish_bundle(
        candidate=_candidate(candidate_id="candidate-model"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(
            bundle_kind="model-specific",
            aggregate_score=0.75,
            model_scores={"mock-small": 0.75},
        ),
        registry_root=registry_root,
        minimum_model_improvement=0.02,
    )

    assert common.published is True
    assert model_specific.published is True
    assert model_specific.bundle is not None
    assert model_specific.effective_score == pytest.approx(0.75)


def test_publish_bundle_requires_milestone_pass_for_model_specific_promotion(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    publish_bundle(
        candidate=_candidate(candidate_id="candidate-common"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="common", aggregate_score=0.90),
        registry_root=registry_root,
    )

    decision = publish_bundle(
        candidate=_candidate(candidate_id="candidate-model"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(
            bundle_kind="model-specific",
            aggregate_score=0.95,
            milestone_passed=False,
        ),
        registry_root=registry_root,
        minimum_model_improvement=0.03,
    )

    assert decision.published is False
    assert decision.bundle is None
    assert "milestone" in decision.reason


def test_publish_bundle_enforces_minimum_model_specific_improvement_and_orders_registry(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    common = publish_bundle(
        candidate=_candidate(candidate_id="candidate-common"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="common", aggregate_score=0.90),
        registry_root=registry_root,
    )
    below_threshold = publish_bundle(
        candidate=_candidate(candidate_id="candidate-specific-low"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="model-specific", aggregate_score=0.92),
        registry_root=registry_root,
        minimum_model_improvement=0.03,
    )
    above_threshold = publish_bundle(
        candidate=_candidate(candidate_id="candidate-specific-high"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="model-specific", aggregate_score=0.95),
        registry_root=registry_root,
        minimum_model_improvement=0.03,
    )

    assert common.published is True
    assert below_threshold.published is False
    assert "improvement" in below_threshold.reason
    assert above_threshold.published is True
    assert above_threshold.bundle is not None

    index_payload = json.loads((registry_root / "index.json").read_text(encoding="utf-8"))
    assert [entry["bundle_kind"] for entry in index_payload["bundles"]] == ["common", "model-specific"]
    assert [entry["candidate_id"] for entry in index_payload["bundles"]] == [
        "candidate-common",
        "candidate-specific-high",
    ]



def test_render_scoreboard_generates_markdown_table(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    common = publish_bundle(
        candidate=_candidate(candidate_id="candidate-common"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="common", aggregate_score=0.90),
        registry_root=registry_root,
    )
    specific = publish_bundle(
        candidate=_candidate(candidate_id="candidate-specific"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="model-specific", aggregate_score=0.95),
        registry_root=registry_root,
        minimum_model_improvement=0.03,
    )

    assert common.bundle is not None
    assert specific.bundle is not None

    scoreboard = render_scoreboard([common.bundle, specific.bundle])
    assert scoreboard == (registry_root / "scoreboard.md").read_text(encoding="utf-8")
    assert "# Prompt Registry Scoreboard" in scoreboard
    assert "| Model family | Model digest | Bundle kind | Candidate | Aggregate | pass^3 | pass^5 |" in scoreboard
    assert "| mock-small | sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef | common | candidate-common | 0.900 | 1.000 | 1.000 |" in scoreboard
    assert "| mock-small | sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef | model-specific | candidate-specific | 0.950 | 1.000 | 1.000 |" in scoreboard


def test_render_scoreboard_uses_aggregate_score_for_prompt_bundle_rows(tmp_path: Path) -> None:
    bundle = PromptBundle(
        schema_version=1,
        bundle_kind="common",
        version="pb-unsafe000000000",
        model_family="mock-small",
        model_digest="sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        candidate_id="candidate-unsafe",
        aggregate_score=0.973,
        effective_score=0.0,
        pass_at_3=1.0,
        pass_at_5=1.0,
        hard_safety_failures=1,
        bundle_dir=tmp_path / "registry" / "bundles" / "mock-small" / "pb-unsafe000000000",
        prompt_bundle_path=tmp_path / "registry" / "bundles" / "mock-small" / "pb-unsafe000000000" / "prompt-bundle.yaml",
        evaluation_summary_path=tmp_path / "registry" / "bundles" / "mock-small" / "pb-unsafe000000000" / "evaluation-summary.json",
    )

    scoreboard = render_scoreboard([bundle])

    assert "| mock-small | sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef | common | candidate-unsafe | 0.973 | 1.000 | 1.000 |" in scoreboard


@pytest.mark.parametrize(("field_name", "repetitions"), [("pass_at_3", 3), ("pass_at_5", 5)])
def test_publish_bundle_rejects_insufficient_pass_hat_k_evidence(
    tmp_path: Path, field_name: str, repetitions: int
) -> None:
    with pytest.raises(ValueError, match=f"{field_name} requires {repetitions} recorded repetitions"):
        publish_bundle(
            candidate=_candidate(),
            campaign=_campaign(),
            model_metadata=_model_metadata(),
            evaluation_summary=_evaluation_summary(**{field_name: None}),
            registry_root=tmp_path / "registry",
        )

    assert not (tmp_path / "registry").exists()


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_publish_bundle_rejects_pass_hat_k_outside_the_unit_interval(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError, match="pass_at_3"):
        publish_bundle(
            candidate=_candidate(),
            campaign=_campaign(),
            model_metadata=_model_metadata(),
            evaluation_summary=_evaluation_summary(pass_at_3=value),
            registry_root=tmp_path / "registry",
        )


def test_publish_bundle_rejects_overlapping_train_and_validation_case_sets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="case_sets train and validation must be disjoint"):
        publish_bundle(
            candidate=_candidate(),
            campaign=_campaign(),
            model_metadata=_model_metadata(),
            evaluation_summary=_evaluation_summary(
                case_sets={
                    "train": ["case-train-a", "case-val-a"],
                    "validation": ["case-val-a", "case-val-b"],
                    "milestone": ["case-train-a", "case-val-a"],
                }
            ),
            registry_root=tmp_path / "registry",
        )

    assert not (tmp_path / "registry").exists()


@pytest.mark.parametrize("missing_key", ["train", "validation"])
def test_publish_bundle_requires_recorded_train_and_validation_case_sets(
    tmp_path: Path, missing_key: str
) -> None:
    case_sets = {
        "train": ["case-train-a"],
        "validation": ["case-val-a"],
        "milestone": ["case-train-a", "case-val-a"],
    }
    case_sets.pop(missing_key)

    with pytest.raises(ValueError, match=f"case_sets must record {missing_key}"):
        publish_bundle(
            candidate=_candidate(),
            campaign=_campaign(),
            model_metadata=_model_metadata(),
            evaluation_summary=_evaluation_summary(case_sets=case_sets),
            registry_root=tmp_path / "registry",
        )


@pytest.mark.parametrize("split", ["train", "validation"])
def test_publish_bundle_rejects_case_sets_outside_the_campaign(tmp_path: Path, split: str) -> None:
    case_sets = {
        "train": ["case-train-a"],
        "validation": ["case-val-a"],
        "milestone": ["case-train-a", "case-val-a"],
    }
    case_sets[split] = [*case_sets[split], "case-not-in-campaign"]

    with pytest.raises(ValueError, match="case-not-in-campaign"):
        publish_bundle(
            candidate=_candidate(),
            campaign=_campaign(),
            model_metadata=_model_metadata(),
            evaluation_summary=_evaluation_summary(case_sets=case_sets),
            registry_root=tmp_path / "registry",
        )


def _publish_common(registry_root: Path, score: float) -> PromotionDecision:
    return publish_bundle(
        candidate=_candidate(candidate_id="candidate-common"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="common", aggregate_score=score),
        registry_root=registry_root,
    )


def test_publish_bundle_rejects_a_model_specific_improvement_equal_to_the_threshold(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    common = _publish_common(registry_root, 0.5)

    tie = publish_bundle(
        candidate=_candidate(candidate_id="candidate-specific-tie"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="model-specific", aggregate_score=0.75),
        registry_root=registry_root,
        minimum_model_improvement=0.25,
    )

    assert common.published is True
    assert tie.published is False
    assert tie.bundle is None
    assert "improvement" in tie.reason


def test_publish_bundle_defaults_to_a_non_zero_minimum_model_improvement(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    common = _publish_common(registry_root, 0.5)

    marginal = publish_bundle(
        candidate=_candidate(candidate_id="candidate-specific-marginal"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="model-specific", aggregate_score=0.5078125),
        registry_root=registry_root,
    )
    clear_win = publish_bundle(
        candidate=_candidate(candidate_id="candidate-specific-clear"),
        campaign=_campaign(),
        model_metadata=_model_metadata(),
        evaluation_summary=_evaluation_summary(bundle_kind="model-specific", aggregate_score=0.75),
        registry_root=registry_root,
    )

    assert DEFAULT_MINIMUM_MODEL_IMPROVEMENT > 0.0
    assert common.published is True
    assert marginal.published is False
    assert "improvement" in marginal.reason
    assert clear_win.published is True
