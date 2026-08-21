from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.contracts import Candidate, Campaign, EvalCase, ProcessServing
from korvid_prompt_lab.publish import PromptBundle, PromotionDecision, publish_bundle, render_scoreboard


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
    case = EvalCase(
        case_id="case-1",
        template_id="template-1",
        prompt="Confirm the postcondition.",
        models=("mock-small",),
    )
    return Campaign(
        schema_version=1,
        campaign_id="campaign-1",
        repetitions=3,
        models=("mock-small",),
        cases=(case,),
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
            "train": ["train-1"],
            "validation": ["val-1"],
            "milestone": ["milestone-1"],
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
                "train": ["train-2", "train-1"],
                "validation": ["val-2", "val-1"],
                "milestone": ["milestone-2", "milestone-1"],
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
                "milestone": ["milestone-1", "milestone-2"],
                "validation": ["val-1", "val-2"],
                "train": ["train-1", "train-2"],
            },
        ),
        registry_root=registry_root,
    )

    assert isinstance(first, PromotionDecision)
    assert isinstance(first.bundle, PromptBundle)
    assert first.bundle is not None
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
