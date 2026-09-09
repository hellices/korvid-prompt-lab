from __future__ import annotations

from dataclasses import fields
from types import MappingProxyType

import pytest

from korvid_prompt_lab import contracts
from korvid_prompt_lab.contracts import (
    GEPA_REFLECTION_MINIBATCH_SIZE,
    MODEL_OPTION_FIELDS,
    Campaign,
    Candidate,
    EvalCase,
    KorvidUpstreamServing,
    SearchStage,
    _immutable_model_options,
    _require_timeout,
)


def _serving() -> KorvidUpstreamServing:
    return KorvidUpstreamServing(
        backend="korvid_upstream",
        source_root="/reviewed/korvid",
        base_url="http://127.0.0.1:11434",
        korvid_revision="33c483e041006eb20259a024ed85a9323e52c8f0",
        timeout_seconds=240.0,
        model_options=MappingProxyType({"temperature": 0.0}),
    )


def _case() -> EvalCase:
    return EvalCase(
        case_id="image-pull-typo",
        template_id="korvid-scenario",
        prompt="Why is the pod stuck in ImagePullBackOff?",
        models=("ollama/qwen3:0.6b",),
    )


def test_legacy_contract_types_and_aliases_are_absent() -> None:
    for name in (
        "ProcessServing",
        "AKSPortForwardServing",
        "KorvidReadonlyServing",
        "KorvidNavigationServing",
        "KorvidNativeServing",
        "DEFAULT_BRIDGE_TIMEOUT_SECONDS",
        "NATIVE_MODEL_OPTION_FIELDS",
        "_immutable_native_model_options",
        "_require_bridge_timeout",
    ):
        assert not hasattr(contracts, name)


def test_candidate_accepts_only_a_non_blank_tier_pack() -> None:
    candidate = Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "candidate-1",
            "components": {"tier_pack": "Use source evidence before answering."},
            "metadata": {"source": "baseline"},
        }
    )

    assert candidate.schema_version == 1
    assert candidate.components == {"tier_pack": "Use source evidence before answering."}
    assert candidate.metadata == {"source": "baseline"}
    assert len(candidate.fingerprint) == 64


@pytest.mark.parametrize(
    "components",
    (
        {"system": "legacy"},
        {"append": "legacy"},
        {"rules": "legacy"},
        {"tool.scale_resource": "legacy"},
        {"tier_pack": "ok", "system": "legacy"},
        {"tier_pack": ""},
    ),
)
def test_candidate_rejects_every_legacy_component_path(
    components: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="tier_pack"):
        Candidate.from_mapping(
            {
                "schema_version": 1,
                "candidate_id": "candidate-1",
                "components": components,
            }
        )


def test_direct_candidate_construction_cannot_bypass_current_schema() -> None:
    with pytest.raises(ValueError, match="tier_pack"):
        Candidate(1, "bad", (("system", "legacy"),))
    with pytest.raises(ValueError, match="schema_version"):
        Candidate(2, "bad", (("tier_pack", "text"),))


def test_candidate_fingerprint_is_stable_for_mapping_order() -> None:
    left = Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "candidate-1",
            "components": {"tier_pack": "prompt"},
            "metadata": {"b": "2", "a": "1"},
        }
    )
    right = Candidate.from_mapping(
        {
            "metadata": {"a": "1", "b": "2"},
            "components": {"tier_pack": "prompt"},
            "candidate_id": "candidate-1",
            "schema_version": 1,
        }
    )

    assert left.fingerprint == right.fingerprint


def test_eval_cases_are_source_identity_records_only() -> None:
    assert _case().template_id == "korvid-scenario"
    journey = EvalCase(
        case_id="tui-follow",
        template_id="korvid-journey",
        prompt='["Find the pod.", "Show its logs."]',
        models=("ollama/qwen3:0.6b",),
    )
    assert journey.template_id == "korvid-journey"

    with pytest.raises(ValueError, match="template"):
        EvalCase("legacy", "scale-deployment-up", "scale it", ("model",))


def test_campaign_has_only_current_fields_and_exactly_one_model() -> None:
    campaign = Campaign(
        campaign_id="source-campaign",
        repetitions=5,
        models=("ollama/qwen3:0.6b",),
        cases=(_case(),),
        serving=_serving(),
    )

    assert [field.name for field in fields(Campaign)] == [
        "campaign_id",
        "repetitions",
        "models",
        "cases",
        "serving",
        "evaluation_splits",
    ]
    assert campaign.evaluation_splits == ()
    assert not hasattr(campaign, "schema_version")
    assert not hasattr(campaign, "bridge_timeout_seconds")

    with pytest.raises(ValueError, match="exactly one"):
        Campaign(
            campaign_id="bad",
            repetitions=1,
            models=("small", "large"),
            cases=(_case(),),
            serving=_serving(),
        )


def test_upstream_serving_allows_an_unbound_endpoint_during_staging() -> None:
    serving = KorvidUpstreamServing(
        backend="korvid_upstream",
        source_root="/reviewed/korvid",
        base_url="",
        korvid_revision="33c483e041006eb20259a024ed85a9323e52c8f0",
        timeout_seconds=240.0,
    )

    assert serving.base_url == ""


def test_campaign_requires_source_cases_for_its_single_model() -> None:
    mismatched = EvalCase(
        case_id="image-pull-typo",
        template_id="korvid-scenario",
        prompt="Why?",
        models=("different-model",),
    )
    with pytest.raises(ValueError, match="campaign"):
        Campaign(
            campaign_id="bad",
            repetitions=1,
            models=("ollama/qwen3:0.6b",),
            cases=(mismatched,),
            serving=_serving(),
        )


def test_campaign_evaluation_splits_reference_declared_source_cases() -> None:
    campaign = Campaign(
        campaign_id="split",
        repetitions=5,
        models=("ollama/qwen3:0.6b",),
        cases=(_case(),),
        serving=_serving(),
        evaluation_splits=(("validation", ("image-pull-typo",)),),
    )
    assert campaign.evaluation_splits == (
        ("validation", ("image-pull-typo",)),
    )

    with pytest.raises(ValueError, match="unknown"):
        Campaign(
            campaign_id="bad-split",
            repetitions=5,
            models=("ollama/qwen3:0.6b",),
            cases=(_case(),),
            serving=_serving(),
            evaluation_splits=(("validation", ("not-a-source-case",)),),
        )


def test_search_stage_is_the_small_current_campaign_contract() -> None:
    stage = SearchStage.from_mapping(
        {"name": "search", "metric_calls": 9, "seeds": [0, 7]},
        index=2,
    )
    assert stage == SearchStage(name="search", metric_calls=9, seeds=(0, 7))
    assert GEPA_REFLECTION_MINIBATCH_SIZE == 3

    with pytest.raises(ValueError, match="unknown"):
        SearchStage.from_mapping(
            {"name": "search", "metric_calls": 9, "seeds": [0], "legacy": True},
            index=0,
        )
    with pytest.raises(ValueError, match="missing required field"):
        SearchStage.from_mapping(
            {"name": "search", "metric_calls": 9},
            index=0,
        )
    with pytest.raises(ValueError, match="positive"):
        SearchStage.from_mapping(
            {"name": "search", "metric_calls": 0, "seeds": [0]},
            index=0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        SearchStage.from_mapping(
            {"name": "search", "metric_calls": 1, "seeds": [-1]},
            index=0,
        )


def test_model_options_and_timeout_use_only_current_names() -> None:
    assert MODEL_OPTION_FIELDS == frozenset(
        {
            "native_thinking",
            "think",
            "num_ctx",
            "temperature",
            "seed",
            "keep_alive",
            "num_predict",
        }
    )
    options = _immutable_model_options(
        {"temperature": 0, "seed": 4, "native_thinking": False}
    )
    assert dict(options) == {
        "temperature": 0,
        "seed": 4,
        "native_thinking": False,
    }
    with pytest.raises(TypeError):
        options["seed"] = 5  # type: ignore[index]

    assert _require_timeout(12, "runtime.timeout_seconds") == 12.0
    for invalid in (0, -1, True, float("nan")):
        with pytest.raises(ValueError, match="positive"):
            _require_timeout(invalid, "runtime.timeout_seconds")
