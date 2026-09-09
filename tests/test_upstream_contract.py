from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from korvid_prompt_lab.upstream_contract import (
    KORVID_REVISION,
    KORVID_VERSION,
    load_source_cases,
    prompt_candidate,
    tier_pack_from_candidate,
)

SOURCE_ROOT = Path(os.environ.get("KORVID_NATIVE_SOURCE_ROOT", ""))
requires_source = pytest.mark.skipif(
    not os.environ.get("KORVID_NATIVE_SOURCE_ROOT"),
    reason="set KORVID_NATIVE_SOURCE_ROOT to the Korvid v0.4.1 source checkout",
)


def test_upstream_contract_owns_the_current_source_pin() -> None:
    assert KORVID_VERSION == "0.4.1"
    assert KORVID_REVISION == "33c483e041006eb20259a024ed85a9323e52c8f0"


@requires_source
def test_source_cases_copy_original_questions_and_hashes() -> None:
    scenario, journey = load_source_cases(
        SOURCE_ROOT,
        ("scenarios/image-pull-typo", "journeys/tui-follow"),
    )

    assert scenario.reference == "scenarios/image-pull-typo"
    assert scenario.case_id == "image-pull-typo"
    assert scenario.kind == "scenario"
    assert scenario.prompt == (
        "Pod web-1 in namespace front is stuck in ImagePullBackOff. What is the cause?"
    )
    assert scenario.questions == (scenario.prompt,)
    assert scenario.eval_case("ollama/qwen3:0.6b").template_id == "korvid-scenario"

    assert journey.reference == "journeys/tui-follow"
    assert journey.case_id == "tui-follow"
    assert journey.kind == "journey"
    assert json.loads(journey.prompt) == [
        "The web pod in namespace front will not start. Find out why.",
        "Put the failing pod's details on screen so I can read them myself.",
        "Now show me the log for that container.",
    ]
    assert journey.questions == tuple(json.loads(journey.prompt))
    assert journey.eval_case("ollama/qwen3:0.6b").template_id == "korvid-journey"

    for case in (scenario, journey):
        source = SOURCE_ROOT / case.source_path
        assert case.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
        assert case.eval_case("ollama/qwen3:0.6b").case_id == case.case_id


@pytest.mark.parametrize(
    "reference",
    (
        "",
        "../scenarios/image-pull-typo",
        "scenarios/../journeys/tui-follow",
        "operations/scale-deployment-up",
        "scenarios/not-real",
        "scenarios/image-pull-typo.yaml",
    ),
)
def test_source_case_references_are_exact_tracked_original_names(reference: str) -> None:
    with pytest.raises(ValueError):
        load_source_cases(SOURCE_ROOT, (reference,))


@requires_source
def test_source_case_ids_must_be_unique() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        load_source_cases(
            SOURCE_ROOT,
            ("scenarios/image-pull-typo", "scenarios/image-pull-typo"),
        )


def test_prompt_candidate_is_only_the_actual_tier_pack_text() -> None:
    text = "Operate from the authored fixture and cite decisive evidence."
    candidate = prompt_candidate(text)

    assert candidate.candidate_id == "korvid-upstream-prompt"
    assert candidate.components == {"tier_pack": text}
    assert tier_pack_from_candidate(candidate) == text
    assert candidate.metadata == {
        "scope": "korvid-upstream-tier-pack",
        "korvid_version": "0.4.1",
        "korvid_revision": "33c483e041006eb20259a024ed85a9323e52c8f0",
    }
