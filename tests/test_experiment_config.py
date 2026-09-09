from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from korvid_prompt_lab.contracts import KorvidUpstreamServing
from korvid_prompt_lab.experiment_config import (
    AKSConnection,
    LoopbackConnection,
    ModelProfile,
    ReflectionProfile,
    load_experiment,
)
from korvid_prompt_lab.upstream_contract import KORVID_REVISION

TARGET_DIGEST = f"sha256:{'1' * 64}"
REFLECTION_DIGEST = f"sha256:{'2' * 64}"


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 2,
        "campaign_id": "unified-native",
        "runtime": {
            "backend": "korvid_upstream",
            "source_root": "env:KORVID_NATIVE_SOURCE_ROOT",
            "korvid_revision": KORVID_REVISION,
            "timeout_seconds": 240,
        },
        "serving": {
            "backend": "loopback",
            "base_url": "env:KORVID_NATIVE_MODEL_URL",
        },
        "model": {
            "reference": "ollama/qwen3:0.6b",
            "digest": TARGET_DIGEST,
            "options": {
                "native_thinking": True,
                "think": False,
                "num_ctx": 16384,
                "temperature": 0.0,
            },
        },
        "reflection": {
            "reference": "ollama_chat/qwen3:14b",
            "digest": REFLECTION_DIGEST,
            "timeout_seconds": 180,
            "options": {
                "reasoning_effort": "disable",
                "num_ctx": 4096,
                "temperature": 0.2,
                "max_tokens": 512,
            },
        },
        "evaluation": {
            "repetitions": 5, "seed": 0,
            "train": ["scenarios/healthy-deployment"],
            "validation": ["journeys/tui-follow"],
            "holdout": ["journeys/compare-namespaces"],
        },
        "search": {
            "stages": [
                {"name": "explore", "metric_calls": 16, "seeds": [0, 1]},
                {"name": "refine", "metric_calls": 24, "seeds": [2]},
            ],
            "total_metric_calls": 96,
            "max_evaluations": 256,
            "max_proposals": 12,
            "wall_clock_seconds": 3600,
            "stagnation_attempt_limit": 3,
        },
    }


def _load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]
):
    monkeypatch.setenv("KORVID_NATIVE_SOURCE_ROOT", str(tmp_path / "korvid"))
    if "KORVID_NATIVE_MODEL_URL" not in os.environ:
        monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434/")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_experiment(path)


def test_load_experiment_builds_native_campaign_with_profile_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _load(tmp_path, monkeypatch, _manifest())

    assert isinstance(spec.runtime, KorvidUpstreamServing)
    assert spec.runtime.base_url == ""
    assert isinstance(spec.serving, LoopbackConnection)
    assert spec.serving.base_url == "http://127.0.0.1:11434"
    assert spec.model.model_tag == "qwen3:0.6b"
    assert spec.reflection.model_tag == "qwen3:14b"
    assert spec.repetitions == 5
    assert spec.evaluation_seed == 0
    assert spec.total_metric_calls == 96
    assert spec.max_proposals == 12
    assert spec.manifest["search"]["max_proposals"] == 12
    assert spec.to_mapping() == spec.manifest
    json.dumps(spec.to_mapping(), sort_keys=True)
    assert re.fullmatch(r"[0-9a-f]{64}", spec.fingerprint)

    from korvid_prompt_lab.upstream_contract import SourceCase

    original_cases = tuple(SourceCase(
        reference=ref, case_id=ref.split("/")[1],
        kind="journey" if ref.startswith("journeys/") else "scenario",
        prompt="Unit metadata fixture", source_path=f"src/korvid/evals/{ref}.yaml",
        source_sha256="1" * 64, questions=("Unit metadata fixture",),
    ) for ref in spec.references)
    monkeypatch.setattr("korvid_prompt_lab.upstream_contract.load_source_cases", lambda root, refs: original_cases)
    campaign = spec.campaign("http://127.0.0.1:43123")
    assert campaign.campaign_id == "unified-native"
    assert campaign.repetitions == 5
    assert campaign.models == ("ollama/qwen3:0.6b",)
    assert [case.case_id for case in campaign.cases] == ["healthy-deployment", "tui-follow", "compare-namespaces"]
    assert all(case.template_id.startswith("korvid-") for case in campaign.cases)
    assert isinstance(campaign.serving, KorvidUpstreamServing)
    assert dict(campaign.evaluation_splits)["holdout"] == ("compare-namespaces",)
    assert campaign.serving.base_url == "http://127.0.0.1:43123"
    assert dict(campaign.serving.model_options) == spec.model.options


def test_profile_options_are_immutable_and_fingerprint_is_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _manifest()
    first = _load(tmp_path, monkeypatch, payload)
    second = _load(tmp_path, monkeypatch, copy.deepcopy(payload))
    assert first.fingerprint == second.fingerprint
    assert first.manifest["baseline"] == "unchanged_korvid_prompt_pack"
    assert first.manifest["evaluation"]["validation"] == ["journeys/tui-follow"]

    with pytest.raises(TypeError):
        first.model.options["num_ctx"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        first.reflection.options["max_tokens"] = 1  # type: ignore[index]

    changed = copy.deepcopy(payload)
    changed["model"]["options"]["num_ctx"] = 8192  # type: ignore[index]
    assert _load(tmp_path, monkeypatch, changed).fingerprint != first.fingerprint


def test_aks_connection_is_parsed_without_allocating_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _manifest()
    payload["serving"] = {
        "backend": "aks_port_forward",
        "resource_group": "rg-eval",
        "cluster_name": "aks-eval",
        "namespace": "models",
        "service": "ollama",
        "node_pool": "modeleval",
    }
    spec = _load(tmp_path, monkeypatch, payload)
    assert spec.serving == AKSConnection(
        backend="aks_port_forward",
        resource_group="rg-eval",
        cluster_name="aks-eval",
        namespace="models",
        service="ollama",
        node_pool="modeleval",
    )
    assert "base_url" not in spec.manifest["serving"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"unexpected": 1}), "unknown field"),
        (lambda value: value["runtime"].update({"korvid_revision": "bad"}), "revision"),
        (lambda value: value["runtime"].update({"source_root": "/literal"}), "source_root"),
        (lambda value: value["evaluation"].update({"repetitions": 3}), "repetitions"),
        (lambda value: value["evaluation"].update({"seed": True}), "seed"),
        (lambda value: value["search"].update({"wall_clock_seconds": 0}), "wall_clock"),
        (lambda value: value["search"].update({"max_proposals": 0}), "max_proposals"),
        (lambda value: value["search"].update({"max_proposals": True}), "max_proposals"),
        (lambda value: value["search"]["stages"][1].update({"name": "explore"}), "duplicate"),
        (lambda value: value["search"]["stages"][1].update({"seeds": [1, 2]}), "duplicate"),
        (lambda value: value["model"].update({"reference": "openai/qwen"}), "ollama/"),
        (lambda value: value["model"].update({"digest": "1" * 64}), "sha256"),
        (lambda value: value["model"]["options"].update({"native_thinking": False}), "native_thinking"),
        (lambda value: value["model"]["options"].update({"num_ctx": True}), "num_ctx"),
        (lambda value: value["model"]["options"].update({"temperature": float("nan")}), "temperature"),
        (lambda value: value["model"]["options"].update({"api_base": "http://bad"}), "unknown field"),
        (lambda value: value["reflection"].update({"reference": "ollama/qwen"}), "ollama_chat/"),
        (lambda value: value["reflection"]["options"].update({"tools": "all"}), "unknown field"),
    ],
)
def test_load_experiment_rejects_invalid_or_untrusted_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    payload = _manifest()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, monkeypatch, payload)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:11434",
        "http://model.internal:11434",
        "http://127.0.0.1:11434/v1",
        "http://user:password@127.0.0.1:11434",
    ],
)
def test_loopback_connection_requires_env_resolved_http_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", url)
    with pytest.raises(ValueError, match="loopback|root|credentials"):
        _load(tmp_path, monkeypatch, _manifest())


def test_profiles_reject_reserved_fields_when_constructed_directly() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        ModelProfile(
            reference="ollama/qwen3:0.6b",
            digest=TARGET_DIGEST,
            options={"native_thinking": True, "credentials": "literal"},
        )
    with pytest.raises(ValueError, match="unknown field"):
        ReflectionProfile(
            reference="ollama_chat/qwen3:14b",
            digest=REFLECTION_DIGEST,
            timeout_seconds=180,
            options={"api_base": "http://bad"},
        )


def test_model_tag_is_the_complete_suffix_after_provider_prefix() -> None:
    target = ModelProfile(
        reference="ollama/team/qwen3:0.6b",
        digest=TARGET_DIGEST,
        options={"native_thinking": True},
    )
    reflection = ReflectionProfile(
        reference="ollama_chat/team/qwen3:14b",
        digest=REFLECTION_DIGEST,
        timeout_seconds=180,
        options={},
    )
    assert target.model_tag == "team/qwen3:0.6b"
    assert reflection.model_tag == "team/qwen3:14b"
