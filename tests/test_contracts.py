from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.config import load_candidate, load_campaign
from korvid_prompt_lab.contracts import Candidate


ROOT = Path(__file__).resolve().parents[1]


def _sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def test_load_candidate_from_example_yaml() -> None:
    candidate = load_candidate(ROOT / "examples/candidates/shipped-small.yaml")

    assert candidate.candidate_id == "shipped-small"
    assert candidate.components == {
        "system": "You are korvid's bounded Kubernetes operator.",
        "append": "Verify the postcondition before reporting completion.",
        "tool.scale_resource": "Request an approval-gated replica-count change.",
    }
    assert candidate.metadata == {"source": "shipped"}
    assert candidate.fingerprint == _sha256(
        {
            "schema_version": 1,
            "candidate_id": "shipped-small",
            "components": {
                "append": "Verify the postcondition before reporting completion.",
                "system": "You are korvid's bounded Kubernetes operator.",
                "tool.scale_resource": "Request an approval-gated replica-count change.",
            },
        }
    )


def test_frozen_dataclasses_use_slots() -> None:
    candidate = load_candidate(ROOT / "examples/candidates/shipped-small.yaml")
    local = load_campaign(ROOT / "examples/campaigns/local-smoke.yaml")

    assert not hasattr(candidate, "__dict__")
    assert not hasattr(local.cases[0], "__dict__")
    assert not hasattr(local.serving, "__dict__")
    assert not hasattr(local, "__dict__")


def test_candidate_fingerprint_is_stable_for_mapping_order() -> None:
    shuffled = {
        "candidate_id": "tiny",
        "components": {
            "tool.scale_resource": "Scale only when approved.",
            "system": "Be brief.",
            "append": "Check the postcondition.",
        },
        "schema_version": 1,
    }
    reordered = {
        "schema_version": 1,
        "candidate_id": "tiny",
        "components": {
            "append": "Check the postcondition.",
            "system": "Be brief.",
            "tool.scale_resource": "Scale only when approved.",
        },
    }

    left = Candidate.from_mapping(shuffled)
    right = Candidate.from_mapping(reordered)

    assert left.fingerprint == right.fingerprint
    assert left.fingerprint == _sha256(
        {
            "schema_version": 1,
            "candidate_id": "tiny",
            "components": {
                "append": "Check the postcondition.",
                "system": "Be brief.",
                "tool.scale_resource": "Scale only when approved.",
            },
        }
    )


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        (
            {
                "schema_version": 1,
                "candidate_id": "bad",
                "components": {"system": "ok"},
                "unknown": "field",
            },
            "unknown",
        ),
        (
            {
                "schema_version": 1,
                "candidate_id": "bad",
                "components": {"tool.": "empty"},
            },
            "component",
        ),
    ],
)
def test_candidate_from_mapping_rejects_invalid_input(mapping: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Candidate.from_mapping(mapping)


def test_load_campaign_from_example_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KORVID_AKS_NAMESPACE", "korvid")
    monkeypatch.setenv("KORVID_AKS_SERVICE", "korvid-api")
    monkeypatch.setenv("KORVID_AKS_MODEL", "qwen3-4b")

    local = load_campaign(ROOT / "examples/campaigns/local-smoke.yaml")
    aks = load_campaign(ROOT / "examples/campaigns/aks-shared-runners.yaml")

    assert local.campaign_id == "local-smoke"
    assert local.repetitions == 1
    assert local.models == ("mock-small",)
    assert local.cases[0].case_id == "smoke-happy"
    assert local.cases[0].models == ("mock-small",)
    assert local.serving.backend == "process"
    assert local.serving.command == (
        "python",
        "tests/fixtures/fake_korvid_bridge.py",
        "--request",
        "{request}",
        "--response",
        "{response}",
    )

    assert aks.campaign_id == "aks-shared-runners"
    assert aks.repetitions == 2
    assert aks.models == ("qwen3-4b",)
    assert aks.serving.backend == "aks_port_forward"
    assert aks.serving.resource_group == "rg-pension-guard"
    assert aks.serving.cluster_name == "aks-shared-runners"
    assert aks.serving.namespace == "korvid"
    assert aks.serving.service == "korvid-api"
    assert aks.serving.model == "qwen3-4b"


def test_load_campaign_rejects_whitespace_only_env_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KORVID_AKS_NAMESPACE", "   ")
    monkeypatch.setenv("KORVID_AKS_SERVICE", "korvid-api")
    monkeypatch.setenv("KORVID_AKS_MODEL", "qwen3-4b")

    path = tmp_path / "campaign.yaml"
    path.write_text(
        """
schema_version: 1
campaign_id: aks-shared-runners
repetitions: 1
models: [qwen3-4b]
cases:
  - case_id: case-a
    template_id: template-a
    prompt: one
    models: [qwen3-4b]
serving:
  backend: aks_port_forward
  resource_group: rg-pension-guard
  cluster_name: aks-shared-runners
  namespace: env:KORVID_AKS_NAMESPACE
  service: env:KORVID_AKS_SERVICE
  model: env:KORVID_AKS_MODEL
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="serving.namespace"):
        load_campaign(path)


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        (
            """
schema_version: 1
campaign_id: duplicate-cases
repetitions: 1
models: [mock-small]
cases:
  - case_id: case-a
    template_id: template-a
    prompt: one
    models: [mock-small]
  - case_id: case-a
    template_id: template-b
    prompt: two
    models: [mock-small]
serving:
  backend: process
  unexpected: true
  command: [python, -c, "print('ok')"]
""",
            "duplicate",
        ),
        (
            """
schema_version: 1
campaign_id: bad-repetitions
repetitions: 0
models: [mock-small]
cases:
  - case_id: case-a
    template_id: template-a
    prompt: one
    models: [mock-small]
serving:
  backend: process
  command: [python, -c, "print('ok')"]
""",
            "repetitions",
        ),
        (
            """
schema_version: 1
campaign_id: missing-model-coverage
repetitions: 1
models: [mock-small, mock-large]
cases:
  - case_id: case-a
    template_id: template-a
    prompt: one
    models: [mock-small]
serving:
  backend: process
  command: [python, -c, "print('ok')"]
""",
            "coverage",
        ),
        (
            """
schema_version: 1
campaign_id: unknown-field
repetitions: 1
models: [mock-small]
cases:
  - case_id: case-a
    template_id: template-a
    prompt: one
    models: [mock-small]
    unexpected: true
serving:
  backend: process
  command: [python, -c, "print('ok')"]
""",
            "unknown",
        ),
    ],
)
def test_load_campaign_rejects_invalid_campaigns(
    tmp_path: Path, yaml_text: str, message: str
) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml_text.strip() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_campaign(path)
