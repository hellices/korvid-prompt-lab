from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.config import load_campaign, load_candidate
from korvid_prompt_lab.contracts import (
    DEFAULT_BRIDGE_TIMEOUT_SECONDS,
    AKSPortForwardServing,
    Candidate,
    KorvidReadonlyServing,
    ProcessServing,
)

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
    monkeypatch.setenv("KORVID_AKS_NAMESPACE", "ollama")
    monkeypatch.setenv("KORVID_AKS_SERVICE", "ollama")
    monkeypatch.setenv("KORVID_AKS_MODEL", "qwen3:4b")

    local = load_campaign(ROOT / "examples/campaigns/local-smoke.yaml")
    aks = load_campaign(ROOT / "examples/campaigns/aks-shared-runners.yaml")

    assert local.campaign_id == "local-smoke"
    assert local.repetitions == 5
    assert local.bridge_timeout_seconds == pytest.approx(60.0)
    assert local.models == ("mock-small",)
    assert [case.case_id for case in local.cases] == ["smoke-happy", "smoke-guardrail"]
    assert local.cases[0].models == ("mock-small",)
    assert isinstance(local.serving, ProcessServing)
    assert local.serving.backend == "process"
    assert local.serving.command == (
        "python3",
        "tests/fixtures/fake_korvid_bridge.py",
        "--request",
        "{request}",
        "--response",
        "{response}",
    )

    assert aks.campaign_id == "aks-shared-runners"
    assert aks.repetitions == 5
    assert aks.bridge_timeout_seconds == pytest.approx(900.0)
    assert aks.models == ("qwen3:4b",)
    assert [case.case_id for case in aks.cases] == ["aks-scale-deployment-up", "aks-restart-denied"]
    # The template ids must be real Korvid operation journeys and the prompts must be
    # those journeys' own first turns, or `korvid-bridge` refuses the case.
    assert [case.template_id for case in aks.cases] == ["scale-deployment-up", "restart-denied"]
    assert [case.prompt for case in aks.cases] == [
        "Scale checkout-a in shop-a from 2 to 3 replicas.",
        "Restart the api deployment in shop-a.",
    ]
    assert isinstance(aks.serving, AKSPortForwardServing)
    assert aks.serving.backend == "aks_port_forward"
    assert aks.serving.resource_group == "rg-pension-guard"
    assert aks.serving.cluster_name == "aks-shared-runners"
    assert aks.serving.namespace == "ollama"
    assert aks.serving.service == "ollama"
    assert aks.serving.model == "qwen3:4b"
    assert aks.serving.command == (
        "korvid-bridge",
        "--request",
        "{request}",
        "--response",
        "{response}",
        "--turn-timeout",
        "300",
    )


def test_load_qualification_campaign_from_example_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KORVID_AKS_NAMESPACE", "ollama")
    monkeypatch.setenv("KORVID_AKS_SERVICE", "ollama")
    monkeypatch.setenv("KORVID_AKS_MODEL", "qwen3:0.6b")

    campaign = load_campaign(ROOT / "examples/campaigns/aks-small-operator-qualification.yaml")

    assert campaign.campaign_id == "aks-small-operator-qualification"
    assert campaign.repetitions == 5
    assert campaign.bridge_timeout_seconds == pytest.approx(900.0)
    assert campaign.models == ("qwen3:0.6b",)
    assert [case.case_id for case in campaign.cases] == [
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
    ]
    assert [case.template_id for case in campaign.cases] == [
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
    ]
    assert isinstance(campaign.serving, AKSPortForwardServing)
    assert campaign.serving.namespace == "ollama"
    assert campaign.serving.service == "ollama"
    assert campaign.serving.model == "qwen3:0.6b"


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


def _timeout_campaign_yaml(timeout_line: str) -> str:
    return (
        """
schema_version: 1
campaign_id: timeout-campaign
repetitions: 1
"""
        + timeout_line
        + """models: [mock-small]
cases:
  - case_id: case-a
    template_id: template-a
    prompt: one
    models: [mock-small]
serving:
  backend: process
  command: [python3, bridge.py, --request, "{request}", --response, "{response}"]
"""
    ).strip() + "\n"


def test_load_campaign_defaults_the_bridge_timeout_to_the_reviewed_value(tmp_path: Path) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(_timeout_campaign_yaml(""), encoding="utf-8")

    campaign = load_campaign(path)

    assert DEFAULT_BRIDGE_TIMEOUT_SECONDS == 300.0
    assert campaign.bridge_timeout_seconds == DEFAULT_BRIDGE_TIMEOUT_SECONDS


def test_load_campaign_parses_an_explicit_bridge_timeout(tmp_path: Path) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(_timeout_campaign_yaml("bridge_timeout_seconds: 45.5\n"), encoding="utf-8")

    campaign = load_campaign(path)

    assert campaign.bridge_timeout_seconds == pytest.approx(45.5)


@pytest.mark.parametrize(
    "timeout_value",
    ["0", "-1", "-0.5", '"30"', "true", "null", "[30]", ".nan", ".inf"],
)
def test_load_campaign_rejects_non_positive_bridge_timeouts(tmp_path: Path, timeout_value: str) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(_timeout_campaign_yaml(f"bridge_timeout_seconds: {timeout_value}\n"), encoding="utf-8")

    with pytest.raises(ValueError, match="bridge_timeout_seconds must be a positive number"):
        load_campaign(path)


def _aks_campaign_yaml(serving_lines: str) -> str:
    return (
        """
schema_version: 1
campaign_id: aks-campaign
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
  namespace: korvid
  service: korvid-api
  model: qwen3-4b
"""
        + serving_lines
    ).strip() + "\n"


def test_load_campaign_parses_the_explicit_local_bridge_command_for_aks_serving(tmp_path: Path) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(
        _aks_campaign_yaml(
            """
  command: [korvid-bridge, --request, "{request}", --response, "{response}"]
"""
        ),
        encoding="utf-8",
    )

    campaign = load_campaign(path)

    assert isinstance(campaign.serving, AKSPortForwardServing)
    assert campaign.serving.command == (
        "korvid-bridge",
        "--request",
        "{request}",
        "--response",
        "{response}",
    )


@pytest.mark.parametrize(
    ("serving_lines", "message"),
    [
        ("", "serving.command"),
        ("""\n  command: []\n""", "serving.command"),
        ("""\n  command: [korvid-bridge, --request, "{request}"]\n""", r"\{response\}"),
        ("""\n  command: [korvid-bridge, "env:KORVID_BRIDGE_ARGS", --request, "{request}", --response, "{response}"]\n""", "env:"),
    ],
)
def test_load_campaign_rejects_unusable_aks_bridge_commands(
    tmp_path: Path, serving_lines: str, message: str
) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(_aks_campaign_yaml(serving_lines), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_campaign(path)


def test_load_campaign_rejects_process_commands_without_artifact_placeholders(tmp_path: Path) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(
        """
schema_version: 1
campaign_id: process-campaign
repetitions: 1
models: [mock-small]
cases:
  - case_id: case-a
    template_id: template-a
    prompt: one
    models: [mock-small]
serving:
  backend: process
  command: [python3, bridge.py]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\{request\}"):
        load_campaign(path)


def _korvid_readonly_campaign_yaml(serving_lines: str) -> str:
    return (
        """
schema_version: 1
campaign_id: korvid-readonly-campaign
repetitions: 1
models: [qwen3-4b]
cases:
  - case_id: oom-killed
    template_id: template-a
    prompt: "The database pod keeps restarting. Why?"
    models: [qwen3-4b]
serving:
  backend: korvid_readonly
  provider: ollama
  base_url: env:KORVID_READONLY_BASE_URL
  profile: small
  timeout_seconds: 90
"""
        + serving_lines
    ).strip() + "\n"


def test_load_campaign_parses_korvid_readonly_serving(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KORVID_READONLY_BASE_URL", "http://127.0.0.1:11434")
    path = tmp_path / "campaign.yaml"
    path.write_text(_korvid_readonly_campaign_yaml(""), encoding="utf-8")

    campaign = load_campaign(path)

    assert isinstance(campaign.serving, KorvidReadonlyServing)
    assert campaign.serving.backend == "korvid_readonly"
    assert campaign.serving.provider == "ollama"
    assert campaign.serving.base_url == "http://127.0.0.1:11434"
    assert campaign.serving.profile == "small"
    assert campaign.serving.timeout_seconds == pytest.approx(90.0)
    assert not hasattr(campaign.serving, "__dict__")


def test_load_campaign_rejects_korvid_readonly_missing_base_url_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KORVID_READONLY_BASE_URL", raising=False)
    path = tmp_path / "campaign.yaml"
    path.write_text(_korvid_readonly_campaign_yaml(""), encoding="utf-8")

    with pytest.raises(ValueError, match="serving.base_url"):
        load_campaign(path)


def test_load_campaign_rejects_literal_korvid_readonly_base_url(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(
        _korvid_readonly_campaign_yaml("").replace(
            "env:KORVID_READONLY_BASE_URL", "http://127.0.0.1:11434"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"serving\.base_url.*env:"):
        load_campaign(path)


@pytest.mark.parametrize(
    ("field_overrides", "message"),
    [
        ("  provider: grpc\n", "provider"),
        ("  profile: medium\n", "profile"),
        ("  timeout_seconds: 0\n", "timeout_seconds"),
        ("  timeout_seconds: -5\n", "timeout_seconds"),
        ("  unexpected: true\n", "unknown"),
    ],
)
def test_load_campaign_rejects_invalid_korvid_readonly_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field_overrides: str, message: str
) -> None:
    monkeypatch.setenv("KORVID_READONLY_BASE_URL", "http://127.0.0.1:11434")
    path = tmp_path / "campaign.yaml"
    path.write_text(_korvid_readonly_campaign_yaml(field_overrides), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_campaign(path)


def test_load_campaign_rejects_openai_compat_provider_with_missing_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KORVID_READONLY_BASE_URL", "http://127.0.0.1:8000/v1")
    path = tmp_path / "campaign.yaml"
    path.write_text(
        (
            """
schema_version: 1
campaign_id: korvid-readonly-campaign
repetitions: 1
models: [qwen3-4b]
cases:
  - case_id: oom-killed
    template_id: template-a
    prompt: "The database pod keeps restarting. Why?"
    models: [qwen3-4b]
serving:
  backend: korvid_readonly
  provider: openai-compat
  base_url: env:KORVID_READONLY_BASE_URL
  profile: full
"""
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="serving.timeout_seconds"):
        load_campaign(path)


def test_readonly_small_example_matches_installed_bundled_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checked-in example never vendors scenario text: it hand-copies exact

    case ids and authored questions from whichever Korvid wheel is installed.
    This test re-derives that same catalog from the installed wheel so a
    Korvid dependency bump that silently reworded a scenario fails this test
    visibly instead of corrupting the example campaign's identity silently.
    """
    from korvid.evals.scenario import bundled_scenarios_dir, load_scenario

    monkeypatch.setenv("KORVID_READONLY_BASE_URL", "http://127.0.0.1:11434")
    campaign = load_campaign(ROOT / "examples/campaigns/korvid-readonly-small.yaml")

    assert isinstance(campaign.serving, KorvidReadonlyServing)
    assert campaign.serving.backend == "korvid_readonly"
    assert campaign.serving.provider == "ollama"
    assert campaign.serving.profile == "small"

    bundled_questions = {
        scenario.id: scenario.question
        for scenario in (
            load_scenario(path) for path in bundled_scenarios_dir().glob("*.yaml")
        )
    }
    assert bundled_questions, "installed Korvid wheel exposed no bundled scenarios"

    assert len(campaign.cases) >= 4
    case_ids = [case.case_id for case in campaign.cases]
    assert len(set(case_ids)) == len(case_ids), "example campaign cases must be unique"

    for case in campaign.cases:
        assert case.case_id in bundled_questions, (
            f"{case.case_id!r} is not a scenario shipped by the installed Korvid "
            "wheel; update examples/campaigns/korvid-readonly-small.yaml to match "
            "the currently installed korvid[agent] distribution"
        )
        assert case.prompt == bundled_questions[case.case_id], (
            f"{case.case_id!r}'s authored question changed in the installed "
            "Korvid wheel; update examples/campaigns/korvid-readonly-small.yaml's "
            "prompt to match verbatim rather than silently drifting from it"
        )

    train_case_ids = {"oom-killed", "crashloop-app-panic"}
    validation_case_ids = {"image-pull-typo", "healthy-deployment"}
    assert train_case_ids <= set(case_ids)
    assert validation_case_ids <= set(case_ids)
    assert train_case_ids.isdisjoint(validation_case_ids)
