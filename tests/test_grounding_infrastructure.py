"""Offline contract tests for the Prompt Lab ARC runner infrastructure files."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VALUES = ROOT / "infra/arc/prompt-lab-runners-values.yaml"
SERVICE_ACCOUNT = ROOT / "infra/arc/prompt-lab-runner-service-account.yaml"
RUNNER_DOCKERFILE = ROOT / "infra/arc/runner/Dockerfile"


def test_prompt_lab_runner_values_are_repo_scoped_and_serial() -> None:
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    assert values["githubConfigUrl"] == "https://github.com/hellices/korvid-prompt-lab"
    assert values["githubConfigSecret"] == "prompt-lab-runners-github-app"
    assert values["runnerScaleSetName"] == "prompt-lab-runners"
    assert values["minRunners"] == 0
    assert values["maxRunners"] == 1


def test_prompt_lab_runners_cannot_schedule_on_model_compute() -> None:
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    pod = values["template"]["spec"]
    assert pod["serviceAccountName"] == "prompt-lab-runners-no-permission"
    assert pod["automountServiceAccountToken"] is False
    assert pod["nodeSelector"] == {"workload": "gha-runner"}
    assert pod["tolerations"] == [
        {
            "key": "gha-runner",
            "operator": "Equal",
            "value": "true",
            "effect": "NoSchedule",
        }
    ]
    assert all(item["key"] != "workload" for item in pod["tolerations"])
    assert pod["containers"][0]["image"] == (
        "acrpensionguard.azurecr.io/runner-base:prompt-lab-v1"
    )


def test_runner_service_account_is_tokenless_and_role_free() -> None:
    docs = list(yaml.safe_load_all(SERVICE_ACCOUNT.read_text(encoding="utf-8")))
    assert [doc["kind"] for doc in docs] == ["Namespace", "ServiceAccount"]
    assert docs[0]["metadata"]["name"] == "arc-runners-prompt-lab"
    assert docs[1]["metadata"]["namespace"] == "arc-runners-prompt-lab"
    assert docs[1]["automountServiceAccountToken"] is False


def test_prompt_lab_runner_image_pins_required_tools_and_non_root_user() -> None:
    body = RUNNER_DOCKERFILE.read_text(encoding="utf-8")
    assert body.startswith(
        "FROM ghcr.io/astral-sh/uv:0.10.9 AS uv\n"
        "FROM acrpensionguard.azurecr.io/runner-base:v1"
    )
    assert "--kubectl-version v1.35.6" in body
    assert "--kubelogin-version v0.2.19" in body
    assert "COPY --from=uv /uv /uvx /usr/local/bin/" in body
    assert body.rstrip().endswith("USER runner")
