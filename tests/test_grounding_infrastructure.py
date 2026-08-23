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


def test_runner_container_runs_non_root() -> None:
    """Fix 1: container securityContext must enforce non-root execution."""
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    sc = values["template"]["spec"]["containers"][0]["securityContext"]
    assert sc["runAsNonRoot"] is True
    assert sc["allowPrivilegeEscalation"] is False


def test_controller_service_account_cross_namespace_discovery() -> None:
    """Fix 2: explicit controllerServiceAccount avoids cross-namespace discovery failure."""
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    csa = values["controllerServiceAccount"]
    assert csa["name"] == "arc-gha-rs-controller"
    assert csa["namespace"] == "arc-systems"


def test_runner_container_security_context_has_numeric_uid_gid() -> None:
    """Task 1 cannot-verify fix: add runAsUser/runAsGroup so Kubernetes can
    enforce non-root numerically even when the image USER is the string 'runner'."""
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    sc = values["template"]["spec"]["containers"][0]["securityContext"]
    assert sc["runAsUser"] == 1001
    assert sc["runAsGroup"] == 1001


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


# ---------------------------------------------------------------------------
# Task 2: Grounding-round workflow routes to the Prompt Lab runner scale set
# ---------------------------------------------------------------------------
WORKFLOW = ROOT / ".github/workflows/grounding-round.yml"


def test_grounding_workflow_routes_to_prompt_lab_runners() -> None:
    """The grounding-round job must run on prompt-lab-runners, not korvid-runners."""
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    runs_on = wf["jobs"]["grounding"]["runs-on"]
    assert runs_on == "prompt-lab-runners", (
        f"Expected 'prompt-lab-runners' but got {runs_on!r}. "
        "Update .github/workflows/grounding-round.yml runs-on."
    )


def test_grounding_workflow_preserves_environment_and_concurrency() -> None:
    """Environment and concurrency settings must be preserved after runner change."""
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert wf["jobs"]["grounding"]["environment"] == "aks-grounding"
    assert wf["jobs"]["grounding"]["timeout-minutes"] == 180
    assert wf["concurrency"]["cancel-in-progress"] is False


def test_grounding_workflow_preserves_permissions() -> None:
    """Top-level permissions must remain unchanged."""
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    perms = wf["permissions"]
    assert perms["contents"] == "read"
    assert perms["id-token"] == "write"
    assert perms["pull-requests"] == "write"
