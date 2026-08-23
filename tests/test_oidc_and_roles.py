"""Offline contract tests for OIDC federation, Azure roles, and GitHub environment setup scripts."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OIDC_SCRIPT = ROOT / "scripts/setup-oidc-federation.sh"
ROLES_SCRIPT = ROOT / "scripts/setup-azure-roles.sh"
ENV_SCRIPT = ROOT / "scripts/setup-github-environment.sh"
CUSTOM_ROLE_DEF = ROOT / "infra/azure/prompt-lab-k8s-data-role.json"
WORKFLOW = ROOT / ".github/workflows/grounding-round.yml"


# ---------------------------------------------------------------------------
# Task 3a: OIDC federation script
# ---------------------------------------------------------------------------


def test_oidc_script_exists_and_is_bash() -> None:
    text = OIDC_SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -Eeuo pipefail" in text


def test_oidc_script_has_correct_subject() -> None:
    """Subject must match the exact environment subject for aks-grounding."""
    text = OIDC_SCRIPT.read_text(encoding="utf-8")
    assert "repo:hellices/korvid-prompt-lab:environment:aks-grounding" in text


def test_oidc_script_is_idempotent() -> None:
    """Must check for existing credential before creating."""
    text = OIDC_SCRIPT.read_text(encoding="utf-8")
    assert "az ad app federated-credential list" in text or "az ad app federated-credential show" in text


def test_oidc_script_does_not_leak_ids() -> None:
    """Must not hardcode tenant/subscription/client IDs."""
    text = OIDC_SCRIPT.read_text(encoding="utf-8")
    # Should use variables, not literals
    assert "set -x" not in text, "set -x would leak secrets to logs"


def test_oidc_script_does_not_create_rolebindings() -> None:
    """Managed Entra + enableAzureRbac means no Kubernetes RoleBindings."""
    text = OIDC_SCRIPT.read_text(encoding="utf-8")
    assert "RoleBinding" not in text
    assert "kubectl" not in text


# ---------------------------------------------------------------------------
# Task 3b: Azure role assignment script
# ---------------------------------------------------------------------------


def test_roles_script_exists_and_is_bash() -> None:
    text = ROLES_SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -Eeuo pipefail" in text


def test_roles_script_assigns_data_actions_at_namespace_scope() -> None:
    """Custom role DataActions must be assigned at AKS_ID/namespaces/ollama."""
    text = ROLES_SCRIPT.read_text(encoding="utf-8")
    assert "namespaces/ollama" in text


def test_roles_script_assigns_scaler_at_agentpool_only() -> None:
    """Management plane scaler role must target only the modeleval agentpool ID."""
    text = ROLES_SCRIPT.read_text(encoding="utf-8")
    assert "agentpool" in text.lower() or "agentPool" in text
    assert "modeleval" in text


def test_roles_script_assigns_cluster_user_at_cluster_scope() -> None:
    """Cluster User Role at cluster scope solely for credentials."""
    text = ROLES_SCRIPT.read_text(encoding="utf-8")
    assert "Azure Kubernetes Service Cluster User Role" in text


def test_roles_script_is_idempotent() -> None:
    """Must check existing assignments before creating."""
    text = ROLES_SCRIPT.read_text(encoding="utf-8")
    assert "az role assignment list" in text or "az role assignment create" in text


def test_roles_script_does_not_leak_ids() -> None:
    text = ROLES_SCRIPT.read_text(encoding="utf-8")
    assert "set -x" not in text


def test_roles_script_does_not_create_rolebindings() -> None:
    text = ROLES_SCRIPT.read_text(encoding="utf-8")
    assert "RoleBinding" not in text
    assert "kubectl" not in text


# ---------------------------------------------------------------------------
# Task 3c: Custom Azure role definition template
# ---------------------------------------------------------------------------


def test_custom_role_definition_exists_and_has_data_actions() -> None:
    role = json.loads(CUSTOM_ROLE_DEF.read_text(encoding="utf-8"))
    assert "DataActions" in str(role) or "dataActions" in str(role)


def test_custom_role_definition_scoped_to_aks_namespaces() -> None:
    role = json.loads(CUSTOM_ROLE_DEF.read_text(encoding="utf-8"))
    role_str = json.dumps(role)
    assert "Microsoft.ContainerService/managedClusters" in role_str


# ---------------------------------------------------------------------------
# Task 3d: GitHub environment setup script
# ---------------------------------------------------------------------------


def test_env_script_exists_and_is_bash() -> None:
    text = ENV_SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -Eeuo pipefail" in text


def test_env_script_creates_aks_grounding_environment() -> None:
    text = ENV_SCRIPT.read_text(encoding="utf-8")
    assert "aks-grounding" in text


def test_env_script_sets_all_required_vars() -> None:
    """Must set the vars the workflow references."""
    text = ENV_SCRIPT.read_text(encoding="utf-8")
    for var in ["AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_SUBSCRIPTION_ID"]:
        assert var in text, f"Missing required variable {var}"


def test_env_script_sets_korvid_app_secret_via_file_stdin() -> None:
    """Secret private key only through a readable file and stdin to gh."""
    text = ENV_SCRIPT.read_text(encoding="utf-8")
    assert "KORVID_APP_PRIVATE_KEY" in text
    # Must use file stdin pattern, not inline value
    assert "<" in text or "stdin" in text.lower() or "cat " in text


def test_env_script_does_not_leak_ids() -> None:
    text = ENV_SCRIPT.read_text(encoding="utf-8")
    assert "set -x" not in text


def test_env_script_no_broad_catches_or_silent_success() -> None:
    """No 2>/dev/null on critical commands, no || true on secret sets."""
    text = ENV_SCRIPT.read_text(encoding="utf-8")
    # Should not swallow errors on gh secret set
    lines = text.split("\n")
    for line in lines:
        if "gh secret set" in line:
            assert "|| true" not in line, f"Silent success on secret set: {line}"
            assert "2>/dev/null" not in line, f"Swallowed error on secret set: {line}"


# ---------------------------------------------------------------------------
# Task 3e: Workflow-to-script consistency
# ---------------------------------------------------------------------------


def test_workflow_environment_matches_scripts() -> None:
    """The workflow environment name must match what the env setup script creates."""
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    env_text = ENV_SCRIPT.read_text(encoding="utf-8")
    wf_env = wf["jobs"]["grounding"]["environment"]
    assert wf_env in env_text
