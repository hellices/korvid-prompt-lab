#!/usr/bin/env bash
# setup-azure-roles.sh — idempotent Azure role assignments for the Prompt Lab
# workflow identity.
#
# Three role assignments at three distinct scopes:
#
#   1. Custom DataActions role  →  ${AKS_ID}/namespaces/ollama
#      Grants Kubernetes API access (pods, services, port-forward) through
#      Azure RBAC DataActions — no Kubernetes role bindings needed because the
#      cluster has managed Entra with enableAzureRbac=true.
#
#   2. Management-plane scaler  →  exact modeleval agentpool resource ID
#      Allows `az aks nodepool show/scale` on the GPU pool only.
#
#   3. Azure Kubernetes Service Cluster User Role  →  cluster scope
#      Solely for `az aks get-credentials` (kubeconfig download).
#
# Required environment variables:
#   AZURE_CLIENT_ID        — Service principal / managed identity client ID
#   AZURE_SUBSCRIPTION_ID  — Subscription ID
#   AKS_RESOURCE_GROUP     — AKS cluster resource group (e.g. rg-pension-guard)
#   AKS_CLUSTER_NAME       — AKS cluster name (e.g. aks-shared-runners)
#   CUSTOM_ROLE_DEF_FILE   — Path to the custom role definition JSON
#
# Usage:  scripts/setup-azure-roles.sh

set -Eeuo pipefail

: "${AZURE_CLIENT_ID:?AZURE_CLIENT_ID is required}"
: "${AZURE_SUBSCRIPTION_ID:?AZURE_SUBSCRIPTION_ID is required}"
: "${AKS_RESOURCE_GROUP:?AKS_RESOURCE_GROUP is required}"
: "${AKS_CLUSTER_NAME:?AKS_CLUSTER_NAME is required}"
: "${CUSTOM_ROLE_DEF_FILE:?CUSTOM_ROLE_DEF_FILE is required}"

AKS_ID="/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${AKS_RESOURCE_GROUP}/providers/Microsoft.ContainerService/managedClusters/${AKS_CLUSTER_NAME}"
AGENTPOOL_ID="${AKS_ID}/agentPools/modeleval"

# Resolve the principal ID from the client ID
PRINCIPAL_ID="$(az ad sp show --id "${AZURE_CLIENT_ID}" --query id --output tsv)"

# ---------------------------------------------------------------------------
# Helper: idempotent role assignment
# ---------------------------------------------------------------------------
assign_role_if_absent() {
  local role_name="$1"
  local scope="$2"
  local description="$3"

  existing="$(az role assignment list \
    --assignee "${PRINCIPAL_ID}" \
    --role "${role_name}" \
    --scope "${scope}" \
    --query "length(@)" \
    --output tsv)"

  if [[ "${existing}" -gt 0 ]]; then
    echo "${description}: already assigned — skipping."
    return 0
  fi

  echo "${description}: creating assignment..."
  az role assignment create \
    --assignee "${PRINCIPAL_ID}" \
    --role "${role_name}" \
    --scope "${scope}"
  echo "${description}: assigned successfully."
}

# ---------------------------------------------------------------------------
# 1. Create or update the custom DataActions role definition
# ---------------------------------------------------------------------------
echo "Ensuring custom DataActions role definition exists..."
ROLE_NAME="$(jq -r '.Name // .roleName // .name' "${CUSTOM_ROLE_DEF_FILE}")"

existing_role="$(az role definition list \
  --name "${ROLE_NAME}" \
  --query "length(@)" \
  --output tsv)"

if [[ "${existing_role}" -gt 0 ]]; then
  echo "Custom role '${ROLE_NAME}' exists — updating definition..."
  az role definition update --role-definition "@${CUSTOM_ROLE_DEF_FILE}"
else
  echo "Creating custom role '${ROLE_NAME}'..."
  az role definition create --role-definition "@${CUSTOM_ROLE_DEF_FILE}"
fi

# ---------------------------------------------------------------------------
# 2. Assign custom DataActions role at namespace scope
# ---------------------------------------------------------------------------
assign_role_if_absent \
  "${ROLE_NAME}" \
  "${AKS_ID}/namespaces/ollama" \
  "DataActions role at namespaces/ollama"

# ---------------------------------------------------------------------------
# 3. Assign scaler role at exact modeleval agentpool ID
# ---------------------------------------------------------------------------
assign_role_if_absent \
  "Prompt Lab Nodepool Scaler" \
  "${AGENTPOOL_ID}" \
  "Scaler role at modeleval agentpool"

# ---------------------------------------------------------------------------
# 4. Azure Kubernetes Service Cluster User Role at cluster scope
# ---------------------------------------------------------------------------
assign_role_if_absent \
  "Azure Kubernetes Service Cluster User Role" \
  "${AKS_ID}" \
  "Cluster User Role at cluster scope (credentials only)"

echo "All role assignments verified."
