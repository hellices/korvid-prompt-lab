#!/usr/bin/env bash
# setup-github-environment.sh — idempotent GitHub environment, variables,
# and secrets for the aks-grounding deployment target.
#
# Creates (or verifies) the aks-grounding environment on the repository and
# sets the variables and secrets the grounding-round workflow consumes.
#
# Secrets are set through file-based stdin — no values appear in process
# arguments or shell history.  Debug tracing is never enabled.
#
# Required environment variables:
#   GH_REPO                     — owner/repo (e.g. hellices/korvid-prompt-lab)
#   AZURE_CLIENT_ID             — Entra application client ID
#   AZURE_TENANT_ID             — Entra tenant ID
#   AZURE_SUBSCRIPTION_ID       — Azure subscription ID
#   KORVID_APP_ID               — GitHub App ID for Korvid repository access
#   KORVID_APP_PRIVATE_KEY_FILE — Path to the GitHub App private key PEM file
#   KORVID_AKS_NAMESPACE        — Kubernetes namespace for the campaign
#   KORVID_AKS_SERVICE          — Kubernetes service name for the campaign
#
# Optional (set only for optimize rounds):
#   GROUNDING_REFLECTION_MODEL      — Reflection LLM model identifier
#   GROUNDING_REFLECTION_CRED_FILE  — Path to file containing the reflection credential
#
# Usage:  scripts/setup-github-environment.sh

set -Eeuo pipefail

: "${GH_REPO:?GH_REPO is required (e.g. hellices/korvid-prompt-lab)}"
: "${AZURE_CLIENT_ID:?AZURE_CLIENT_ID is required}"
: "${AZURE_TENANT_ID:?AZURE_TENANT_ID is required}"
: "${AZURE_SUBSCRIPTION_ID:?AZURE_SUBSCRIPTION_ID is required}"
: "${KORVID_APP_ID:?KORVID_APP_ID is required}"
: "${KORVID_APP_PRIVATE_KEY_FILE:?KORVID_APP_PRIVATE_KEY_FILE is required}"
: "${KORVID_AKS_NAMESPACE:?KORVID_AKS_NAMESPACE is required}"
: "${KORVID_AKS_SERVICE:?KORVID_AKS_SERVICE is required}"

ENV_NAME="aks-grounding"

# ---------------------------------------------------------------------------
# 1. Ensure the environment exists
# ---------------------------------------------------------------------------
echo "Ensuring environment '${ENV_NAME}' exists on ${GH_REPO}..."
if gh api "repos/${GH_REPO}/environments/${ENV_NAME}" --silent 2>/dev/null; then
  echo "Environment '${ENV_NAME}' already exists."
else
  echo "Creating environment '${ENV_NAME}'..."
  gh api --method PUT "repos/${GH_REPO}/environments/${ENV_NAME}"
  echo "Environment '${ENV_NAME}' created."
fi

# ---------------------------------------------------------------------------
# 2. Set environment variables (idempotent — gh variable set overwrites)
# ---------------------------------------------------------------------------
echo "Setting environment variables..."

gh variable set AZURE_CLIENT_ID \
  --repo "${GH_REPO}" --env "${ENV_NAME}" --body "${AZURE_CLIENT_ID}"

gh variable set AZURE_TENANT_ID \
  --repo "${GH_REPO}" --env "${ENV_NAME}" --body "${AZURE_TENANT_ID}"

gh variable set AZURE_SUBSCRIPTION_ID \
  --repo "${GH_REPO}" --env "${ENV_NAME}" --body "${AZURE_SUBSCRIPTION_ID}"

gh variable set KORVID_APP_ID \
  --repo "${GH_REPO}" --env "${ENV_NAME}" --body "${KORVID_APP_ID}"

gh variable set KORVID_AKS_NAMESPACE \
  --repo "${GH_REPO}" --env "${ENV_NAME}" --body "${KORVID_AKS_NAMESPACE}"

gh variable set KORVID_AKS_SERVICE \
  --repo "${GH_REPO}" --env "${ENV_NAME}" --body "${KORVID_AKS_SERVICE}"

echo "Environment variables set."

# ---------------------------------------------------------------------------
# 3. Set secrets via file stdin (no inline values)
# ---------------------------------------------------------------------------
echo "Setting KORVID_APP_PRIVATE_KEY secret from file..."

if [[ ! -r "${KORVID_APP_PRIVATE_KEY_FILE}" ]]; then
  echo "Error: KORVID_APP_PRIVATE_KEY_FILE is not readable: ${KORVID_APP_PRIVATE_KEY_FILE}" >&2
  exit 1
fi

gh secret set KORVID_APP_PRIVATE_KEY \
  --repo "${GH_REPO}" --env "${ENV_NAME}" \
  < "${KORVID_APP_PRIVATE_KEY_FILE}"

echo "KORVID_APP_PRIVATE_KEY secret set."

# ---------------------------------------------------------------------------
# 4. Optional: reflection credential for optimize rounds
# ---------------------------------------------------------------------------
if [[ -n "${GROUNDING_REFLECTION_MODEL:-}" ]]; then
  gh variable set GROUNDING_REFLECTION_MODEL \
    --repo "${GH_REPO}" --env "${ENV_NAME}" --body "${GROUNDING_REFLECTION_MODEL}"
  echo "GROUNDING_REFLECTION_MODEL variable set."
fi

if [[ -n "${GROUNDING_REFLECTION_CRED_FILE:-}" && -r "${GROUNDING_REFLECTION_CRED_FILE}" ]]; then
  gh secret set GROUNDING_REFLECTION_CREDENTIAL \
    --repo "${GH_REPO}" --env "${ENV_NAME}" \
    < "${GROUNDING_REFLECTION_CRED_FILE}"
  echo "GROUNDING_REFLECTION_CREDENTIAL secret set."
fi

echo "GitHub environment '${ENV_NAME}' fully configured."
