#!/usr/bin/env bash
# setup-oidc-federation.sh — idempotent OIDC federated credential for the
# Prompt Lab workflow identity.
#
# Creates a federated identity credential on the Entra application so GitHub
# Actions can authenticate via OIDC from the aks-grounding environment.
#
# The cluster uses managed Entra with enableAzureRbac=true, so Kubernetes
# API authorization is handled entirely through Azure role DataActions — no
# Kubernetes role bindings are created.
#
# Required environment variables:
#   AZURE_APP_OBJECT_ID  — Object ID of the Entra application registration
#   AZURE_CLIENT_ID      — Client (application) ID
#
# Usage:  scripts/setup-oidc-federation.sh

set -Eeuo pipefail

: "${AZURE_APP_OBJECT_ID:?AZURE_APP_OBJECT_ID is required}"
: "${AZURE_CLIENT_ID:?AZURE_CLIENT_ID is required}"

CREDENTIAL_NAME="prompt-lab-aks-grounding"
SUBJECT="repo:hellices/korvid-prompt-lab:environment:aks-grounding"
ISSUER="https://token.actions.githubusercontent.com"
AUDIENCE="api://AzureADTokenExchange"

echo "Checking for existing federated credential '${CREDENTIAL_NAME}'..."

existing="$(az ad app federated-credential list \
  --id "${AZURE_APP_OBJECT_ID}" \
  --query "[?name=='${CREDENTIAL_NAME}'].name" \
  --output tsv)"

if [[ -n "$existing" ]]; then
  echo "Federated credential '${CREDENTIAL_NAME}' already exists — skipping creation."
  exit 0
fi

echo "Creating federated credential '${CREDENTIAL_NAME}'..."
az ad app federated-credential create \
  --id "${AZURE_APP_OBJECT_ID}" \
  --parameters "{
    \"name\": \"${CREDENTIAL_NAME}\",
    \"issuer\": \"${ISSUER}\",
    \"subject\": \"${SUBJECT}\",
    \"audiences\": [\"${AUDIENCE}\"],
    \"description\": \"GitHub Actions OIDC for korvid-prompt-lab aks-grounding environment\"
  }"

echo "Federated credential '${CREDENTIAL_NAME}' created successfully."
