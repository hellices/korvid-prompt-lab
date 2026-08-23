#!/usr/bin/env bash
# install-prompt-lab-runner.sh — install the Prompt Lab ARC runner scale set.
#
# Secrets reach kubectl only through mode-0600 files and --from-file, never as
# command-line arguments or log output.
#
# Required environment:
#   ARC_GITHUB_APP_ID                 GitHub App id
#   ARC_GITHUB_APP_INSTALLATION_ID    GitHub App installation id
#   ARC_GITHUB_APP_PRIVATE_KEY_FILE   path to a PEM private key file
#
# Usage:  scripts/install-prompt-lab-runner.sh

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

AKS_RESOURCE_GROUP="rg-pension-guard"
AKS_CLUSTER_NAME="aks-shared-runners"
NAMESPACE="arc-runners-prompt-lab"
RELEASE_NAME="prompt-lab-runners"
VALUES_FILE="${REPO_ROOT}/infra/arc/prompt-lab-runners-values.yaml"
SA_MANIFEST="${REPO_ROOT}/infra/arc/prompt-lab-runner-service-account.yaml"

die() {
  printf 'install-prompt-lab-runner: %s\n' "$*" >&2
  exit 1
}

# ── Validate required environment ──────────────────────────────────────────
: "${ARC_GITHUB_APP_ID:?ARC_GITHUB_APP_ID is required}"
: "${ARC_GITHUB_APP_INSTALLATION_ID:?ARC_GITHUB_APP_INSTALLATION_ID is required}"
: "${ARC_GITHUB_APP_PRIVATE_KEY_FILE:?ARC_GITHUB_APP_PRIVATE_KEY_FILE is required}"

[[ -r "${ARC_GITHUB_APP_PRIVATE_KEY_FILE}" ]] \
  || die "private key file is not readable: ${ARC_GITHUB_APP_PRIVATE_KEY_FILE}"

# ── Verify cluster context ────────────────────────────────────────────────
printf 'Verifying AKS cluster context …\n'
current_sub="$(az account show --query id -o tsv 2>/dev/null)" \
  || die "no active Azure subscription"
printf 'Subscription: (active)\n'

az aks get-credentials \
  --resource-group "${AKS_RESOURCE_GROUP}" \
  --name "${AKS_CLUSTER_NAME}" \
  --overwrite-existing >/dev/null 2>&1 \
  || die "cannot get credentials for ${AKS_CLUSTER_NAME}"

kubectl cluster-info >/dev/null 2>&1 \
  || die "cluster is not reachable"

printf 'Cluster: %s\n' "${AKS_CLUSTER_NAME}"

# ── Prepare temporary directory for secret files ──────────────────────────
tmp="$(mktemp -d)"
chmod 0700 "${tmp}"

cleanup() {
  rm -rf "${tmp}"
}
trap cleanup EXIT

# Write secret values to mode-0600 files — never as argv
printf '%s' "${ARC_GITHUB_APP_ID}" > "${tmp}/github_app_id"
chmod 0600 "${tmp}/github_app_id"

printf '%s' "${ARC_GITHUB_APP_INSTALLATION_ID}" > "${tmp}/github_app_installation_id"
chmod 0600 "${tmp}/github_app_installation_id"

cp -- "${ARC_GITHUB_APP_PRIVATE_KEY_FILE}" "${tmp}/github_app_private_key"
chmod 0600 "${tmp}/github_app_private_key"

# ── Apply namespace and service account ───────────────────────────────────
printf 'Applying namespace and service account …\n'
kubectl apply -f "${SA_MANIFEST}"

# ── Create/update the GitHub App secret via --from-file ───────────────────
printf 'Creating GitHub App secret …\n'
kubectl -n "${NAMESPACE}" create secret generic \
  prompt-lab-runners-github-app \
  --from-file=github_app_id="${tmp}/github_app_id" \
  --from-file=github_app_installation_id="${tmp}/github_app_installation_id" \
  --from-file=github_app_private_key="${tmp}/github_app_private_key" \
  --dry-run=client -o yaml |
  kubectl apply -f -

# ── Install the pinned ARC runner scale set chart ─────────────────────────
printf 'Installing ARC runner scale set (chart 0.14.2) …\n'
helm upgrade --install "${RELEASE_NAME}" \
  --namespace "${NAMESPACE}" \
  --version 0.14.2 \
  --values "${VALUES_FILE}" \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --wait --timeout 10m

# ── Post-install verification ─────────────────────────────────────────────
printf 'Verifying deployment …\n'

config_url="$(kubectl -n "${NAMESPACE}" get autoscalingrunnersets.actions.github.com \
  "${RELEASE_NAME}" -o jsonpath='{.spec.githubConfigUrl}')"
printf 'githubConfigUrl: %s\n' "${config_url}"

min_runners="$(kubectl -n "${NAMESPACE}" get autoscalingrunnersets.actions.github.com \
  "${RELEASE_NAME}" -o jsonpath='{.spec.minRunners}')"
max_runners="$(kubectl -n "${NAMESPACE}" get autoscalingrunnersets.actions.github.com \
  "${RELEASE_NAME}" -o jsonpath='{.spec.maxRunners}')"
printf 'minRunners: %s  maxRunners: %s\n' "${min_runners}" "${max_runners}"

# Wait for listener pod readiness
kubectl -n "${NAMESPACE}" wait pod \
  -l actions.github.com/scale-set-name="${RELEASE_NAME}" \
  --for=condition=Ready --timeout=120s \
  || die "listener pod did not become ready"

printf 'install-prompt-lab-runner: done\n'
