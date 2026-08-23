#!/usr/bin/env bash
# install-prompt-lab-runner.sh — install the Prompt Lab ARC runner scale set.
#
# The script owns its own credentials end to end: it downloads a kubeconfig
# into a private temporary directory, converts it with kubelogin, exports it
# for its own kubectl and helm calls, and deletes it on exit.  The operator's
# ~/.kube/config is never read, merged, or overwritten, so running this on a
# workstation cannot repoint an unrelated shell at the shared cluster.
#
# Secrets reach kubectl only through mode-0600 files and --from-file, never as
# command-line arguments and never on stdout or stderr.
#
# After the release is installed the script re-reads the AutoscalingRunnerSet
# and compares every field the design depends on against the exact expected
# value, then waits for the listener pod in the ARC controller namespace
# (arc-systems) — listeners never run in the runner namespace, so waiting
# there would hang until the timeout.
#
# Required environment:
#   ARC_GITHUB_APP_ID                 GitHub App id
#   ARC_GITHUB_APP_INSTALLATION_ID    GitHub App installation id
#   ARC_GITHUB_APP_PRIVATE_KEY_FILE   path to a readable PEM private key file
#
# Usage:  scripts/install-prompt-lab-runner.sh

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

AKS_RESOURCE_GROUP="rg-pension-guard"
AKS_CLUSTER_NAME="aks-shared-runners"

RUNNER_NAMESPACE="arc-runners-prompt-lab"
# The gha-runner-scale-set-controller and every listener pod live here.
CONTROLLER_NAMESPACE="arc-systems"
RELEASE_NAME="prompt-lab-runners"
SECRET_NAME="prompt-lab-runners-github-app"
CHART_REFERENCE="oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set"

# Expected post-install shape.  These mirror infra/arc/prompt-lab-runners-values.yaml
# and are compared against the live object, so a hand-edited release fails here.
EXPECTED_CONFIG_URL="https://github.com/hellices/korvid-prompt-lab"
EXPECTED_MIN_RUNNERS="0"
EXPECTED_MAX_RUNNERS="1"
EXPECTED_SERVICE_ACCOUNT="prompt-lab-runners-no-permission"
EXPECTED_NODE_SELECTOR="gha-runner"
EXPECTED_RUN_AS_USER="1001"
EXPECTED_RUN_AS_GROUP="1001"
LISTENER_TIMEOUT="180s"

VALUES_FILE="${REPO_ROOT}/infra/arc/prompt-lab-runners-values.yaml"
SA_MANIFEST="${REPO_ROOT}/infra/arc/prompt-lab-runner-service-account.yaml"

die() {
  printf 'install-prompt-lab-runner: %s\n' "$*" >&2
  exit 1
}

ok() {
  printf '  ✓ %s\n' "$*"
}

section() {
  printf '\n── %s ──\n' "$*"
}

require_command() {
  local command_name
  for command_name in "$@"; do
    command -v "${command_name}" >/dev/null 2>&1 \
      || die "required command not found: ${command_name}"
  done
}

# ── Preflight: nothing is mutated before this block completes ─────────────
require_command az helm jq kubectl kubelogin

: "${ARC_GITHUB_APP_ID:?ARC_GITHUB_APP_ID is required}"
: "${ARC_GITHUB_APP_INSTALLATION_ID:?ARC_GITHUB_APP_INSTALLATION_ID is required}"
: "${ARC_GITHUB_APP_PRIVATE_KEY_FILE:?ARC_GITHUB_APP_PRIVATE_KEY_FILE is required}"

[[ -r "${ARC_GITHUB_APP_PRIVATE_KEY_FILE}" ]] \
  || die "ARC_GITHUB_APP_PRIVATE_KEY_FILE is not a readable file"
[[ -s "${ARC_GITHUB_APP_PRIVATE_KEY_FILE}" ]] \
  || die "ARC_GITHUB_APP_PRIVATE_KEY_FILE is empty"
[[ -r "${VALUES_FILE}" ]] \
  || die "missing chart values: infra/arc/prompt-lab-runners-values.yaml"
[[ -r "${SA_MANIFEST}" ]] \
  || die "missing manifest: infra/arc/prompt-lab-runner-service-account.yaml"

# ── Private workspace for the kubeconfig and every secret file ────────────
# An explicit template keeps the directory inside TMPDIR on every platform:
# BSD mktemp ignores TMPDIR when no template is given.
tmp_root="${TMPDIR:-/tmp}"
tmp="$(mktemp -d "${tmp_root%/}/prompt-lab-runner.XXXXXXXX")"
chmod 700 "${tmp}"

cleanup() {
  rm -rf "${tmp}"
}
trap cleanup EXIT

# ── Cluster credentials that belong to this run only ──────────────────────
section "Cluster access"

az account show --output none \
  || die "no active Azure subscription: run 'az login' first"

az aks show \
  --resource-group "${AKS_RESOURCE_GROUP}" \
  --name "${AKS_CLUSTER_NAME}" \
  --output none \
  || die "cannot see ${AKS_CLUSTER_NAME} in ${AKS_RESOURCE_GROUP}"

kubeconfig="${tmp}/kubeconfig"
az aks get-credentials \
  --resource-group "${AKS_RESOURCE_GROUP}" \
  --name "${AKS_CLUSTER_NAME}" \
  --file "${kubeconfig}" \
  --overwrite-existing \
  --only-show-errors \
  --output none \
  || die "cannot download credentials for ${AKS_CLUSTER_NAME}"
chmod 600 "${kubeconfig}"

export KUBECONFIG="${kubeconfig}"
kubelogin convert-kubeconfig -l azurecli \
  || die "kubelogin could not convert the kubeconfig"

kubectl cluster-info >/dev/null \
  || die "the cluster is not reachable with the downloaded credentials"
ok "reached ${AKS_CLUSTER_NAME} through a private kubeconfig"

# ── Secret material: mode-0600 files, never argv ──────────────────────────
printf '%s' "${ARC_GITHUB_APP_ID}" >"${tmp}/github_app_id"
chmod 600 "${tmp}/github_app_id"

printf '%s' "${ARC_GITHUB_APP_INSTALLATION_ID}" >"${tmp}/github_app_installation_id"
chmod 600 "${tmp}/github_app_installation_id"

cp -- "${ARC_GITHUB_APP_PRIVATE_KEY_FILE}" "${tmp}/github_app_private_key"
chmod 600 "${tmp}/github_app_private_key"

# ── Namespace, service account, and the GitHub App secret ─────────────────
section "Runner namespace and credentials"

kubectl apply --filename "${SA_MANIFEST}" \
  || die "cannot apply the namespace and service account manifest"
ok "namespace ${RUNNER_NAMESPACE} and service account ${EXPECTED_SERVICE_ACCOUNT}"

kubectl --namespace "${RUNNER_NAMESPACE}" create secret generic \
  "${SECRET_NAME}" \
  --from-file=github_app_id="${tmp}/github_app_id" \
  --from-file=github_app_installation_id="${tmp}/github_app_installation_id" \
  --from-file=github_app_private_key="${tmp}/github_app_private_key" \
  --dry-run=client -o yaml |
  kubectl apply -f - \
  || die "cannot create or update the ${SECRET_NAME} secret"
ok "secret ${SECRET_NAME} (name only — values never printed)"

# ── Pinned chart install ──────────────────────────────────────────────────
section "Runner scale set"

# The chart version is written literally so a contract test catches any drift.
helm upgrade --install "${RELEASE_NAME}" \
  --namespace "${RUNNER_NAMESPACE}" \
  --version 0.14.2 \
  --values "${VALUES_FILE}" \
  "${CHART_REFERENCE}" \
  --wait --timeout 10m \
  || die "the runner scale set release did not install cleanly"
ok "release ${RELEASE_NAME} installed from the pinned 0.14.2 chart"

# ── Post-install verification against exact expectations ──────────────────
section "Installed scale set"

runner_set_json="$(kubectl --namespace "${RUNNER_NAMESPACE}" \
  get autoscalingrunnersets.actions.github.com "${RELEASE_NAME}" -o json)" \
  || die "cannot read the installed AutoscalingRunnerSet"

field() {
  local expression="$1" value
  value="$(printf '%s' "${runner_set_json}" | jq -r "${expression}")" \
    || die "cannot read ${expression} from the AutoscalingRunnerSet"
  printf '%s' "${value}"
}

assert_field() {
  local expression="$1" expected="$2" actual
  actual="$(field "${expression}")"
  [[ "${actual}" == "${expected}" ]] \
    || die "AutoscalingRunnerSet ${expression} is '${actual}', expected '${expected}'"
  ok "${expression} = ${actual}"
}

assert_field '.spec.githubConfigUrl' "${EXPECTED_CONFIG_URL}"
assert_field '.spec.minRunners' "${EXPECTED_MIN_RUNNERS}"
assert_field '.spec.maxRunners' "${EXPECTED_MAX_RUNNERS}"
assert_field '.spec.template.spec.serviceAccountName' "${EXPECTED_SERVICE_ACCOUNT}"
assert_field '.spec.template.spec.automountServiceAccountToken' "false"
assert_field '.spec.template.spec.nodeSelector.workload' "${EXPECTED_NODE_SELECTOR}"

# Select the runner container by name: the controller is free to reorder or
# add containers, so an index would silently check the wrong one.
runner_container_json="$(field '
  [ .spec.template.spec.containers[]? | select(.name == "runner") ] | first // empty')"
[[ -n "${runner_container_json}" ]] \
  || die "the AutoscalingRunnerSet has no container named 'runner'"

assert_container_field() {
  local expression="$1" expected="$2" actual
  actual="$(printf '%s' "${runner_container_json}" | jq -r "${expression}")" \
    || die "cannot read ${expression} from the runner container"
  [[ "${actual}" == "${expected}" ]] \
    || die "runner container ${expression} is '${actual}', expected '${expected}'"
  ok "runner ${expression} = ${actual}"
}

assert_container_field '.securityContext.runAsNonRoot' "true"
assert_container_field '.securityContext.runAsUser' "${EXPECTED_RUN_AS_USER}"
assert_container_field '.securityContext.runAsGroup' "${EXPECTED_RUN_AS_GROUP}"
assert_container_field '.securityContext.allowPrivilegeEscalation' "false"

# Runners must tolerate neither taint the model nodes carry, or a grounding
# round could land its runner pod on the GPU node it is meant to drive.
model_tolerations="$(field '
  [ .spec.template.spec.tolerations[]?
    | select((.key == "workload" and .value == "ollama")
             or .key == "kubernetes.azure.com/scalesetpriority")
    | .key ]
  | join(", ")')"
[[ -z "${model_tolerations}" ]] \
  || die "the runner template tolerates model-node taints: ${model_tolerations}"
ok "runner template tolerates no model-node taint"

# ── Listener readiness in the ARC controller namespace ────────────────────
section "Listener"

kubectl --namespace "${CONTROLLER_NAMESPACE}" wait pod \
  --selector "actions.github.com/scale-set-name=${RELEASE_NAME},actions.github.com/scale-set-namespace=${RUNNER_NAMESPACE}" \
  --for=condition=Ready \
  --timeout="${LISTENER_TIMEOUT}" \
  || die "the ${RELEASE_NAME} listener is not Ready in ${CONTROLLER_NAMESPACE}"
ok "listener Ready in ${CONTROLLER_NAMESPACE}"

printf '\ninstall-prompt-lab-runner: done ✓\n'
