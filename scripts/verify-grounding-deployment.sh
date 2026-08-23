#!/usr/bin/env bash
# verify-grounding-deployment.sh — read-only verification of the Prompt Lab
# ARC runner scale set and grounding infrastructure.
#
# This script is strictly read-only: it never applies, deletes, upgrades, or
# scales any resource.  It prints identities and secret names only, never
# secret values.
#
# Usage:  scripts/verify-grounding-deployment.sh

set -Eeuo pipefail

NAMESPACE="arc-runners-prompt-lab"
RELEASE_NAME="prompt-lab-runners"
REPOSITORY="hellices/korvid-prompt-lab"
AKS_RESOURCE_GROUP="rg-pension-guard"
AKS_CLUSTER_NAME="aks-shared-runners"
MODEL_NODE_POOL="modeleval"

die() {
  printf 'verify-grounding-deployment: FAIL — %s\n' "$*" >&2
  exit 1
}

ok() {
  printf '  ✓ %s\n' "$*"
}

section() {
  printf '\n── %s ──\n' "$*"
}

# ── 1. ARC release status ─────────────────────────────────────────────────
section "ARC release status"

release_status="$(helm status "${RELEASE_NAME}" -n "${NAMESPACE}" -o json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["status"])')" \
  || die "helm release ${RELEASE_NAME} not found"

[[ "${release_status}" == "deployed" ]] \
  || die "release status is '${release_status}', expected 'deployed'"
ok "release ${RELEASE_NAME} is deployed"

# ── 2. Scale set URL ──────────────────────────────────────────────────────
section "Scale set configuration"

config_url="$(kubectl -n "${NAMESPACE}" get autoscalingrunnersets.actions.github.com \
  "${RELEASE_NAME}" -o jsonpath='{.spec.githubConfigUrl}')" \
  || die "cannot read AutoscalingRunnerSet"

[[ "${config_url}" == "https://github.com/${REPOSITORY}" ]] \
  || die "githubConfigUrl is '${config_url}', expected 'https://github.com/${REPOSITORY}'"
ok "githubConfigUrl: ${config_url}"

# ── 3. Min/max runners ────────────────────────────────────────────────────
min_runners="$(kubectl -n "${NAMESPACE}" get autoscalingrunnersets.actions.github.com \
  "${RELEASE_NAME}" -o jsonpath='{.spec.minRunners}')"
max_runners="$(kubectl -n "${NAMESPACE}" get autoscalingrunnersets.actions.github.com \
  "${RELEASE_NAME}" -o jsonpath='{.spec.maxRunners}')"

[[ "${min_runners}" == "0" ]] || die "minRunners is ${min_runners}, expected 0"
[[ "${max_runners}" == "1" ]] || die "maxRunners is ${max_runners}, expected 1"
ok "minRunners: ${min_runners}  maxRunners: ${max_runners}"

# ── 4. Runner template scheduling ─────────────────────────────────────────
section "Runner template scheduling"

node_selector="$(kubectl -n "${NAMESPACE}" get autoscalingrunnersets.actions.github.com \
  "${RELEASE_NAME}" -o jsonpath='{.spec.template.spec.nodeSelector.workload}')"

[[ "${node_selector}" == "gha-runner" ]] \
  || die "runner nodeSelector workload is '${node_selector}', expected 'gha-runner'"
ok "nodeSelector workload=gha-runner"

# ── 5. Runner must NOT tolerate workload=ollama ───────────────────────────
tolerations_json="$(kubectl -n "${NAMESPACE}" get autoscalingrunnersets.actions.github.com \
  "${RELEASE_NAME}" -o jsonpath='{.spec.template.spec.tolerations}')"

if echo "${tolerations_json}" | grep -q '"key":"ollama"'; then
  die "runner tolerates ollama taint — runners must not schedule on model nodes"
fi
ok "no ollama toleration on runners"

# ── 6. Security context ───────────────────────────────────────────────────
section "Runner security"

run_as_non_root="$(kubectl -n "${NAMESPACE}" get autoscalingrunnersets.actions.github.com \
  "${RELEASE_NAME}" -o jsonpath='{.spec.template.spec.containers[0].securityContext.runAsNonRoot}')"

[[ "${run_as_non_root}" == "true" ]] \
  || die "runner container is not runAsNonRoot"
ok "runner runs as non-root"

# ── 7. Workflow uses prompt-lab-runners label ─────────────────────────────
section "Workflow configuration"

workflow_file="$(find "$(git rev-parse --show-toplevel)/.github/workflows" \
  -name '*.yml' -o -name '*.yaml' | head -20)"

found_label=false
for wf in ${workflow_file}; do
  if grep -q 'runs-on:.*prompt-lab-runners' "${wf}" 2>/dev/null; then
    ok "workflow $(basename "${wf}") uses prompt-lab-runners"
    found_label=true
  fi
done
[[ "${found_label}" == "true" ]] \
  || die "no workflow found with runs-on: prompt-lab-runners"

# ── 8. GitHub Environment variable/secret names ──────────────────────────
section "GitHub Environment"

env_vars="$(gh api "repos/${REPOSITORY}/environments/aks-grounding/variables" \
  --jq '.[].name' 2>/dev/null || true)"
env_secrets="$(gh api "repos/${REPOSITORY}/environments/aks-grounding/secrets" \
  --jq '.[].name' 2>/dev/null || true)"

if [[ -n "${env_vars}" ]]; then
  printf '  variables: %s\n' "${env_vars}" | tr '\n' ', '
  printf '\n'
fi
if [[ -n "${env_secrets}" ]]; then
  printf '  secrets (names only): %s\n' "${env_secrets}" | tr '\n' ', '
  printf '\n'
fi
ok "environment variable/secret names listed (values never printed)"

# ── 9. modeleval node pool ────────────────────────────────────────────────
section "modeleval node pool"

pool_count="$(az aks nodepool show \
  --resource-group "${AKS_RESOURCE_GROUP}" \
  --cluster-name "${AKS_CLUSTER_NAME}" \
  --name "${MODEL_NODE_POOL}" \
  --query count -o tsv 2>/dev/null)" \
  || die "cannot query ${MODEL_NODE_POOL} node pool"

pool_state="$(az aks nodepool show \
  --resource-group "${AKS_RESOURCE_GROUP}" \
  --cluster-name "${AKS_CLUSTER_NAME}" \
  --name "${MODEL_NODE_POOL}" \
  --query provisioningState -o tsv 2>/dev/null)"

[[ "${pool_count}" == "0" || "${pool_count}" == "1" ]] \
  || die "modeleval count is ${pool_count}, expected 0 or 1"
[[ "${pool_state}" == "Succeeded" ]] \
  || die "modeleval provisioningState is '${pool_state}', expected 'Succeeded'"
ok "modeleval count=${pool_count} provisioningState=${pool_state}"

# ── 10. Ollama scheduling targets modeleval ───────────────────────────────
section "Ollama scheduling"

ollama_ns="ollama"
ollama_selector="$(kubectl -n "${ollama_ns}" get deploy ollama \
  -o jsonpath='{.spec.template.spec.nodeSelector.workload}' 2>/dev/null)" \
  || die "cannot read Ollama deployment"

[[ "${ollama_selector}" == "modeleval" ]] \
  || die "Ollama nodeSelector workload is '${ollama_selector}', expected 'modeleval'"
ok "Ollama nodeSelector workload=modeleval"

ollama_tolerations="$(kubectl -n "${ollama_ns}" get deploy ollama \
  -o jsonpath='{.spec.template.spec.tolerations}' 2>/dev/null)"

if ! echo "${ollama_tolerations}" | grep -q 'modeleval'; then
  die "Ollama does not tolerate modeleval taint"
fi
ok "Ollama tolerates modeleval"

# ── 11. Artifact upload path ends at safe-evidence/ ──────────────────────
section "Safe evidence path"

safe_evidence_ok=false
for wf in ${workflow_file}; do
  if grep -q 'safe-evidence' "${wf}" 2>/dev/null; then
    safe_evidence_ok=true
    ok "workflow $(basename "${wf}") references safe-evidence path"
  fi
done
[[ "${safe_evidence_ok}" == "true" ]] \
  || die "no workflow references safe-evidence artifact path"

# ── Done ──────────────────────────────────────────────────────────────────
printf '\nverify-grounding-deployment: all checks passed ✓\n'
