#!/usr/bin/env bash
# verify-grounding-deployment.sh — read-only audit of the Prompt Lab grounding
# deployment: the ARC runner scale set, the model node pool, the Ollama
# deployment it serves, the protected GitHub Environment, and the workflow
# contract that ties them together.
#
# The script only reads.  It never creates, changes, deletes, or scales
# anything, and it prints identities and variable/secret *names* only — never a
# value.  Every gh, az, kubectl, and helm call is fatal on failure: a check that
# cannot run is a failed check, not a skipped one.
#
# Like the installer it downloads its own kubeconfig into a private temporary
# directory and deletes it on exit, so the operator's ~/.kube/config is never
# read or rewritten.
#
# Usage:  scripts/verify-grounding-deployment.sh

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

REPOSITORY="hellices/korvid-prompt-lab"
ENVIRONMENT_NAME="aks-grounding"

AKS_RESOURCE_GROUP="rg-pension-guard"
AKS_CLUSTER_NAME="aks-shared-runners"
MODEL_NODE_POOL="modeleval"

RUNNER_NAMESPACE="arc-runners-prompt-lab"
RELEASE_NAME="prompt-lab-runners"

MODEL_NAMESPACE="ollama"
MODEL_DEPLOYMENT="ollama"
# Live scheduling of the model deployment on the shared cluster.
MODEL_NODE_SELECTOR_KEY="purpose"
MODEL_NODE_SELECTOR_VALUE="korvid-model-eval"

EXPECTED_CONFIG_URL="https://github.com/${REPOSITORY}"
EXPECTED_MIN_RUNNERS="0"
EXPECTED_MAX_RUNNERS="1"
EXPECTED_SERVICE_ACCOUNT="prompt-lab-runners-no-permission"
EXPECTED_NODE_SELECTOR="gha-runner"
EXPECTED_RUN_AS_USER="1001"
EXPECTED_RUN_AS_GROUP="1001"
# The reviewed runner image.  Anything else means the live scale set would run
# code that never went through the image review.
EXPECTED_RUNNER_IMAGE="acrpensionguard.azurecr.io/runner-base:prompt-lab-v1"

WORKFLOW_FILE="${REPO_ROOT}/.github/workflows/grounding-round.yml"
SAFE_EVIDENCE_DIR="prompt-lab/artifacts/grounding-round/safe-evidence"

REQUIRED_VARIABLES=(
  AZURE_CLIENT_ID
  AZURE_SUBSCRIPTION_ID
  AZURE_TENANT_ID
  KORVID_AKS_NAMESPACE
  KORVID_AKS_SERVICE
  KORVID_APP_ID
)
REQUIRED_SECRET="KORVID_APP_PRIVATE_KEY"
OPTIONAL_SECRET="GROUNDING_REFLECTION_CREDENTIAL"

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

require_command() {
  local command_name
  for command_name in "$@"; do
    command -v "${command_name}" >/dev/null 2>&1 \
      || die "required command not found: ${command_name}"
  done
}

require_command az gh helm jq kubectl kubelogin python3

python3 -c 'import yaml' >/dev/null 2>&1 \
  || die "python3 with PyYAML is required to parse the workflow contract"

[[ -r "${WORKFLOW_FILE}" ]] \
  || die "missing workflow: .github/workflows/grounding-round.yml"

# ── Private workspace for this run's kubeconfig ───────────────────────────
# An explicit template keeps the directory inside TMPDIR on every platform:
# BSD mktemp ignores TMPDIR when no template is given.
tmp_root="${TMPDIR:-/tmp}"
tmp="$(mktemp -d "${tmp_root%/}/verify-grounding.XXXXXXXX")"
chmod 700 "${tmp}"

cleanup() {
  rm -rf "${tmp}"
}
trap cleanup EXIT

section "Cluster access"

az account show --output none \
  || die "no active Azure subscription: run 'az login' first"

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
ok "reached ${AKS_CLUSTER_NAME} through a private kubeconfig"

# ── 1. Release status ─────────────────────────────────────────────────────
section "ARC release"

release_json="$(helm status "${RELEASE_NAME}" --namespace "${RUNNER_NAMESPACE}" --output json)" \
  || die "helm release ${RELEASE_NAME} is not installed in ${RUNNER_NAMESPACE}"
release_status="$(printf '%s' "${release_json}" | jq -r '.info.status')" \
  || die "cannot read the release status"
[[ "${release_status}" == "deployed" ]] \
  || die "release ${RELEASE_NAME} is '${release_status}', expected 'deployed'"
ok "release ${RELEASE_NAME} is deployed"

# ── 2. Scale set shape ────────────────────────────────────────────────────
section "Runner scale set"

runner_set_json="$(kubectl --namespace "${RUNNER_NAMESPACE}" \
  get autoscalingrunnersets.actions.github.com "${RELEASE_NAME}" -o json)" \
  || die "cannot read the AutoscalingRunnerSet ${RELEASE_NAME}"

runner_field() {
  local expression="$1" value
  value="$(printf '%s' "${runner_set_json}" | jq -r "${expression}")" \
    || die "cannot read ${expression} from the AutoscalingRunnerSet"
  printf '%s' "${value}"
}

assert_runner_field() {
  local expression="$1" expected="$2" actual
  actual="$(runner_field "${expression}")"
  [[ "${actual}" == "${expected}" ]] \
    || die "AutoscalingRunnerSet ${expression} is '${actual}', expected '${expected}'"
  ok "${expression} = ${actual}"
}

assert_runner_field '.spec.githubConfigUrl' "${EXPECTED_CONFIG_URL}"
assert_runner_field '.spec.minRunners' "${EXPECTED_MIN_RUNNERS}"
assert_runner_field '.spec.maxRunners' "${EXPECTED_MAX_RUNNERS}"
assert_runner_field '.spec.template.spec.serviceAccountName' "${EXPECTED_SERVICE_ACCOUNT}"
assert_runner_field '.spec.template.spec.automountServiceAccountToken' "false"
assert_runner_field '.spec.template.spec.nodeSelector.workload' "${EXPECTED_NODE_SELECTOR}"

section "Runner security context"

# Select the runner container by name: the controller is free to reorder or
# add containers, so an index would silently check the wrong one.
runner_container_json="$(runner_field '
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
assert_container_field '.image' "${EXPECTED_RUNNER_IMAGE}"

# The runner must tolerate neither taint the model nodes carry.
model_tolerations="$(runner_field '
  [ .spec.template.spec.tolerations[]?
    | select((.key == "workload" and .value == "ollama")
             or .key == "kubernetes.azure.com/scalesetpriority")
    | .key ]
  | join(", ")')"
[[ -z "${model_tolerations}" ]] \
  || die "the runner template tolerates model-node taints: ${model_tolerations}"
ok "runner template tolerates no model-node taint"

# ── 3. Model node pool ────────────────────────────────────────────────────
section "Model node pool"

node_pool_json="$(az aks nodepool show \
  --resource-group "${AKS_RESOURCE_GROUP}" \
  --cluster-name "${AKS_CLUSTER_NAME}" \
  --name "${MODEL_NODE_POOL}" \
  --output json)" \
  || die "cannot read the ${MODEL_NODE_POOL} node pool"

pool_count="$(printf '%s' "${node_pool_json}" | jq -r '.count')" \
  || die "cannot read the ${MODEL_NODE_POOL} count"
pool_state="$(printf '%s' "${node_pool_json}" | jq -r '.provisioningState')" \
  || die "cannot read the ${MODEL_NODE_POOL} provisioningState"

[[ "${pool_count}" == "0" || "${pool_count}" == "1" ]] \
  || die "${MODEL_NODE_POOL} count is ${pool_count}, expected 0 or 1"
[[ "${pool_state}" == "Succeeded" ]] \
  || die "${MODEL_NODE_POOL} provisioningState is '${pool_state}', expected 'Succeeded'"
ok "${MODEL_NODE_POOL} count=${pool_count} provisioningState=${pool_state}"

# ── 4. Ollama still targets the model nodes ───────────────────────────────
section "Ollama scheduling"

ollama_json="$(kubectl --namespace "${MODEL_NAMESPACE}" \
  get deployment "${MODEL_DEPLOYMENT}" -o json)" \
  || die "cannot read the ${MODEL_DEPLOYMENT} deployment in ${MODEL_NAMESPACE}"

ollama_selector="$(printf '%s' "${ollama_json}" \
  | jq -r --arg key "${MODEL_NODE_SELECTOR_KEY}" '.spec.template.spec.nodeSelector[$key]')" \
  || die "cannot read the Ollama nodeSelector"
[[ "${ollama_selector}" == "${MODEL_NODE_SELECTOR_VALUE}" ]] \
  || die "Ollama nodeSelector ${MODEL_NODE_SELECTOR_KEY} is '${ollama_selector}', expected '${MODEL_NODE_SELECTOR_VALUE}'"
ok "Ollama nodeSelector ${MODEL_NODE_SELECTOR_KEY}=${ollama_selector}"

assert_ollama_toleration() {
  local key="$1" value="$2" matches
  matches="$(printf '%s' "${ollama_json}" | jq -r --arg key "${key}" --arg value "${value}" '
    [ .spec.template.spec.tolerations[]?
      | select(.key == $key and .value == $value and .effect == "NoSchedule") ]
    | length')" \
    || die "cannot read the Ollama tolerations"
  [[ "${matches}" != "0" ]] \
    || die "Ollama does not tolerate ${key}=${value}:NoSchedule"
  ok "Ollama tolerates ${key}=${value}:NoSchedule"
}

assert_ollama_toleration "workload" "ollama"
assert_ollama_toleration "kubernetes.azure.com/scalesetpriority" "spot"

# ── 5. Protected GitHub Environment ───────────────────────────────────────
section "GitHub Environment"

environment_name="$(gh api "repos/${REPOSITORY}/environments/${ENVIRONMENT_NAME}" --jq '.name')" \
  || die "cannot read the ${ENVIRONMENT_NAME} Environment of ${REPOSITORY}"
[[ "${environment_name}" == "${ENVIRONMENT_NAME}" ]] \
  || die "the environment endpoint answered '${environment_name}', expected '${ENVIRONMENT_NAME}'"
ok "environment ${ENVIRONMENT_NAME} exists"

# The environment endpoints answer with objects, so the names live under
# .variables[] and .secrets[] — never a bare array.
variable_names="$(gh api "repos/${REPOSITORY}/environments/${ENVIRONMENT_NAME}/variables" \
  --paginate --jq '.variables[].name')" \
  || die "cannot list the ${ENVIRONMENT_NAME} Environment variables"

for required_variable in "${REQUIRED_VARIABLES[@]}"; do
  grep -Fxq -- "${required_variable}" <<<"${variable_names}" \
    || die "environment variable missing: ${required_variable}"
  ok "variable ${required_variable}"
done

secret_names="$(gh api "repos/${REPOSITORY}/environments/${ENVIRONMENT_NAME}/secrets" \
  --paginate --jq '.secrets[].name')" \
  || die "cannot list the ${ENVIRONMENT_NAME} Environment secrets"

grep -Fxq -- "${REQUIRED_SECRET}" <<<"${secret_names}" \
  || die "environment secret missing: ${REQUIRED_SECRET}"
ok "secret ${REQUIRED_SECRET} (name only — no value is ever read)"

if grep -Fxq -- "${OPTIONAL_SECRET}" <<<"${secret_names}"; then
  ok "secret ${OPTIONAL_SECRET} present — optimize-evaluate rounds are enabled"
else
  ok "secret ${OPTIONAL_SECRET} absent — evaluate-only rounds"
fi

# ── 6. Workflow contract ──────────────────────────────────────────────────
section "Workflow contract"

# Parsed structurally: the grounding job of this one workflow must name the
# scale set and upload exactly the safe-evidence directory.  A grep over every
# workflow would accept a decoy file or a comment.
if ! python3 - "${WORKFLOW_FILE}" "${RELEASE_NAME}" "${SAFE_EVIDENCE_DIR}" <<'PY'
import sys

import yaml

workflow_path, scale_set, safe_evidence_dir = sys.argv[1], sys.argv[2], sys.argv[3]

with open(workflow_path, encoding="utf-8") as handle:
    workflow = yaml.safe_load(handle)

problems = []
job = (workflow.get("jobs") or {}).get("grounding")

if job is None:
    problems.append("the workflow has no 'grounding' job")
else:
    runs_on = job.get("runs-on")
    if runs_on != scale_set:
        problems.append(f"runs-on is {runs_on!r}, expected {scale_set!r}")

    evidence_dir = str((job.get("env") or {}).get("GROUNDING_SAFE_EVIDENCE_DIR", ""))
    if not evidence_dir.endswith("/" + safe_evidence_dir):
        problems.append(
            f"GROUNDING_SAFE_EVIDENCE_DIR is {evidence_dir!r}, "
            f"expected it to end with {safe_evidence_dir!r}"
        )

    uploads = [
        step
        for step in (job.get("steps") or [])
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    if len(uploads) != 1:
        problems.append(f"expected exactly one upload-artifact step, found {len(uploads)}")
    else:
        uploaded = str((uploads[0].get("with") or {}).get("path", "")).strip()
        if uploaded.rstrip("/") != safe_evidence_dir:
            problems.append(
                f"the artifact upload path is {uploaded!r}, expected the "
                f"safe-evidence directory {safe_evidence_dir + '/'!r}"
            )

for problem in problems:
    print(problem, file=sys.stderr)

sys.exit(1 if problems else 0)
PY
then
  die "the grounding workflow does not match the Prompt Lab runner contract"
fi
ok "grounding job runs on ${RELEASE_NAME} and uploads only ${SAFE_EVIDENCE_DIR}/"

printf '\nverify-grounding-deployment: all checks passed ✓\n'
