#!/usr/bin/env bash
# configure-grounding-access.sh — bootstrap the least-privilege identity and the
# protected GitHub Environment a Prompt Lab grounding round runs with.
#
# This is the only grounding-access bootstrap entry point.  It is idempotent:
# every step first asks Azure or GitHub what already exists and only then
# creates, replaces, or updates it, so a second run is a no-op apart from
# create-or-update calls that carry the reviewed definition.
#
# What it produces
#   * Entra application and service principal `korvid-prompt-lab-grounding`
#   * a federated credential bound to the `aks-grounding` Environment subject
#     of `hellices/korvid-prompt-lab` — never a client secret
#   * a custom Kubernetes DataActions role assigned only at the ollama
#     namespace scope of the AKS cluster
#   * a custom management-plane role assigned only at the modeleval agent pool
#   * `Azure Kubernetes Service Cluster User Role` only at the cluster, so the
#     workflow can download a kubeconfig and nothing else
#   * the protected `aks-grounding` Environment, its six variables, and the
#     Korvid App private key
#
# Required environment
#   KORVID_APP_ID                GitHub App id for the read-only Korvid checkout
#   KORVID_APP_PRIVATE_KEY_FILE  readable PEM file holding that App's key
#
# Optional, optimize-evaluate rounds only
#   GROUNDING_REFLECTION_MODEL            LiteLLM model string
#     Ollama (ollama/* or ollama_chat/*): model only, no credential needed
#     Hosted providers (openai/*, anthropic/*, etc.): model + credential file
#   GROUNDING_REFLECTION_CREDENTIAL_FILE  readable file holding API key
#     Required for hosted providers; must NOT be set for Ollama models
#
# Secrets reach GitHub only as stdin streamed from a file, never as a command
# line argument.  Progress is reported by name: no subscription, tenant,
# client, principal, or resource id and no secret value is written to stdout,
# stderr, or a file outside the private render directory.  Tracing is never
# enabled for the same reason.
#
# Usage:  scripts/configure-grounding-access.sh

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/reflection-provider.sh
source "${SCRIPT_DIR}/lib/reflection-provider.sh"

REPOSITORY="hellices/korvid-prompt-lab"
ENVIRONMENT_NAME="aks-grounding"

APP_DISPLAY_NAME="korvid-prompt-lab-grounding"
FEDERATED_CREDENTIAL_NAME="github-aks-grounding"
FEDERATED_ISSUER="https://token.actions.githubusercontent.com"
FEDERATED_AUDIENCE="api://AzureADTokenExchange"

AKS_RESOURCE_GROUP="rg-pension-guard"
AKS_CLUSTER_NAME="aks-shared-runners"
MODEL_NODE_POOL="modeleval"
MODEL_NAMESPACE="ollama"
MODEL_SERVICE="ollama"

CLUSTER_USER_ROLE="Azure Kubernetes Service Cluster User Role"
NODE_POOL_ROLE_NAME="Korvid Prompt Lab Grounding Node Pool Scaler"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
KUBERNETES_ROLE_TEMPLATE="${REPO_ROOT}/infra/azure/grounding-kubernetes-role.json.tpl"

# Entra and ARM replicate a new principal asynchronously, so the first role
# assignment after a fresh registration can legitimately fail once.
RETRY_ATTEMPTS="${_GROUNDING_RETRY_ATTEMPTS:-6}"
RETRY_DELAY_SECONDS="${_GROUNDING_RETRY_DELAY_SECONDS:-5}"

die() {
  printf 'configure-grounding-access: %s\n' "$*" >&2
  exit 1
}

require_command() {
  local command_name
  for command_name in "$@"; do
    command -v "${command_name}" >/dev/null 2>&1 \
      || die "required command not found: ${command_name}"
  done
}

# Run a read-only CLI query and normalise the several ways the CLIs spell
# "nothing found" into an empty string.  The explicit `if` keeps a failing
# query a failure even where `set -e` is suspended, such as inside `retry`.
lookup() {
  local value
  if ! value="$("$@")"; then
    return 1
  fi
  value="$(printf '%s' "${value}" | tr -d '\r\n')"
  if [[ "${value}" == "None" || "${value}" == "null" ]]; then
    value=""
  fi
  printf '%s' "${value}"
}

retry() {
  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi
    if (( attempt >= RETRY_ATTEMPTS )); then
      return 1
    fi
    attempt=$((attempt + 1))
    sleep "${RETRY_DELAY_SECONDS}"
  done
}

# Create an empty private file before anything is rendered into it, so a
# rendered role definition or credential is never briefly world readable.
private_file() {
  local path="$1"
  : >"${path}"
  chmod 600 "${path}"
}

require_expanded() {
  local path="$1"
  if grep -Eq '__[A-Z0-9_]+__' "${path}"; then
    die "refusing to send an unexpanded template placeholder to az: $(basename "${path}")"
  fi
}

require_command gh az jq kubectl kubelogin

: "${KORVID_APP_ID:?KORVID_APP_ID is required}"
: "${KORVID_APP_PRIVATE_KEY_FILE:?KORVID_APP_PRIVATE_KEY_FILE is required}"

[[ -r "${KORVID_APP_PRIVATE_KEY_FILE}" ]] \
  || die "KORVID_APP_PRIVATE_KEY_FILE is not a readable file"
[[ -s "${KORVID_APP_PRIVATE_KEY_FILE}" ]] \
  || die "KORVID_APP_PRIVATE_KEY_FILE is empty"

REFLECTION_MODEL="${GROUNDING_REFLECTION_MODEL:-}"
REFLECTION_CREDENTIAL_FILE="${GROUNDING_REFLECTION_CREDENTIAL_FILE:-}"
REFLECTION_REQUIRES_CREDENTIAL=false

if [[ -n "$REFLECTION_CREDENTIAL_FILE" && -z "$REFLECTION_MODEL" ]]; then
  die "GROUNDING_REFLECTION_MODEL is required with GROUNDING_REFLECTION_CREDENTIAL_FILE"
fi

if [[ -n "$REFLECTION_MODEL" ]]; then
  validate_reflection_model "$REFLECTION_MODEL" \
    || die "invalid reflection model: $REFLECTION_MODEL"
  if reflection_requires_credential "$REFLECTION_MODEL"; then
    REFLECTION_REQUIRES_CREDENTIAL=true
    [[ -n "$REFLECTION_CREDENTIAL_FILE" ]] \
      || die "GROUNDING_REFLECTION_CREDENTIAL_FILE is required with hosted GROUNDING_REFLECTION_MODEL"
    [[ -r "$REFLECTION_CREDENTIAL_FILE" ]] \
      || die "GROUNDING_REFLECTION_CREDENTIAL_FILE is not a readable file"
    [[ -s "$REFLECTION_CREDENTIAL_FILE" ]] \
      || die "GROUNDING_REFLECTION_CREDENTIAL_FILE is empty"
  elif [[ -n "$REFLECTION_CREDENTIAL_FILE" ]]; then
    die "GROUNDING_REFLECTION_CREDENTIAL_FILE must not be set for Ollama reflection"
  fi
fi

[[ -r "${KUBERNETES_ROLE_TEMPLATE}" ]] \
  || die "missing role template: infra/azure/grounding-kubernetes-role.json.tpl"

RENDER_DIR="$(mktemp -d)"
chmod 700 "${RENDER_DIR}"
cleanup() {
  rm -rf "${RENDER_DIR}"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 1. Resolve the repository's current default OIDC subject prefix
# ---------------------------------------------------------------------------
oidc_config="$(gh api "repos/${REPOSITORY}/actions/oidc/customization/sub")"
oidc_uses_default="$(printf '%s' "${oidc_config}" | jq -r '.use_default // false')"
[[ "${oidc_uses_default}" == "true" ]] \
  || die "the repository must use GitHub's default OIDC subject format"

oidc_subject_prefix="$(printf '%s' "${oidc_config}" | jq -r '.sub_claim_prefix // empty')"
oidc_config=""
if [[ "${oidc_subject_prefix}" != "repo:hellices/korvid-prompt-lab" ]] \
  && [[ ! "${oidc_subject_prefix}" =~ ^repo:hellices@[0-9]+/korvid-prompt-lab@[0-9]+$ ]]; then
  die "GitHub returned an unexpected OIDC subject prefix for the repository"
fi
federated_subject="${oidc_subject_prefix}:environment:${ENVIRONMENT_NAME}"

# ---------------------------------------------------------------------------
# 2. Discover the active subscription and tenant without printing either
# ---------------------------------------------------------------------------
account_json="$(az account show --output json)"
subscription_id="$(printf '%s' "${account_json}" | jq -r '.id // empty')"
tenant_id="$(printf '%s' "${account_json}" | jq -r '.tenantId // empty')"
account_json=""

[[ -n "${subscription_id}" ]] || die "could not discover the signed-in Azure subscription"
[[ -n "${tenant_id}" ]] || die "could not discover the signed-in Azure tenant"
subscription_scope="/subscriptions/${subscription_id}"

# ---------------------------------------------------------------------------
# 3. Create or reuse the Entra application and its service principal
# ---------------------------------------------------------------------------
printf 'Ensuring Entra application %s\n' "${APP_DISPLAY_NAME}"
app_id="$(lookup az ad app list \
  --filter "displayName eq '${APP_DISPLAY_NAME}'" \
  --query "[0].appId" \
  --output tsv)"

if [[ -z "${app_id}" ]]; then
  app_id="$(lookup az ad app create \
    --display-name "${APP_DISPLAY_NAME}" \
    --query appId \
    --output tsv)"
  [[ -n "${app_id}" ]] || die "the Entra application was created without an application id"
  printf '  registration created\n'
else
  printf '  reusing the existing registration\n'
fi

app_object_id="$(lookup az ad app show --id "${app_id}" --query id --output tsv)"
[[ -n "${app_object_id}" ]] || die "could not resolve the Entra application object id"

sp_object_id="$(lookup az ad sp list \
  --filter "appId eq '${app_id}'" \
  --query "[0].id" \
  --output tsv)"

if [[ -z "${sp_object_id}" ]]; then
  sp_object_id="$(retry lookup az ad sp create --id "${app_id}" --query id --output tsv)"
  [[ -n "${sp_object_id}" ]] || die "the service principal was created without an object id"
  printf '  service principal created\n'
else
  printf '  reusing the existing service principal\n'
fi

# ---------------------------------------------------------------------------
# 4. Bind the credential to the aks-grounding Environment subject
# ---------------------------------------------------------------------------
printf 'Ensuring federated credential %s\n' "${FEDERATED_CREDENTIAL_NAME}"
federated_file="${RENDER_DIR}/federated-credential.json"
private_file "${federated_file}"
jq -n \
  --arg name "${FEDERATED_CREDENTIAL_NAME}" \
  --arg issuer "${FEDERATED_ISSUER}" \
  --arg subject "${federated_subject}" \
  --arg audience "${FEDERATED_AUDIENCE}" \
  '{
     name: $name,
     issuer: $issuer,
     subject: $subject,
     audiences: [$audience],
     description: "GitHub Actions OIDC for the aks-grounding Environment"
   }' >"${federated_file}"
require_expanded "${federated_file}"

existing_credential="$(az ad app federated-credential list \
  --id "${app_object_id}" \
  --output json \
  | jq -c --arg name "${FEDERATED_CREDENTIAL_NAME}" \
    'map(select(.name == $name)) | first // empty')"

if [[ -z "${existing_credential}" ]]; then
  az ad app federated-credential create \
    --id "${app_object_id}" \
    --parameters "@${federated_file}" \
    --output none
  printf '  credential created\n'
else
  credential_matches="$(printf '%s' "${existing_credential}" \
    | jq -r --slurpfile desired "${federated_file}" '
        if .issuer == $desired[0].issuer
           and .subject == $desired[0].subject
           and ((.audiences // []) == $desired[0].audiences)
        then "yes" else "no" end')"
  if [[ "${credential_matches}" == "yes" ]]; then
    printf '  already bound to the aks-grounding Environment subject\n'
  else
    credential_id="$(printf '%s' "${existing_credential}" | jq -r '.id // empty')"
    [[ -n "${credential_id}" ]] || die "the drifted federated credential has no id"
    az ad app federated-credential delete \
      --id "${app_object_id}" \
      --federated-credential-id "${credential_id}" \
      --output none
    az ad app federated-credential create \
      --id "${app_object_id}" \
      --parameters "@${federated_file}" \
      --output none
    printf '  drifted credential replaced\n'
  fi
fi

# ---------------------------------------------------------------------------
# 5. Resolve the exact scopes; never build a resource id by hand
# ---------------------------------------------------------------------------
aks_id="$(lookup az aks show \
  --resource-group "${AKS_RESOURCE_GROUP}" \
  --name "${AKS_CLUSTER_NAME}" \
  --query id \
  --output tsv)"
[[ "${aks_id}" == /subscriptions/* ]] || die "unexpected AKS cluster resource id"

node_pool_id="$(lookup az aks nodepool show \
  --resource-group "${AKS_RESOURCE_GROUP}" \
  --cluster-name "${AKS_CLUSTER_NAME}" \
  --name "${MODEL_NODE_POOL}" \
  --query id \
  --output tsv)"
[[ "${node_pool_id}" == *"/agentPools/modeleval" ]] \
  || die "the ${MODEL_NODE_POOL} node pool did not return its own agent pool resource id"

kubernetes_scope="${aks_id}/namespaces/ollama"
[[ "${kubernetes_scope}" == "${aks_id}/namespaces/${MODEL_NAMESPACE}" ]] \
  || die "the assigned namespace scope drifted from KORVID_AKS_NAMESPACE"

# ---------------------------------------------------------------------------
# 6. Define both custom roles, then assign each at its own scope
# ---------------------------------------------------------------------------
kubernetes_role_file="${RENDER_DIR}/grounding-kubernetes-role.json"
private_file "${kubernetes_role_file}"
jq --arg scope "${subscription_scope}" \
  '.AssignableScopes = [$scope]' \
  "${KUBERNETES_ROLE_TEMPLATE}" >"${kubernetes_role_file}"

kubernetes_role_name="$(jq -r '.Name // empty' "${kubernetes_role_file}")"
[[ -n "${kubernetes_role_name}" ]] || die "the Kubernetes role template has no Name"

node_pool_role_file="${RENDER_DIR}/grounding-node-pool-role.json"
private_file "${node_pool_role_file}"
jq -n \
  --arg name "${NODE_POOL_ROLE_NAME}" \
  --arg scope "${subscription_scope}" \
  '{
     Name: $name,
     Description: "Read and scale only the modeleval agent pool that serves a grounding round.",
     Actions: [
       "Microsoft.ContainerService/managedClusters/agentPools/read",
       "Microsoft.ContainerService/managedClusters/agentPools/write"
     ],
     NotActions: [],
     DataActions: [],
     NotDataActions: [],
     AssignableScopes: [$scope]
   }' >"${node_pool_role_file}"

ensure_role_definition() {
  local file="$1" role_name existing
  role_name="$(jq -r '.Name // empty' "${file}")"
  [[ -n "${role_name}" ]] || die "rendered role definition has no Name"
  require_expanded "${file}"

  existing="$(lookup az role definition list \
    --name "${role_name}" \
    --scope "${subscription_scope}" \
    --query "[0].roleName" \
    --output tsv)"

  if [[ -n "${existing}" ]]; then
    az role definition update --role-definition "@${file}" --output none
    printf '  role %s updated\n' "${role_name}"
  else
    az role definition create --role-definition "@${file}" --output none
    printf '  role %s created\n' "${role_name}"
  fi
}

assign_role() {
  local role_name="$1" scope="$2" description="$3" existing
  existing="$(lookup az role assignment list \
    --assignee-object-id "${sp_object_id}" \
    --role "${role_name}" \
    --scope "${scope}" \
    --query "length(@)" \
    --output tsv)"

  if [[ "${existing}" =~ ^[0-9]+$ ]] && (( existing > 0 )); then
    printf '  %s: already assigned\n' "${description}"
    return 0
  fi

  retry az role assignment create \
    --assignee-object-id "${sp_object_id}" \
    --assignee-principal-type ServicePrincipal \
    --role "${role_name}" \
    --scope "${scope}" \
    --output none
  printf '  %s: assigned\n' "${description}"
}

printf 'Ensuring custom Azure role definitions\n'
ensure_role_definition "${kubernetes_role_file}"
ensure_role_definition "${node_pool_role_file}"

printf 'Ensuring role assignments\n'
assign_role "${kubernetes_role_name}" "${kubernetes_scope}" \
  "Kubernetes data access in the ${MODEL_NAMESPACE} namespace"
assign_role "${NODE_POOL_ROLE_NAME}" "${node_pool_id}" \
  "agent pool read and write on ${MODEL_NODE_POOL}"
assign_role "${CLUSTER_USER_ROLE}" "${aks_id}" \
  "cluster user credentials on the cluster"

# ---------------------------------------------------------------------------
# 7. Protect the Environment with the authenticated GitHub user as reviewer
# ---------------------------------------------------------------------------
printf 'Ensuring GitHub Environment %s\n' "${ENVIRONMENT_NAME}"
reviewer_id="$(lookup gh api user --jq '.id')"
[[ "${reviewer_id}" =~ ^[0-9]+$ ]] || die "could not resolve the authenticated GitHub user"

environment_file="${RENDER_DIR}/environment.json"
private_file "${environment_file}"
jq -n --argjson reviewer "${reviewer_id}" \
  '{
     wait_timer: 0,
     prevent_self_review: false,
     reviewers: [{type: "User", id: $reviewer}],
     deployment_branch_policy: null
   }' >"${environment_file}"

gh api \
  --method PUT "repos/${REPOSITORY}/environments/${ENVIRONMENT_NAME}" \
  --input "${environment_file}" \
  --silent
printf '  the authenticated GitHub user is a required reviewer\n'

# ---------------------------------------------------------------------------
# 8. Publish the six Environment variables; values travel on stdin
# ---------------------------------------------------------------------------
printf 'Setting Environment variables\n'
printf '%s' "${app_id}" | gh variable set AZURE_CLIENT_ID --env aks-grounding --repo "${REPOSITORY}"
printf '%s' "${tenant_id}" | gh variable set AZURE_TENANT_ID --env aks-grounding --repo "${REPOSITORY}"
printf '%s' "${subscription_id}" | gh variable set AZURE_SUBSCRIPTION_ID --env aks-grounding --repo "${REPOSITORY}"
printf '%s' "${KORVID_APP_ID}" | gh variable set KORVID_APP_ID --env aks-grounding --repo "${REPOSITORY}"
printf '%s' "${MODEL_NAMESPACE}" | gh variable set KORVID_AKS_NAMESPACE --env aks-grounding --repo "${REPOSITORY}"
printf '%s' "${MODEL_SERVICE}" | gh variable set KORVID_AKS_SERVICE --env aks-grounding --repo "${REPOSITORY}"

# ---------------------------------------------------------------------------
# 9. Stream the secrets from their files; no value is ever an argument
# ---------------------------------------------------------------------------
printf 'Storing the Korvid App private key\n'
cat "$KORVID_APP_PRIVATE_KEY_FILE" | gh secret set KORVID_APP_PRIVATE_KEY --env aks-grounding --repo "${REPOSITORY}"

if [[ -n "$REFLECTION_MODEL" ]]; then
  printf 'Storing the optimize-evaluate reflection model\n'
  printf '%s' "$REFLECTION_MODEL" |
    gh variable set GROUNDING_REFLECTION_MODEL --env aks-grounding --repo "$REPOSITORY"
  if [[ "$REFLECTION_REQUIRES_CREDENTIAL" == "true" ]]; then
    printf 'Storing the optimize-evaluate reflection credential\n'
    cat "$REFLECTION_CREDENTIAL_FILE" |
      gh secret set GROUNDING_REFLECTION_CREDENTIAL --env aks-grounding --repo "$REPOSITORY"
  fi
fi

printf 'Grounding access is configured for the %s Environment.\n' "${ENVIRONMENT_NAME}"
