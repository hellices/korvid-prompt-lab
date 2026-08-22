#!/usr/bin/env bash
# run-grounding-round.sh — protected grounding lifecycle orchestrator
#
# Required environment variables:
#   GROUNDING_MODEL            — allowed: qwen3:{0.6b,1.7b,4b,8b,14b}
#   GROUNDING_CANDIDATE        — path or label of the candidate YAML
#   GROUNDING_ROUND_TYPE       — evaluate | optimize-evaluate
#   KORVID_SOURCE_ROOT         — path to Korvid repository checkout
#   GROUNDING_ARTIFACT_ROOT    — directory for evaluation artifacts
#   WORKFLOW_RUN_URL           — GitHub Actions run URL (for report)
#   PROMPT_LAB_REVISION        — prompt-lab git revision (for report)
#   KORVID_REVISION            — korvid git revision (for report)
#
# Additional required for optimize-evaluate:
#   GROUNDING_REFLECTION_MODEL      — reflection LLM model identifier
#   GROUNDING_REFLECTION_CREDENTIAL — API credential for reflection model

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

: "${GROUNDING_MODEL:?GROUNDING_MODEL is required}"
: "${GROUNDING_CANDIDATE:?GROUNDING_CANDIDATE is required}"
: "${GROUNDING_ROUND_TYPE:?GROUNDING_ROUND_TYPE is required}"
: "${KORVID_SOURCE_ROOT:?KORVID_SOURCE_ROOT is required}"
: "${GROUNDING_ARTIFACT_ROOT:?GROUNDING_ARTIFACT_ROOT is required}"
: "${WORKFLOW_RUN_URL:?WORKFLOW_RUN_URL is required}"
: "${PROMPT_LAB_REVISION:?PROMPT_LAB_REVISION is required}"
: "${KORVID_REVISION:?KORVID_REVISION is required}"

case "$GROUNDING_MODEL" in
  qwen3:0.6b|qwen3:1.7b|qwen3:4b|qwen3:8b|qwen3:14b) ;;
  *) echo "unsupported grounding model: $GROUNDING_MODEL" >&2; exit 2 ;;
esac

case "$GROUNDING_ROUND_TYPE" in
  evaluate|optimize-evaluate) ;;
  *) echo "unsupported round type: $GROUNDING_ROUND_TYPE" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Node-count snapshot and conditional cleanup trap
# ---------------------------------------------------------------------------

original_count="$(
  az aks nodepool show \
    --resource-group rg-pension-guard \
    --cluster-name aks-shared-runners \
    --name modeleval \
    --query count \
    --output tsv
)"

case "$original_count" in
  0|1) ;;
  *) echo "unexpected modeleval node count: $original_count" >&2; exit 2 ;;
esac

scaled_by_round=false
_cleanup_ran=false

resolve_optimize_best_candidate() {
  local artifact_root="$1"
  local invocations_root="${artifact_root}/invocations"
  local -a matches=()
  local match

  if [[ ! -d "$invocations_root" || -L "$invocations_root" ]]; then
    return 1
  fi

  while IFS= read -r -d '' match; do
    matches+=("$match")
  done < <(find -P "$invocations_root" -mindepth 2 -maxdepth 2 -type f -name 'best-candidate.yaml' -print0)

  if (( ${#matches[@]} != 1 )); then
    return 1
  fi

  if [[ -L "${matches[0]}" ]]; then
    return 1
  fi

  if [[ "$(dirname "$(dirname "${matches[0]}")")" != "$invocations_root" ]]; then
    return 1
  fi

  printf '%s\n' "${matches[0]}"
}

cleanup() {
  if [[ "$_cleanup_ran" == true ]]; then return; fi
  _cleanup_ran=true
  if [[ "$scaled_by_round" == true ]]; then
    az aks nodepool scale \
      --resource-group rg-pension-guard \
      --cluster-name aks-shared-runners \
      --name modeleval \
      --node-count 0
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------------------------------------------------------------------------
# Scale up when the pool is empty
# ---------------------------------------------------------------------------

if [[ "$original_count" == "0" ]]; then
  az aks nodepool scale \
    --resource-group rg-pension-guard \
    --cluster-name aks-shared-runners \
    --name modeleval \
    --node-count 1
  scaled_by_round=true
fi

# ---------------------------------------------------------------------------
# Preflight: AKS readiness check (bounded 15-minute deadline)
# ---------------------------------------------------------------------------

_preflight_deadline=$(( $(date +%s) + ${_AKS_CHECK_DEADLINE_SECONDS:-900} ))
_aks_check_poll_interval="${_AKS_CHECK_POLL_INTERVAL:-10}"
while true; do
  if korvid-prompt-lab aks-check \
       --korvid-source-root "$KORVID_SOURCE_ROOT" \
       --model "$GROUNDING_MODEL"; then
    break
  fi
  if (( $(date +%s) >= _preflight_deadline )); then
    echo "aks-check timed out after 15 minutes" >&2
    exit 1
  fi
  sleep "$_aks_check_poll_interval"
done

# ---------------------------------------------------------------------------
# optimize step (optimize-evaluate only)
# ---------------------------------------------------------------------------

_candidate="$GROUNDING_CANDIDATE"
_opt_artifact_root=""

if [[ "$GROUNDING_ROUND_TYPE" == "optimize-evaluate" ]]; then
  : "${GROUNDING_REFLECTION_MODEL:?GROUNDING_REFLECTION_MODEL is required for optimize-evaluate}"
  : "${GROUNDING_REFLECTION_CREDENTIAL:?GROUNDING_REFLECTION_CREDENTIAL is required for optimize-evaluate}"

  # Map the credential to the provider-standard environment variable so that
  # dspy.LM picks it up from the environment — never pass secrets in argv.
  _reflection_provider="${GROUNDING_REFLECTION_MODEL%%/*}"
  if [[ "$_reflection_provider" == "$GROUNDING_REFLECTION_MODEL" ]]; then
    _reflection_provider="openai"
  fi
  _reflection_provider_lc="$(printf '%s' "$_reflection_provider" | tr '[:upper:]' '[:lower:]')"
  case "$_reflection_provider_lc" in
    openai)         _reflection_cred_var="OPENAI_API_KEY" ;;
    anthropic)      _reflection_cred_var="ANTHROPIC_API_KEY" ;;
    cohere)         _reflection_cred_var="COHERE_API_KEY" ;;
    gemini|google)  _reflection_cred_var="GEMINI_API_KEY" ;;
    *)              _reflection_cred_var="OPENAI_API_KEY" ;;
  esac

  _opt_artifact_root="${GROUNDING_ARTIFACT_ROOT}/optimize"
  mkdir -p "$_opt_artifact_root"
  _optimize_args=(
    optimize
    --candidate "$_candidate"
    --reflection-model "$GROUNDING_REFLECTION_MODEL"
    --korvid-source-root "$KORVID_SOURCE_ROOT"
    --artifact-root "$_opt_artifact_root"
  )

  # optimize — never fall back to seed on failure; credential scoped to subprocess only
  env "${_reflection_cred_var}=${GROUNDING_REFLECTION_CREDENTIAL}" korvid-prompt-lab "${_optimize_args[@]}"

  # Resolve exactly one new best-candidate.yaml
  if ! _best_candidate="$(resolve_optimize_best_candidate "$_opt_artifact_root")"; then
    echo "optimize did not produce exactly one regular best-candidate.yaml under ${_opt_artifact_root}/invocations — aborting" >&2
    exit 1
  fi
  _candidate="$_best_candidate"
  _opt_report_root="$(dirname "$_best_candidate")"
fi

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

_eval_artifact_root="${GROUNDING_ARTIFACT_ROOT}/evaluate"
mkdir -p "$_eval_artifact_root"
_evaluate_args=(
  evaluate
  --candidate "$_candidate"
  --model "$GROUNDING_MODEL"
  --korvid-source-root "$KORVID_SOURCE_ROOT"
  --artifact-root "$_eval_artifact_root"
)

evaluate_exit=0
korvid-prompt-lab "${_evaluate_args[@]}" || evaluate_exit=$?

# exit 1 is an expected safety signal; any other non-zero exit is systemic
if (( evaluate_exit != 0 && evaluate_exit != 1 )); then
  echo "evaluate returned unexpected exit code $evaluate_exit (systemic)" >&2
  exit "$evaluate_exit"
fi

# ---------------------------------------------------------------------------
# Report (runs even when evaluate exits 1)
# ---------------------------------------------------------------------------

_report_args=(
  --artifact-root "$_eval_artifact_root"
  --safe-output "${GROUNDING_ARTIFACT_ROOT}/safe-evidence"
  --prompt-lab-revision "$PROMPT_LAB_REVISION"
  --korvid-revision "$KORVID_REVISION"
  --workflow-run-url "$WORKFLOW_RUN_URL"
)
if [[ -n "${_opt_report_root:-}" ]]; then
  _report_args+=(--optimize-artifact-root "$_opt_report_root")
fi

korvid-grounding-report "${_report_args[@]}"

exit "$evaluate_exit"
