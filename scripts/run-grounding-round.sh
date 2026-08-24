#!/usr/bin/env bash
# run-grounding-round.sh — protected grounding lifecycle orchestrator
#
# Required environment variables:
#   GROUNDING_MODEL            — allowed: qwen3:{0.6b,1.7b,4b,8b,14b}
#   GROUNDING_CANDIDATE        — path of the candidate YAML (checkout-relative)
#   GROUNDING_CAMPAIGN         — path of the campaign YAML (checkout-relative)
#   GROUNDING_ROUND_TYPE       — evaluate | optimize-evaluate
#   GROUNDING_TRAIN_CASE_ID       — case id forming the train split
#   GROUNDING_VALIDATION_CASE_ID  — case id forming the validation split
#   GROUNDING_MILESTONE_CASE_IDS  — comma-separated milestone-pack case ids
#   GROUNDING_MAX_METRIC_CALLS    — GEPA metric-call budget (positive integer)
#   GROUNDING_SEED                — GEPA search seed (non-negative integer)
#   KORVID_SOURCE_ROOT         — path to Korvid repository checkout
#   KORVID_AKS_MODEL           — model the campaign serves; must equal GROUNDING_MODEL
#   KORVID_AKS_NAMESPACE       — namespace the campaign port-forwards into
#   KORVID_AKS_SERVICE         — service the campaign port-forwards to
#   GROUNDING_ARTIFACT_ROOT    — directory for evaluation artifacts
#   WORKFLOW_RUN_URL           — GitHub Actions run URL (for report)
#   PROMPT_LAB_REVISION        — prompt-lab git revision (for report)
#   KORVID_REVISION            — korvid git revision (for report)
#
# Additional required for optimize-evaluate:
#   GROUNDING_REFLECTION_MODEL      — reflection LLM model identifier
#   GROUNDING_REFLECTION_CREDENTIAL — API credential for hosted reflection models
#
# The campaign resolves `models`, `serving.namespace`, `serving.service`, and
# `serving.model` through `env:` references, so the KORVID_AKS_* variables are as
# load-bearing as the campaign path itself: without them `load_campaign()` fails
# and every subcommand exits 2.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/reflection-provider.sh
source "${SCRIPT_DIR}/lib/reflection-provider.sh"

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

: "${GROUNDING_MODEL:?GROUNDING_MODEL is required}"
: "${GROUNDING_CANDIDATE:?GROUNDING_CANDIDATE is required}"
: "${GROUNDING_CAMPAIGN:?GROUNDING_CAMPAIGN is required}"
: "${GROUNDING_ROUND_TYPE:?GROUNDING_ROUND_TYPE is required}"
: "${GROUNDING_TRAIN_CASE_ID:?GROUNDING_TRAIN_CASE_ID is required}"
: "${GROUNDING_VALIDATION_CASE_ID:?GROUNDING_VALIDATION_CASE_ID is required}"
: "${GROUNDING_MILESTONE_CASE_IDS:?GROUNDING_MILESTONE_CASE_IDS is required}"
: "${GROUNDING_MAX_METRIC_CALLS:?GROUNDING_MAX_METRIC_CALLS is required}"
: "${GROUNDING_SEED:?GROUNDING_SEED is required}"
: "${KORVID_SOURCE_ROOT:?KORVID_SOURCE_ROOT is required}"
: "${KORVID_AKS_MODEL:?KORVID_AKS_MODEL is required by the campaign env: references}"
: "${KORVID_AKS_NAMESPACE:?KORVID_AKS_NAMESPACE is required by the campaign env: references}"
: "${KORVID_AKS_SERVICE:?KORVID_AKS_SERVICE is required by the campaign env: references}"
: "${GROUNDING_ARTIFACT_ROOT:?GROUNDING_ARTIFACT_ROOT is required}"
: "${WORKFLOW_RUN_URL:?WORKFLOW_RUN_URL is required}"
: "${PROMPT_LAB_REVISION:?PROMPT_LAB_REVISION is required}"
: "${KORVID_REVISION:?KORVID_REVISION is required}"

case "$GROUNDING_MODEL" in
  qwen3:0.6b|qwen3:1.7b|qwen3:4b|qwen3:8b|qwen3:14b) ;;
  *) echo "unsupported grounding model: $GROUNDING_MODEL" >&2; exit 2 ;;
esac

# The allowlist only binds the round if the model the campaign actually serves is
# the same one: otherwise the report would name one model and grade another.
if [[ "$KORVID_AKS_MODEL" != "$GROUNDING_MODEL" ]]; then
  echo "KORVID_AKS_MODEL ($KORVID_AKS_MODEL) must equal GROUNDING_MODEL ($GROUNDING_MODEL)" >&2
  exit 2
fi

case "$GROUNDING_ROUND_TYPE" in
  evaluate|optimize-evaluate) ;;
  *) echo "unsupported round type: $GROUNDING_ROUND_TYPE" >&2; exit 2 ;;
esac

_CASE_ID_PATTERN='^[A-Za-z0-9][A-Za-z0-9._-]*$'
_CASE_ID_LIST_PATTERN='^[A-Za-z0-9][A-Za-z0-9._-]*(,[A-Za-z0-9][A-Za-z0-9._-]*)*$'
_RELATIVE_YAML_PATTERN='^[A-Za-z0-9._/-]+\.(yaml|yml)$'

if [[ ! "$GROUNDING_CAMPAIGN" =~ $_RELATIVE_YAML_PATTERN ]] \
  || [[ "$GROUNDING_CAMPAIGN" == /* ]] \
  || [[ "$GROUNDING_CAMPAIGN" == *".."* ]]; then
  echo "GROUNDING_CAMPAIGN must be a relative YAML path inside the checkout: $GROUNDING_CAMPAIGN" >&2
  exit 2
fi

require_case_id() {
  local value="$1" label="$2"
  if [[ ! "$value" =~ $_CASE_ID_PATTERN ]]; then
    echo "invalid $label: case ids must match $_CASE_ID_PATTERN (got: $value)" >&2
    exit 2
  fi
}

# Splits a comma-separated list into the global array `split_case_ids_out`.
# `IFS=',' read -r -a` splits on commas only: no eval, no word splitting on
# whitespace, and no pathname expansion, so a hostile value can never become an
# extra argv token.  Every element is then checked against the case-id
# vocabulary, which rejects whitespace, globs, and option-looking values.
split_case_ids() {
  local raw="$1" label="$2"
  local -a parts=()
  local part
  split_case_ids_out=()

  # Validate the whole list first: `read -a` silently drops empty leading and
  # trailing fields, so "a," or ",a" would otherwise pass unnoticed.
  if [[ ! "$raw" =~ $_CASE_ID_LIST_PATTERN ]]; then
    echo "invalid $label: expected a comma-separated case id list (got: $raw)" >&2
    exit 2
  fi

  IFS=',' read -r -a parts <<< "$raw"
  for part in ${parts[@]+"${parts[@]}"}; do
    require_case_id "$part" "$label"
    split_case_ids_out+=("$part")
  done

  if (( ${#split_case_ids_out[@]} == 0 )); then
    echo "invalid $label: at least one case id is required" >&2
    exit 2
  fi
}

require_case_id "$GROUNDING_TRAIN_CASE_ID" "GROUNDING_TRAIN_CASE_ID"
require_case_id "$GROUNDING_VALIDATION_CASE_ID" "GROUNDING_VALIDATION_CASE_ID"

if [[ "$GROUNDING_TRAIN_CASE_ID" == "$GROUNDING_VALIDATION_CASE_ID" ]]; then
  echo "train and validation case sets must be disjoint: $GROUNDING_TRAIN_CASE_ID" >&2
  exit 2
fi

split_case_ids "$GROUNDING_MILESTONE_CASE_IDS" "GROUNDING_MILESTONE_CASE_IDS"
_milestone_args=()
for _milestone_case_id in "${split_case_ids_out[@]}"; do
  _milestone_args+=(--milestone-case-id "$_milestone_case_id")
done

if [[ ! "$GROUNDING_MAX_METRIC_CALLS" =~ ^[1-9][0-9]{0,4}$ ]]; then
  echo "GROUNDING_MAX_METRIC_CALLS must be a positive integer: $GROUNDING_MAX_METRIC_CALLS" >&2
  exit 2
fi

if [[ ! "$GROUNDING_SEED" =~ ^[0-9]{1,9}$ ]]; then
  echo "GROUNDING_SEED must be a non-negative integer: $GROUNDING_SEED" >&2
  exit 2
fi

# Optimization credentials are validated here — before the pool is read, scaled,
# or waited on — so a misconfigured round never costs cluster time.
_reflection_env=()
if [[ "$GROUNDING_ROUND_TYPE" == "optimize-evaluate" ]]; then
  : "${GROUNDING_REFLECTION_MODEL:?GROUNDING_REFLECTION_MODEL is required for optimize-evaluate}"
  if ! validate_reflection_model "$GROUNDING_REFLECTION_MODEL"; then
    echo "invalid reflection model: $GROUNDING_REFLECTION_MODEL" >&2
    exit 2
  fi

  if reflection_requires_credential "$GROUNDING_REFLECTION_MODEL"; then
    : "${GROUNDING_REFLECTION_CREDENTIAL:?GROUNDING_REFLECTION_CREDENTIAL is required for optimize-evaluate}"
    _reflection_cred_var="$(reflection_credential_env_name "$GROUNDING_REFLECTION_MODEL")"
    _reflection_env=("${_reflection_cred_var}=${GROUNDING_REFLECTION_CREDENTIAL}")
  else
    for _dns_label in "$KORVID_AKS_SERVICE" "$KORVID_AKS_NAMESPACE"; do
      if (( ${#_dns_label} > 63 )) || [[ ! "$_dns_label" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
        echo "invalid Kubernetes DNS label for Ollama reflection: $_dns_label" >&2
        exit 2
      fi
    done
    _reflection_env=(
      "OLLAMA_API_BASE=http://${KORVID_AKS_SERVICE}.${KORVID_AKS_NAMESPACE}.svc.cluster.local:11434"
    )
  fi
fi

# ---------------------------------------------------------------------------
# Prerequisite: verify required tools are available before scaling
# ---------------------------------------------------------------------------

for _required_tool in az kubectl kubelogin uv; do
  if ! command -v "$_required_tool" >/dev/null 2>&1; then
    echo "required tool not found: $_required_tool" >&2
    exit 1
  fi
done

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

# Runs one evaluate invocation under an identical contract (same campaign,
# case splits, and milestone pack) regardless of whether the candidate is the
# incoming seed or optimize's best candidate, so seed and best evidence are
# always comparable.
#
# Returns:
#   0   — evaluation completed, no hard safety failures
#   1   — validated safety result: evaluate reported hard safety failures and
#         evaluation-summary.json is internally consistent with that exit code
#   70  — orchestrator-internal systemic-evidence code: evaluate exited with
#         an unexpected code, produced no summary, or produced a summary that
#         is malformed or inconsistent with its own exit code. Never confused
#         with the validated safety exit 1 above.
run_evaluation() {
  local candidate="$1"
  local artifact_root="$2"
  local exit_code=0
  local summary

  mkdir -p "$artifact_root"
  korvid-prompt-lab evaluate \
    --candidate "$candidate" \
    --campaign "$GROUNDING_CAMPAIGN" \
    --artifact-root "$artifact_root" \
    --train-case-id "$GROUNDING_TRAIN_CASE_ID" \
    --validation-case-id "$GROUNDING_VALIDATION_CASE_ID" \
    "${_milestone_args[@]}" || exit_code=$?

  if (( exit_code != 0 && exit_code != 1 )); then
    echo "evaluate returned unexpected exit code $exit_code (systemic)" >&2
    return 70
  fi
  summary="${artifact_root}/evaluation-summary.json"
  if [[ ! -f "$summary" ]]; then
    echo "evaluate did not produce evaluation-summary.json (systemic error)" >&2
    return 70
  fi
  if ! evaluation_summary_matches_exit "$summary" "$exit_code"; then
    echo "evaluate summary is inconsistent or reports systemic failures" >&2
    return 70
  fi
  return "$exit_code"
}

# Validates that evaluation-summary.json is internally consistent with the
# exit code evaluate returned: zero non-negative safety/systemic counts, no
# systemic failures, and hard-safety-failure counts that agree with exit 0/1.
evaluation_summary_matches_exit() {
  python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
exit_code = int(sys.argv[2])
systemic = payload.get("systemic_failures")
hard = payload.get("hard_safety_failures")
if (
    type(systemic) is not int
    or systemic < 0
    or type(hard) is not int
    or hard < 0
    or systemic != 0
    or exit_code not in (0, 1)
):
    raise SystemExit(1)
raise SystemExit(0 if (exit_code == 0 and hard == 0) or (exit_code == 1 and hard > 0) else 1)
PY
}

# Reads an optimization-summary.json and prints "true"/"false" for whether the
# best candidate differs from the seed candidate, validated against the
# summary's own fingerprints. Exits non-zero on a malformed or inconsistent
# summary so callers never optimize on unverifiable evidence.
optimization_changed() {
  python3 - "$1" <<'PY'
import json
import re
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
seed = payload.get("seed_candidate_fingerprint")
best = payload.get("best_candidate_fingerprint")
changed = payload.get("best_candidate_differs_from_seed")
fingerprint = re.compile(r"[0-9a-f]{64}")
if (
    not isinstance(seed, str)
    or fingerprint.fullmatch(seed) is None
    or not isinstance(best, str)
    or fingerprint.fullmatch(best) is None
    or type(changed) is not bool
    or changed != (seed != best)
):
    raise SystemExit(1)
print("true" if changed else "false")
PY
}

cleanup() {
  local _status=$?
  if [[ "$_cleanup_ran" == true ]]; then exit "$_status"; fi
  _cleanup_ran=true
  if [[ "$scaled_by_round" == true ]]; then
    az aks nodepool scale \
      --resource-group rg-pension-guard \
      --cluster-name aks-shared-runners \
      --name modeleval \
      --node-count 0
  fi
  # Restore the status the shell was exiting with: some bash versions otherwise
  # replace it with the trap's own status, turning a failure into a green run.
  exit "$_status"
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
#
# exit 75 (EX_TEMPFAIL) means "not ready yet" and is retried until the deadline.
# exit 1 is a permanent preflight failure (cluster identity, credentials, missing
# tools) and exit 2 is usage/configuration — neither is retried.
# ---------------------------------------------------------------------------

_aks_check_artifact_root="${GROUNDING_ARTIFACT_ROOT}/aks-check"
mkdir -p "$_aks_check_artifact_root"
_aks_check_args=(
  aks-check
  --campaign "$GROUNDING_CAMPAIGN"
  --artifact-root "$_aks_check_artifact_root"
)

_preflight_deadline=$(( $(date +%s) + ${_AKS_CHECK_DEADLINE_SECONDS:-900} ))
_aks_check_poll_interval="${_AKS_CHECK_POLL_INTERVAL:-10}"
while true; do
  _preflight_exit=0
  korvid-prompt-lab "${_aks_check_args[@]}" || _preflight_exit=$?
  if (( _preflight_exit == 0 )); then
    break
  fi
  if (( _preflight_exit != 75 )); then
    echo "aks-check failed with exit $_preflight_exit (permanent or configuration); not retrying" >&2
    exit "$_preflight_exit"
  fi
  if (( $(date +%s) >= _preflight_deadline )); then
    echo "aks-check timed out after 15 minutes" >&2
    exit 1
  fi
  sleep "$_aks_check_poll_interval"
done

# ---------------------------------------------------------------------------
# optimize step (optimize-evaluate only) and evaluate (both round types)
# ---------------------------------------------------------------------------

_candidate="$GROUNDING_CANDIDATE"
_opt_artifact_root=""
_opt_report_root=""
_before_eval_artifact_root=""

if [[ "$GROUNDING_ROUND_TYPE" == "optimize-evaluate" ]]; then
  # Seed evidence: evaluate the incoming candidate before optimization touches
  # it, under the identical evaluation contract the best candidate will later
  # run under. This is same-round evidence, not a cached prior run, so the
  # comparison always has a fresh baseline — even when optimization leaves the
  # candidate unchanged.
  _before_eval_artifact_root="${GROUNDING_ARTIFACT_ROOT}/evaluate-before"
  before_exit=0
  run_evaluation "$_candidate" "$_before_eval_artifact_root" || before_exit=$?
  if (( before_exit == 70 )); then
    exit "$before_exit"
  fi

  _opt_artifact_root="${GROUNDING_ARTIFACT_ROOT}/optimize"
  mkdir -p "$_opt_artifact_root"
  _optimize_args=(
    optimize
    --candidate "$_candidate"
    --campaign "$GROUNDING_CAMPAIGN"
    --artifact-root "$_opt_artifact_root"
    --max-metric-calls "$GROUNDING_MAX_METRIC_CALLS"
    --seed "$GROUNDING_SEED"
    --reflection-model "$GROUNDING_REFLECTION_MODEL"
    --train-case-id "$GROUNDING_TRAIN_CASE_ID"
    --validation-case-id "$GROUNDING_VALIDATION_CASE_ID"
  )

  # optimize — never fall back to seed on failure; provider env scoped to subprocess only
  env "${_reflection_env[@]}" korvid-prompt-lab "${_optimize_args[@]}"

  # Resolve exactly one new best-candidate.yaml
  if ! _best_candidate="$(resolve_optimize_best_candidate "$_opt_artifact_root")"; then
    echo "optimize did not produce exactly one regular best-candidate.yaml under ${_opt_artifact_root}/invocations — aborting" >&2
    exit 1
  fi
  _opt_report_root="$(dirname "$_best_candidate")"
  _opt_summary="${_opt_report_root}/optimization-summary.json"

  if ! _best_changed="$(optimization_changed "$_opt_summary")"; then
    echo "optimize produced an invalid or inconsistent optimization summary at ${_opt_summary} — aborting" >&2
    exit 1
  fi

  if [[ "$_best_changed" == "true" ]]; then
    # The best candidate differs from the seed: it must clear the same
    # evaluation contract on its own evidence before it can be compared.
    _candidate="$_best_candidate"
    _eval_artifact_root="${GROUNDING_ARTIFACT_ROOT}/evaluate"
    evaluate_exit=0
    run_evaluation "$_candidate" "$_eval_artifact_root" || evaluate_exit=$?
    if (( evaluate_exit == 70 )); then
      exit "$evaluate_exit"
    fi
  else
    # Unchanged: the seed evaluation already is the best-candidate evaluation.
    # Reuse it rather than re-running an identical evaluation twice.
    _eval_artifact_root="$_before_eval_artifact_root"
    evaluate_exit="$before_exit"
  fi
else
  _eval_artifact_root="${GROUNDING_ARTIFACT_ROOT}/evaluate"
  evaluate_exit=0
  run_evaluation "$_candidate" "$_eval_artifact_root" || evaluate_exit=$?
  # evaluate-only rounds have no before/after comparison, so a systemic
  # evidence failure is reported as the conventional non-zero exit rather
  # than the orchestrator-internal 70 sentinel.
  if (( evaluate_exit == 70 )); then
    exit 1
  fi
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
if [[ -n "$_opt_report_root" ]]; then
  _report_args+=(--optimize-artifact-root "$_opt_report_root")
fi
if [[ -n "$_before_eval_artifact_root" ]]; then
  _report_args+=(--before-artifact-root "$_before_eval_artifact_root")
fi

korvid-grounding-report "${_report_args[@]}"

exit "$evaluate_exit"
