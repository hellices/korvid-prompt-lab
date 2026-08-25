#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/reflection-provider.sh
source "${SCRIPT_DIR}/lib/reflection-provider.sh"

: "${CAMPAIGN_CONTROL:?CAMPAIGN_CONTROL is required}"
: "${CAMPAIGN_STATE:?CAMPAIGN_STATE is required}"
: "${CAMPAIGN_CANDIDATE:?CAMPAIGN_CANDIDATE is required}"
: "${CAMPAIGN_OUTPUT_ROOT:?CAMPAIGN_OUTPUT_ROOT is required}"
: "${CAMPAIGN_EXPECTED_PRIOR_HASH:?CAMPAIGN_EXPECTED_PRIOR_HASH is required}"
: "${GROUNDING_CAMPAIGN:?GROUNDING_CAMPAIGN is required}"
: "${KORVID_AKS_NAMESPACE:?KORVID_AKS_NAMESPACE is required}"
: "${KORVID_AKS_SERVICE:?KORVID_AKS_SERVICE is required}"

if [[ -e "$CAMPAIGN_OUTPUT_ROOT" || -L "$CAMPAIGN_OUTPUT_ROOT" ]]; then
  echo "campaign output already exists: $CAMPAIGN_OUTPUT_ROOT" >&2
  exit 70
fi

_output_parent="$(cd -- "$(dirname -- "$CAMPAIGN_OUTPUT_ROOT")" && pwd)"
_work_root="${_output_parent}/.campaign-step-${$}"
if [[ -e "$_work_root" || -L "$_work_root" ]]; then
  echo "campaign step work path already exists: $_work_root" >&2
  exit 70
fi
mkdir -m 700 "$_work_root"
_cleanup_work=true
cleanup_step() {
  local status=$?
  if [[ "$_cleanup_work" == true && -d "$_work_root" && ! -L "$_work_root" ]]; then
    rm -rf -- "$_work_root"
  fi
  exit "$status"
}
trap cleanup_step EXIT

_action_path="${_work_root}/action.json"
if ! korvid-campaign plan \
    --control "$CAMPAIGN_CONTROL" \
    --state "$CAMPAIGN_STATE" \
    --output "$_action_path"; then
  echo "campaign planning failed" >&2
  exit 70
fi

_binding_path="${_work_root}/binding.json"
if ! python3 - "$CAMPAIGN_CONTROL" "$CAMPAIGN_STATE" "$_action_path" \
    "$GROUNDING_CAMPAIGN" "$CAMPAIGN_CANDIDATE" \
    "$CAMPAIGN_EXPECTED_PRIOR_HASH" "$_binding_path" <<'PY'
import json
import sys
from pathlib import Path

from korvid_prompt_lab.campaign_cli import _load_state
from korvid_prompt_lab.campaigns import (
    load_optimization_campaign,
    next_action,
    state_hash,
)
from korvid_prompt_lab.config import load_campaign, load_candidate

control_path, state_path, action_path, evaluation_path, candidate_path, expected, output = (
    sys.argv[1:]
)
evaluation = load_campaign(evaluation_path)
control = load_optimization_campaign(control_path, evaluation)
state = _load_state(Path(state_path))
current_hash = state_hash(state)
if current_hash != expected:
    raise SystemExit(
        f"expected prior hash mismatch: got {expected}, state has {current_hash}"
    )
payload = json.loads(Path(action_path).read_text(encoding="utf-8"))
if payload.get("terminal") is True:
    raise SystemExit("campaign has no expensive action to execute")
if set(payload) != {
    "action_id",
    "kind",
    "expected_state_hash",
    "stage_index",
    "seed_index",
    "tier_index",
    "metric_calls",
}:
    raise SystemExit("planned action has an invalid shape")
planned = next_action(control, state, __import__("datetime").datetime.now(
    tz=__import__("datetime").UTC
))
if planned is None:
    raise SystemExit("campaign has no expensive action to execute")
expected_action = {
    "action_id": planned.action_id,
    "kind": planned.kind.value,
    "expected_state_hash": planned.expected_state_hash,
    "stage_index": planned.stage_index,
    "seed_index": planned.seed_index,
    "tier_index": planned.tier_index,
    "metric_calls": planned.metric_calls,
}
if payload != expected_action or payload["expected_state_hash"] != expected:
    raise SystemExit("planned action does not exactly bind the prior state")
candidate = load_candidate(candidate_path)
if candidate.fingerprint != state.champion_fingerprint:
    raise SystemExit("candidate fingerprint does not match the state champion")
tier = control.model_tiers[planned.tier_index]
if (
    state.model_identity.name,
    state.model_identity.model,
    state.model_identity.digest,
) != (tier.name, tier.model, tier.digest):
    raise SystemExit("planned model tier does not match state model identity")
stage = control.stages[planned.stage_index]
seed = stage.seeds[planned.seed_index] if planned.kind.value == "search" else 0
round_type = "optimize-evaluate" if planned.kind.value == "search" else "evaluate"
evaluation_ids = (
    control.validation_case_ids
    if planned.kind.value == "search"
    else control.milestone_case_ids
)
binding = {
    "action_id": planned.action_id,
    "kind": planned.kind.value.upper(),
    "round_type": round_type,
    "model": tier.model,
    "digest": tier.digest,
    "metric_calls": planned.metric_calls,
    "seed": seed,
    "train_case_ids": list(control.train_case_ids),
    "validation_case_ids": list(control.validation_case_ids),
    "milestone_case_ids": list(control.milestone_case_ids),
    "evaluation_case_ids": list(evaluation_ids),
}
Path(output).write_text(json.dumps(binding), encoding="utf-8")
PY
then
  echo "campaign action binding failed" >&2
  exit 70
fi

json_scalar() {
  python3 - "$_binding_path" "$1" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]
if isinstance(value, (dict, list)) or value is None:
    raise SystemExit("binding scalar expected")
print(value)
PY
}
json_lines() {
  python3 - "$_binding_path" "$1" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]
if not isinstance(value, list) or not value or any(
    not isinstance(item, str) or not item for item in value
):
    raise SystemExit("binding string array expected")
print("\n".join(value))
PY
}

export GROUNDING_ACTION_KIND="$(json_scalar kind)"
export GROUNDING_ROUND_TYPE="$(json_scalar round_type)"
export GROUNDING_MODEL="$(json_scalar model)"
export KORVID_AKS_MODEL="$GROUNDING_MODEL"
if [[ "$GROUNDING_ACTION_KIND" == "SEARCH" ]]; then
  export GROUNDING_MAX_METRIC_CALLS="$(json_scalar metric_calls)"
else
  unset GROUNDING_MAX_METRIC_CALLS
fi
export GROUNDING_SEED="$(json_scalar seed)"
export GROUNDING_TRAIN_CASE_IDS="$(json_lines train_case_ids)"
export GROUNDING_VALIDATION_CASE_IDS="$(json_lines validation_case_ids)"
export GROUNDING_MILESTONE_CASE_IDS="$(json_lines milestone_case_ids)"
export GROUNDING_EVALUATION_CASE_IDS="$(json_lines evaluation_case_ids)"
export GROUNDING_CAMPAIGN_ACTION_ID="$(json_scalar action_id)"
export GROUNDING_CANDIDATE="$CAMPAIGN_CANDIDATE"
export GROUNDING_OPTIMIZATION_CAMPAIGN="$CAMPAIGN_CONTROL"
export GROUNDING_ARTIFACT_ROOT="${_work_root}/round"
export GROUNDING_MODEL_ENDPOINT="${GROUNDING_MODEL_ENDPOINT:-http://${KORVID_AKS_SERVICE}.${KORVID_AKS_NAMESPACE}.svc.cluster.local:11434}"

_round_script="${GROUNDING_ROUND_SCRIPT:-${SCRIPT_DIR}/run-grounding-round.sh}"
_round_exit=0
_config_error=""
if [[ "$GROUNDING_ACTION_KIND" == "SEARCH" ]]; then
  if [[ -z "${GROUNDING_REFLECTION_MODEL:-}" ]] \
      || ! validate_reflection_model "$GROUNDING_REFLECTION_MODEL"; then
    _config_error="reflection model is missing or invalid"
  elif reflection_requires_credential "$GROUNDING_REFLECTION_MODEL" \
      && [[ -z "${GROUNDING_REFLECTION_CREDENTIAL:-}" ]]; then
    _config_error="reflection credential is required"
  fi
fi
if [[ -n "$_config_error" ]]; then
  mkdir -p "$GROUNDING_ARTIFACT_ROOT"
  printf '%s\n' "config_error" > "${GROUNDING_ARTIFACT_ROOT}/outcome-kind"
  echo "$_config_error" >&2
  _round_exit=70
else
  bash "$_round_script" || _round_exit=$?
fi

_outcome_kind=evidence
_advance_args=()
case "$_round_exit" in
  0|1)
    _safe_evidence="${GROUNDING_ARTIFACT_ROOT}/safe-evidence"
    if korvid-campaign validate-evidence \
        --control "$CAMPAIGN_CONTROL" \
        --state "$CAMPAIGN_STATE" \
        --action "$_action_path" \
        --evidence "$_safe_evidence" \
        --expected-prior-hash "$CAMPAIGN_EXPECTED_PRIOR_HASH"; then
      _advance_args+=(--evidence "$_safe_evidence")
    else
      echo "grounding safe evidence failed strict validation" >&2
      _outcome_kind=system_error
    fi
    ;;
  70)
    if [[ -f "${GROUNDING_ARTIFACT_ROOT}/outcome-kind" ]] \
        && [[ "$(<"${GROUNDING_ARTIFACT_ROOT}/outcome-kind")" == "config_error" ]]; then
      _outcome_kind=config_error
    else
      _outcome_kind=system_error
    fi
    ;;
  *)
    echo "grounding round returned unclassified exit $_round_exit" >&2
    _outcome_kind=system_error
    _round_exit=70
    ;;
esac

if [[ "$_outcome_kind" != "evidence" ]]; then
  _advance_args+=(--outcome-kind "$_outcome_kind" --error-message "grounding round exit ${_round_exit}")
fi

_next_state="${_work_root}/next-state.json"
if ! korvid-campaign advance \
    --control "$CAMPAIGN_CONTROL" \
    --state "$CAMPAIGN_STATE" \
    --action "$_action_path" \
    "${_advance_args[@]}" \
    --output-state "$_next_state" \
    --expected-prior-hash "$CAMPAIGN_EXPECTED_PRIOR_HASH"; then
  echo "campaign state advance failed" >&2
  exit 70
fi

_rendered="${_work_root}/rendered"
if ! korvid-campaign render \
    --control "$CAMPAIGN_CONTROL" \
    --state "$_next_state" \
    --output-dir "$_rendered"; then
  echo "campaign artifact rendering failed" >&2
  exit 70
fi
cp "$_action_path" "${_rendered}/campaign-action.json"
if [[ -d "${GROUNDING_ARTIFACT_ROOT}/safe-evidence" ]]; then
  cp -R "${GROUNDING_ARTIFACT_ROOT}/safe-evidence" "${_rendered}/round-evidence"
fi
mv "$_rendered" "$CAMPAIGN_OUTPUT_ROOT"

_status="$(python3 - "$CAMPAIGN_OUTPUT_ROOT/campaign-state.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)"

if [[ "$_outcome_kind" != "evidence" ]]; then
  exit 70
fi
case "$_status" in
  running|qualified) exit 0 ;;
  not_converged) exit 1 ;;
  *) exit 70 ;;
esac
