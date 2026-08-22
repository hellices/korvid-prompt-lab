#!/usr/bin/env bash
# Re-prove the reviewed Korvid pin against the live GitHub API.
#
# `src/korvid_prompt_lab/korvid_pin.py` is a *dated snapshot*: the contract tests
# replay it offline so they stay deterministic, which by construction cannot
# notice that the upstream world moved.  This script is the live half.  Run it
# before trusting the pin again — after a Korvid pull request is merged, force
# pushed, or closed, or whenever a grounding round is rejected at the provenance
# gate.
#
# It proves the two independent properties a usable pin needs:
#
#   provenance    — the commit is authoritative Korvid code: contained in the
#                   default branch, or in the head of an open pull request of the
#                   authoritative repository itself (never a fork) targeting that
#                   default branch.  This is the same rule the workflow's
#                   pre-credential trust gate enforces.
#   compatibility — every Korvid source path the bridge worker imports exists at
#                   that exact commit.
#
# Requires only `gh` (already required to work with this repository) and the
# `python3` that runs the test suite.  It installs nothing and writes nothing.
#
# Usage:  scripts/verify-korvid-pin.sh
# Exit:   0 = pin re-proven, 1 = pin no longer provable (see the reported reason)

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

if ! command -v gh >/dev/null 2>&1; then
  echo "verify-korvid-pin: the GitHub CLI (gh) is required" >&2
  exit 1
fi

# Read the reviewed declaration rather than restating it: a second copy of the
# SHA in this file is exactly the drift the pin exists to prevent.
pin_field() {
  PYTHONPATH="$REPO_ROOT/src" "$PYTHON" -c '
import sys
from korvid_prompt_lab import korvid_pin

field = sys.argv[1]
if field == "sha":
    print(korvid_pin.APPROVED_KORVID_SHA)
elif field == "repository":
    print(korvid_pin.KORVID_REPOSITORY)
elif field == "default_branch":
    print(korvid_pin.KORVID_DEFAULT_BRANCH)
elif field == "pull_request":
    print(korvid_pin.APPROVED_KORVID_PROVENANCE.pull_request)
elif field == "paths":
    print("\n".join(korvid_pin.REQUIRED_KORVID_SOURCE_PATHS))
elif field == "summary":
    print(korvid_pin.approved_pin_summary())
else:
    raise SystemExit(f"unknown pin field: {field}")
' "$1"
}

KORVID_SHA="$(pin_field sha)"
KORVID_REPO="$(pin_field repository)"
DEFAULT_BRANCH="$(pin_field default_branch)"
DECLARED_PR="$(pin_field pull_request)"

echo "Verifying $(pin_field summary)"
echo

# --- provenance: default branch ---------------------------------------------
compare_status() {
  # $1 = base ref, $2 = head ref.  Prints the compare status, or nothing when the
  # API cannot resolve the pair (an unresolvable compare is not a pass).
  gh api "repos/${KORVID_REPO}/compare/${1}...${2}" --jq '.status' 2>/dev/null || true
}

default_status="$(compare_status "$KORVID_SHA" "$DEFAULT_BRANCH")"
echo "provenance: compare ${KORVID_SHA}...${DEFAULT_BRANCH} => ${default_status:-unresolvable}"

provenance_route=""
if [[ "$default_status" == "identical" || "$default_status" == "ahead" ]]; then
  provenance_route="default branch ${DEFAULT_BRANCH}"
else
  # --- provenance: open, same-repository pull requests -----------------------
  # `--jq` filters on the authoritative repository and base branch, so a fork
  # head can never reach the compare below.
  candidates="$(
    gh api "repos/${KORVID_REPO}/pulls?state=open&base=${DEFAULT_BRANCH}&per_page=100" \
      --jq ".[] | select(.head.repo.full_name == \"${KORVID_REPO}\") | select(.base.ref == \"${DEFAULT_BRANCH}\") | \"\(.number) \(.head.sha)\"" \
      2>/dev/null || true
  )"

  if [[ -z "$candidates" ]]; then
    echo "provenance: no open same-repository pull request targets ${DEFAULT_BRANCH}" >&2
  fi

  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    pr_number="${candidate%% *}"
    pr_head="${candidate##* }"
    pr_status="$(compare_status "$KORVID_SHA" "$pr_head")"
    echo "provenance: compare ${KORVID_SHA}...${pr_head} (PR #${pr_number}) => ${pr_status:-unresolvable}"
    if [[ "$pr_status" == "identical" || "$pr_status" == "ahead" ]]; then
      provenance_route="open pull request #${pr_number} (head ${pr_head})"
      if [[ "$pr_number" != "$DECLARED_PR" ]]; then
        echo "note: the pin declares PR #${DECLARED_PR} but PR #${pr_number} vouches for it now;" \
             "update korvid_pin.APPROVED_KORVID_PROVENANCE" >&2
      fi
      break
    fi
  done <<< "$candidates"
fi

if [[ -z "$provenance_route" ]]; then
  echo >&2
  echo "FAIL: ${KORVID_SHA} is no longer provable as authoritative ${KORVID_REPO} code." >&2
  echo "      The grounding round would be rejected at the pre-credential trust gate." >&2
  echo "      Repin to a commit that is contained in ${DEFAULT_BRANCH} or in an open" >&2
  echo "      same-repository pull request head, and update korvid_pin.py." >&2
  exit 1
fi

echo "provenance: PROVEN via ${provenance_route}"
echo

# --- compatibility: the bridge's Korvid source paths exist at the pin --------
missing=0
while IFS= read -r path; do
  [[ -n "$path" ]] || continue
  if gh api "repos/${KORVID_REPO}/contents/${path}?ref=${KORVID_SHA}" --jq '.sha' >/dev/null 2>&1; then
    echo "compatibility: present  ${path}"
  else
    echo "compatibility: MISSING  ${path}" >&2
    missing=$((missing + 1))
  fi
done < <(pin_field paths)

if [[ "$missing" -ne 0 ]]; then
  echo >&2
  echo "FAIL: ${missing} bridge dependency path(s) are absent at ${KORVID_SHA}." >&2
  echo "      A round would pass the trust gate and then die with" >&2
  echo "      'korvid operation harness is not importable' after spending credentials." >&2
  exit 1
fi

echo
echo "OK: ${KORVID_SHA} is authoritative ${KORVID_REPO} code and carries every bridge dependency."
