#!/usr/bin/env bash
# Re-prove the reviewed Korvid pin against the live GitHub API.
#
# `src/korvid_prompt_lab/korvid_pin.py` is a *dated snapshot*: the contract tests
# replay it offline so they stay deterministic, which by construction cannot
# notice that the upstream world moved. This script is the live half. Run it
# before trusting the pin again — after a Korvid pull request is merged, force
# pushed, or closed, or whenever a grounding round is rejected at the provenance
# gate.
#
# It proves the two independent properties a usable pin needs:
#
#   provenance    — the commit is authoritative Korvid code: contained in the
#                   default branch, or in the head of an open pull request of the
#                   authoritative repository itself (never a fork) targeting that
#                   default branch. This is the same rule the workflow's
#                   pre-credential trust gate enforces.
#   compatibility — every Korvid source path the bridge needs exists at that
#                   exact commit, and the checked-in bridge import contract still
#                   resolves from the source text there.
#
# Requires only `gh` (already required to work with this repository) and the
# `python3` that runs the test suite. It installs nothing and writes nothing.
#
# Usage:  scripts/verify-korvid-pin.sh
# Exit:   0 = pin re-proven, 1 = pin no longer provable (see the reported reason)

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
PIN_PATH="$REPO_ROOT/src/korvid_prompt_lab/korvid_pin.py"
BRIDGE_WORKER_PATH="$REPO_ROOT/src/korvid_prompt_lab/bridge_worker.py"

if ! command -v gh >/dev/null 2>&1; then
  echo "verify-korvid-pin: the GitHub CLI (gh) is required" >&2
  exit 1
fi

# Read the reviewed declaration rather than restating it: a second copy of the
# SHA in this file is exactly the drift the pin exists to prevent.
pin_field() {
  "$PYTHON" -c '
import ast
import sys

pin_path = sys.argv[1]
field = sys.argv[2]
bridge_worker_path = sys.argv[3]

with open(pin_path, encoding="utf-8") as pin_file:
    pin_tree = ast.parse(pin_file.read(), filename=pin_path)
with open(bridge_worker_path, encoding="utf-8") as bridge_worker_file:
    bridge_tree = ast.parse(bridge_worker_file.read(), filename=bridge_worker_path)

values = {}

def literal(node):
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {literal(key): literal(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "KorvidProvenance":
            return {keyword.arg: literal(keyword.value) for keyword in node.keywords}
        if node.func.id == "MappingProxyType" and len(node.args) == 1:
            return literal(node.args[0])
    return ast.literal_eval(node)

for statement in pin_tree.body:
    target = None
    value = None
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target, value = statement.targets[0], statement.value
    elif isinstance(statement, ast.AnnAssign):
        target, value = statement.target, statement.value
    if isinstance(target, ast.Name) and value is not None:
        try:
            values[target.id] = literal(value)
        except (KeyError, ValueError, TypeError):
            pass

def module_for_path(path):
    normalized = path.removeprefix("src/")
    if not normalized.endswith(".py"):
        raise ValueError(path)
    return normalized.removesuffix(".py").replace("/", ".")

paths = tuple(values["REQUIRED_KORVID_SOURCE_PATHS"])
module_to_path = {module_for_path(path): path for path in paths}

bridge_import = None
for node in ast.walk(bridge_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_import_korvid":
        bridge_import = node
        break
if bridge_import is None:
    raise SystemExit("bridge_worker._import_korvid not found")

submodule_aliases = {}
for node in ast.walk(bridge_import):
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            candidate = f"{module}.{alias.name}" if module else alias.name
            if candidate in module_to_path:
                submodule_aliases[alias.asname or alias.name] = candidate

checks = {}
for module, names in values["REQUIRED_KORVID_IMPORTS"].items():
    for name in names:
        candidate = f"{module}.{name}"
        if candidate in module_to_path:
            continue
        path = module_to_path.get(module)
        if path is None:
            continue
        checks.setdefault(path, set()).add(name)

for node in ast.walk(bridge_import):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        module_name = submodule_aliases.get(node.value.id)
        if module_name is None:
            continue
        checks.setdefault(module_to_path[module_name], set()).add(node.attr)

checks.setdefault(module_to_path["tests.evals.operation_app"], set()).add(
    values["REQUIRED_PATCHABLE_ATTRIBUTE"]
)

provenance = values["APPROVED_KORVID_PROVENANCE"]
if provenance["kind"] == values["PROVENANCE_OPEN_PULL_REQUEST"]:
    summary = (
        values["KORVID_REPOSITORY"]
        + "@"
        + values["APPROVED_KORVID_SHA"]
        + " (open PR #"
        + str(provenance["pull_request"])
        + " "
        + provenance["branch"]
        + " -> "
        + provenance["base_branch"]
        + "; compare vs "
        + values["KORVID_DEFAULT_BRANCH"]
        + ": "
        + provenance["default_branch_compare_status"]
        + "; PR head compare: "
        + provenance["head_compare_status"]
        + "; verified "
        + provenance["verified_on"]
        + ")"
    )
else:
    summary = (
        values["KORVID_REPOSITORY"]
        + "@"
        + values["APPROVED_KORVID_SHA"]
        + " (default branch "
        + values["KORVID_DEFAULT_BRANCH"]
        + "; compare vs "
        + values["KORVID_DEFAULT_BRANCH"]
        + ": "
        + provenance["default_branch_compare_status"]
        + "; verified "
        + provenance["verified_on"]
        + ")"
    )

if field == "sha":
    print(values["APPROVED_KORVID_SHA"])
elif field == "repository":
    print(values["KORVID_REPOSITORY"])
elif field == "default_branch":
    print(values["KORVID_DEFAULT_BRANCH"])
elif field == "pull_request":
    if provenance["kind"] != values["PROVENANCE_OPEN_PULL_REQUEST"]:
        raise SystemExit("pull_request is only defined for open-pull-request provenance")
    if provenance.get("pull_request") is None:
        raise SystemExit("open-pull-request provenance is missing pull_request")
    print(provenance["pull_request"])
elif field == "paths":
    print("\n".join(paths))
elif field == "summary":
    print(summary)
elif field == "import_checks":
    for path, names in sorted(checks.items()):
        if names:
            print(path + "\t" + ",".join(sorted(names)))
else:
    raise SystemExit("unknown pin field: " + field)
  ' "$PIN_PATH" "$1" "$BRIDGE_WORKER_PATH"
}

python_bindings() {
  local module_path="$1"
  shift
  "$PYTHON" -c '
import ast
import sys

module_path = sys.argv[1]
required = set(sys.argv[2:])
source = sys.stdin.read()
tree = ast.parse(source, filename=module_path)

bound = set()
for statement in tree.body:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bound.add(statement.name)
    elif isinstance(statement, ast.Assign):
        for target in statement.targets:
            if isinstance(target, ast.Name):
                bound.add(target.id)
    elif isinstance(statement, ast.AnnAssign):
        if isinstance(statement.target, ast.Name):
            bound.add(statement.target.id)
    elif isinstance(statement, ast.Import):
        for alias in statement.names:
            bound.add(alias.asname or alias.name.split(".")[0])
    elif isinstance(statement, ast.ImportFrom):
        for alias in statement.names:
            bound.add(alias.asname or alias.name)

missing = sorted(required - bound)
if missing:
    print(", ".join(missing))
    raise SystemExit(1)
  ' "$module_path" "$@"
}

fetch_raw() {
  gh api -H "Accept: application/vnd.github.raw" \
    "repos/${KORVID_REPO}/contents/$1?ref=${KORVID_SHA}" 2>/dev/null
}

KORVID_SHA="$(pin_field sha)"
KORVID_REPO="$(pin_field repository)"
DEFAULT_BRANCH="$(pin_field default_branch)"

echo "Verifying $(pin_field summary)"
echo

# --- provenance: default branch ---------------------------------------------
compare_status() {
  # $1 = base ref, $2 = head ref. Prints the compare status, or nothing when the
  # API cannot resolve the pair (an unresolvable compare is not a pass).
  gh api "repos/${KORVID_REPO}/compare/${1}...${2}" --jq ".status" 2>/dev/null || true
}

default_status="$(compare_status "$KORVID_SHA" "$DEFAULT_BRANCH")"
echo "provenance: compare ${KORVID_SHA}...${DEFAULT_BRANCH} => ${default_status:-unresolvable}"

provenance_route=""
if [[ "$default_status" == "identical" || "$default_status" == "ahead" ]]; then
  provenance_route="default branch ${DEFAULT_BRANCH}"
else
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
      declared_pr="$(pin_field pull_request)"
      if [[ "$pr_number" != "$declared_pr" ]]; then
        echo "note: the pin declares PR #${declared_pr} but PR #${pr_number} vouches for it now;" \
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

# --- compatibility: the bridge's Korvid source paths and symbols resolve -----
missing_paths=0
while IFS= read -r path; do
  [[ -n "$path" ]] || continue
  if gh api "repos/${KORVID_REPO}/contents/${path}?ref=${KORVID_SHA}" --jq ".sha" >/dev/null 2>&1; then
    echo "compatibility: present  ${path}"
  else
    echo "compatibility: MISSING  ${path}" >&2
    missing_paths=$((missing_paths + 1))
  fi
done < <(pin_field paths)

if [[ "$missing_paths" -ne 0 ]]; then
  echo >&2
  echo "FAIL: ${missing_paths} bridge dependency path(s) are absent at ${KORVID_SHA}." >&2
  echo "      A round would pass the trust gate and then die with" >&2
  echo "      'korvid operation harness is not importable' after spending credentials." >&2
  exit 1
fi

missing_bindings=0
while IFS=$'\t' read -r path csv_names; do
  [[ -n "$path" ]] || continue
  IFS=',' read -r -a required_names <<< "$csv_names"
  if ! raw_source="$(fetch_raw "$path")"; then
    echo "compatibility: UNREADABLE ${path}" >&2
    missing_bindings=$((missing_bindings + 1))
    continue
  fi
  if missing_names="$(printf '%s' "$raw_source" | python_bindings "$path" "${required_names[@]}")"; then
    echo "compatibility: binds    ${path} :: ${csv_names}"
  else
    echo "compatibility: MISSING  ${path} :: ${missing_names}" >&2
    missing_bindings=$((missing_bindings + 1))
  fi
done < <(pin_field import_checks)

if [[ "$missing_bindings" -ne 0 ]]; then
  echo >&2
  echo "FAIL: ${missing_bindings} bridge import-contract check(s) do not resolve at ${KORVID_SHA}." >&2
  echo "      Source-path existence alone is insufficient; repin to a commit whose" >&2
  echo "      runtime symbols satisfy bridge_worker._import_korvid and operation_app patching." >&2
  exit 1
fi

echo
echo "OK: ${KORVID_SHA} is authoritative ${KORVID_REPO} code and satisfies the bridge import contract."
