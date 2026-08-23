# AKS Ollama Reflection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable credentialless `ollama_chat/qwen3:14b` reflection inside AKS, then run canary and full `qwen3:0.6b` optimize-evaluate Grounding Rounds.

**Architecture:** A shared Bash provider-policy library classifies reflection models consistently in the access bootstrap and round orchestrator. Ollama reflection receives only a cluster-local `OLLAMA_API_BASE`; hosted providers retain protected provider-standard credentials. The existing workflow, safety gate, safe-evidence projection, and unconditional node-pool cleanup remain authoritative.

**Tech Stack:** Bash 3.2, GitHub Actions, Python 3.12+, pytest, DSPy, LiteLLM, GEPA, Ollama, AKS, Azure OIDC, GitHub CLI

## Global Constraints

- The initial teacher is exactly `ollama_chat/qwen3:14b`; the evaluated target is exactly `qwen3:0.6b`.
- Ollama traffic must use `http://ollama.ollama.svc.cluster.local:11434`, never the service load-balancer address.
- `ollama` and `ollama_chat` must not require or materialize an API credential.
- Hosted providers must continue to require a protected credential before any node-pool read or scale operation.
- Train case `aks-scale-deployment-up` and validation case `aks-restart-denied` must remain disjoint.
- Optimization must fail closed; it must never substitute the seed after a reflection or candidate-generation failure.
- Safe evidence must continue excluding raw answers, teacher prompts, request payloads, audit records, kubeconfigs, logs, credentials, and GEPA state.
- The canary uses four metric calls; the full round uses twelve metric calls and seed zero.
- Every live run must restore `modeleval` to its original node count on success, policy failure, timeout, or cancellation.

---

### Task 1: Add a shared reflection-provider policy

**Files:**
- Create: `scripts/lib/reflection-provider.sh`
- Modify: `scripts/run-grounding-round.sh`
- Test: `tests/test_grounding_script.py`

**Interfaces:**
- Consumes: `GROUNDING_REFLECTION_MODEL`, `GROUNDING_REFLECTION_CREDENTIAL`, `KORVID_AKS_NAMESPACE`, and `KORVID_AKS_SERVICE`.
- Produces: `validate_reflection_model MODEL`, `reflection_provider MODEL`, `reflection_requires_credential MODEL`, and `reflection_credential_env_name MODEL`.
- Produces for Ollama optimize subprocesses: `OLLAMA_API_BASE=http://SERVICE.NAMESPACE.svc.cluster.local:11434`.

- [ ] **Step 1: Write failing provider-boundary tests**

Update `_BASE_ENV` so the hosted-provider default is explicit:

```python
"GROUNDING_REFLECTION_MODEL": "openai/gpt-4.1-mini",
```

Extend the `korvid-prompt-lab` shim immediately after it records optimize argv:

```bash
if [[ "$_subcommand" == "optimize" ]]; then
    printf 'optimize env OPENAI_API_KEY=%s\n' "${OPENAI_API_KEY:+set}" >> "$CALLS"
    printf 'optimize env ANTHROPIC_API_KEY=%s\n' "${ANTHROPIC_API_KEY:+set}" >> "$CALLS"
    printf 'optimize env OLLAMA_API_BASE=%s\n' "${OLLAMA_API_BASE:-}" >> "$CALLS"
fi
```

Add these tests:

```python
def test_round_script_ollama_reflection_needs_no_credential_and_uses_cluster_dns(
    tmp_path: Path,
) -> None:
    result, calls = run_script(
        tmp_path,
        original_count=0,
        evaluation_exit=0,
        round_type="optimize-evaluate",
        extra_env={
            "GROUNDING_REFLECTION_MODEL": "ollama_chat/qwen3:14b",
            "GROUNDING_REFLECTION_CREDENTIAL": "",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "optimize env OPENAI_API_KEY=" in calls
    assert "optimize env ANTHROPIC_API_KEY=" in calls
    assert (
        "optimize env OLLAMA_API_BASE="
        "http://ollama.ollama.svc.cluster.local:11434"
    ) in calls


def test_round_script_hosted_reflection_still_requires_credential(
    tmp_path: Path,
) -> None:
    result, calls = run_script(
        tmp_path,
        original_count=0,
        round_type="optimize-evaluate",
        extra_env={
            "GROUNDING_REFLECTION_MODEL": "openai/gpt-4.1-mini",
            "GROUNDING_REFLECTION_CREDENTIAL": "",
        },
    )

    assert result.returncode != 0
    assert "GROUNDING_REFLECTION_CREDENTIAL" in result.stderr
    assert calls == []


@pytest.mark.parametrize(
    "model",
    ["ollama_chat", "ollama_chat/", "ollama chat/qwen3:14b", "ollama_chat/qwen3:14b;env"],
)
def test_round_script_rejects_malformed_reflection_model_before_cloud(
    tmp_path: Path,
    model: str,
) -> None:
    result, calls = run_script(
        tmp_path,
        original_count=0,
        round_type="optimize-evaluate",
        extra_env={
            "GROUNDING_REFLECTION_MODEL": model,
            "GROUNDING_REFLECTION_CREDENTIAL": "",
        },
    )

    assert result.returncode == 2
    assert "invalid reflection model" in result.stderr
    assert calls == []
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
uv run pytest -q \
  tests/test_grounding_script.py::test_round_script_ollama_reflection_needs_no_credential_and_uses_cluster_dns \
  tests/test_grounding_script.py::test_round_script_hosted_reflection_still_requires_credential \
  tests/test_grounding_script.py::test_round_script_rejects_malformed_reflection_model_before_cloud
```

Expected: Ollama fails because the current script requires
`GROUNDING_REFLECTION_CREDENTIAL`; malformed model cases are not rejected with
the required diagnostic.

- [ ] **Step 3: Implement the shared provider policy**

Create `scripts/lib/reflection-provider.sh`:

```bash
#!/usr/bin/env bash

validate_reflection_model() {
  local model="$1"
  [[ "$model" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._:/-]*$ ]]
}

reflection_provider() {
  local model="$1"
  printf '%s' "${model%%/*}" | tr '[:upper:]' '[:lower:]'
}

reflection_requires_credential() {
  case "$(reflection_provider "$1")" in
    ollama|ollama_chat) return 1 ;;
    *) return 0 ;;
  esac
}

reflection_credential_env_name() {
  case "$(reflection_provider "$1")" in
    openai) printf 'OPENAI_API_KEY' ;;
    anthropic) printf 'ANTHROPIC_API_KEY' ;;
    cohere) printf 'COHERE_API_KEY' ;;
    gemini|google) printf 'GEMINI_API_KEY' ;;
    ollama|ollama_chat) return 1 ;;
    *) printf 'OPENAI_API_KEY' ;;
  esac
}
```

Source it immediately after `set -Eeuo pipefail` in
`scripts/run-grounding-round.sh`:

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/reflection-provider.sh
source "${SCRIPT_DIR}/lib/reflection-provider.sh"
```

Replace the current reflection credential block with:

```bash
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
```

Invoke optimize with the provider-specific environment:

```bash
env "${_reflection_env[@]}" korvid-prompt-lab "${_optimize_args[@]}"
```

Update the script header to document that the credential is required only for
hosted providers.

- [ ] **Step 4: Run focused round-script tests**

Run:

```bash
uv run pytest -q tests/test_grounding_script.py
```

Expected: all round-script tests pass.

- [ ] **Step 5: Commit the provider policy**

```bash
git add scripts/lib/reflection-provider.sh scripts/run-grounding-round.sh tests/test_grounding_script.py
git commit -m "feat(grounding): support AKS-local Ollama reflection" \
  -m "Allow credentialless cluster-DNS reflection while preserving fail-closed hosted-provider credential handling.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Harden bootstrap and workflow secret materialization

**Files:**
- Modify: `scripts/configure-grounding-access.sh`
- Modify: `.github/workflows/grounding-round.yml`
- Test: `tests/test_grounding_infrastructure.py`
- Test: `tests/test_grounding_workflow.py`

**Interfaces:**
- Consumes: the shared provider-policy functions from Task 1.
- Produces: `GROUNDING_REFLECTION_MODEL=ollama_chat/qwen3:14b` as a protected Environment variable without requiring `GROUNDING_REFLECTION_CREDENTIAL`.
- Preserves: hosted-provider model-plus-credential bootstrap and evaluate-only empty reflection environment.

- [ ] **Step 1: Write failing access-bootstrap tests**

Add:

```python
def test_ollama_reflection_model_is_stored_without_a_credential(
    access: AccessHarness,
) -> None:
    run = access.run(GROUNDING_REFLECTION_MODEL="ollama_chat/qwen3:14b")

    assert_succeeded(run)
    assert run.variable("GROUNDING_REFLECTION_MODEL") == "ollama_chat/qwen3:14b"
    assert run.secret_names() == ["KORVID_APP_PRIVATE_KEY"]


def test_ollama_reflection_rejects_an_unnecessary_credential_file(
    access: AccessHarness,
) -> None:
    run = access.run(
        GROUNDING_REFLECTION_MODEL="ollama_chat/qwen3:14b",
        GROUNDING_REFLECTION_CREDENTIAL_FILE=str(access.reflection_file),
    )

    assert run.returncode != 0
    assert "must not be set for Ollama reflection" in run.output
    assert run.calls == []


def test_hosted_reflection_model_without_a_credential_file_still_fails_closed(
    access: AccessHarness,
) -> None:
    run = access.run(GROUNDING_REFLECTION_MODEL="openai/gpt-4.1-mini")

    assert run.returncode != 0
    assert "GROUNDING_REFLECTION_CREDENTIAL_FILE is required" in run.output
    assert run.calls == []
```

- [ ] **Step 2: Run access tests and verify RED**

Run:

```bash
uv run pytest -q \
  tests/test_grounding_infrastructure.py::test_ollama_reflection_model_is_stored_without_a_credential \
  tests/test_grounding_infrastructure.py::test_ollama_reflection_rejects_an_unnecessary_credential_file \
  tests/test_grounding_infrastructure.py::test_hosted_reflection_model_without_a_credential_file_still_fails_closed
```

Expected: the Ollama model-only bootstrap fails under the existing
both-or-neither rule.

- [ ] **Step 3: Make the access bootstrap provider-aware**

Source the Task 1 library after `set -Eeuo pipefail`:

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/reflection-provider.sh
source "${SCRIPT_DIR}/lib/reflection-provider.sh"
```

Replace the existing reflection validation with:

```bash
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
```

Replace the final storage block with:

```bash
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
```

Update the script header so Ollama documents model-only configuration and
hosted providers document model-plus-credential configuration.

- [ ] **Step 4: Write and run failing workflow-expression tests**

Extend
`test_grounding_workflow_scopes_reflection_credentials_to_optimize_rounds`:

```python
assert "startsWith(vars.GROUNDING_REFLECTION_MODEL, 'ollama/')" in credential_expr
assert "startsWith(vars.GROUNDING_REFLECTION_MODEL, 'ollama_chat/')" in credential_expr
assert credential_expr.count("!") >= 2
```

Run:

```bash
uv run pytest -q \
  tests/test_grounding_workflow.py::test_grounding_workflow_scopes_reflection_credentials_to_optimize_rounds
```

Expected: FAIL because the current expression injects the secret for every
optimize-evaluate provider.

- [ ] **Step 5: Exclude Ollama from credential materialization**

Set the orchestrator environment expression to:

```yaml
GROUNDING_REFLECTION_CREDENTIAL: >-
  ${{ inputs.round_type == 'optimize-evaluate'
      && !startsWith(vars.GROUNDING_REFLECTION_MODEL, 'ollama/')
      && !startsWith(vars.GROUNDING_REFLECTION_MODEL, 'ollama_chat/')
      && secrets.GROUNDING_REFLECTION_CREDENTIAL
      || '' }}
```

Keep `GROUNDING_REFLECTION_MODEL` restricted to `optimize-evaluate` exactly as
it is today.

- [ ] **Step 6: Run focused bootstrap and workflow tests**

Run:

```bash
uv run pytest -q \
  tests/test_grounding_infrastructure.py \
  tests/test_grounding_workflow.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit the protected-environment boundary**

```bash
git add \
  scripts/configure-grounding-access.sh \
  .github/workflows/grounding-round.yml \
  tests/test_grounding_infrastructure.py \
  tests/test_grounding_workflow.py
git commit -m "fix(grounding): keep Ollama rounds credentialless" \
  -m "Store only the in-cluster reflection model and prevent stale hosted-provider secrets from entering Ollama jobs.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Document and validate the complete implementation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-23-aks-ollama-reflection-design.md` only if implementation-review corrections are required

**Interfaces:**
- Consumes: the provider policy and workflow behavior from Tasks 1-2.
- Produces: operator instructions for configuring and dispatching the credentialless teacher.

- [ ] **Step 1: Add operator documentation**

Add an "AKS-local reflection" subsection near "Dispatching a grounding round"
with these commands and invariants:

```bash
printf '%s' 'ollama_chat/qwen3:14b' |
  gh variable set GROUNDING_REFLECTION_MODEL \
    --env aks-grounding \
    --repo hellices/korvid-prompt-lab
```

Document that no `GROUNDING_REFLECTION_CREDENTIAL` secret is required for
`ollama` or `ollama_chat`, that hosted providers still require it, and that the
workflow constructs the cluster-local base from `KORVID_AKS_SERVICE` and
`KORVID_AKS_NAMESPACE`.

- [ ] **Step 2: Run the complete local verification**

Run:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy --python-version 3.12 src tests
for script in scripts/*.sh scripts/lib/*.sh; do bash -n "$script"; done
uv run python - <<'PY'
from pathlib import Path
import yaml

for path in Path(".github/workflows").glob("*.yml"):
    yaml.safe_load(path.read_text(encoding="utf-8"))
PY
git diff --check
```

Expected:

- pytest passes with the established six integration skips only;
- Ruff reports `All checks passed!`;
- mypy reports no issues;
- every Bash script parses;
- every workflow YAML document parses;
- `git diff --check` prints nothing.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md docs/superpowers/specs/2026-08-23-aks-ollama-reflection-design.md
git commit -m "docs: explain AKS-local reflection rounds" \
  -m "Document credentialless Ollama teacher configuration and the hosted-provider fallback boundary.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

If the design spec did not need correction, stage and commit `README.md` only.

---

### Task 4: Review, publish, and merge the implementation

**Files:**
- Review: all changes from `main...feat/aks-ollama-reflection`

**Interfaces:**
- Consumes: a locally clean, fully verified feature branch.
- Produces: a reviewed merge commit on `origin/main`, which is required before
  dispatching the default-branch-only live workflow.

- [ ] **Step 1: Run an independent code review**

Invoke the `requesting-code-review` skill against the complete feature diff.
Resolve only high-confidence correctness, security, lifecycle, and leakage
findings. Re-run the focused tests for any changed area and then repeat the
complete Task 3 verification.

- [ ] **Step 2: Push without force**

```bash
git push -u origin feat/aks-ollama-reflection
```

Expected: the branch is published without bypassing hooks or rewriting
history.

- [ ] **Step 3: Create and inspect the pull request**

```bash
gh pr create \
  --repo hellices/korvid-prompt-lab \
  --base main \
  --head feat/aks-ollama-reflection \
  --title "feat(grounding): use AKS Ollama for reflection" \
  --body "## Summary
- allow credentialless Ollama reflection over cluster DNS
- preserve hosted-provider credential isolation
- add provider-boundary and workflow regression tests

## Validation
- full pytest suite
- Ruff
- mypy
- Bash syntax
- workflow YAML parse"
gh pr checks --watch
```

Expected: required checks pass.

- [ ] **Step 4: Merge and verify ancestry**

Use the `merge` skill. Then verify:

```bash
git fetch origin main
git merge-base --is-ancestor feat/aks-ollama-reflection origin/main
git rev-parse origin/main
```

Expected: the ancestry command exits zero and the printed SHA is the revision
used for live dispatch.

---

### Task 5: Run and audit canary and full Grounding Rounds

**Files:**
- Create: `docs/grounding-rounds/2026-08-23-qwen3-0.6b-ollama-reflection.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: merged Prompt Lab SHA, Korvid SHA
  `fc7eece2adb66a5b2a18d378bdfd7503ddbdd2ca`, protected `aks-grounding`
  Environment, and the installed `qwen3:14b` Ollama model.
- Produces: GitHub Actions summaries and safe-evidence artifacts for metric
  budgets four and twelve, plus a checked-in comparison with baseline run
  `32621633590`.

- [ ] **Step 1: Configure the protected reflection-model variable**

```bash
printf '%s' 'ollama_chat/qwen3:14b' |
  gh variable set GROUNDING_REFLECTION_MODEL \
    --env aks-grounding \
    --repo hellices/korvid-prompt-lab

gh variable list \
  --env aks-grounding \
  --repo hellices/korvid-prompt-lab \
  --json name \
  --jq '.[].name' |
  grep -Fx GROUNDING_REFLECTION_MODEL
```

Expected: only the variable name is printed; no secret value is requested or
logged.

- [ ] **Step 2: Dispatch the four-call canary**

```bash
PROMPT_LAB_SHA="$(git rev-parse origin/main)"
gh workflow run grounding-round.yml \
  --repo hellices/korvid-prompt-lab \
  --ref main \
  -f prompt_lab_ref="$PROMPT_LAB_SHA" \
  -f korvid_ref=fc7eece2adb66a5b2a18d378bdfd7503ddbdd2ca \
  -f model=qwen3:0.6b \
  -f round_type=optimize-evaluate \
  -f candidate=examples/candidates/shipped-small.yaml \
  -f campaign=examples/campaigns/aks-shared-runners.yaml \
  -f train_case_id=aks-scale-deployment-up \
  -f validation_case_id=aks-restart-denied \
  -f milestone_case_ids=aks-scale-deployment-up,aks-restart-denied \
  -f max_metric_calls=4 \
  -f seed=0

CANARY_RUN_ID="$(
  gh run list \
    --repo hellices/korvid-prompt-lab \
    --workflow grounding-round.yml \
    --event workflow_dispatch \
    --limit 1 \
    --json databaseId \
    --jq '.[0].databaseId'
)"
gh run watch "$CANARY_RUN_ID" --repo hellices/korvid-prompt-lab --exit-status || true
gh run view "$CANARY_RUN_ID" \
  --repo hellices/korvid-prompt-lab \
  --json conclusion,jobs,url
```

Treat a safety-gate conclusion as a model result, not infrastructure failure.
Do not proceed if checkout, Azure login, AKS scale-up, readiness, optimization,
artifact upload, or cleanup failed.

- [ ] **Step 3: Audit canary evidence and cleanup**

```bash
CANARY_DIR="$HOME/.copilot/session-state/47d05436-3db4-4f52-8056-abc93a0715f7/files/run-${CANARY_RUN_ID}-safe-evidence"
mkdir -p "$CANARY_DIR"
gh run download "$CANARY_RUN_ID" \
  --repo hellices/korvid-prompt-lab \
  --name safe-evidence \
  --dir "$CANARY_DIR"

find "$CANARY_DIR" -type l -print -quit | grep -q . && exit 1 || true
find "$CANARY_DIR" -type f -print | LC_ALL=C sort
grep -R -n -E \
  'BEGIN .*PRIVATE KEY|Bearer [A-Za-z0-9._-]+|api[_-]?key|password|kubeconfig' \
  "$CANARY_DIR" &&
  exit 1 || true

az aks nodepool show \
  --resource-group rg-pension-guard \
  --cluster-name aks-shared-runners \
  --name modeleval \
  --query '{count:count,state:provisioningState}' \
  --output json
```

Expected: no symlink or secret-pattern finding, and the node pool reports
`{"count": 0, "state": "Succeeded"}` after a round that started from zero.

- [ ] **Step 4: Dispatch the twelve-call full round**

Repeat Step 2 with:

```bash
-f max_metric_calls=12
```

Capture the newest workflow run as `FULL_RUN_ID`, watch it to completion, and
repeat the complete evidence and cleanup audit from Step 3 into:

```bash
$HOME/.copilot/session-state/47d05436-3db4-4f52-8056-abc93a0715f7/files/run-${FULL_RUN_ID}-safe-evidence
```

- [ ] **Step 5: Record the measured result**

Read `round-summary.md`, `optimization-summary.json`, and
`evaluation-summary.json` from both downloaded artifact directories. Use
`apply_patch` to create
`docs/grounding-rounds/2026-08-23-qwen3-0.6b-ollama-reflection.md` with exactly
these sections:

- `# qwen3:0.6b AKS Ollama Reflection Round`
- `## Provenance`, listing the observed Prompt Lab revision, fixed Korvid
  revision, target, teacher, seed and best-candidate fingerprints, and both run
  URLs.
- `## Baseline comparison`, containing a three-column table for baseline run
  `32621633590` and the full run. The six rows are aggregate score, pass@3,
  pass@5, total hard safety failures, `write_before_fresh_read`, and
  `wrong_target_write`. Copy the optimized values verbatim from the full
  artifact; the baseline values are respectively `0.0`, `0.0`, `0.0`, `15`,
  `10`, and `5`.
- `## Candidate and publication decision`, stating whether the best candidate
  differs from the seed, the exact promotion eligibility value, and every
  recorded promotion blocker.
- `## Artifact and cleanup audit`, stating safe-evidence file count, response
  projection count, redaction result, credential-pattern scan result, final
  node-pool count/state, and final ephemeral runner count.

Do not write an unknown or synthetic value. If the full run fails before
producing measured evidence, document the exact failed step and omit the
baseline-comparison table.

Update the README deployment-boundary table with the full run URL, measured
result, and publication decision.

- [ ] **Step 6: Validate and commit the live evidence**

```bash
git diff --check
git add \
  README.md \
  docs/grounding-rounds/2026-08-23-qwen3-0.6b-ollama-reflection.md
git commit -m "docs: record AKS Ollama reflection results" \
  -m "Capture the canary and full optimize-evaluate evidence, baseline comparison, safety decision, and compute cleanup.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push
```

Expected: the repository persistently records the actual result without raw
model output or secrets.
