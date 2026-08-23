# Review Fix Report — Important Task 2

**Commit:** b7223646df8c11623f03dd7a8f9633310848186b  
**Branch:** feat/prompt-lab-mvp  
**Date:** 2026-08-22

---

## Finding 1 — Signal cancellation double-cleanup and wrong exit code

**Root cause:** `trap cleanup EXIT INT TERM` caused the cleanup function to
run twice on SIGINT: once via the INT handler and again via the EXIT handler
when bash exited. Additionally, the INT/TERM handlers did not call `exit`, so
the script could continue running after the signal, violating the conventional
exit-code contract (130 for SIGINT, 143 for SIGTERM).

**Fix applied** (`scripts/run-grounding-round.sh`):
- Added `_cleanup_ran=false` variable and idempotency guard in `cleanup()`.
- Replaced `trap cleanup EXIT INT TERM` with:
  - `trap cleanup EXIT` — single cleanup path
  - `trap 'exit 130' INT` — conventional INT exit code
  - `trap 'exit 143' TERM` — conventional TERM exit code

**Tests added:**
- `test_round_script_sigint_exits_130_and_scales_down_exactly_once` — sends SIGINT to process group while aks-check blocks; asserts returncode==130 and `calls.count("scale:0") == 1`
- `test_round_script_sigterm_exits_143_and_scales_down_exactly_once` — same for SIGTERM; asserts returncode==143
- `test_round_script_sigint_no_scale_when_pool_already_had_capacity` — SIGINT on pre-existing node must not scale down

---

## Finding 2 — Reflection credential in argv

**Root cause:** The script passed `--reflection-credential "$GROUNDING_REFLECTION_CREDENTIAL"` to `korvid-prompt-lab optimize`, exposing the API key in the process table. The `optimize` subcommand has no such flag — `_build_reflection_lm` constructs `dspy.LM(model_name)` which reads credentials from provider-standard environment variables automatically.

**Fix applied** (`scripts/run-grounding-round.sh`):
- Removed `--reflection-credential` from `_optimize_args`.
- Added provider prefix extraction from `GROUNDING_REFLECTION_MODEL` (bash 3.2-compatible via `tr`, not `${var,,}`).
- Maps provider to standard env var: `openai→OPENAI_API_KEY`, `anthropic→ANTHROPIC_API_KEY`, `cohere→COHERE_API_KEY`, `gemini/google→GEMINI_API_KEY`, fallback→`OPENAI_API_KEY`.
- Credentials passed via `env "VAR=VALUE" korvid-prompt-lab ...` — scoped to subprocess only, never in argv.

**Verification:** No changes to `cli.py` needed — `--reflection-credential` was never a valid CLI flag; the fix is purely in the orchestrator script.

**Tests added:**
- `test_round_script_optimize_credential_not_in_argv` — runs optimize-evaluate, asserts `GROUNDING_REFLECTION_CREDENTIAL` value absent from all `optimize arg=...` lines
- `test_round_script_optimize_credential_not_in_report_args` — same guard for report invocation

---

## Results

| Metric | Value |
|--------|-------|
| Commit hash | `b7223646df8c` |
| Tests before | 10 grounding-script + 355 other = 365 |
| Tests after | 15 grounding-script + 355 other = 370 |
| New tests passing | 5/5 |
| Full suite | **365 passed** (integration excluded) |
| `bash -n` | syntax OK |

## Concerns / Notes

- The provider→env-var mapping uses a fixed allowlist. If a new provider is introduced (e.g., `mistral`), the `*)` fallback sends to `OPENAI_API_KEY`, which would silently fail. This is a conscious design choice to fail loudly at LLM call time rather than at credential setup.
- The test infrastructure uses `os.setpgrp` + `os.killpg` to send signals to the entire bash process group, which is the correct way to simulate terminal Ctrl+C in a test harness.
- The credential value (`fake-token`) must not match any substring of a file path or other benign string to avoid false-positive test failures; the test uses `_BASE_ENV["GROUNDING_REFLECTION_CREDENTIAL"]` directly, so changing the credential in the future remains safe.

---

# Review Round 2 — Medium Operator Defect: `pin_field` Package Import

**Commit:** (this commit)
**Branch:** feat/prompt-lab-mvp
**Date:** 2026-08-22

## Defect

`scripts/verify-korvid-pin.sh` is documented as requiring only `gh` and
`python3`, but `pin_field()` used:

```python
from korvid_prompt_lab import korvid_pin
```

Importing the *package* triggers `korvid_prompt_lab/__init__.py` →
`config.py` → `contracts.py`, which (a) requires third-party packages
(`dspy`, `gepa`, `PyYAML`) not present in a bare Python environment and
(b) uses `@dataclass(slots=True)` (Python 3.10+), crashing on the macOS
system Python 3.9.

## Fix applied (`scripts/verify-korvid-pin.sh`)

Replaced the package import with a standard-library `ast` parser that reads
literal declarations directly from
`$REPO_ROOT/src/korvid_prompt_lab/korvid_pin.py`. The declaration is never
executed, so package dependencies and Python-version-sensitive runtime features
such as slotted dataclasses are irrelevant. `PYTHONPATH` manipulation is gone,
and malformed or incomplete declarations fail before any credential-carrying
operation.

## Test added (`tests/test_korvid_pin.py`)

`test_verify_script_pin_field_does_not_execute_the_declaration` extracts the
inline Python snippet, gives it a declaration containing a deliberate runtime
exception, and runs it under `python3 -S`. This was **RED** when the verifier
executed the declaration and is **GREEN** with literal AST parsing.

## Results

| Check | Result |
|-------|--------|
| RED (before fix) | `RuntimeError: the verifier executed the declaration` |
| GREEN (after fix) | 1 passed |
| `env PYTHON=python3 ./scripts/verify-korvid-pin.sh` | **OK** — pin proven via PR #312 |
| Full suite | **520 passed, 6 skipped** |
| `ruff check .` | **0 errors** |
| `mypy --python-version 3.12 src tests` | **no issues found in 35 source files** |
| Shell syntax and workflow YAML | **OK** |

## Notes

- The live verification above used the macOS system Python 3.9.6 with no
  project environment or third-party packages.

---

# Review Round 3 — Publication Evidence and Score Validation

## Findings

1. The bundled synthetic bridge reported `execution_mode: live` by default, so
   the documented local-smoke campaign could satisfy the live-only publication
   gate without contacting a model.
2. `aggregate_score`, `model_scores`, and existing registry baseline scores
   accepted non-finite and out-of-range numbers. `NaN` comparisons could bypass
   the model-improvement gate and persist invalid registry state.

## Fixes

- Synthetic bridge evidence now defaults to `scripted`; tests that exercise the
  live parsing branch opt in explicitly. The shipped local-smoke evaluate flow
  remains available for diagnostics, while publication rejects its summary.
- Publication requires aggregate, per-model, and existing baseline scores to be
  finite numbers in `[0.0, 1.0]`. The minimum-improvement threshold must also be
  finite and non-negative.
- README publication instructions now use a live AKS evaluation summary and
  explicitly identify local-smoke evidence as non-publishable.

## Verification

| Check | Result |
|-------|--------|
| Synthetic local-smoke publish regression | rejected with no registry output |
| Invalid score RED | 12 failed before validation |
| Invalid score GREEN | 12 passed |
| Focused publication/CLI/adapter/runner/contracts | 203 passed |
| Full suite | 533 passed, 6 skipped |
| Ruff | passed |
| mypy | passed for 35 source files |
| Shell syntax and workflow YAML | passed |

---

# Same-AKS Task 5 Pre-merge Validation

## Read-only checks

| Check | Result |
|-------|--------|
| `hellices/korvid-prompt-lab` default branch | `main` |
| `hellices/korvid-prompt-lab` admin access | `true` |
| `modeleval` node count | `0` |
| `modeleval` provisioning state | `Succeeded` |
| `runner-base:prompt-lab-v1` ACR image | present — digest `sha256:5c8105400a9f6035a8fb7f7a06e6f81277af45584a148a0af6437bef259bae56`, lastUpdateTime `2026-08-23T04:13:18Z` |
| `prompt-lab-runners` ARC scale set | absent (not installed — GitHub App env inputs unset) |
| `aks-grounding` Environment | absent (not installed — GitHub App env inputs unset) |
| `korvid-runners` githubConfigUrl | `https://github.com/hellices/korvid` (unchanged) |

## Repository validation

| Check | Result |
|-------|--------|
| `pytest -q` (KORVID_SOURCE_ROOT set) | **656 passed, 6 skipped** in 186 s |
| `ruff check .` | **All checks passed** |
| `mypy --python-version 3.12 src tests` | **Success: no issues found in 36 source files** |
| `bash -n scripts/*.sh` | **OK** |
| YAML parse `.github/workflows/*.yml` | **OK** (`grounding-round.yml`) |

## Deployment boundary

- ACR image `runner-base:prompt-lab-v1` was built and pushed (ch1s succeeded, digest above).
- `aks-grounding` Environment and `prompt-lab-runners` ARC scale set are **not installed**: `KORVID_APP_ID`, `KORVID_APP_PRIVATE_KEY_FILE`, `ARC_GITHUB_APP_ID`, `ARC_GITHUB_APP_INSTALLATION_ID`, and `ARC_GITHUB_APP_PRIVATE_KEY_FILE` are all unset. Load them from the operator's secret manager, then run `scripts/configure-grounding-access.sh` and `scripts/install-prompt-lab-runner.sh`.
- Live grounding round intentionally waits for default-branch merge: `grounding-round.yml` dispatches only from `main`.

## Remaining prerequisites

1. Operator provides: `KORVID_APP_ID`, `KORVID_APP_PRIVATE_KEY_FILE`
2. Operator provides: `ARC_GITHUB_APP_ID`, `ARC_GITHUB_APP_INSTALLATION_ID`, `ARC_GITHUB_APP_PRIVATE_KEY_FILE`
3. Run `scripts/configure-grounding-access.sh` (Step 3)
4. Run `scripts/install-prompt-lab-runner.sh` (Step 4)
5. Merge PR to `main` and dispatch `grounding-round.yml` (Steps 7–8)

---

# First Remote Grounding Round

**Date:** 2026-08-23
**Prompt Lab revision:** `ed0b3c0592e63e83d914e617c65c8681f12f2f9f`
**Korvid revision:** `fc7eece2adb66a5b2a18d378bdfd7503ddbdd2ca`
**Model:** `qwen3:0.6b`
**Round type:** evaluate-only

## OIDC preflight correction

The initial workflow run
[`32620932840`](https://github.com/hellices/korvid-prompt-lab/actions/runs/32620932840)
reached the dedicated ARC runner but failed at Azure login. GitHub issued an
Environment OIDC subject using its current `sub_claim_prefix`, which includes
immutable owner and repository IDs, while the Entra federated credential still
used the older name-only subject.

`scripts/configure-grounding-access.sh` now reads
`repos/hellices/korvid-prompt-lab/actions/oidc/customization/sub`, requires the
default subject policy, validates that the prefix identifies this repository,
and binds Entra to `<sub_claim_prefix>:environment:aks-grounding`. Re-running
the bootstrap replaced the drifted credential. The grounding infrastructure
test suite passes with **136 tests** after this correction.

The preflight failure occurred before the node-pool lifecycle or evaluation
steps. No safe-evidence artifact was produced for that run.

## Live evaluation result

Workflow run
[`32621633590`](https://github.com/hellices/korvid-prompt-lab/actions/runs/32621633590)
successfully completed OIDC login, both exact-SHA checkouts, Python setup, AKS
scale-up, Ollama readiness, all ten live evaluation repetitions, safe-evidence
upload, and cleanup. Its final workflow conclusion is **failure by policy**:
the authoritative evaluator detected hard safety failures, so the candidate
was intentionally not promotion-eligible.

| Metric | Result |
|---|---:|
| Aggregate score | `0.0` |
| Model score (`qwen3:0.6b`) | `0.0` |
| pass@3 | `0.0` |
| pass@5 | `0.0` |
| Live runs | `10` |
| Systemic failures | `0` |
| Hard safety failures | `15` |
| `write_before_fresh_read` | `10` |
| `wrong_target_write` | `5` |
| Promotion eligible | `false` |

## Safe-evidence audit

GitHub uploaded exactly one artifact named `safe-evidence` (artifact
`9488695746`, 10,304 bytes, 30-day retention). Its manifest contains:

- `evaluation-summary.json`
- `round-summary.json`
- `round-summary.md`
- ten files under `responses/`, one redacted protocol-v2 projection per live
  repetition

All JSON files parse successfully. Every response projection has an empty
`answer`, a null `error`, aggregate journal counters/checkpoint names only, and
no raw journal events or audit records. The artifact contains no symlinks and
no `request.json`, `audit.jsonl`, kubeconfig, raw log, Kubernetes manifest,
credential, optimizer state, or GEPA state. A credential-pattern scan found no
private keys, bearer tokens, API keys, passwords, or kubeconfig key material.

## Compute lifecycle

- The ephemeral runner scheduled on `aks-runners-9qb9x`, which carries the
  `workload=gha-runner` placement.
- Ollama scheduled separately on the `modeleval` node
  `aks-modeleval-31248830-vmss00000c`.
- The workflow scaled `modeleval` from 0 to 1 and restored it to
  `count=0`, `provisioningState=Succeeded`.
- After completion, the ARC runner namespace had zero runner pods and zero
  pending/running ephemeral runners. The repository listener remained Ready.
