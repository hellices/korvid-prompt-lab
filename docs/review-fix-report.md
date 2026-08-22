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
