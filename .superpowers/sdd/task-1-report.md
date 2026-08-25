STATUS: DONE

COMMITS:
- e1542a58f8718cb5937ef811ef0f6a9e1b86f889 fix(grounding): pin importable merged Korvid

FILES CHANGED:
- .github/workflows/grounding-round.yml
- README.md
- scripts/verify-korvid-pin.sh
- src/korvid_prompt_lab/bridge.py
- src/korvid_prompt_lab/bridge_worker.py
- src/korvid_prompt_lab/korvid_pin.py
- tests/test_bridge.py
- tests/test_bridge_worker.py
- tests/test_grounding_workflow.py
- tests/test_korvid_pin.py

RED EVIDENCE:
1) Diagnostic/setup command (needed because the worktree initially lacked dev test tools)
   Command:
   uv sync --python 3.12 --extra dev
   Output:
   Resolved 69 packages in 4ms
   Downloaded pygments
   Installed 10 packages in 94ms
   + pytest==9.1.1
   + mypy==2.3.1
   + ruff==0.16.3
   ...

2) Missing bridge import-preflight builder before implementation
   Command:
   uv run --python 3.12 pytest tests/test_bridge.py::test_bridge_check_imports_builds_worker_preflight -q
   Output:
   ERROR collecting tests/test_bridge.py
   ImportError: cannot import name 'build_worker_import_check' from 'korvid_prompt_lab.bridge'

3) Stale pin + missing worker/workflow interfaces before implementation
   Command:
   uv run --python 3.12 pytest \
     tests/test_korvid_pin.py::test_approved_pin_is_reviewed_squash_merge_on_default_branch \
     tests/test_bridge_worker.py::test_worker_check_imports_reports_missing_name_without_traceback \
     tests/test_grounding_workflow.py::test_grounding_workflow_preflights_korvid_runtime_imports_before_azure_and_scaling -q
   Output:
   FFF [100%]
   FAILED tests/test_korvid_pin.py::test_approved_pin_is_reviewed_squash_merge_on_default_branch
     AssertionError: assert 'fc7eece2adb66a5b2a18d378bdfd7503ddbdd2ca' == '62bd3cbee2e27369bb81abc0957dae341c2aa434'
   FAILED tests/test_bridge_worker.py::test_worker_check_imports_reports_missing_name_without_traceback
     argparse/SystemExit because --check-imports did not exist and --request/--response were still required
   FAILED tests/test_grounding_workflow.py::test_grounding_workflow_preflights_korvid_runtime_imports_before_azure_and_scaling
     AssertionError: workflow must preflight `korvid-bridge --check-imports` before Azure/model credentials or any AKS node-pool operation

GREEN EVIDENCE:
1) Focused regression suite
   Command:
   uv run --python 3.12 pytest \
     tests/test_korvid_pin.py \
     tests/test_bridge.py \
     tests/test_bridge_worker.py \
     tests/test_grounding_workflow.py -q
   Output:
   ........................................................................ [ 36%]
   ........................................................................ [ 72%]
   ........................................................                 [100%]
   200 passed in 22.26s

2) Live Korvid pin + import-contract verification
   Command:
   scripts/verify-korvid-pin.sh
   Output:
   Verifying hellices/korvid@62bd3cbee2e27369bb81abc0957dae341c2aa434 (default branch main; compare vs main: identical; verified 2026-08-26)
   provenance: compare 62bd3cbee2e27369bb81abc0957dae341c2aa434...main => identical
   provenance: PROVEN via default branch main
   compatibility: present  src/korvid/agent/profiles.py
   compatibility: present  src/korvid/evals/operation.py
   compatibility: present  src/korvid/evals/scripted.py
   compatibility: present  src/korvid/providers/openai_compat.py
   compatibility: present  src/korvid/providers/static_creds.py
   compatibility: present  tests/evals/operation_app.py
   compatibility: present  tests/evals/operation_campaign.py
   compatibility: present  tests/evals/operation_scripts.py
   compatibility: present  tests/ui/waits.py
   compatibility: binds    src/korvid/agent/profiles.py :: PromptOverrides
   compatibility: binds    src/korvid/evals/operation.py :: LIFECYCLE_CHECKPOINTS,bundled_operations_dir,load_operation_journeys
   compatibility: binds    src/korvid/evals/scripted.py :: ScriptedProvider
   compatibility: binds    src/korvid/providers/openai_compat.py :: OpenAICompatProvider,ProviderError
   compatibility: binds    src/korvid/providers/static_creds.py :: StaticHeaderSource
   compatibility: binds    tests/evals/operation_app.py :: build_profile,run_operation_journey
   compatibility: binds    tests/evals/operation_campaign.py :: approval_timeout_for
   compatibility: binds    tests/evals/operation_scripts.py :: OPERATION_SCRIPTS
   compatibility: binds    tests/ui/waits.py :: WaitTimeout
   OK: 62bd3cbee2e27369bb81abc0957dae341c2aa434 is authoritative hellices/korvid code and satisfies the bridge import contract.

3) Static checks
   Command:
   uv run --python 3.12 ruff check src/korvid_prompt_lab/bridge.py \
     src/korvid_prompt_lab/bridge_worker.py src/korvid_prompt_lab/korvid_pin.py \
     tests/test_bridge.py tests/test_bridge_worker.py tests/test_korvid_pin.py
   Output:
   All checks passed!

4) Type checks
   Command:
   uv run --python 3.12 mypy src/korvid_prompt_lab/bridge.py \
     src/korvid_prompt_lab/bridge_worker.py src/korvid_prompt_lab/korvid_pin.py
   Output:
   Success: no issues found in 3 source files

5) Shell syntax check
   Command:
   bash -n scripts/verify-korvid-pin.sh
   Output:
   (no output; exit 0)

SELF-REVIEW:
- The bridge now has a dedicated `build_worker_import_check()` path that reuses the exact `uv run --project <checkout> --no-sync`, `PYTHONPATH`, and `PYTHONDONTWRITEBYTECODE` contract used by real rounds.
- The worker exposes `--check-imports` as a mutually exclusive preflight path and reports sanitized symbol/module failures such as `korvid import failed: korvid.evals.operation: LIFECYCLE_CHECKPOINTS` without a traceback.
- `_import_korvid()` no longer hides the specific import failure behind a generic configuration error; the preflight and main entrypoint surface the real contract breach.
- `korvid_pin.py` now records the merged default-branch squash commit `62bd3cbee2e27369bb81abc0957dae341c2aa434` and default-branch provenance verified on `2026-08-26`.
- `scripts/verify-korvid-pin.sh` now proves both source-path existence and runtime symbol bindings by parsing the checked-in import contract and matching it against live GitHub raw source text (including annotated assignments, `async def`, imported patch targets, and the `build_profile` rebinding contract).
- The workflow now runs `korvid-bridge --check-imports` after Korvid checkout/dependency setup and before Azure OIDC, model credentials, or any AKS node-pool operation.
- README text was updated because the shipped user-facing contract and default pin are directly coupled to the workflow/pin changes and are asserted by existing tests.

CONCERNS:
- None for Task 1 itself.
- Note: `.superpowers/sdd/progress.md` had a pre-existing unstaged modification and was intentionally left out of the commit.

REVIEW FIXES:
- Commit: c43265a36ef0b00e47d9d9f9a709ecf6da2e9aad fix(grounding): apply task 1 review fixes

RED EVIDENCE:
1) Credential scrubbing regressions and dead default-branch provenance fields were reproduced first.
   Command:
   uv run --python 3.12 pytest \
     tests/test_bridge_worker.py::test_sanitize_import_error_redacts_structured_captures \
     tests/test_bridge_worker.py::test_sanitize_import_error_redacts_fallback_name_path \
     tests/test_bridge_worker.py::test_worker_check_imports_redacts_configured_secret_values \
     tests/test_korvid_pin.py::test_default_branch_provenance_does_not_carry_pull_request_only_fields \
     tests/test_korvid_pin.py::test_open_pull_request_provenance_requires_pull_request_fields -q
   Result:
   6 failed in 0.14s
   - structured import-error paths leaked `hunter2-secret`
   - `--check-imports` stderr leaked `hunter2-secret`
   - default-branch provenance still carried PR-only values (`pull_request == 312`)
   - `KorvidProvenance` did not reject missing PR fields for the open-PR route

GREEN EVIDENCE:
1) Focused review-fix regressions passed after the code changes.
   Command:
   uv run --python 3.12 pytest \
     tests/test_bridge_worker.py::test_sanitize_import_error_redacts_structured_captures \
     tests/test_bridge_worker.py::test_sanitize_import_error_redacts_fallback_name_path \
     tests/test_bridge_worker.py::test_worker_check_imports_redacts_configured_secret_values \
     tests/test_korvid_pin.py::test_default_branch_provenance_does_not_carry_pull_request_only_fields \
     tests/test_korvid_pin.py::test_open_pull_request_provenance_requires_pull_request_fields -q
   Result:
   6 passed in 0.11s

2) Covering test suite passed.
   Command:
   uv run --python 3.12 pytest \
     tests/test_bridge_worker.py \
     tests/test_korvid_pin.py \
     tests/test_bridge.py \
     tests/test_grounding_workflow.py -q
   Result:
   206 passed in 22.36s

3) Static/type/live verification passed.
   Command:
   uv run --python 3.12 ruff check \
     src/korvid_prompt_lab/bridge.py \
     src/korvid_prompt_lab/bridge_worker.py \
     src/korvid_prompt_lab/korvid_pin.py \
     tests/test_bridge.py tests/test_bridge_worker.py tests/test_korvid_pin.py tests/test_grounding_workflow.py
   Result:
   All checks passed!

   Command:
   uv run --python 3.12 mypy \
     src/korvid_prompt_lab/bridge.py \
     src/korvid_prompt_lab/bridge_worker.py \
     src/korvid_prompt_lab/korvid_pin.py
   Result:
   Success: no issues found in 3 source files

   Command:
   scripts/verify-korvid-pin.sh
   Result:
   Verifying hellices/korvid@62bd3cbee2e27369bb81abc0957dae341c2aa434 (default branch main; compare vs main: identical; verified 2026-08-26)
   provenance: compare 62bd3cbee2e27369bb81abc0957dae341c2aa434...main => identical
   provenance: PROVEN via default branch main
   compatibility: present  src/korvid/agent/profiles.py
   compatibility: present  src/korvid/evals/operation.py
   compatibility: present  src/korvid/evals/scripted.py
   compatibility: present  src/korvid/providers/openai_compat.py
   compatibility: present  src/korvid/providers/static_creds.py
   compatibility: present  tests/evals/operation_app.py
   compatibility: present  tests/evals/operation_campaign.py
   compatibility: present  tests/evals/operation_scripts.py
   compatibility: present  tests/ui/waits.py
   compatibility: binds    src/korvid/agent/profiles.py :: PromptOverrides
   compatibility: binds    src/korvid/evals/operation.py :: LIFECYCLE_CHECKPOINTS,bundled_operations_dir,load_operation_journeys
   compatibility: binds    src/korvid/evals/scripted.py :: ScriptedProvider
   compatibility: binds    src/korvid/providers/openai_compat.py :: OpenAICompatProvider,ProviderError
   compatibility: binds    src/korvid/providers/static_creds.py :: StaticHeaderSource
   compatibility: binds    tests/evals/operation_app.py :: build_profile,run_operation_journey
   compatibility: binds    tests/evals/operation_campaign.py :: approval_timeout_for
   compatibility: binds    tests/evals/operation_scripts.py :: OPERATION_SCRIPTS
   compatibility: binds    tests/ui/waits.py :: WaitTimeout
   OK: 62bd3cbee2e27369bb81abc0957dae341c2aa434 is authoritative hellices/korvid code and satisfies the bridge import contract.

SELF-REVIEW:
- `check_korvid_imports()` now matches `_import_korvid()`'s real contract: only import/attribute failures are handled there, and the main entrypoint passes a concrete environment mapping into the sanitization path.
- `sanitize_import_error()` now scrubs secret-bearing structured captures and the `error.name` fallback path while preserving the safe `module: missing-name` shape.
- Default-branch provenance now has its own invariant: successful default-branch compare status and no PR-only fields. Open-PR provenance is validated to require its route-specific fields.
- The verifier no longer assumes PR data exists for default-branch provenance, but still fails closed if an open-PR route ever needs a missing `pull_request` declaration.
