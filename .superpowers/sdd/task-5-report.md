# Task 5 Report

## Status

PASS — exactly one planned AKS action is bound, executed, classified, and
advanced through one cooperative-CAS transition.

## Commit

- `6c1c970` — `feat(campaigns): execute one holdout-safe action`
- `cbd3fcd` — `fix(campaigns): safely classify campaign evidence`
- `8c953b5` — `fix(workflow): gate GEPA budget to optimization`

## RED / GREEN

- RED: campaign CLI rejected `--outcome-kind`; grounding still required
  singular split variables; the campaign-step wrapper did not exist.
- RED: scoped evaluation rejected planned split metadata outside the selected
  validation cases.
- RED: system/config retry accounting, reflection preflight classification,
  and evaluation-campaign evidence identity tests failed.
- RED (review fix): milestone/confirm rejected the controller's zero GEPA
  budget, standalone evaluate required campaign action metadata, and malformed
  evidence could fail before state advancement.
- GREEN (review fix): real wrapper process tests reach exact milestone-only
  evaluation with no optimize; missing, malformed, and contradictory evidence
  preflight into exactly one SYSTEM_ERROR advance.
- GREEN: all focused process, workflow, controller-boundary, and CLI tests pass.

## Verification

```text
uv run --python 3.12 pytest tests/test_grounding_script.py tests/test_grounding_workflow.py tests/test_optimization_campaign_script.py tests/test_campaign_artifacts.py tests/test_campaign_cli.py tests/test_cli.py -q
# 262 passed in 72.80s

bash -n scripts/run-grounding-round.sh scripts/run-optimization-campaign-step.sh

uv run --python 3.12 ruff check src/korvid_prompt_lab/campaign_cli.py src/korvid_prompt_lab/cli.py src/korvid_prompt_lab/campaign_artifacts.py tests/test_campaign_cli.py tests/test_cli.py tests/test_campaign_artifacts.py tests/test_grounding_script.py tests/test_grounding_workflow.py tests/test_optimization_campaign_script.py
# All checks passed!

uv run --python 3.12 mypy src/korvid_prompt_lab/campaign_cli.py src/korvid_prompt_lab/cli.py src/korvid_prompt_lab/campaign_artifacts.py
# Success: no issues found in 3 source files
```

## Holdout proof

- SEARCH receives repeated train/validation IDs and exact validation
  `--case-id` values.
- SEARCH optimize/evaluate subprocesses receive no milestone argv and run with
  `GROUNDING_MILESTONE_CASE_IDS` removed from their environment.
- MILESTONE/CONFIRM select exact milestone `--case-id` values and use
  evaluate-only mode without exporting or passing a GEPA budget.
- Split emptiness, duplicates, pairwise overlap, unknown IDs, and scope/action
  mismatch fail before node-pool inspection.
- Live selected-tier digest validation runs after AKS readiness and before seed
  evaluation, optimization, or reflection.

## Exit, CAS, and cleanup invariants

- Grounding evidence exits are only `0`/`1`; systemic/config failures are `70`.
- Grounding exit `70` advances SYSTEM_ERROR directly. Exit `0`/`1` first uses
  read-only `validate-evidence`; validation failures advance SYSTEM_ERROR once
  against the unchanged prior state.
- A failure after successful evidence pre-validation is ambiguous and fails
  closed without a fallback advance, preventing double advancement.
- Optimizer failure never falls back to the seed.
- System errors increment retry count without metric/score movement; config
  errors are terminal without budget or retry consumption.
- The wrapper invokes the Grounding round at most once and `advance` once.
- Explicit expected-prior hash is checked before execution and again by
  `korvid-campaign advance`.
- Existing Grounding traps restore only capacity allocated by the round on
  success, hard-safety failure, systemic failure, digest mismatch, and signal.

## Self-review

- Confirmed no singular split environment names remain in workflow/script.
- Confirmed milestone flags occur only in the non-SEARCH evaluation branch.
- Restored standalone workflow default `round_type=evaluate`; campaign action
  metadata and GEPA budget are absent from ordinary evaluate execution.
- Corrected evidence campaign identity to bind the evaluation campaign while
  action ID/CAS bind evidence to the optimization controller.
- Confirmed the pre-existing `.superpowers/sdd/progress.md` change was not
  staged or modified by Task 5.

## Concerns

None.
