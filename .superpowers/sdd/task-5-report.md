# Task 5 Report

## Status

PASS — exactly one planned AKS action is bound, executed, classified, and
advanced through one cooperative-CAS transition.

## Commit

- `6c1c970` — `feat(campaigns): execute one holdout-safe action`

## RED / GREEN

- RED: campaign CLI rejected `--outcome-kind`; grounding still required
  singular split variables; the campaign-step wrapper did not exist.
- RED: scoped evaluation rejected planned split metadata outside the selected
  validation cases.
- RED: system/config retry accounting, reflection preflight classification,
  and evaluation-campaign evidence identity tests failed.
- GREEN: all focused process, workflow, controller-boundary, and CLI tests pass.

## Verification

```text
uv run --python 3.12 pytest tests/test_grounding_script.py tests/test_grounding_workflow.py tests/test_optimization_campaign_script.py tests/test_campaign_artifacts.py tests/test_campaign_cli.py tests/test_cli.py -q
# 251 passed in 65.11s

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
  evaluate-only mode with zero metric calls.
- Split emptiness, duplicates, pairwise overlap, unknown IDs, and scope/action
  mismatch fail before node-pool inspection.
- Live selected-tier digest validation runs after AKS readiness and before seed
  evaluation, optimization, or reflection.

## Exit, CAS, and cleanup invariants

- Grounding evidence exits are only `0`/`1`; systemic/config failures are `70`.
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
- Corrected evidence campaign identity to bind the evaluation campaign while
  action ID/CAS bind evidence to the optimization controller.
- Confirmed the pre-existing `.superpowers/sdd/progress.md` change was not
  staged or modified by Task 5.

## Concerns

None.
