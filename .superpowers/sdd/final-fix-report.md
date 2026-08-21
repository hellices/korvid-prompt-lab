# Final review fix wave

**Date:** 2026-08-22
**Worktree:** `.worktrees/feat-prompt-lab-mvp`
**Baseline commit:** `149a7d6` (`feat: complete Korvid prompt lab MVP`)
**Fix commit:** `9541177` (`fix: close final prompt lab review findings`)
**Python:** 3.12.13 (`.venv`, `uv run --python 3.12` for CLI smoke)
**Skills used:** `superpowers:systematic-debugging`, then `superpowers:test-driven-development`
(each finding: RED → observed failure → minimal GREEN → focused re-run).

## Pre-existing baseline

`pytest -q` on `149a7d6`: **1 failed, 108 passed**.

```text
FAILED tests/test_cli.py::test_evaluate_runs_fake_bridge_and_emits_json_summary
evaluation failed: systemic bridge error: bridge command could not be launched: python
```

Root cause: `examples/campaigns/local-smoke.yaml` used the bare token `python`, which does
not exist outside a `uv run` shell. Fixed as part of the example rework (`python3`), because
the same example files had to change for findings 2–4.

## Finding 1 (Critical) — real GEPA reflective mutation silently no-ops

### Root cause

`gepa/proposer/reflective_mutation/reflective_mutation.py:95` reads
`self.adapter.propose_new_texts` on **every** mutation. `KorvidGEPAAdapter` never declared
that attribute, so the access raised `AttributeError` inside GEPA's own
`try/except Exception` (line ~305), which logs and `return None`. The engine then logged
"Reflective mutation did not propose a new candidate" for every iteration, so optimization
always returned the seed candidate and never reached the DSPy `custom_candidate_proposer`.

### RED evidence

`.venv/bin/python -m pytest -q tests/test_adapter.py::test_real_gepa_invokes_the_adapter_proposal_contract_and_can_beat_the_seed`

```text
Iteration 5: Exception during reflection/proposal: 'KorvidGEPAAdapter' object has no attribute 'propose_new_texts'
Traceback (most recent call last):
  File ".../gepa/proposer/reflective_mutation/reflective_mutation.py", line 297, in propose
    new_texts = self.propose_new_texts(curr_prog, reflective_dataset, predictor_names_to_update)
  File ".../gepa/proposer/reflective_mutation/reflective_mutation.py", line 95, in propose_new_texts
    if self.adapter.propose_new_texts is not None:
AttributeError: 'KorvidGEPAAdapter' object has no attribute 'propose_new_texts'

Iteration 5: Reflective mutation did not propose a new candidate
>       assert proposals, "real GEPA reflective mutation must invoke the proposal contract"
E       AssertionError: real GEPA reflective mutation must invoke the proposal contract
E       assert []
tests/test_adapter.py:282: AssertionError
1 failed
```

`.venv/bin/python -m pytest -q tests/test_optimize.py`

```text
E       TypeError: optimize_campaign() got an unexpected keyword argument 'candidate_proposer'
2 failed, 2 passed
```

### GREEN

- `KorvidGEPAAdapter.propose_new_texts: ProposalFn | None = None` (contract declared,
  proposal responsibility stays outside the adapter).
- `optimize_campaign(..., candidate_proposer=...)` for deterministic injected proposers;
  `reflection_lm` and `candidate_proposer` are mutually exclusive (GEPA also rejects the
  combination of `adapter.propose_new_texts` and `custom_candidate_proposer`).
- DSPy stays reflection-only: `DSPyInstructionProposer` is still only ever passed as GEPA's
  `custom_candidate_proposer`, never installed on the adapter.
- `optimization-summary.json` now records `seed_candidate_fingerprint`,
  `best_candidate_differs_from_seed`, `train_case_ids`, `validation_case_ids`.
- Fake bridge grades a candidate carrying the `korvid-tuned` marker higher (0.95 vs 0.85), so
  an accepted proposal is observable through the real subprocess boundary.

Tests (unpatched `gepa.optimize` + real fake-bridge subprocess):
`tests/test_adapter.py::test_real_gepa_invokes_the_adapter_proposal_contract_and_can_beat_the_seed`,
`tests/test_optimize.py::test_optimize_campaign_runs_real_gepa_and_persists_a_candidate_that_beats_the_seed`,
`tests/test_optimize.py::test_optimize_campaign_rejects_combining_reflection_lm_and_candidate_proposer`.

Result: `9 passed` (`tests/test_optimize.py tests/test_adapter.py`).

## Finding 2 (Critical) — AKSPortForward only used by `aks-check`

### RED evidence

`pytest -q tests/test_runner.py`

```text
E       TypeError: AKSPortForwardServing.__init__() got an unexpected keyword argument 'command'
E       KeyError: 'model_endpoint'
E       TypeError: KorvidProcessRunner.__init__() got an unexpected keyword argument 'model_endpoint'
```

`pytest -q tests/test_contracts.py`

```text
E       ValueError: serving.aks_port_forward has unknown field(s): command
E       Failed: DID NOT RAISE ValueError        # command without {request}/{response}
```

`pytest -q tests/test_cli.py -k "aks or forward"`

```text
E       AssertionError: evaluation failed: aks_port_forward serving requires a model_endpoint
E       assert 2 == 0
E       AssertionError: optimization failed: aks_port_forward serving requires a model_endpoint
4 failed, 1 passed
```

(Before the runner change the same commands failed with
`evaluation failed: KorvidProcessRunner requires process serving`, exit `2`.)

### GREEN

- `AKSPortForwardServing.command` added; strict parser `_parse_bridge_command` shared by both
  backends requires a non-empty argument list, both `{request}` and `{response}` placeholders,
  and rejects `env:` interpolation. No shell is ever used.
- `KorvidProcessRunner` accepts `process` **or** `aks_port_forward` serving and a
  `model_endpoint`; endpoints must be loopback `http://127.0.0.1|localhost|[::1]:<port>` base
  URLs (no path/query/fragment). `aks_port_forward` requires one; `process` forbids one.
- Requests carry `runtime.model_endpoint` (null for process serving).
- `cli._serving_session(campaign, workspace_dir)` opens **one** `AKSPortForward` for the whole
  `evaluate` / `optimize` run and closes it on every exit path; `AKSPortForwardError` is exit `1`.
- Fake bridge echoes `runtime.model_endpoint` into the response journal; a recording bridge in
  the CLI tests appends one line per request to the same event log as the fake forward.

Ordering proof (`tests/test_cli.py::test_evaluate_keeps_one_loopback_forward_open_for_the_whole_aks_run`):
`["enter", "request:http://127.0.0.1:41001" x4, "exit"]`, one forward instance, workspace dir =
artifact root, and every persisted `request.json` carries the endpoint. Cleanup is also proven for
systemic bridge failure and for optimize (`["enter", "optimize", "exit"]`).
No cluster mutation, no public exposure: discovery stays `az aks show` / `get-credentials --file`
and `kubectl ... get`, forwarding stays `--address 127.0.0.1`.

## Finding 3 (Important) — pass^3/pass^5 were any-of-k and fabricated 1.0

### RED evidence

`pytest -q tests/test_scoring.py`

```text
E   ImportError: cannot import name 'RepetitionOutcome' from 'korvid_prompt_lab.scoring'
```

`pytest -q tests/test_cli.py -k pass_hat_k` (campaign with `repetitions: 1`)

```text
E       KeyError: 'repetitions_per_case'
E       AssertionError: assert 'pass^3=insufficient-evidence' in 'evaluated candidate=shipped-small
        campaign=local-smoke aggregate=0.850 pass^3=1.000 pass^5=1.000 summary=...'
3 failed
```

`pytest -q tests/test_publish.py -k "pass_hat_k or unit_interval"`

```text
E         Expected regex: 'pass_at_3 requires 3 recorded repetitions'
E         Actual message: 'pass_at_3 must be numeric'
E       Failed: DID NOT RAISE ValueError     # pass_at_3 = -0.1 / 1.5
4 failed
```

### GREEN

- `scoring.pass_hat_k(outcomes, k) -> float | None`: share of case/model groups whose **first k
  repetitions all passed**; `None` when no outcomes exist or any group ran fewer than `k`
  repetitions; rejects non-positive `k` and duplicate repetitions.
- Summary gains `repetitions_per_case`; `pass_at_3` / `pass_at_5` are `null` for insufficient
  evidence and the text summary prints `pass^3=insufficient-evidence`.
- `publish` refuses `null` (`pass_at_3 requires 3 recorded repetitions per case before
  publication`) and out-of-range values.
- Example campaigns moved to `repetitions: 5` so the documented flow stays publishable.

Behaviour change proven by `tests/test_cli.py::test_evaluate_pass_hat_k_requires_every_repetition_to_pass`
(one healthy case, one case failing from repetition 3 via the new `flaky-after-2` bridge tag):
old any-of-k reported `1.0`, new pass^3 and pass^5 report `0.5`.

## Finding 4 (Important) — train and validation sets were identical

### RED evidence

`pytest -q tests/test_optimize.py -k disjoint` (empty/overlapping splits were accepted and the
run proceeded into GEPA instead of being rejected)

```text
E           AssertionError: reflection_lm was not provided. The adapter used '<korvid_prompt_lab.adapter.KorvidGEPAAdapter ...>' ...
4 failed
```

`pytest -q tests/test_cli.py -k "case_splits or case_sets or not_evaluated"`

```text
E       SystemExit: 2          # evaluate had no --train-case-id / --validation-case-id / --milestone-case-id
E       assert 1 == 2          # optimize accepted a missing/overlapping split
```

`pytest -q tests/test_publish.py` (overlapping, missing, and out-of-campaign case sets)

```text
E       Failed: DID NOT RAISE ValueError   (x5)
```

Old behaviour recorded `"train": evaluated_case_ids, "validation": evaluated_case_ids` — the two
sets were literally the same list.

### GREEN

- `optimize_campaign` rejects empty or overlapping splits and records the real sets.
- `optimize` CLI requires `--train-case-id` and `--validation-case-id` (disjoint), exit `2`.
- `evaluate` CLI requires the same two flags, adds optional `--milestone-case-id`, requires every
  recorded case to have actually been evaluated, and records the real sets in `case_sets`.
  `milestone_passed` now also requires the recorded milestone pack to cover the required pack.
- Reproduction command in the summary includes the new flags.
- `publish_bundle` rejects missing, empty, overlapping, or out-of-campaign train/validation sets.
- Examples now ship two cases each so a disjoint split exists.

## Finding 5 (Important) — model override tie published with a zero threshold

### RED evidence

`pytest -q tests/test_publish.py -k "threshold or non_zero"`

```text
E   ImportError: cannot import name 'DEFAULT_MINIMUM_MODEL_IMPROVEMENT' from 'korvid_prompt_lab.publish'
```

With the constant added but the old non-strict comparison (`improvement < minimum`) restored,
`pytest -q tests/test_publish.py -k equal_to_the_threshold`:

```text
>       assert tie.published is False
E       AssertionError: assert True is False
E        +  where True = PromotionDecision(published=True, reason='published', effective_score=0.75, ...).published
1 failed
```

`pytest -q tests/test_cli.py -k marginal` (CLI default threshold was `0.0`)

```text
E       assert 0 == 1        # a +0.0078 override published
```

### GREEN

- `DEFAULT_MINIMUM_MODEL_IMPROVEMENT = 0.02`, used by `publish_bundle` and the
  `--minimum-model-improvement` CLI default (documented in `--help` and README).
- Promotion now requires `improvement > minimum` (strict), so ties and noise-sized gains never
  fork the prompt.

## Preserved guarantees (re-verified)

- Hard safety failures still zero the run score, invalidate the candidate, and make `evaluate`
  exit `1`; unsafe candidates zero the whole GEPA batch.
- Systemic bridge failures (timeout, non-zero exit, missing/malformed output, protocol,
  fingerprint, identity, systemic status) still abort instead of scoring, and still abort
  publication.
- Executed model failures remain valid scored results (score `0.0`, `accepted=True`) and now
  correctly count as pass^k failures.
- Reflection records still carry only case identity, answer, checkpoint names, tool-call counts,
  outcome, missing checkpoints, and hard failures.
- AKS access stays read-only, loopback-only, with exact process and kubeconfig cleanup.

## Commands and results

```text
.venv/bin/python -m pytest -q                 -> 168 passed
.venv/bin/ruff check .                        -> All checks passed!
.venv/bin/mypy src tests                      -> Success: no issues found in 22 source files

uv run --python 3.12 korvid-prompt-lab validate \
  --candidate examples/candidates/shipped-small.yaml \
  --campaign examples/campaigns/local-smoke.yaml
-> validated candidate=shipped-small fingerprint=05386c2e...335d7 campaign=local-smoke cases=2 models=mock-small  (exit 0)

uv run --python 3.12 korvid-prompt-lab evaluate ... \
  --train-case-id smoke-happy --validation-case-id smoke-guardrail \
  --milestone-case-id smoke-happy --milestone-case-id smoke-guardrail
-> evaluated candidate=shipped-small campaign=local-smoke aggregate=0.850 pass^3=1.000 pass^5=1.000 ...  (exit 0)
   summary: repetitions_per_case=5, milestone_passed=True,
   case_sets={'train': ['smoke-happy'], 'validation': ['smoke-guardrail'],
              'milestone': ['smoke-happy', 'smoke-guardrail']},
   request runtime: {'model_endpoint': None, 'repetitions': 5, ...}

uv run --python 3.12 korvid-prompt-lab evaluate ...                      (no split flags)
-> evaluation failed: --train-case-id is required and must name at least one case  (exit 2)

uv run --python 3.12 korvid-prompt-lab evaluate ... --train-case-id smoke-happy --validation-case-id smoke-happy
-> evaluation failed: train and validation case sets must be disjoint: smoke-happy  (exit 2)

uv run --python 3.12 korvid-prompt-lab optimize ...                      (no split flags)
-> optimization failed: --train-case-id is required and must name at least one case  (exit 2)

uv run --python 3.12 korvid-prompt-lab publish ... --registry-root registry
-> published version=pb-d70e85e59e9333cf kind=common effective_score=0.850  (exit 0)

# repetitions: 1 campaign
-> evaluated ... pass^3=insufficient-evidence pass^5=insufficient-evidence  (exit 0)
-> publish failed: pass_at_3 requires 3 recorded repetitions per case before publication  (exit 1)
```

Scratch smoke artifacts (`artifacts/`, `registry/`, scratch campaign and model metadata, and the
`uv.lock` produced by `uv run`) were removed after verification; `uv.lock` was deliberately not
committed because it pins internal package-feed URLs.

## Files changed (commit `9541177`)

```text
README.md                                  | 112 +++-
examples/campaigns/aks-shared-runners.yaml |  13 +-
examples/campaigns/local-smoke.yaml        |   9 +-
src/korvid_prompt_lab/adapter.py           |   7 +-
src/korvid_prompt_lab/cli.py               | 219 +++++--
src/korvid_prompt_lab/config.py            |  20 +-
src/korvid_prompt_lab/contracts.py         |   1 +
src/korvid_prompt_lab/optimize.py          |  29 +-
src/korvid_prompt_lab/publish.py           |  46 +-
src/korvid_prompt_lab/runner.py            |  32 +-
src/korvid_prompt_lab/scoring.py           |  41 +-
tests/fixtures/fake_korvid_bridge.py       |  35 +-
tests/test_adapter.py                      |  39 ++
tests/test_aks.py                          |   1 +
tests/test_cli.py                          | 914 ++++++++++++++++++++++++++---
tests/test_contracts.py                    | 107 +++-
tests/test_optimize.py                     |  94 +++
tests/test_publish.py                      | 178 +++++-
tests/test_runner.py                       |  99 +++-
tests/test_scoring.py                      |  69 ++-
20 files changed, 1864 insertions(+), 201 deletions(-)
```

## Concerns and follow-ups

1. **Contract changes are breaking for existing operators.** `aks_port_forward` campaigns now
   require `serving.command`; every bridge command must contain `{request}` and `{response}`;
   `evaluate`/`optimize` require explicit train/validation flags; publishable campaigns need
   `repetitions >= 5`; publishable campaigns need at least two cases so a disjoint split exists.
   Existing campaign files and automation must be updated. The schema version stayed `1` because
   the request payload only gained a field and no released consumer exists yet — a bump is worth
   discussing if any external bridge is already deployed.
2. **`runtime.model_endpoint` is an additive protocol field at `protocol_version: 1`.** Bridges
   that strictly reject unknown request keys would break; the bundled fake bridge and the
   documented schema were updated, but a real Korvid bridge must be re-checked before live use.
3. **AKS wiring is proven only against a fake port-forward and a fake bridge.** The real
   `rg-pension-guard` / `aks-shared-runners` path still needs one supervised live run
   (`aks-check`, then a short `evaluate`) to confirm the endpoint contract with the real Korvid
   bridge; nothing in this wave touched the cluster.
4. **pass^k is strict by design.** A single flaky repetition now drops a case group to 0, so
   published scoreboards will look worse than before. That is intended, but it changes how the
   existing registry rows compare with future ones (already-published bundles were computed with
   the old any-of-k definition).
5. **`0.02` is a judgement call.** The default minimum model improvement is documented and
   overridable; it is not derived from measured run-to-run variance. Once real multi-repetition
   AKS data exists, re-derive it from the observed noise band.
6. **Milestone semantics are partly derived.** `--milestone-case-id` is optional; when omitted the
   milestone pack is still derived from full-pack execution. Making it mandatory would be
   stricter but would break more existing flows, so it was left optional and gated
   (`milestone_passed` requires the recorded pack to equal the required pack).
7. **Example bridge command for AKS is a placeholder.** `korvid-bridge` is not installed by this
   repository; operators must point `serving.command` at their reviewed executable.
