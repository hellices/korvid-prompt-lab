# AGENTS.md

Operational boundaries for agents. Schema, runbook, and budget details are in
README.md. The user-agreed role takes precedence over any implementation description.

## Role (non-negotiable)

- Import the actual Korvid prompt and verify baseline composition fidelity.
  Optimize the original operating text through the source-supported `tier_pack`
  replacement. Use original scenario outcomes / journey success, not a new oracle
  or additive-rules-only substitute.
- Korvid owns: prompt composition, runtime, tools, safety (immutable), common
  context (immutable), dynamic context (Korvid assembler), scenarios, fixtures,
  journey turn order, grading. Lab owns: acquisition, candidate generation,
  experiment orchestration, result comparison, artifacts.
- Preserve original IDs, questions, fixtures, assertions, and whole journeys.
  Select original references; never invent variants, translate cases, or split
  dependent turns into independent examples. Lab fixtures test plumbing only.
- `upstream.py` exports `inspect_upstream`, `KorvidUpstreamRunner`, and
  `export_upstream_prompt`. `upstream_contract.py` holds source-reference and
  prompt-pack metadata helpers. `source_runtime.py` is the subprocess launcher
  (`validate_source`, `run_upstream_request`). `upstream_worker.py` invokes the
  original Korvid loaders/runners/grader in the source environment.
  CLI: `uv run --extra dev korvid-prompt-lab run --experiment <file> [--check-only]`.
- `export_upstream_prompt()` writes files only; it does not run inference.
  `prompt_override_verified` is set by the separately budgeted pipeline step
  (one original train case run with `prompt_path` reload). `product_application_verified`
  is always false. `QUALIFIED` is an evaluation gate, never automatic product deployment.
- Missing upstream API or product application hook: report the limitation and
  request user approval where needed. Do not change the goal to fit a convenient API.

## Provenance checks before any live run

- Pinned SHA `33c483e041006eb20259a024ed85a9323e52c8f0`; verify with
  `git -C /path/to/korvid-native rev-parse HEAD` and confirm clean worktree.
- `NO_GRIND`: verify copied tier_pack equals the live resolved low pack before
  spending model budget.
- Record Korvid revision, source paths, and file hashes in artifacts.
  Test counts do not prove these contracts.
- Read README.md before acting. Flag implementation/role mismatches; neither
  an implementation nor an assistant-written README constitutes user approval.

## Boundaries

- Korvid runtime/tool/tier-pack/benchmark changes require separate product review
  and user approval. Do not change scoring or fixtures to make a candidate pass.
- Do not reproduce Korvid with DSPy `ReAct`. DSPy and the teacher propose text;
  standalone `gepa.optimize` searches using original Korvid outcomes.
- `prompt_improved` and `qualified` reset to false on cleanup/system failure.
  Never relax score/pass criteria or fabricate evidence. Do not invent test
  counts or completion status.
- No automatic registry publication. No overwriting a user's Korvid config.

## Verification commands

```
uv run --extra dev python -m pytest -q
uv run --extra dev mypy src tests
uv run --extra dev ruff check src tests

KORVID_NATIVE_SOURCE_ROOT=/path/to/korvid-native \
  uv run --extra dev python -m pytest \
  tests/test_upstream_contract.py tests/test_upstream_application.py tests/test_upstream_runner.py \
  tests/test_experiment_http.py -q

KORVID_NATIVE_SOURCE_ROOT=/path/to/korvid-native \
  PYTHONPATH="$PWD/src:/path/to/korvid-native/src" \
  /path/to/korvid-native/.venv/bin/python -m pytest \
  tests/test_upstream_worker.py -q

MYPYPATH="$PWD/src/korvid_prompt_lab:/path/to/korvid-native/src" \
  uv run --extra dev mypy --explicit-package-bases \
  --python-executable /path/to/korvid-native/.venv/bin/python \
  src/korvid_prompt_lab/upstream_worker.py tests/test_upstream_worker.py
```

Always pass `--extra dev` to `uv run`. Tests verify transport integration only;
they are not model-quality evidence. Claim checks passed only after running them.
