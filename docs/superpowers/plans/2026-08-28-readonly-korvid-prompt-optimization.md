# Read-Only Korvid Prompt Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect Korvid's shipped read-only scenario evals to Prompt Lab's existing evaluate and GEPA optimize flows.

**Architecture:** Add a typed campaign serving backend and runner that execute one installed Korvid scenario with private prompt override files, then normalize its JSON into the existing `BridgeResult`. Generate the seed from the installed profile so before/after measurements share one execution path.

**Tech Stack:** Python 3.12, Korvid 0.3, subprocess, pytest, GEPA/DSPy

## Global Constraints

- Read-only evals use the installed Korvid wheel, never a source checkout.
- Write/approval journeys and their SHA pin remain unchanged.
- Raw upstream output and credentials are never published.
- Usage/system/model outcomes remain distinct and fail closed.
- Existing strict publication gates remain unchanged.

---

### Task 1: Baseline Candidate

- [x] Add tests for current small/full profile materialization, metadata, stable
  fingerprint, output collision, and exact YAML.
- [x] Implement `korvid-baseline --profile --output`.
- [x] Verify against the installed Korvid wheel and commit.

### Task 2: Campaign Backend and Runner

- [x] Add `KorvidReadonlyServing` schema tests and parsing.
- [x] Add runner RED tests for exact scenario/question, private files, CLI/env,
  timeout, cleanup, concurrency, and exit classification.
- [x] Implement one-case execution and JSON normalization to `BridgeResult`.
- [x] Run runner/scoring tests and commit.

### Task 3: GEPA Integration

- [x] Generalize adapter/optimizer runner typing to the shared runner protocol.
- [x] Add reflection trace fields for read-only diagnosis/evidence/citation
  feedback without exposing raw cluster data.
- [x] Prove a deterministic fake Korvid CLI candidate can beat the baseline.
- [x] Commit after adapter/optimize tests.

### Task 4: Examples and Documentation

- [x] Add a read-only small-profile campaign with disjoint train/validation
  scenario IDs.
- [x] Document baseline, evaluate, and optimize commands.
- [x] Run full tests, Ruff, mypy, lock/diff checks, and independent review.
- [ ] Merge after the pull request is approved.

### Task 5: Live Canary

- [x] Execute the baseline and three bounded optimization rounds on the AKS
  `qwen3:0.6b` model.
- [x] Verify candidate count, before/after score, and failure movement.
- [x] Continue only while the search signal is non-flat.
- [x] Preserve safe evidence and restore AKS capacity.

Live canary result:

- The installed Korvid `small` baseline scored `0.375` on the four-case
  milestone set, with zero systemic and zero hard-safety failures.
- Two GEPA searches using `qwen3:0.6b` as the reflection model retained the
  seed (`num_candidates=1`). The generated proposals were blank or copied
  scenario-specific failure text instead of producing a general instruction.
- A manually bounded evidence-first append candidate scored `0.525` once but
  `0.300` on an identical repeat. It is therefore not a promotion candidate.
- Replacing the system prompt with a concise evidence-first prompt scored
  `0.175` and was rejected.
- A final GEPA search using `qwen3:4b` for reflection timed out while proposing
  and retained the seed (`num_candidates=1`, `best_validation_score=0.775`).
- All live evaluations kept systemic and hard-safety failure counts at zero.
  The installed-scenario execution path and non-flat scoring signal are
  verified, but no statistically stable prompt improvement is claimed.
- Raw model responses, fixture state, credentials, and kubeconfig were not
  preserved. Only fingerprints, aggregate movements, bounded search outcomes,
  and model identity were retained.
