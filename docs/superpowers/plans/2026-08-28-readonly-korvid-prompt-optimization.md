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
- [ ] Run full tests, Ruff, mypy, lock/diff checks, independent review, and
  merge.

### Task 5: Live Canary

- [ ] Execute baseline and one bounded optimization round on the AKS model.
- [ ] Verify candidate count, before/after score, and failure movement.
- [ ] Continue only if the search signal is non-flat.
- [ ] Preserve evidence and restore AKS capacity.
