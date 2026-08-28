"""Deterministic, no-network contract test: a distinct prompt candidate can
score above the shipped korvid-baseline seed and become GEPA's best candidate
when evaluated through :class:`KorvidReadonlyRunner` against a fake, in-process
``python -m korvid.evals`` CLI.

This is the Task 3 proof that the read-only backend is a fully usable
alternative evidence source for the existing GEPA search/optimizer control
plane: no separate optimizer, no mocked GEPA internals, and no real model or
cluster contact anywhere in the run.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.baseline import build_baseline_candidate
from korvid_prompt_lab.config import load_candidate
from korvid_prompt_lab.contracts import Campaign, EvalCase, KorvidReadonlyServing
from korvid_prompt_lab.korvid_readonly import KorvidReadonlyRunner
from korvid_prompt_lab.optimize import optimize_campaign

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from fake_korvid_evals import (
    TUNED_MARKER,
)

FAKE_EVALS = ROOT / "tests" / "fixtures" / "fake_korvid_evals.py"

#: Real scenarios shipped with the installed Korvid wheel, split disjointly
#: into train/validation the same way a checked-in read-only campaign would.
_TRAIN_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("bad-command-crash", "The exporter pod in namespace metrics crashes instantly on every start. Why?"),
    ("crashloop-app-panic", "checkout-1 in namespace shop keeps restarting. What is wrong with it?"),
    (
        "crashloop-dependency-unreachable",
        "The billing pod in namespace shop keeps crashing shortly after start. What is causing the crashes?",
    ),
)
_VALIDATION_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("crashloop-missing-env", "The orders pod in namespace shop keeps restarting. What is wrong with it?"),
    ("healthy-deployment", "Is the checkout deployment in namespace shop healthy right now?"),
)


def _cases(scenarios: tuple[tuple[str, str], ...]) -> list[EvalCase]:
    return [
        EvalCase(case_id=case_id, template_id="readonly-template", prompt=prompt, models=("mock-small",))
        for case_id, prompt in scenarios
    ]


def _readonly_runner(cases: Sequence[EvalCase]) -> KorvidReadonlyRunner:
    campaign = Campaign(
        schema_version=1,
        campaign_id="campaign-readonly-gepa",
        repetitions=1,
        models=("mock-small",),
        cases=tuple(cases),
        serving=KorvidReadonlyServing(
            backend="korvid_readonly",
            provider="openai-compat",
            base_url="http://127.0.0.1:41001/v1",
            profile="small",
            timeout_seconds=120.0,
        ),
    )
    return KorvidReadonlyRunner(campaign)


def _tuning_proposer(proposals: list[list[str]]) -> Any:
    """A deterministic, non-LM candidate proposer: append TUNED_MARKER to every
    component GEPA asks to update. Mirrors the ``_recording_proposer`` pattern
    already used by ``tests/test_optimize.py`` and ``tests/test_adapter.py``
    for the write/approval bridge, but exercised here against the read-only
    backend so this file needs no reflection LM or network access."""

    def propose(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        proposals.append(list(components_to_update))
        return {name: f"{candidate[name]} {TUNED_MARKER}" for name in components_to_update}

    return propose


def test_readonly_backed_gepa_search_finds_a_tuned_candidate_that_beats_the_shipped_baseline(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "korvid_prompt_lab.korvid_readonly._KORVID_EVALS_COMMAND",
        (sys.executable, str(FAKE_EVALS)),
    )
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "prompt-driven")

    seed_candidate = build_baseline_candidate("small")
    train_cases = _cases(_TRAIN_SCENARIOS)
    validation_cases = _cases(_VALIDATION_SCENARIOS)
    runner = _readonly_runner(train_cases + validation_cases)
    proposals: list[list[str]] = []

    artifacts = optimize_campaign(
        runner=runner,
        seed_candidate=seed_candidate,
        train_cases=train_cases,
        validation_cases=validation_cases,
        artifact_root=tmp_path / "artifacts",
        max_metric_calls=16,
        candidate_proposer=_tuning_proposer(proposals),
    )

    assert proposals, "real GEPA must invoke the injected proposal contract"
    assert artifacts.best_candidate.components != seed_candidate.components
    assert TUNED_MARKER in "".join(artifacts.best_candidate.components.values())

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["seed_candidate_fingerprint"] == seed_candidate.fingerprint
    assert summary["best_candidate_differs_from_seed"] is True
    assert summary["train_case_ids"] == [case_id for case_id, _ in _TRAIN_SCENARIOS]
    assert summary["validation_case_ids"] == [case_id for case_id, _ in _VALIDATION_SCENARIOS]

    persisted = load_candidate(artifacts.best_candidate_path)
    assert persisted.components == artifacts.best_candidate.components
    assert not list((tmp_path / "artifacts").rglob("korvid-eval-output.json"))
    for path in (tmp_path / "artifacts").rglob("*.json"):
        assert "diagnosis complete with cited evidence" not in path.read_text(
            encoding="utf-8"
        )
