from __future__ import annotations

import hashlib
import json
import shutil
import sys
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab import stable_rollover
from korvid_prompt_lab.contracts import Candidate
from korvid_prompt_lab.stable_rollover import (
    PriorCampaignEvidence,
    PriorFinalistEvidence,
    load_prior_campaign_evidence,
)
from korvid_prompt_lab.stable_scenarios import (
    RolloverScenarioManifest,
    ScenarioAssignment,
    ScenarioClass,
    ScenarioManifest,
    ScenarioSplitSummary,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def scratch() -> Iterator[Path]:
    directory = ROOT / "artifacts" / f"stable-rollover-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _candidate_entry(
    *,
    candidate_id: str,
    append: str,
    axes: list[str] | None = None,
    fingerprint: str | None = None,
) -> tuple[dict[str, Any], str]:
    candidate = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "components": {
            "system": "installed",
            "append": append,
        },
        "metadata": {"korvid_version": "0.3.0", "profile": "small"},
    }
    computed = Candidate.from_mapping(candidate).fingerprint
    payload = {
        "axes": axes if axes is not None else candidate_id.split("+"),
        "candidate": {
            **candidate,
            "candidate_fingerprint": computed if fingerprint is None else fingerprint,
        },
    }
    return payload, computed


def _qualification_entry(
    *,
    candidate_id: str,
    candidate_validation: float,
    baseline_validation: float,
    candidate_milestone: float,
    baseline_milestone: float,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "candidate_validation": {"mean_score": candidate_validation},
        "baseline_validation": {"mean_score": baseline_validation},
        "candidate_milestone": {"mean_score": candidate_milestone},
        "baseline_milestone": {"mean_score": baseline_milestone},
    }


def _build_prior_root(root: Path) -> dict[str, Any]:
    summary = {
        "schema_version": 1,
        "campaign_id": "stable-search-korvid-small",
        "decision": {"status": "no_stable_winner", "candidate_id": None},
    }
    scenario_manifest = {
        "korvid_version": "0.3.0",
        "assignments": [
            {
                "scenario_id": "used-a",
                "scenario_class": "workload-health",
                "split": "train",
                "question_sha256": "a" * 64,
                "fixture_sha256": "b" * 64,
                "korvid_version": "0.3.0",
            }
        ],
        "train": ["used-a"],
        "validation": [],
        "milestone": [],
        "split_summaries": [],
    }
    alpha_entry, alpha_fingerprint = _candidate_entry(
        candidate_id="alpha",
        append="gather the highest-value evidence before stating a conclusion.",
        axes=["alpha"],
    )
    finalist_entry, finalist_fingerprint = _candidate_entry(
        candidate_id="cite-before-conclusion+stop-with-uncertainty",
        append="name observed evidence. stop if evidence is insufficient.",
        axes=["cite-before-conclusion", "stop-with-uncertainty"],
    )
    beta_entry, beta_fingerprint = _candidate_entry(
        candidate_id="beta",
        append="inspect pod status before any final conclusion.",
        axes=["beta"],
    )
    candidate_manifest = {
        "schema_version": 1,
        "candidates": [alpha_entry, finalist_entry, beta_entry],
    }
    qualification = {
        "schema_version": 1,
        "stage": "qualification",
        "candidates": [
            _qualification_entry(
                candidate_id="beta",
                candidate_validation=0.35,
                baseline_validation=0.33166666666666667,
                candidate_milestone=0.39,
                baseline_milestone=0.4033333333333333,
            ),
            _qualification_entry(
                candidate_id="cite-before-conclusion+stop-with-uncertainty",
                candidate_validation=0.3333333333333333,
                baseline_validation=0.33166666666666667,
                candidate_milestone=0.2866666666666667,
                baseline_milestone=0.4033333333333333,
            ),
            _qualification_entry(
                candidate_id="alpha",
                candidate_validation=0.34,
                baseline_validation=0.33166666666666667,
                candidate_milestone=0.28,
                baseline_milestone=0.4033333333333333,
            ),
        ],
        "decision": {"status": "no_stable_winner", "candidate_id": None},
    }
    summary_path = _write_json(root / "stable-search-summary.json", summary)
    scenario_path = _write_json(root / "scenario-manifest.json", scenario_manifest)
    _write_json(root / "candidate-manifest.json", candidate_manifest)
    _write_json(root / "stage-c" / "qualification-summary.json", qualification)
    return {
        "summary_path": summary_path,
        "scenario_path": scenario_path,
        "expected_assignments": (
            ScenarioAssignment(
                scenario_id="used-a",
                scenario_class=ScenarioClass.WORKLOAD_HEALTH,
                split="train",
                question_sha256="a" * 64,
                fixture_sha256="b" * 64,
                korvid_version="0.3.0",
            ),
        ),
        "winner": {
            "candidate_id": "beta",
            "candidate_fingerprint": beta_fingerprint,
            "append": "inspect pod status before any final conclusion.",
            "validation_delta": 0.35 - 0.33166666666666667,
            "milestone_delta": 0.39 - 0.4033333333333333,
        },
        "fingerprints": {
            "alpha": alpha_fingerprint,
            "beta": beta_fingerprint,
            "finalist": finalist_fingerprint,
        },
    }


def _rollover_manifest() -> RolloverScenarioManifest:
    assignments = (
        ScenarioAssignment(
            scenario_id="dev-train-b",
            scenario_class=ScenarioClass.IMAGE_CONFIG,
            split="train",
            question_sha256="9" * 64,
            fixture_sha256="9" * 64,
            korvid_version="0.3.0",
        ),
        ScenarioAssignment(
            scenario_id="dev-train-a",
            scenario_class=ScenarioClass.WORKLOAD_HEALTH,
            split="train",
            question_sha256="8" * 64,
            fixture_sha256="8" * 64,
            korvid_version="0.3.0",
        ),
        ScenarioAssignment(
            scenario_id="dev-validation-a",
            scenario_class=ScenarioClass.NETWORKING,
            split="validation",
            question_sha256="7" * 64,
            fixture_sha256="7" * 64,
            korvid_version="0.3.0",
        ),
        ScenarioAssignment(
            scenario_id="fresh-milestone-b",
            scenario_class=ScenarioClass.STORAGE,
            split="milestone",
            question_sha256="6" * 64,
            fixture_sha256="e" * 64,
            korvid_version="0.3.0",
        ),
        ScenarioAssignment(
            scenario_id="fresh-milestone-a",
            scenario_class=ScenarioClass.SCHEDULING_RESOURCES,
            split="milestone",
            question_sha256="5" * 64,
            fixture_sha256="b" * 64,
            korvid_version="0.3.0",
        ),
    )
    manifest = ScenarioManifest(
        korvid_version="0.3.0",
        assignments=assignments,
        train=("dev-train-b", "dev-train-a"),
        validation=("dev-validation-a",),
        milestone=("fresh-milestone-b", "fresh-milestone-a"),
        split_summaries=(
            ScenarioSplitSummary(
                split_name="train",
                classes=(ScenarioClass.IMAGE_CONFIG, ScenarioClass.WORKLOAD_HEALTH),
                scenario_ids=("dev-train-b", "dev-train-a"),
            ),
            ScenarioSplitSummary(
                split_name="validation",
                classes=(ScenarioClass.NETWORKING,),
                scenario_ids=("dev-validation-a",),
            ),
            ScenarioSplitSummary(
                split_name="milestone",
                classes=(ScenarioClass.STORAGE, ScenarioClass.SCHEDULING_RESOURCES),
                scenario_ids=("fresh-milestone-b", "fresh-milestone-a"),
            ),
        ),
    )
    return RolloverScenarioManifest(
        manifest=manifest,
        consumed_ids=("dev-train-a", "dev-train-b", "dev-validation-a"),
        fresh_milestone_ids=("fresh-milestone-b", "fresh-milestone-a"),
        audit_reserve_ids=("fresh-audit",),
    )


def _prior_evidence(root: Path) -> PriorCampaignEvidence:
    return PriorCampaignEvidence(
        artifact_root=root.resolve(),
        campaign_id="stable-search-korvid-small",
        korvid_version="0.3.0",
        summary_sha256="c" * 64,
        scenario_manifest_sha256="d" * 64,
        consumed_assignments=(
            ScenarioAssignment(
                scenario_id="used-b",
                scenario_class=ScenarioClass.IMAGE_CONFIG,
                split="validation",
                question_sha256="2" * 64,
                fixture_sha256="f" * 64,
                korvid_version="0.3.0",
            ),
            ScenarioAssignment(
                scenario_id="used-a",
                scenario_class=ScenarioClass.WORKLOAD_HEALTH,
                split="train",
                question_sha256="1" * 64,
                fixture_sha256="a" * 64,
                korvid_version="0.3.0",
            ),
        ),
        finalist=PriorFinalistEvidence(
            candidate_id="cite-before-conclusion+stop-with-uncertainty",
            candidate_fingerprint="e" * 64,
            append="name the observed evidence and its source before the final conclusion.",
            validation_delta=0.12,
            milestone_delta=-0.08,
        ),
    )


def test_load_prior_campaign_evidence_reads_confined_prior_artifacts(
    scratch: Path,
) -> None:
    expected = _build_prior_root(scratch / "prior-root")

    evidence = load_prior_campaign_evidence(scratch / "prior-root")

    assert evidence.artifact_root == (scratch / "prior-root").resolve()
    assert evidence.campaign_id == "stable-search-korvid-small"
    assert evidence.korvid_version == "0.3.0"
    assert evidence.summary_sha256 == hashlib.sha256(expected["summary_path"].read_bytes()).hexdigest()
    assert evidence.scenario_manifest_sha256 == hashlib.sha256(
        expected["scenario_path"].read_bytes()
    ).hexdigest()
    assert evidence.consumed_assignments == expected["expected_assignments"]
    assert evidence.finalist.candidate_id == expected["winner"]["candidate_id"]
    assert evidence.finalist.candidate_fingerprint == expected["winner"]["candidate_fingerprint"]
    assert evidence.finalist.append == expected["winner"]["append"]
    assert evidence.finalist.validation_delta == pytest.approx(expected["winner"]["validation_delta"])
    assert evidence.finalist.milestone_delta == pytest.approx(expected["winner"]["milestone_delta"])


def test_load_prior_campaign_evidence_breaks_finalist_ties_by_candidate_id(
    scratch: Path,
) -> None:
    _build_prior_root(scratch / "prior-root")
    alpha_entry, alpha_fingerprint = _candidate_entry(
        candidate_id="alpha",
        append="collect one more read-only signal before concluding.",
        axes=["alpha"],
    )
    beta_entry, _ = _candidate_entry(
        candidate_id="beta",
        append="collect one more read-only signal before concluding, then stop.",
        axes=["beta"],
    )
    _write_json(
        scratch / "prior-root" / "candidate-manifest.json",
        {"schema_version": 1, "candidates": [beta_entry, alpha_entry]},
    )
    _write_json(
        scratch / "prior-root" / "stage-c" / "qualification-summary.json",
        {
            "schema_version": 1,
            "stage": "qualification",
            "candidates": [
                _qualification_entry(
                    candidate_id="beta",
                    candidate_validation=0.45,
                    baseline_validation=0.40,
                    candidate_milestone=0.55,
                    baseline_milestone=0.50,
                ),
                _qualification_entry(
                    candidate_id="alpha",
                    candidate_validation=0.45,
                    baseline_validation=0.40,
                    candidate_milestone=0.55,
                    baseline_milestone=0.50,
                ),
            ],
            "decision": {"status": "no_stable_winner", "candidate_id": None},
        },
    )

    evidence = load_prior_campaign_evidence(scratch / "prior-root")

    assert evidence.finalist.candidate_id == "alpha"
    assert evidence.finalist.candidate_fingerprint == alpha_fingerprint


def _replace_json(path: Path, replacer: Callable[[Any], Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _write_json(path, replacer(payload))


def _replace_with_internal_symlink(root: Path) -> None:
    path = root / "stage-c" / "qualification-summary.json"
    real_path = root / "stage-c" / "qualification-summary-real.json"
    original = path.read_text(encoding="utf-8")
    path.unlink()
    real_path.write_text(original, encoding="utf-8")
    path.symlink_to(real_path)


def _replace_with_external_symlink(root: Path) -> None:
    path = root / "stage-c" / "qualification-summary.json"
    outside = root.parent / "outside-qualification.json"
    original = path.read_text(encoding="utf-8")
    path.unlink()
    outside.write_text(original, encoding="utf-8")
    path.symlink_to(outside)


@pytest.mark.parametrize(
    ("case_name", "mutate", "match"),
    [
        (
            "existing symlink in any required path",
            _replace_with_internal_symlink,
            "symlink",
        ),
        (
            "required path escaping prior root",
            _replace_with_external_symlink,
            "escape",
        ),
        (
            "missing required file",
            lambda root: (root / "candidate-manifest.json").unlink(),
            "candidate-manifest.json",
        ),
        (
            "non-object JSON root",
            lambda root: (root / "stable-search-summary.json").write_text("[]\n", encoding="utf-8"),
            "stable-search-summary.json",
        ),
        (
            "schema_version other than 1",
            lambda root: _replace_json(
                root / "stable-search-summary.json",
                lambda payload: {**payload, "schema_version": 2},
            ),
            "schema_version",
        ),
        (
            "decision other than no_stable_winner",
            lambda root: _replace_json(
                root / "stage-c" / "qualification-summary.json",
                lambda payload: {**payload, "decision": {"status": "promote", "candidate_id": "beta"}},
            ),
            "no_stable_winner",
        ),
        (
            "NaN or Infinity in a score",
            lambda root: _replace_json(
                root / "stage-c" / "qualification-summary.json",
                lambda payload: {
                    **payload,
                    "candidates": [
                        {
                            **payload["candidates"][0],
                            "candidate_validation": {"mean_score": float("nan")},
                        },
                        *payload["candidates"][1:],
                    ],
                },
            ),
            "finite",
        ),
        (
            "candidate ID missing from candidate-manifest",
            lambda root: _replace_json(
                root / "candidate-manifest.json",
                lambda payload: {
                    **payload,
                    "candidates": [
                        entry
                        for entry in payload["candidates"]
                        if entry["candidate"]["candidate_id"] != "beta"
                    ],
                },
            ),
            "candidate manifest",
        ),
        (
            "candidate fingerprint mismatch",
            lambda root: _replace_json(
                root / "candidate-manifest.json",
                lambda payload: {
                    **payload,
                    "candidates": [
                        {
                            **entry,
                            "candidate": {
                                **entry["candidate"],
                                "candidate_fingerprint": "d" * 64,
                            },
                        }
                        if entry["candidate"]["candidate_id"] == "beta"
                        else entry
                        for entry in payload["candidates"]
                    ],
                },
            ),
            "fingerprint",
        ),
        (
            "split membership inconsistent with assignments",
            lambda root: _replace_json(
                root / "scenario-manifest.json",
                lambda payload: {**payload, "validation": ["used-a"]},
            ),
            "split membership",
        ),
    ],
)
def test_load_prior_campaign_evidence_rejects_invalid_prior_roots(
    scratch: Path,
    case_name: str,
    mutate: Callable[[Path], None],
    match: str,
) -> None:
    root = scratch / case_name.replace(" ", "-")
    _build_prior_root(root)
    mutate(root)

    with pytest.raises(ValueError, match=match):
        load_prior_campaign_evidence(root)


def test_write_rollover_lineage_serializes_only_bounded_digests(
    scratch: Path,
) -> None:
    prior_root = scratch / "prior-root"
    prior_root.mkdir()
    evidence = _prior_evidence(prior_root)
    rollover = _rollover_manifest()
    path = scratch / "rollover-lineage.json"

    assert hasattr(stable_rollover, "write_rollover_lineage")
    written = stable_rollover.write_rollover_lineage(
        path,
        evidence,
        rollover,
        terminal_reason="qualification_complete",
    )

    assert written == path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "prior": {
            "campaign_id": "stable-search-korvid-small",
            "decision": "no_stable_winner",
            "stable_search_summary_sha256": "c" * 64,
            "scenario_manifest_sha256": "d" * 64,
            "finalist_id": "cite-before-conclusion+stop-with-uncertainty",
            "finalist_fingerprint": "e" * 64,
        },
        "scenario_consumption": {
            "korvid_version": "0.3.0",
            "consumed": ["a" * 64, "f" * 64],
            "fresh_milestone": ["b" * 64, "e" * 64],
            "counts": {
                "train": 2,
                "validation": 1,
                "milestone": 2,
                "audit_reserve": 1,
            },
        },
        "candidate_matrix_version": "rollover-v1",
        "max_target_calls": 306,
        "terminal_reason": "qualification_complete",
    }
    text = path.read_text(encoding="utf-8")
    assert str(prior_root.resolve()) not in text
    assert "used-a" not in text
    assert "fresh-milestone-a" not in text
    for forbidden in ("question", "fixture_state", "endpoint", "raw_answer", "raw_error"):
        assert forbidden not in text


def test_write_rollover_winner_writes_exact_candidate_yaml_and_rejects_collisions(
    scratch: Path,
) -> None:
    candidate = Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "decisive-read-first",
            "components": {
                "system": "Stay safe.",
                "append": "inspect runtime evidence before stating a diagnosis.",
            },
            "metadata": {
                "profile": "small",
                "rollover_from": "c" * 64,
            },
        }
    )
    path = scratch / "winner.yaml"
    existing_path = scratch / "existing-winner.yaml"
    existing_path.write_text("already here\n", encoding="utf-8")
    symlink_path = scratch / "winner-link.yaml"
    symlink_path.symlink_to(existing_path)

    assert hasattr(stable_rollover, "write_rollover_winner")
    written = stable_rollover.write_rollover_winner(path, candidate)

    assert written == path
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "candidate_id": "decisive-read-first",
        "components": {
            "system": "Stay safe.",
            "append": "inspect runtime evidence before stating a diagnosis.",
        },
        "metadata": {
            "profile": "small",
            "rollover_from": "c" * 64,
        },
    }
    with pytest.raises(FileExistsError, match="already exists"):
        stable_rollover.write_rollover_winner(existing_path, candidate)
    with pytest.raises(FileExistsError, match="symlink"):
        stable_rollover.write_rollover_winner(symlink_path, candidate)


@pytest.mark.parametrize(
    "components",
    [
        {"system": "Stay safe."},
        {
            "system": "Stay safe.",
            "append": "inspect runtime evidence before stating a diagnosis.",
            "tool.kubectl": "kubectl get pods",
        },
    ],
)
def test_write_rollover_winner_rejects_candidates_without_exact_system_and_append_components(
    scratch: Path,
    components: dict[str, str],
) -> None:
    candidate = Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "bad-winner",
            "components": components,
            "metadata": {"profile": "small"},
        }
    )

    assert hasattr(stable_rollover, "write_rollover_winner")
    with pytest.raises(ValueError, match="exactly system and append"):
        stable_rollover.write_rollover_winner(scratch / "bad-winner.yaml", candidate)
