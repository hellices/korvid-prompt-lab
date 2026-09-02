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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.contracts import Candidate
from korvid_prompt_lab.stable_rollover import load_prior_campaign_evidence
from korvid_prompt_lab.stable_scenarios import ScenarioAssignment, ScenarioClass

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
