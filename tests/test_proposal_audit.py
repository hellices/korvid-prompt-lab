from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from current_helpers import FakeRunner, candidate, case, fake_gepa_result

from korvid_prompt_lab.experiment_budget import ExperimentBudget
from korvid_prompt_lab.optimize import optimize_campaign
from korvid_prompt_lab.reflection import (
    AuditedProposalSource,
    DuplicateProposal,
    ProposalProviderError,
    ProposalRejected,
)


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def audited_source(
    tmp_path: Path,
    proposer: Any,
    *,
    budget: ExperimentBudget | None = None,
) -> AuditedProposalSource:
    return AuditedProposalSource(
        proposer=proposer,
        source="candidate_proposer",
        invocation_dir=tmp_path,
        seed_candidate=candidate(),
        budget=budget or ExperimentBudget(0, 10, 30.0),
    )


def test_distinct_and_duplicate_tier_pack_proposals_are_audited(
    tmp_path: Path,
) -> None:
    source = audited_source(
        tmp_path,
        lambda *_args: {"tier_pack": "Use decisive source evidence."},
    )
    records = {"tier_pack": [{"Feedback": {"success": False}}]}

    proposal = source(candidate().components, records, ["tier_pack"])
    with pytest.raises(DuplicateProposal):
        source(candidate().components, records, ["tier_pack"])

    assert proposal == {"tier_pack": "Use decisive source evidence."}
    assert source.summary() == {
        "proposal_attempts": 2,
        "distinct_proposals": 1,
        "duplicate_proposals": 1,
        "invalid_proposals": 0,
        "provider_errors": 0,
        "error_labels": {},
        "candidate_fingerprints": [
            read_json(tmp_path / "proposals/0001.json")[
                "proposed_candidate_fingerprint"
            ]
        ],
        "proposal_record_paths": ["proposals/0001.json", "proposals/0002.json"],
    }
    assert read_json(tmp_path / "proposals/0001.json")["status"] == "valid"
    assert read_json(tmp_path / "proposals/0002.json")["status"] == "duplicate"
    assert read_json(tmp_path / "proposal-audit.json")["duplicate_proposals"] == 1


@pytest.mark.parametrize(
    ("proposal", "label"),
    [
        (None, "malformed_proposal"),
        ({"system": "legacy component"}, "malformed_proposal"),
        ({"tier_pack": "   "}, "blank_proposal"),
        ({"tier_pack": 1}, "malformed_proposal"),
    ],
)
def test_invalid_proposals_are_audited_and_never_returned(
    tmp_path: Path,
    proposal: object,
    label: str,
) -> None:
    source = audited_source(tmp_path, lambda *_args: proposal)

    with pytest.raises(ProposalRejected) as error:
        source(candidate().components, {"tier_pack": []}, ["tier_pack"])

    assert error.value.error_label == label
    record = read_json(tmp_path / "proposals/0001.json")
    assert record["status"] == "invalid"
    assert record["error_label"] == label
    assert source.summary()["invalid_proposals"] == 1


def test_candidate_fingerprints_preserve_exact_tier_pack_text(tmp_path: Path) -> None:
    proposals = iter(
        [
            {"tier_pack": "Keep exact punctuation."},
            {"tier_pack": "Keep exact punctuation.\n"},
        ]
    )
    source = audited_source(tmp_path, lambda *_args: next(proposals))

    first = source(candidate().components, {"tier_pack": []}, ["tier_pack"])
    second = source(candidate().components, {"tier_pack": []}, ["tier_pack"])

    records = [
        read_json(tmp_path / "proposals/0001.json"),
        read_json(tmp_path / "proposals/0002.json"),
    ]
    assert first != second
    assert records[0]["status"] == records[1]["status"] == "valid"
    assert (
        records[0]["proposed_candidate_fingerprint"]
        != records[1]["proposed_candidate_fingerprint"]
    )


@pytest.mark.parametrize(
    "failure",
    [ProposalProviderError("provider_timeout"), RuntimeError("programming bug")],
)
def test_proposal_failures_are_persisted_and_propagated_unchanged(
    tmp_path: Path,
    failure: Exception,
) -> None:
    def fail(*_args: object) -> dict[str, str]:
        raise failure

    source = audited_source(tmp_path, fail)

    with pytest.raises(type(failure)) as error:
        source(candidate().components, {"tier_pack": []}, ["tier_pack"])

    assert error.value is failure
    record = read_json(tmp_path / "proposals/0001.json")
    assert record["status"] == "provider_error"
    assert "programming bug" not in json.dumps(record)


def test_optimizer_always_emits_current_proposal_audit_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, validation = case("train"), case("validation")
    runner = FakeRunner(
        [train, validation],
        splits=(("train", ("train",)), ("validation", ("validation",))),
    )

    def fake_optimize(**kwargs: object) -> Any:
        return fake_gepa_result(
            cast(str, kwargs["run_dir"]),
            [candidate().components],
        )

    monkeypatch.setattr("korvid_prompt_lab.optimize.gepa.optimize", fake_optimize)
    artifacts = optimize_campaign(
        runner=cast(Any, runner),
        seed_candidate=candidate(),
        train_cases=[train],
        validation_cases=[validation],
        artifact_root=tmp_path,
        max_metric_calls=4,
        candidate_proposer=lambda current, *_args: dict(current),
        budget=ExperimentBudget(10, 1, 30.0),
    )

    summary = read_json(artifacts.summary_path)
    assert {
        "proposal_attempts",
        "distinct_proposals",
        "duplicate_proposals",
        "invalid_proposals",
        "provider_errors",
        "error_labels",
        "candidate_fingerprints",
        "proposal_record_paths",
    } <= summary.keys()
    assert (artifacts.invocation_dir / "proposal-audit.json").is_file()


def test_run_context_changes_only_the_current_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, validation = case("train"), case("validation")
    runner = FakeRunner(
        [train, validation],
        splits=(("train", ("train",)), ("validation", ("validation",))),
    )

    monkeypatch.setattr(
        "korvid_prompt_lab.optimize.gepa.optimize",
        lambda **kwargs: fake_gepa_result(
            cast(str, kwargs["run_dir"]),
            [candidate().components],
        ),
    )
    common = {
        "runner": cast(Any, runner),
        "seed_candidate": candidate(),
        "train_cases": [train],
        "validation_cases": [validation],
        "max_metric_calls": 4,
        "candidate_proposer": lambda current, *_args: dict(current),
        "budget": ExperimentBudget(10, 1, 30.0),
    }
    first = optimize_campaign(
        **common,
        artifact_root=tmp_path / "one",
        run_context={"profile": "small", "manifest_fingerprint": "1" * 64},
    )
    common["budget"] = ExperimentBudget(10, 1, 30.0)
    second = optimize_campaign(
        **common,
        artifact_root=tmp_path / "two",
        run_context={"profile": "large", "manifest_fingerprint": "1" * 64},
    )

    assert first.run_id != second.run_id
    assert read_json(first.invocation_dir / "run-identity.json")["run_context"] == {
        "profile": "small",
        "manifest_fingerprint": "1" * 64,
    }


def test_run_context_rejects_compatibility_fields_before_creating_artifacts(
    tmp_path: Path,
) -> None:
    train, validation = case("train"), case("validation")
    runner = FakeRunner(
        [train, validation],
        splits=(("train", ("train",)), ("validation", ("validation",))),
    )

    with pytest.raises(ValueError, match="unknown.*exact_summary"):
        optimize_campaign(
            runner=cast(Any, runner),
            seed_candidate=candidate(),
            train_cases=[train],
            validation_cases=[validation],
            artifact_root=tmp_path,
            max_metric_calls=4,
            candidate_proposer=lambda current, *_args: dict(current),
            budget=ExperimentBudget(10, 1, 30.0),
            run_context={"exact_summary": True},
        )

    assert not (tmp_path / "invocations").exists()
