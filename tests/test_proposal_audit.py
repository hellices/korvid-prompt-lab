from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from current_helpers import (
    MODEL,
    FakeRunner,
    candidate,
    case,
    fake_gepa_result,
    serving,
)

from korvid_prompt_lab.contracts import Candidate, KorvidUpstreamServing
from korvid_prompt_lab.experiment_budget import BudgetExhausted, ExperimentBudget
from korvid_prompt_lab.optimize import optimize_campaign
from korvid_prompt_lab.reflection import (
    AuditedProposalSource,
    DuplicateProposal,
    ProposalProviderError,
    ProposalRejected,
)
from korvid_prompt_lab.runner import BridgeInvocationError


@pytest.fixture(autouse=True)
def stub_source_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "korvid_prompt_lab.reflection.validate_upstream_candidate",
        lambda *_args: None,
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
        serving=serving(),
        model=MODEL,
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


def test_source_validation_is_deadline_bounded_and_does_not_consume_evaluations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    budget = ExperimentBudget(0, 1, 5.0, clock=lambda: clock[0])
    proposed = candidate("Keep exact text.\n")
    validated: list[Candidate] = []

    def propose(*_args: Any) -> dict[str, str]:
        clock[0] = 2.0
        return proposed.components

    def validate(
        bounded: KorvidUpstreamServing, model: str, value: Candidate,
    ) -> None:
        assert bounded.timeout_seconds == 3.0
        assert model == MODEL
        assert budget.proposals == 1
        assert budget.evaluations == 0
        validated.append(value)

    monkeypatch.setattr("korvid_prompt_lab.reflection.validate_upstream_candidate", validate)
    source = audited_source(tmp_path, propose, budget=budget)
    assert source(candidate().components, {}, ["tier_pack"]) == proposed.components
    assert validated[0].fingerprint == proposed.fingerprint
    assert budget.evaluations == 0
    assert source.summary()["distinct_proposals"] == 1


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("broken source configuration"),
        RuntimeError("programming error"),
        BridgeInvocationError("worker timeout"),
        ProposalProviderError("provider_timeout"),
    ],
)
def test_composition_validation_failures_remain_fatal_and_are_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception,
) -> None:
    def fail(*_args: Any) -> None:
        raise failure

    monkeypatch.setattr("korvid_prompt_lab.reflection.validate_upstream_candidate", fail)
    budget = ExperimentBudget(0, 1, 30)
    source = audited_source(tmp_path, lambda *_args: {"tier_pack": "Revised."}, budget=budget)
    with pytest.raises(type(failure)) as error:
        source(candidate().components, {}, ["tier_pack"])
    assert error.value is failure
    assert source.pending_exception is failure
    assert source.summary()["provider_errors"] == 1
    assert source.summary()["invalid_proposals"] == 0
    assert budget.proposals == 1
    assert budget.evaluations == 0


@pytest.mark.parametrize("verdict", [None, "static_prompt_too_large", "timeout"])
def test_composition_deadline_expiry_is_fatal_not_a_rejected_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str | None,
) -> None:
    clock = [0.0]
    budget = ExperimentBudget(0, 1, 5, clock=lambda: clock[0])

    def validate(*_args: Any) -> str | None:
        clock[0] = 5.0
        if verdict == "timeout":
            raise BridgeInvocationError("worker timed out")
        return verdict

    monkeypatch.setattr("korvid_prompt_lab.reflection.validate_upstream_candidate", validate)
    source = audited_source(tmp_path, lambda *_args: {"tier_pack": "Revised."}, budget=budget)
    with pytest.raises(BudgetExhausted) as error:
        source(candidate().components, {}, ["tier_pack"])
    assert error.value.reason == "wall_clock"
    assert source.pending_exception is error.value
    assert source.summary()["error_labels"] == {"budget_wall_clock": 1}
    assert source.summary()["invalid_proposals"] == 0
    assert budget.proposals == 1
    assert budget.evaluations == 0


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
