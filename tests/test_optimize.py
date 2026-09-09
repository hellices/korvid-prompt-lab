from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import yaml  # type: ignore[import-untyped]
from current_helpers import (
    IMPROVED_TEXT,
    FakeRunner,
    candidate,
    case,
    fake_gepa_result,
)

from korvid_prompt_lab.contracts import Candidate, EvalCase
from korvid_prompt_lab.experiment_budget import ExperimentBudget
from korvid_prompt_lab.optimize import OptimizationArtifacts, optimize_campaign
from korvid_prompt_lab.scoring import EvaluationResult


def source_runner(
    train: list[EvalCase],
    validation: list[EvalCase],
    holdout: list[EvalCase] | None = None,
    **kwargs: Any,
) -> FakeRunner:
    holdout = holdout or []
    return FakeRunner(
        [*train, *validation, *holdout],
        splits=(
            ("train", tuple(item.case_id for item in train)),
            ("validation", tuple(item.case_id for item in validation)),
            *(
                (("holdout", tuple(item.case_id for item in holdout)),)
                if holdout
                else ()
            ),
        ),
        **kwargs,
    )


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_optimize_requires_budget_and_exactly_one_proposal_source(
    tmp_path: Path,
) -> None:
    train, validation = [case("train")], [case("validation")]
    runner = source_runner(train, validation)
    required = {
        "runner": cast(Any, runner),
        "seed_candidate": candidate(),
        "train_cases": train,
        "validation_cases": validation,
        "artifact_root": tmp_path,
        "max_metric_calls": 4,
    }

    with pytest.raises(TypeError):
        optimize_campaign(  # type: ignore[call-arg]
            **required,
            candidate_proposer=lambda current, *_args: dict(current),
        )
    with pytest.raises(ValueError, match="exactly one"):
        optimize_campaign(
            **required,
            budget=ExperimentBudget(10, 1, 30.0),
        )
    with pytest.raises(ValueError, match="exactly one"):
        optimize_campaign(
            **required,
            reflection_lm=object(),
            candidate_proposer=lambda current, *_args: dict(current),
            budget=ExperimentBudget(10, 1, 30.0),
        )


def test_rejected_later_proposal_preserves_a_fully_evaluated_winner(tmp_path: Path) -> None:
    train = [case("train-a"), case("train-b")]
    validation = [case("validation-a"), case("validation-b")]
    original = candidate()
    runner = source_runner(
        train, validation,
        success=lambda proposed, selected: (
            proposed.fingerprint != original.fingerprint and selected.case_id.endswith("-a")
        ),
    )
    proposals = iter([
        {"tier_pack": "Use the original evidence carefully."},
        {"tier_pack": ""},
    ])
    result = optimize_campaign(
        runner=runner, seed_candidate=original, train_cases=train, validation_cases=validation,
        artifact_root=tmp_path, max_metric_calls=40,
        candidate_proposer=lambda *_args: next(proposals),
        budget=ExperimentBudget(100, 4, 30),
    )
    assert result.best_candidate.fingerprint != original.fingerprint
    assert result.result.val_aggregate_scores[result.result.best_idx] == 0.5
    summary = read_json(result.summary_path)
    assert summary["invalid_proposals"] == 1
    assert summary["stop_reason"] == "proposal_rejected"


def test_optimize_passes_current_contract_to_gepa_and_persists_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, validation = [case("train")], [case("validation", journey=True)]
    runner = source_runner(train, validation)
    seed = candidate()
    captured: dict[str, Any] = {}

    def fake_optimize(**kwargs: object) -> Any:
        captured.update(kwargs)
        return fake_gepa_result(
            cast(str, kwargs["run_dir"]),
            [seed.components, {"tier_pack": IMPROVED_TEXT}],
            scores=[0.0, 1.0],
        )

    monkeypatch.setattr("korvid_prompt_lab.optimize.gepa.optimize", fake_optimize)
    artifacts = optimize_campaign(
        runner=cast(Any, runner),
        seed_candidate=seed,
        train_cases=train,
        validation_cases=validation,
        artifact_root=tmp_path,
        max_metric_calls=8,
        seed=7,
        candidate_proposer=lambda *_args: {"tier_pack": IMPROVED_TEXT},
        budget=ExperimentBudget(20, 2, 30.0),
    )

    assert isinstance(artifacts, OptimizationArtifacts)
    assert captured["seed_candidate"] == {"tier_pack": seed.components["tier_pack"]}
    assert captured["trainset"] == train
    assert captured["valset"] == validation
    assert captured["seed"] == 7
    assert captured["raise_on_exception"] is True
    assert captured["reflection_minibatch_size"] == 3
    persisted = yaml.safe_load(
        artifacts.best_candidate_path.read_text(encoding="utf-8")
    )
    assert persisted["components"] == {"tier_pack": IMPROVED_TEXT}
    summary = read_json(artifacts.summary_path)
    assert summary["best_candidate_fingerprint"] == artifacts.best_candidate.fingerprint
    assert summary["seed_candidate_fingerprint"] == seed.fingerprint
    assert summary["best_candidate_differs_from_seed"] is True
    assert summary["train_case_ids"] == ["train"]
    assert summary["validation_case_ids"] == ["validation"]


@pytest.mark.parametrize(
    ("splits", "selected_train", "message"),
    [
        ((), "train", "declared train and validation"),
        (
            (
                ("train", ("train",)),
                ("validation", ("validation",)),
                ("holdout", ("holdout",)),
            ),
            "holdout",
            "never holdout",
        ),
    ],
)
def test_optimize_enforces_declared_source_splits_and_never_uses_holdout(
    tmp_path: Path,
    splits: tuple[tuple[str, tuple[str, ...]], ...],
    selected_train: str,
    message: str,
) -> None:
    cases = {
        "train": case("train"),
        "validation": case("validation", journey=True),
        "holdout": case("holdout", journey=True),
    }
    runner = FakeRunner(list(cases.values()), splits=splits)

    with pytest.raises(ValueError, match=message):
        optimize_campaign(
            runner=cast(Any, runner),
            seed_candidate=candidate(),
            train_cases=[cases[selected_train]],
            validation_cases=[cases["validation"]],
            artifact_root=tmp_path,
            max_metric_calls=8,
            candidate_proposer=lambda current, *_args: dict(current),
            budget=ExperimentBudget(20, 1, 30.0),
        )


def test_optimize_preserves_source_kind_and_identity(tmp_path: Path) -> None:
    train, validation = case("train"), case("validation", journey=True)
    runner = source_runner([train], [validation])
    wrong_kind = EvalCase(
        case_id=train.case_id,
        template_id="korvid-journey",
        prompt=train.prompt,
        models=train.models,
    )

    with pytest.raises(ValueError, match="source identity"):
        optimize_campaign(
            runner=cast(Any, runner),
            seed_candidate=candidate(),
            train_cases=[wrong_kind],
            validation_cases=[validation],
            artifact_root=tmp_path,
            max_metric_calls=8,
            candidate_proposer=lambda current, *_args: dict(current),
            budget=ExperimentBudget(20, 1, 30.0),
        )


def test_optimize_refuses_to_reuse_source_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, validation = [case("train")], [case("validation")]
    runner = source_runner(train, validation)
    seed = candidate()

    monkeypatch.setattr(
        "korvid_prompt_lab.optimize.gepa.optimize",
        lambda **kwargs: fake_gepa_result(
            cast(str, kwargs["run_dir"]),
            [seed.components],
        ),
    )

    def run() -> OptimizationArtifacts:
        return optimize_campaign(
            runner=cast(Any, runner),
            seed_candidate=seed,
            train_cases=train,
            validation_cases=validation,
            artifact_root=tmp_path,
            max_metric_calls=8,
            seed=3,
            candidate_proposer=lambda current, *_args: dict(current),
            budget=ExperimentBudget(20, 1, 30.0),
        )

    first = run()
    with pytest.raises(ValueError, match="already exists|never resumes"):
        run()

    assert first.summary_path.is_file()
    assert (
        read_json(first.invocation_dir / "run-identity.json")["run_id"] == first.run_id
    )


def test_optimizer_does_not_double_count_a_runner_owned_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, validation = [case("train")], [case("validation")]
    budget = ExperimentBudget(5, 1, 30.0)
    delegate = source_runner(train, validation)

    class BudgetOwningRunner:
        campaign = delegate.campaign
        evaluation_budget = budget
        calls = 0

        def run(
            self,
            candidate_value: Candidate,
            case_value: EvalCase,
            run_dir: Path | str,
            *,
            repetition: int = 1,
            seed: int = 0,
        ) -> EvaluationResult:
            self.evaluation_budget.consume_evaluation()
            self.calls += 1
            return delegate.run(
                candidate_value,
                case_value,
                run_dir,
                repetition=repetition,
                seed=seed,
            )

    runner = BudgetOwningRunner()

    def fake_optimize(**kwargs: object) -> Any:
        cast(Any, kwargs["adapter"]).evaluate(
            validation,
            candidate().components,
        )
        return fake_gepa_result(
            cast(str, kwargs["run_dir"]),
            [candidate().components],
        )

    monkeypatch.setattr("korvid_prompt_lab.optimize.gepa.optimize", fake_optimize)
    optimize_campaign(
        runner=cast(Any, runner),
        seed_candidate=candidate(),
        train_cases=train,
        validation_cases=validation,
        artifact_root=tmp_path,
        max_metric_calls=8,
        candidate_proposer=lambda current, *_args: dict(current),
        budget=budget,
    )

    assert runner.calls == 1
    assert budget.evaluations == 1


def test_real_gepa_selects_a_distinct_plaintext_prompt_at_the_proposal_cap(
    tmp_path: Path,
) -> None:
    train = [case(f"train-{index}") for index in range(3)]
    validation = [case(f"validation-{index}") for index in range(2)]
    budget = ExperimentBudget(40, 1, 30.0)
    runner = source_runner(
        train,
        validation,
        success=lambda value, _case: value.components["tier_pack"] == IMPROVED_TEXT,
    )

    artifacts = optimize_campaign(
        runner=cast(Any, runner),
        seed_candidate=candidate(),
        train_cases=train,
        validation_cases=validation,
        artifact_root=tmp_path,
        max_metric_calls=40,
        candidate_proposer=lambda *_args: {"tier_pack": IMPROVED_TEXT},
        budget=budget,
    )

    assert artifacts.best_candidate.components == {"tier_pack": IMPROVED_TEXT}
    assert artifacts.best_candidate.fingerprint != candidate().fingerprint
    assert budget.proposals == 1


def test_real_gepa_preserves_the_winner_when_an_atomic_iteration_will_not_fit(
    tmp_path: Path,
) -> None:
    train = [case(f"train-{index}") for index in range(3)]
    validation = [case(f"validation-{index}") for index in range(2)]
    budget = ExperimentBudget(40, 10, 30.0)
    delegate = source_runner(
        train,
        validation,
        success=lambda value, _case: value.components["tier_pack"] == IMPROVED_TEXT,
    )

    class SearchLimitedRunner:
        campaign = delegate.campaign
        evaluation_budget = budget

        def __init__(self) -> None:
            self.remaining_search_calls = 10
            self.calls = 0

        def run(
            self,
            candidate_value: Candidate,
            case_value: EvalCase,
            run_dir: Path | str,
            *,
            repetition: int = 1,
            seed: int = 0,
        ) -> EvaluationResult:
            if self.remaining_search_calls <= 0:
                raise AssertionError("GEPA started an iteration that could not finish")
            self.remaining_search_calls -= 1
            self.evaluation_budget.consume_evaluation()
            self.calls += 1
            return delegate.run(
                candidate_value,
                case_value,
                run_dir,
                repetition=repetition,
                seed=seed,
            )

    runner = SearchLimitedRunner()
    artifacts = optimize_campaign(
        runner=cast(Any, runner),
        seed_candidate=candidate(),
        train_cases=train,
        validation_cases=validation,
        artifact_root=tmp_path,
        max_metric_calls=40,
        candidate_proposer=lambda *_args: {"tier_pack": IMPROVED_TEXT},
        budget=budget,
    )

    assert runner.calls == 10
    assert runner.remaining_search_calls == 0
    assert artifacts.best_candidate.components == {"tier_pack": IMPROVED_TEXT}
