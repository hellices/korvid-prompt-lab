from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.experiment_budget import BudgetExhausted, ExperimentBudget


class FakeClock:
    def __init__(self, now: float = 10.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_experiment_budget_starts_immediately_and_tracks_remaining_time() -> None:
    clock = FakeClock()
    budget = ExperimentBudget(3, 2, 8.0, clock=clock)

    assert budget.evaluations == 0
    assert budget.proposals == 0
    assert budget.elapsed_seconds == 0.0
    assert budget.remaining_seconds == 8.0
    assert budget.stop_reason is None

    clock.now += 2.5

    assert budget.elapsed_seconds == 2.5
    assert budget.remaining_seconds == 5.5
    budget.check()


def test_experiment_budget_counts_only_successfully_consumed_work() -> None:
    clock = FakeClock()
    budget = ExperimentBudget(1, 1, 8.0, clock=clock)

    budget.consume_evaluation()
    budget.consume_proposal()

    assert budget.evaluations == 1
    assert budget.proposals == 1

    with pytest.raises(BudgetExhausted) as evaluation_error:
        budget.consume_evaluation()
    assert evaluation_error.value.stop_reason == "max_evaluations"
    assert budget.evaluations == 1

    with pytest.raises(BudgetExhausted) as proposal_error:
        budget.consume_proposal()
    assert proposal_error.value.stop_reason == "max_proposals"
    assert proposal_error.value.reason == "max_proposals"
    assert budget.proposals == 1
    assert budget.stop_reason == "max_proposals"


def test_experiment_budget_checks_deadline_before_incrementing() -> None:
    clock = FakeClock()
    budget = ExperimentBudget(2, 2, 3.0, clock=clock)
    clock.now += 3.0

    with pytest.raises(BudgetExhausted) as check_error:
        budget.check()
    assert check_error.value.stop_reason == "wall_clock"
    assert budget.remaining_seconds == 0.0

    with pytest.raises(BudgetExhausted, match="wall_clock"):
        budget.consume_evaluation()
    with pytest.raises(BudgetExhausted, match="wall_clock"):
        budget.consume_proposal()

    assert budget.evaluations == 0
    assert budget.proposals == 0
    assert budget.stop_reason == "wall_clock"


def test_proposal_exhaustion_does_not_consume_or_block_evaluation_budget() -> None:
    budget = ExperimentBudget(2, 0, 8.0)

    with pytest.raises(BudgetExhausted) as error:
        budget.consume_proposal()

    assert error.value.reason == "max_proposals"
    budget.consume_evaluation()
    budget.consume_evaluation()
    assert budget.evaluations == 2


@pytest.mark.parametrize(
    ("max_evaluations", "max_proposals", "wall_clock_seconds"),
    [
        (-1, 1, 1.0),
        (1, -1, 1.0),
        (True, 1, 1.0),
        (1, False, 1.0),
        (1, 1, -0.1),
        (1, 1, math.inf),
        (1, 1, True),
    ],
)
def test_experiment_budget_rejects_invalid_limits(
    max_evaluations: object,
    max_proposals: object,
    wall_clock_seconds: object,
) -> None:
    with pytest.raises(ValueError):
        ExperimentBudget(
            max_evaluations,  # type: ignore[arg-type]
            max_proposals,  # type: ignore[arg-type]
            wall_clock_seconds,  # type: ignore[arg-type]
        )
