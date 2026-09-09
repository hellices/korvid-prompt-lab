from __future__ import annotations

import math
import time
from collections.abc import Callable


class BudgetExhausted(RuntimeError):
    def __init__(self, stop_reason: str) -> None:
        self.stop_reason = stop_reason
        self.reason = stop_reason
        super().__init__(f"experiment budget exhausted: {stop_reason}")


class ExperimentBudget:
    def __init__(
        self,
        max_evaluations: int,
        max_proposals: int,
        wall_clock_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(max_evaluations, bool)
            or not isinstance(max_evaluations, int)
            or max_evaluations < 0
        ):
            raise ValueError("max_evaluations must be a non-negative integer")
        if (
            isinstance(max_proposals, bool)
            or not isinstance(max_proposals, int)
            or max_proposals < 0
        ):
            raise ValueError("max_proposals must be a non-negative integer")
        if (
            isinstance(wall_clock_seconds, bool)
            or not isinstance(wall_clock_seconds, (int, float))
            or not math.isfinite(float(wall_clock_seconds))
            or wall_clock_seconds < 0
        ):
            raise ValueError("wall_clock_seconds must be finite and non-negative")
        if not callable(clock):
            raise ValueError("clock must be callable")  # noqa: TRY004 - one validation API

        self.max_evaluations = max_evaluations
        self.max_proposals = max_proposals
        self.wall_clock_seconds = float(wall_clock_seconds)
        self._clock = clock
        self._started_at = clock()
        self._evaluations = 0
        self._proposals = 0
        self._stop_reason: str | None = None

    @property
    def evaluations(self) -> int:
        return self._evaluations

    @property
    def proposals(self) -> int:
        return self._proposals

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.wall_clock_seconds - self.elapsed_seconds)

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    def check(self) -> None:
        if self.elapsed_seconds >= self.wall_clock_seconds:
            self._exhaust("wall_clock")

    def consume_evaluation(self) -> None:
        self.check()
        if self._evaluations >= self.max_evaluations:
            self._exhaust("max_evaluations")
        self._evaluations += 1

    def consume_proposal(self) -> None:
        self.check()
        if self._proposals >= self.max_proposals:
            self._exhaust("max_proposals")
        self._proposals += 1

    def _exhaust(self, stop_reason: str) -> None:
        self._stop_reason = stop_reason
        raise BudgetExhausted(stop_reason)
