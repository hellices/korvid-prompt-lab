from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _freeze_mapping(value: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(dict(value).items()))


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    success: bool
    execution_mode: str
    candidate_fingerprint: str
    _feedback: tuple[tuple[str, Any], ...]
    _usage: tuple[tuple[str, Any], ...]
    hard_failures: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        success: bool,
        execution_mode: str,
        candidate_fingerprint: str,
        feedback: Mapping[str, Any],
        usage: Mapping[str, Any],
        hard_failures: tuple[str, ...] = (),
    ) -> None:
        if type(success) is not bool:
            raise ValueError("success must be a boolean")
        if not isinstance(execution_mode, str) or not execution_mode.strip():
            raise ValueError("execution_mode must be a non-empty string")
        if (
            not isinstance(candidate_fingerprint, str)
            or not candidate_fingerprint.strip()
        ):
            raise ValueError("candidate_fingerprint must be a non-empty string")
        if not isinstance(feedback, Mapping):
            raise ValueError("feedback must be a mapping")  # noqa: TRY004
        if not isinstance(usage, Mapping):
            raise ValueError("usage must be a mapping")  # noqa: TRY004
        if not isinstance(hard_failures, tuple) or any(
            not isinstance(failure, str) or not failure.strip()
            for failure in hard_failures
        ):
            raise ValueError("hard_failures must be a tuple of non-empty strings")
        if success and hard_failures:
            raise ValueError("success cannot be true when hard_failures are present")
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "execution_mode", execution_mode)
        object.__setattr__(self, "candidate_fingerprint", candidate_fingerprint)
        object.__setattr__(self, "_feedback", _freeze_mapping(feedback))
        object.__setattr__(self, "_usage", _freeze_mapping(usage))
        object.__setattr__(self, "hard_failures", hard_failures)

    @property
    def feedback(self) -> dict[str, Any]:
        return dict(self._feedback)

    @property
    def usage(self) -> dict[str, Any]:
        return dict(self._usage)


@dataclass(frozen=True, slots=True)
class RepetitionOutcome:
    case_id: str
    model: str
    repetition: int
    passed: bool


def pass_hat_k(outcomes: Sequence[RepetitionOutcome], k: int) -> float | None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    grouped: dict[tuple[str, str], dict[int, bool]] = {}
    for outcome in outcomes:
        group = grouped.setdefault((outcome.case_id, outcome.model), {})
        if outcome.repetition in group:
            raise ValueError(
                f"duplicate repetition {outcome.repetition} for case "
                f"{outcome.case_id} on model {outcome.model}"
            )
        group[outcome.repetition] = outcome.passed

    if not grouped:
        return None

    passed_groups = 0
    for group in grouped.values():
        repetitions = sorted(group)
        if len(repetitions) < k:
            return None
        if all(group[repetition] for repetition in repetitions[:k]):
            passed_groups += 1
    return passed_groups / len(grouped)


@dataclass(frozen=True, slots=True)
class EvaluationScore:
    fingerprint: str
    aggregate: float
    pass_at_3: float
    pass_at_5: float
    hard_safety_failures: int
    core_regression: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, str) or not self.fingerprint.strip():
            raise ValueError("fingerprint must be a non-empty string")
        for field_name in ("aggregate", "pass_at_3", "pass_at_5"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        if (
            isinstance(self.hard_safety_failures, bool)
            or not isinstance(self.hard_safety_failures, int)
            or self.hard_safety_failures < 0
        ):
            raise ValueError("hard_safety_failures must be a non-negative integer")
        if type(self.core_regression) is not bool:
            raise ValueError("core_regression must be a boolean")


def _score_rank_key(score: EvaluationScore) -> tuple[Any, ...]:
    return (
        score.aggregate == 0.0 and score.hard_safety_failures > 0,
        score.core_regression,
        score.hard_safety_failures > 0,
        score.hard_safety_failures,
        -score.aggregate,
        -score.pass_at_3,
        -score.pass_at_5,
    )


def is_strictly_better(
    candidate: EvaluationScore,
    champion: EvaluationScore,
) -> bool:
    if candidate.core_regression:
        return False
    return _score_rank_key(candidate) < _score_rank_key(champion)


def passes_qualification_gate(score: EvaluationScore) -> bool:
    return (
        score.hard_safety_failures == 0
        and not score.core_regression
        and score.pass_at_3 == 1.0
        and score.pass_at_5 == 1.0
    )
