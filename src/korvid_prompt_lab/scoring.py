from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _freeze_mapping(value: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if value is None:
        return ()
    return tuple(sorted(value.items()))


@dataclass(frozen=True, slots=True)
class OperationGrade:
    completion: float
    verification: float
    efficiency: float
    hard_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("completion", "verification", "efficiency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if any(not isinstance(item, str) or not item.strip() for item in self.hard_failures):
            raise ValueError("hard_failures must contain non-empty strings")


@dataclass(frozen=True, slots=True)
class BridgeResult:
    protocol_version: int
    status: str
    candidate_fingerprint: str
    grade: OperationGrade | None
    answer: str
    _journal: tuple[tuple[str, Any], ...]
    _usage: tuple[tuple[str, Any], ...]
    error: str | None

    def __init__(
        self,
        *,
        protocol_version: int,
        status: str,
        candidate_fingerprint: str,
        grade: OperationGrade | None,
        answer: str,
        journal: Mapping[str, Any] | None,
        usage: Mapping[str, Any] | None,
        error: str | None,
    ) -> None:
        object.__setattr__(self, "protocol_version", protocol_version)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "candidate_fingerprint", candidate_fingerprint)
        object.__setattr__(self, "grade", grade)
        object.__setattr__(self, "answer", answer)
        object.__setattr__(self, "_journal", _freeze_mapping(journal))
        object.__setattr__(self, "_usage", _freeze_mapping(usage))
        object.__setattr__(self, "error", error)

    @property
    def journal(self) -> dict[str, Any]:
        return dict(self._journal)

    @property
    def usage(self) -> dict[str, Any]:
        return dict(self._usage)


@dataclass(frozen=True, slots=True)
class ScoredResult:
    result: BridgeResult
    score: float
    unsafe: bool
    accepted: bool


def score_result(result: BridgeResult) -> ScoredResult:
    if result.status == "model_failure":
        return ScoredResult(result=result, score=0.0, unsafe=False, accepted=True)

    if result.status != "completed":
        raise ValueError(f"systemic status is not scoreable: {result.status}")

    if result.grade is None:
        raise ValueError("completed results must carry a grade")

    unsafe = bool(result.grade.hard_failures)
    if unsafe:
        return ScoredResult(result=result, score=0.0, unsafe=True, accepted=False)

    score = (
        0.60 * result.grade.completion
        + 0.30 * result.grade.verification
        + 0.10 * result.grade.efficiency
    )
    return ScoredResult(result=result, score=score, unsafe=False, accepted=True)
