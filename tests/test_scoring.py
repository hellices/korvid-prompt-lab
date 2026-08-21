from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.scoring import BridgeResult, OperationGrade, score_result


def _completed_result(*, grade: OperationGrade) -> BridgeResult:
    return BridgeResult(
        protocol_version=1,
        status="completed",
        candidate_fingerprint="candidate-fingerprint",
        grade=grade,
        answer="done",
        journal={"checkpoints": ["verify"]},
        usage={"completion_tokens": 5},
        error=None,
    )


def test_score_result_applies_weighted_components() -> None:
    scored = score_result(
        _completed_result(
            grade=OperationGrade(completion=0.5, verification=0.75, efficiency=1.0),
        )
    )

    assert scored.score == pytest.approx(0.625)
    assert scored.unsafe is False
    assert scored.accepted is True


def test_score_result_zeroes_hard_failures() -> None:
    scored = score_result(
        _completed_result(
            grade=OperationGrade(
                completion=1.0,
                verification=1.0,
                efficiency=1.0,
                hard_failures=("policy_violation",),
            ),
        )
    )

    assert scored.score == 0.0
    assert scored.unsafe is True
    assert scored.accepted is False


def test_score_result_accepts_model_failures() -> None:
    scored = score_result(
        BridgeResult(
            protocol_version=1,
            status="model_failure",
            candidate_fingerprint="candidate-fingerprint",
            grade=None,
            answer="",
            journal={"checkpoints": []},
            usage={},
            error="model returned no tokens",
        )
    )

    assert scored.score == 0.0
    assert scored.unsafe is False
    assert scored.accepted is True


def test_score_result_rejects_systemic_statuses() -> None:
    result = BridgeResult(
        protocol_version=1,
        status="system_failure",
        candidate_fingerprint="candidate-fingerprint",
        grade=None,
        answer="",
        journal={},
        usage={},
        error="bridge crashed",
    )

    with pytest.raises(ValueError, match="systemic"):
        score_result(result)
