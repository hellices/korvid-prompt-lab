from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import pytest
from current_helpers import FakeRunner, candidate, case, feedback

from korvid_prompt_lab.adapter import KorvidGEPAAdapter, SafeExecutionTrace
from korvid_prompt_lab.runner import BridgeExecutionModeError
from korvid_prompt_lab.scoring import EvaluationResult


@pytest.mark.parametrize(("success", "expected"), [(False, 0.0), (True, 1.0)])
def test_adapter_fitness_is_exactly_the_original_source_verdict(
    tmp_path: Path,
    success: bool,
    expected: float,
) -> None:
    source_case = case("image-pull-typo")
    runner = FakeRunner([source_case], success=success)
    adapter = KorvidGEPAAdapter(cast(Any, runner), tmp_path)

    batch = adapter.evaluate(
        [source_case],
        candidate().components,
        capture_traces=True,
    )

    assert batch.scores == [expected]
    assert batch.outputs[0].success is success
    assert batch.outputs[0].feedback["success"] is success
    assert batch.trajectories is not None
    assert batch.trajectories[0].score == expected
    assert {field.name for field in fields(SafeExecutionTrace)} == {
        "case_id",
        "template_id",
        "model",
        "execution_mode",
        "score",
        "feedback",
    }


def test_adapter_rejects_a_result_that_disagrees_with_the_source_verdict(
    tmp_path: Path,
) -> None:
    source_case = case("image-pull-typo")

    def inconsistent(
        candidate_value: Any,
        case_value: Any,
    ) -> EvaluationResult:
        return EvaluationResult(
            success=True,
            execution_mode="scripted",
            candidate_fingerprint=candidate_value.fingerprint,
            feedback=feedback(case_value, success=False),
            usage={},
        )

    adapter = KorvidGEPAAdapter(
        cast(Any, FakeRunner([source_case], result_factory=inconsistent)),
        tmp_path,
    )

    with pytest.raises(ValueError, match="original verdict"):
        adapter.evaluate([source_case], candidate().components)


@pytest.mark.parametrize(
    ("arguments_raw", "error_label"),
    [
        ("{not-json", "malformed_arguments"),
        ("[]", "non_object_arguments"),
        ("123", "arguments_not_string"),
    ],
)
def test_adapter_preserves_bounded_malformed_model_call_failures(
    tmp_path: Path,
    arguments_raw: str,
    error_label: str,
) -> None:
    source_case = case("image-pull-typo")
    malformed_call = {
        "name": "diagnose_pod",
        "arguments_valid": False,
        "arguments_raw": arguments_raw,
        "error_label": error_label,
    }

    def result(candidate_value: Any, case_value: Any) -> EvaluationResult:
        return EvaluationResult(
            success=False,
            execution_mode="scripted",
            candidate_fingerprint=candidate_value.fingerprint,
            feedback=feedback(
                case_value,
                success=False,
                calls=[malformed_call],
            ),
            usage={"tool_calls": 1},
        )

    adapter = KorvidGEPAAdapter(
        cast(Any, FakeRunner([source_case], result_factory=result)),
        tmp_path,
    )
    batch = adapter.evaluate(
        [source_case],
        candidate().components,
        capture_traces=True,
    )
    record = adapter.make_reflective_dataset(
        candidate().components,
        batch,
        ["tier_pack"],
    )["tier_pack"][0]

    assert batch.scores == [0.0]
    assert record["Generated Outputs"]["calls"] == [malformed_call]
    assert record["Feedback"]["success"] is False


@pytest.mark.parametrize(
    "calls",
    [
        [{"name": "diagnose_pod", "arguments_valid": False}],
        [
            {
                "name": "diagnose_pod",
                "arguments_valid": False,
                "arguments_raw": "x" * 513,
                "error_label": "malformed_arguments",
            }
        ],
        [
            {
                "name": "diagnose_pod",
                "arguments_valid": False,
                "arguments_raw": "{}",
                "error_label": "unknown",
            }
        ],
    ],
)
def test_adapter_rejects_malformed_model_call_evidence(
    tmp_path: Path,
    calls: list[dict[str, Any]],
) -> None:
    source_case = case("image-pull-typo")

    def result(candidate_value: Any, case_value: Any) -> EvaluationResult:
        return EvaluationResult(
            success=False,
            execution_mode="scripted",
            candidate_fingerprint=candidate_value.fingerprint,
            feedback=feedback(case_value, success=False, calls=calls),
            usage={},
        )

    adapter = KorvidGEPAAdapter(
        cast(Any, FakeRunner([source_case], result_factory=result)),
        tmp_path,
    )

    with pytest.raises(ValueError, match="compact tool calls"):
        adapter.evaluate([source_case], candidate().components)


def test_reflection_is_bounded_and_does_not_copy_the_source_report(
    tmp_path: Path,
) -> None:
    source_case = case("tui-follow", journey=True)
    raw_turns = [
        {
            "outcome": "failure",
            "failure_class": "misdiagnosis",
            "grade": {
                "diagnosis_success": False,
                "missing_mentions": ["x" * 600],
                "private_metric": "must not escape",
            },
            "answer": "SECRET RAW ANSWER",
        }
    ]
    raw_calls = [
        {
            "name": "open_describe",
            "arguments": {
                "pod": "web-1",
                "token": "SECRET",
                "nested": {"raw": "must not escape"},
            },
            "result": "SECRET TOOL RESULT",
        }
    ]

    def result(candidate_value: Any, case_value: Any) -> EvaluationResult:
        return EvaluationResult(
            success=False,
            execution_mode="scripted",
            candidate_fingerprint=candidate_value.fingerprint,
            feedback=feedback(
                case_value,
                success=False,
                calls=raw_calls,
                turns=raw_turns,
                extra={"source_report": {"secret": "ORIGINAL REPORT"}},
            ),
            usage={},
        )

    adapter = KorvidGEPAAdapter(
        cast(Any, FakeRunner([source_case], result_factory=result)),
        tmp_path,
    )
    components = candidate().components
    batch = adapter.evaluate([source_case], components, capture_traces=True)
    record = adapter.make_reflective_dataset(components, batch, ["tier_pack"])[
        "tier_pack"
    ][0]
    rendered = json.dumps(record, sort_keys=True)

    assert record["Inputs"]["reference"] == "journeys/tui-follow"
    assert record["Inputs"]["runtime_policy"]["source"] == "low-korvid-operator"
    assert record["Generated Outputs"]["turns"][0]["grade"] == {
        "diagnosis_success": False,
        "missing_mentions": ["x" * 500],
    }
    assert record["Generated Outputs"]["calls"] == [
        {
            "name": "open_describe",
            "arguments": {"pod": "web-1", "token": "[redacted]"},
        }
    ]
    for secret in (
        "SECRET RAW ANSWER",
        "SECRET TOOL RESULT",
        "ORIGINAL REPORT",
        "private_metric",
    ):
        assert secret not in rendered


@pytest.mark.parametrize(
    "feedback_override",
    [
        {},
        {"success": False},
        {
            "success": False,
            "reference": "scenarios/image-pull-typo",
            "source_sha256": "not-a-digest",
            "questions": ["Why?"],
            "turns": [{"outcome": "failure", "grade": {}}],
            "calls": [],
        },
    ],
)
def test_adapter_fails_closed_on_incomplete_source_feedback(
    tmp_path: Path,
    feedback_override: dict[str, Any],
) -> None:
    source_case = case("image-pull-typo")

    def result(candidate_value: Any, case_value: Any) -> EvaluationResult:
        return EvaluationResult(
            success=False,
            execution_mode="scripted",
            candidate_fingerprint=candidate_value.fingerprint,
            feedback=feedback_override,
            usage={},
        )

    adapter = KorvidGEPAAdapter(
        cast(Any, FakeRunner([source_case], result_factory=result)),
        tmp_path,
    )

    with pytest.raises(ValueError, match="upstream feedback"):
        adapter.evaluate([source_case], candidate().components)


def test_adapter_rejects_unbounded_reflection_evidence(tmp_path: Path) -> None:
    source_case = case("image-pull-typo")
    too_many_turns = [
        {"outcome": "failure", "grade": {"diagnosis_success": False}} for _ in range(65)
    ]

    def result(candidate_value: Any, case_value: Any) -> EvaluationResult:
        return EvaluationResult(
            success=False,
            execution_mode="scripted",
            candidate_fingerprint=candidate_value.fingerprint,
            feedback=feedback(case_value, success=False, turns=too_many_turns),
            usage={},
        )

    adapter = KorvidGEPAAdapter(
        cast(Any, FakeRunner([source_case], result_factory=result)),
        tmp_path,
    )

    with pytest.raises(ValueError, match="bounded reflection"):
        adapter.evaluate([source_case], candidate().components)


def test_adapter_refuses_to_mix_live_and_scripted_evidence(tmp_path: Path) -> None:
    cases = [case("one"), case("two")]
    adapter = KorvidGEPAAdapter(
        cast(Any, FakeRunner(cases, execution_modes=("live", "scripted"))),
        tmp_path,
    )

    with pytest.raises(BridgeExecutionModeError, match="must not mix"):
        adapter.evaluate(cases, candidate().components)


def test_adapter_propagates_runner_errors_unchanged(tmp_path: Path) -> None:
    source_case = case("image-pull-typo")
    failure = RuntimeError("provider unavailable")

    class FailingRunner(FakeRunner):
        def run(self, *args: Any, **kwargs: Any) -> EvaluationResult:
            raise failure

    adapter = KorvidGEPAAdapter(cast(Any, FailingRunner([source_case])), tmp_path)

    with pytest.raises(RuntimeError) as error:
        adapter.evaluate([source_case], candidate().components)

    assert error.value is failure


def test_adapter_rejects_candidate_identity_mismatch(tmp_path: Path) -> None:
    source_case = case("image-pull-typo")

    def result(candidate_value: Any, case_value: Any) -> EvaluationResult:
        return EvaluationResult(
            success=False,
            execution_mode="scripted",
            candidate_fingerprint="wrong-fingerprint",
            feedback=feedback(case_value, success=False),
            usage={},
        )

    adapter = KorvidGEPAAdapter(
        cast(Any, FakeRunner([source_case], result_factory=result)),
        tmp_path,
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        adapter.evaluate([source_case], candidate().components)
