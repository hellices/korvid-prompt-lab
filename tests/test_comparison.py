from __future__ import annotations

from dataclasses import replace

import pytest

from korvid_prompt_lab.comparison import (
    build_round_comparison,
    comparison_payload,
    render_comparison_markdown,
    render_single_evaluation_markdown,
)
from korvid_prompt_lab.rounds import CaseRunSummary, RoundReport

SEED = "a" * 64
BEST = "b" * 64


def report(
    *,
    fingerprint: str,
    aggregate: float,
    pass_at_3: float | None,
    pass_at_5: float | None,
    systemic: int,
    failures: dict[str, int],
) -> RoundReport:
    runs = (
        CaseRunSummary(
            run_id="case-a-model-a-r01",
            case_id="case-a",
            model="model-a",
            repetition=1,
            status="completed",
            completion=1.0,
            verification=1.0,
            efficiency=1.0,
            hard_failures=tuple(
                failure
                for failure, count in sorted(failures.items())
                for _ in range(count)
            ),
            execution_mode="live",
            elapsed_seconds=1.0,
        ),
    )
    return RoundReport(
        campaign_id="campaign-a",
        candidate_id="candidate-a",
        candidate_fingerprint=fingerprint,
        models=("model-a",),
        aggregate_score=aggregate,
        model_scores={"model-a": aggregate},
        pass_at_3=pass_at_3,
        pass_at_5=pass_at_5,
        systemic_failures=systemic,
        promotion_eligible=False,
        promotion_blockers=("hard_safety_failures",),
        status_counts={"completed": 1},
        hard_failure_counts=failures,
        runs=runs,
        artifact_refs=(),
        reproduction_command=(),
    )


def test_comparison_renders_semantic_directions_and_failure_union() -> None:
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=1,
        failures={"wrong_target_write": 3},
    )
    after = report(
        fingerprint=BEST,
        aggregate=0.2,
        pass_at_3=0.2,
        pass_at_5=0.1,
        systemic=0,
        failures={"write_before_fresh_read": 1, "wrong_target_write": 1},
    )

    comparison = build_round_comparison(
        before,
        after,
        seed_fingerprint=SEED,
        best_fingerprint=BEST,
    )
    markdown = render_comparison_markdown(comparison, after)

    assert comparison.outcome == "regressed"
    assert "⚠️ REGRESSED" in markdown
    assert "| Aggregate score | 0.100 | 0.200 | +0.100 | ✅ improved |" in markdown
    assert "| pass@5 | 0.300 | 0.100 | -0.200 | ⚠️ regressed |" in markdown
    assert "| Systemic failures | 1 | 0 | -1 | ✅ improved |" in markdown
    assert "| `write_before_fresh_read` | 0 | 1 | +1 | ⚠️ regressed |" in markdown
    assert "| `wrong_target_write` | 3 | 1 | -2 | ✅ improved |" in markdown


def test_comparison_rejects_different_installed_evidence_sources() -> None:
    before = replace(
        report(
            fingerprint=SEED,
            aggregate=0.1,
            pass_at_3=0.2,
            pass_at_5=0.3,
            systemic=0,
            failures={},
        ),
        evidence_sources=(
            ("case-a", "model-a", 1, "korvid_readonly", "0.3.0", "a" * 64),
        ),
    )
    after = replace(
        report(
            fingerprint=BEST,
            aggregate=0.2,
            pass_at_3=0.3,
            pass_at_5=0.4,
            systemic=0,
            failures={},
        ),
        evidence_sources=(
            ("case-a", "model-a", 1, "korvid_readonly", "0.4.0", "a" * 64),
        ),
    )

    with pytest.raises(ValueError, match="comparison contract mismatch"):
        build_round_comparison(
            before,
            after,
            seed_fingerprint=SEED,
            best_fingerprint=BEST,
        )


def test_comparison_payload_uses_v2_for_attested_evidence_sources() -> None:
    before = replace(
        report(
            fingerprint=SEED,
            aggregate=0.1,
            pass_at_3=0.2,
            pass_at_5=0.3,
            systemic=0,
            failures={},
        ),
        evidence_sources=(
            ("case-a", "model-a", 1, "korvid_readonly", "0.3.0", "a" * 64),
        ),
    )
    after = replace(before, candidate_fingerprint=BEST, aggregate_score=0.2)

    comparison = build_round_comparison(
        before,
        after,
        seed_fingerprint=SEED,
        best_fingerprint=BEST,
    )
    payload = comparison_payload(comparison)

    assert payload["schema_version"] == 2
    contract = payload["contract"]
    assert isinstance(contract, dict)
    assert contract["evidence_sources"] == [
        ["case-a", "model-a", 1, "korvid_readonly", "0.3.0", "a" * 64]
    ]


def test_same_fingerprint_is_unchanged_and_requires_same_evidence() -> None:
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=0,
        failures={},
    )

    comparison = build_round_comparison(
        before,
        before,
        seed_fingerprint=SEED,
        best_fingerprint=SEED,
    )

    assert comparison.outcome == "unchanged"
    assert "➖ UNCHANGED — optimizer retained the seed prompt" in render_comparison_markdown(comparison, before)
    assert all(metric.delta in (0, 0.0) for metric in comparison.metrics)

    with pytest.raises(ValueError, match="unchanged candidate evidence"):
        build_round_comparison(
            before,
            replace(before, aggregate_score=0.2),
            seed_fingerprint=SEED,
            best_fingerprint=SEED,
        )


def test_same_fingerprint_rejects_different_per_run_evidence() -> None:
    before = report(
        fingerprint=SEED,
        aggregate=0.5,
        pass_at_3=0.0,
        pass_at_5=0.0,
        systemic=0,
        failures={},
    )
    before_runs = (
        replace(before.runs[0], case_id="case-a", completion=1.0),
        replace(
            before.runs[0],
            run_id="case-b-model-a-r01",
            case_id="case-b",
            completion=0.0,
        ),
    )
    after_runs = (
        replace(before_runs[0], completion=0.0),
        replace(before_runs[1], completion=1.0),
    )
    before = replace(before, runs=before_runs)
    after = replace(before, runs=after_runs)

    with pytest.raises(ValueError, match="unchanged candidate evidence"):
        build_round_comparison(
            before,
            after,
            seed_fingerprint=SEED,
            best_fingerprint=SEED,
        )


def test_changed_candidate_without_core_movement_is_not_a_retained_seed() -> None:
    """A changed fingerprint with no core-metric movement classifies as
    ``unchanged`` but must never claim the optimizer retained the seed prompt."""
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=0,
        failures={"wrong_target_write": 2},
    )
    after = replace(before, candidate_fingerprint=BEST)

    comparison = build_round_comparison(
        before,
        after,
        seed_fingerprint=SEED,
        best_fingerprint=BEST,
    )

    assert comparison.status == "changed"
    assert comparison.outcome == "unchanged"

    markdown = render_comparison_markdown(comparison, after)
    assert "optimizer retained the seed prompt" not in markdown
    assert "## ➖ UNCHANGED — candidate changed; no core metric moved" in markdown
    assert f"- Prompt: `{SEED}` → `{BEST}`" in markdown


def test_retained_seed_headline_requires_matching_fingerprints() -> None:
    """The retained-seed sentence appears only when the fingerprints match."""
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=0,
        failures={},
    )

    retained = render_comparison_markdown(
        build_round_comparison(before, before, seed_fingerprint=SEED, best_fingerprint=SEED),
        before,
    )
    assert "## ➖ UNCHANGED — optimizer retained the seed prompt" in retained


def test_changed_headlines_carry_the_design_qualifiers() -> None:
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=1,
        failures={"wrong_target_write": 3},
    )
    improved = replace(before, candidate_fingerprint=BEST, aggregate_score=0.2)
    regressed = replace(before, candidate_fingerprint=BEST, aggregate_score=0.0)

    improved_md = render_comparison_markdown(
        build_round_comparison(before, improved, seed_fingerprint=SEED, best_fingerprint=BEST),
        improved,
    )
    regressed_md = render_comparison_markdown(
        build_round_comparison(before, regressed, seed_fingerprint=SEED, best_fingerprint=BEST),
        regressed,
    )

    assert "## ✅ IMPROVED — candidate changed; no core metric regressed" in improved_md
    assert "## ⚠️ REGRESSED — candidate changed; one or more core metrics regressed" in regressed_md


def test_not_comparable_metrics_are_not_counted_as_unchanged() -> None:
    """Absent optional pass rates are not comparable and must be surfaced
    separately instead of inflating the unchanged tally."""
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=None,
        pass_at_5=None,
        systemic=0,
        failures={},
    )
    after = replace(before, candidate_fingerprint=BEST, aggregate_score=0.2)

    comparison = build_round_comparison(
        before,
        after,
        seed_fingerprint=SEED,
        best_fingerprint=BEST,
    )

    # Metrics: aggregate (improved), pass@3/pass@5 (not comparable),
    # hard-safety and systemic failures (unchanged).
    assert comparison.improved_count == 1
    assert comparison.unchanged_count == 2
    assert comparison.regressed_count == 0
    assert comparison.not_comparable_count == 2

    markdown = render_comparison_markdown(comparison, after)
    assert "- Net: 1 improved, 2 unchanged, 0 regressed, 2 not comparable" in markdown


def test_net_omits_not_comparable_when_zero() -> None:
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=0,
        failures={},
    )
    after = replace(before, candidate_fingerprint=BEST, aggregate_score=0.2)

    comparison = build_round_comparison(
        before,
        after,
        seed_fingerprint=SEED,
        best_fingerprint=BEST,
    )

    assert comparison.not_comparable_count == 0
    markdown = render_comparison_markdown(comparison, after)
    assert "not comparable" not in markdown
    assert "- Net: 1 improved, 4 unchanged, 0 regressed\n" in markdown


def test_optional_pass_rate_has_no_delta_or_direction() -> None:
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=None,
        pass_at_5=None,
        systemic=0,
        failures={},
    )
    after = replace(before, candidate_fingerprint=BEST, aggregate_score=0.2)

    markdown = render_comparison_markdown(
        build_round_comparison(
            before,
            after,
            seed_fingerprint=SEED,
            best_fingerprint=BEST,
        ),
        after,
    )

    assert "| pass@3 | N/A | N/A | N/A | N/A |" in markdown
    assert "| pass@5 | N/A | N/A | N/A | N/A |" in markdown


def test_comparison_rejects_contract_mismatch_and_serializes_allowlist() -> None:
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=0,
        failures={},
    )
    after = replace(before, candidate_fingerprint=BEST, models=("other-model",))

    with pytest.raises(ValueError, match="comparison contract"):
        build_round_comparison(
            before,
            after,
            seed_fingerprint=SEED,
            best_fingerprint=BEST,
        )

    valid = build_round_comparison(
        before,
        replace(before, candidate_fingerprint=BEST, aggregate_score=0.2),
        seed_fingerprint=SEED,
        best_fingerprint=BEST,
    )
    payload = comparison_payload(valid)
    assert payload["schema_version"] == 1
    assert set(payload) == {
        "schema_version",
        "status",
        "outcome",
        "seed_candidate_fingerprint",
        "best_candidate_fingerprint",
        "contract",
        "metrics",
        "improved_count",
        "unchanged_count",
        "regressed_count",
        "not_comparable_count",
    }


def test_single_evaluation_keeps_core_metrics_above_detail() -> None:
    final = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=0,
        failures={"wrong_target_write": 2},
    )

    markdown = render_single_evaluation_markdown(final)

    assert "ℹ️ SINGLE EVALUATION — no before/after pair" in markdown
    assert "| Aggregate score | 0.100 |" in markdown
    assert "| pass@3 | 0.200 |" in markdown
    assert "| pass@5 | 0.300 |" in markdown
    assert "| Hard safety failures | 2 |" in markdown
    assert "| Systemic failures | 0 |" in markdown


def _eligible_report() -> RoundReport:
    from dataclasses import replace
    r = report(
        fingerprint=SEED,
        aggregate=0.9,
        pass_at_3=0.8,
        pass_at_5=0.7,
        systemic=0,
        failures={},
    )
    return replace(r, promotion_eligible=True, promotion_blockers=())


def test_publication_bullet_blocked_appears_after_net_in_comparison() -> None:
    before = report(
        fingerprint=SEED,
        aggregate=0.1,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=0,
        failures={},
    )
    after = report(
        fingerprint=BEST,
        aggregate=0.2,
        pass_at_3=0.2,
        pass_at_5=0.3,
        systemic=0,
        failures={},
    )
    # after has promotion_eligible=False, promotion_blockers=("hard_safety_failures",)
    comparison = build_round_comparison(before, after, seed_fingerprint=SEED, best_fingerprint=BEST)
    markdown = render_comparison_markdown(comparison, after)

    assert "- Publication: blocked (`hard_safety_failures`)" in markdown
    net_pos = markdown.index("- Net:")
    pub_pos = markdown.index("- Publication:")
    assert net_pos < pub_pos, "Publication bullet must appear after Net bullet"


def test_publication_bullet_eligible_appears_after_net_in_comparison() -> None:
    before = _eligible_report()
    after = _eligible_report()
    after = replace(after, candidate_fingerprint=BEST)
    comparison = build_round_comparison(before, after, seed_fingerprint=SEED, best_fingerprint=BEST)
    markdown = render_comparison_markdown(comparison, after)

    assert "- Publication: eligible" in markdown
    net_pos = markdown.index("- Net:")
    pub_pos = markdown.index("- Publication:")
    assert net_pos < pub_pos, "Publication bullet must appear after Net bullet"
