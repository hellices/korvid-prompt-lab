"""The native experiment lifecycle; Korvid executes and GEPA searches."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from .adapter import _slugify
from .artifacts import write_json_artifact
from .contracts import (
    GEPA_REFLECTION_MINIBATCH_SIZE,
    Campaign,
    Candidate,
    EvalCase,
    KorvidUpstreamServing,
)
from .experiment_budget import BudgetExhausted, ExperimentBudget
from .experiment_config import ExperimentSpec
from .experiment_evaluation import Assessment, assess
from .model_session import model_session
from .optimize import ProposalRejected, optimize_campaign
from .runner import BridgeInvocationError
from .scoring import EvaluationResult
from .source_runtime import validate_source
from .upstream import KorvidUpstreamRunner, export_upstream_prompt, inspect_upstream
from .upstream_contract import prompt_candidate, tier_pack_from_candidate


class SearchLimitReached(RuntimeError):
    """The hard search-call cap was reached, leaving qualification budget intact."""


class ToolPreflightInconclusive(RuntimeError):
    """A real tool event was not observed; no model-quality conclusion follows."""


class ExperimentIntegrityError(ValueError):
    """A measured execution identity changed within an immutable experiment."""


class _BudgetedRunner:
    def __init__(
        self, campaign: Campaign, budget: ExperimentBudget, search_limit: int,
        runtime_fingerprint: str, evaluation_seed: int = 0,
    ) -> None:
        self.campaign = campaign
        self.budget = budget
        self.evaluation_budget = budget
        self.search_limit = search_limit
        self.runtime_fingerprint = runtime_fingerprint
        self.evaluation_seed = evaluation_seed
        self.search_calls = 0
        self.searching = False
        self.prompt_path: Path | None = None

    @property
    def remaining_search_calls(self) -> int:
        return self.search_limit - self.search_calls

    def bounded_campaign(self) -> Campaign:
        self.budget.check()
        serving = self.campaign.serving
        if not isinstance(serving, KorvidUpstreamServing):
            raise ValueError("experiment requires the original Korvid evaluator")  # noqa: TRY004
        return replace(self.campaign, serving=replace(
            serving, timeout_seconds=min(serving.timeout_seconds, self.budget.remaining_seconds),
        ))

    def run(
        self, candidate: Candidate, case: EvalCase, run_dir: Path | str, *,
        repetition: int = 1, seed: int = 0,
    ) -> EvaluationResult:
        if self.searching and self.search_calls >= self.search_limit:
            raise SearchLimitReached("total_metric_calls")
        self.budget.consume_evaluation()
        if self.searching:
            self.search_calls += 1
            seed = self.evaluation_seed
        runner = KorvidUpstreamRunner(self.bounded_campaign(), prompt_path=self.prompt_path)
        try:
            result = runner.run(candidate, case, run_dir, repetition=repetition, seed=seed)
        except BridgeInvocationError:
            self.budget.check()
            raise
        self.budget.check()
        if result.execution_mode != "live":
            raise ValueError("the canonical experiment requires live native evidence")
        identity = json.loads((Path(run_dir) / "upstream-summary.json").read_text(encoding="utf-8"))
        if identity["runtime"]["fingerprint"] != self.runtime_fingerprint:
            raise ExperimentIntegrityError("native_runtime_changed")
        if self.prompt_path is not None and identity.get("evaluation_override_verified") is not True:
            raise ExperimentIntegrityError("exported_prompt_not_applied")
        return result


def lab_identity() -> dict[str, Any]:
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("dspy", "gepa", "litellm", "PyYAML")},
    }


@contextmanager
def _direct_loopback_requests() -> Iterator[None]:
    keys = [
        key for key in os.environ
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    ]
    saved = {key: os.environ.pop(key) for key in keys}
    os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
    try:
        yield
    finally:
        os.environ.pop("NO_PROXY", None)
        os.environ.update(saved)


def build_reflection_lm(spec: ExperimentSpec, endpoint: str, budget: ExperimentBudget, seed: int) -> object:
    budget.check()
    return dspy.LM(
        spec.reflection.reference, api_base=endpoint,
        timeout=min(spec.reflection.timeout_seconds, budget.remaining_seconds),
        num_retries=0, cache=False, seed=seed, **dict(spec.reflection.options),
    )


def _phase(summary: dict[str, Any], root: Path, name: str) -> None:
    summary["phase"] = name
    summary["phases"].append(name)
    write_json_artifact(root / "experiment-summary.json", summary)


def _split_cases(spec: ExperimentSpec, campaign: Campaign, split: str) -> tuple[EvalCase, ...]:
    by_id = {case.case_id: case for case in campaign.cases}
    return tuple(by_id[ref.split("/", 1)[1]] for ref in spec.case_splits[split])


def _tool_preflight(runner: _BudgetedRunner, baseline: Candidate, cases: tuple[EvalCase, ...], root: Path, seed: int) -> None:
    cases = cases[:3]
    for index, case in enumerate(cases):
        result = runner.run(baseline, case, root / f"probe-{index}", seed=seed + index)
        if result.usage["tool_calls"] > 0:
            return
    raise ToolPreflightInconclusive(
        "No structured tool event in the selected original train cases; investigate model behavior "
        "and transport separately. No baseline or prompt-quality score was produced."
    )


def _search(
    spec: ExperimentSpec, runner: _BudgetedRunner, baseline: Candidate,
    before: Assessment, root: Path, summary: dict[str, Any], verify: Callable[[], object],
) -> tuple[Candidate, Assessment]:
    champion, measured = baseline, before
    stagnation = 0
    train_cases = _split_cases(spec, runner.campaign, "train")
    validation_cases = _split_cases(spec, runner.campaign, "validation")
    minimum_invocation_calls = 2 * GEPA_REFLECTION_MINIBATCH_SIZE + 2 * len(validation_cases)
    for stage_index, stage in enumerate(spec.stages):
        for seed_index, seed in enumerate(stage.seeds):
            runner.budget.check()
            if runner.budget.proposals >= spec.max_proposals:
                summary["search_stop_reason"] = "max_proposals"
                return champion, measured
            if runner.remaining_search_calls < minimum_invocation_calls:
                summary["search_stop_reason"] = "insufficient_search_metric_budget"
                return champion, measured
            verify()
            attempt_root = root / "search" / f"{stage_index:02d}-{seed_index:02d}-{_slugify(stage.name)[:80]}"
            attempt: dict[str, Any] = {"stage": stage.name, "seed": seed, "status": "running"}
            summary["attempts"].append(attempt)
            count_before = runner.search_calls
            runner.searching = True
            try:
                result = optimize_campaign(
                    runner=runner, seed_candidate=champion,
                    train_cases=train_cases,
                    validation_cases=validation_cases,
                    artifact_root=attempt_root, max_metric_calls=stage.metric_calls, seed=seed,
                    reflection_lm=build_reflection_lm(spec, summary["endpoint"], runner.budget, seed),
                    budget=runner.budget,
                    run_context={
                        "manifest_fingerprint": spec.fingerprint,
                        "source": {
                            "lab": summary["lab"], "native_runtime_fingerprint": runner.runtime_fingerprint,
                            "cases": [
                                {"reference": item["reference"], "source_sha256": item["source_sha256"]}
                                for item in summary["upstream"]["cases"]
                                if item["reference"] in (*spec.case_splits["train"], *spec.case_splits["validation"])
                            ],
                            "prompt": {
                                "pack_id": summary["upstream"]["prompt"]["pack_id"],
                                "source_sha256": summary["upstream"]["prompt"]["source_sha256"],
                            },
                        },
                        "profile": spec.to_mapping()["model"],
                    },
                )
            except ProposalRejected:
                attempt["status"] = "proposal_rejected"
                stagnation += 1
            except SearchLimitReached:
                attempt["status"] = "search_budget_exhausted"
                summary["search_stop_reason"] = "total_metric_calls"
                return champion, measured
            except BudgetExhausted as exc:
                if exc.reason == "max_proposals":
                    attempt["status"] = "proposal_budget_exhausted"
                    summary["search_stop_reason"] = "max_proposals"
                    return champion, measured
                raise
            else:
                runner.searching = False
                proposed = result.best_candidate
                tier_pack_from_candidate(proposed)
                attempt["candidate_fingerprint"] = proposed.fingerprint
                if proposed.fingerprint == champion.fingerprint:
                    attempt["status"] = "unchanged"
                    stagnation += 1
                else:
                    tested = assess(
                        runner, proposed, validation_cases, attempt_root / "paired-validation",
                        seed=spec.evaluation_seed,
                    )
                    if tested.improves(measured):
                        champion, measured = proposed, tested
                        attempt["status"] = "selected"
                        stagnation = 0
                    else:
                        attempt["status"] = "not_improved"
                        stagnation += 1
            finally:
                runner.searching = False
                attempt["metric_calls"] = runner.search_calls - count_before
                write_json_artifact(attempt_root / "attempt.json", attempt)
            verify()
            if stagnation >= spec.stagnation_attempt_limit:
                summary["search_stop_reason"] = "stagnation"
                return champion, measured
    summary["search_stop_reason"] = "stages_completed"
    return champion, measured


def _coverage(root: Path, baseline: Candidate | None) -> dict[str, Any]:
    fingerprints: set[str] = set()
    invalid = duplicates = valid = provider_errors = 0
    for path in root.glob("search/*/invocations/*/proposal-audit.json"):
        audit = json.loads(path.read_text(encoding="utf-8"))
        fingerprints.update(audit["candidate_fingerprints"])
        invalid += audit["invalid_proposals"]
        duplicates += audit["duplicate_proposals"]
        valid += audit["distinct_proposals"]
        provider_errors += audit["provider_errors"]
    if baseline is not None:
        fingerprints.discard(baseline.fingerprint)
    return {
        "distinct_proposals": len(fingerprints), "invalid_proposals": invalid,
        "duplicate_proposals": duplicates + valid - len(fingerprints),
        "provider_errors": provider_errors,
        "proposed_candidate_fingerprints": sorted(fingerprints),
    }


def run_experiment(
    spec: ExperimentSpec, artifact_root: Path, *, allow_capacity_changes: bool = False,
) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    budget = ExperimentBudget(spec.max_evaluations, spec.max_proposals, spec.wall_clock_seconds)
    baseline: Candidate | None = None
    summary: dict[str, Any] = {
        "schema_version": 2, "campaign_id": spec.campaign_id, "experiment_fingerprint": spec.fingerprint,
        "status": "RUNNING", "phase": "preflight", "phases": [], "attempts": [],
        "pipeline_completed": False, "prompt_improved": False, "validation_improved": False,
        "qualified": False, "prompt_override_verified": False, "holdout_used": False,
        "product_application_verified": False, "lab": lab_identity(),
        "publication": "not_requested", "qualification_scope": "korvid-original-evaluation",
    }
    runner: _BudgetedRunner | None = None
    write_json_artifact(artifact_root / "resolved-experiment.json", spec.to_mapping())
    try:
        _phase(summary, artifact_root, "preflight")
        validate_source(Path(spec.runtime.source_root))
        snapshot = inspect_upstream(
            replace(spec.runtime, base_url="http://127.0.0.1:1"),
            spec.model.reference, spec.references, seed=spec.evaluation_seed,
        )
        if snapshot["prompt"]["baseline_equivalent"] is not True:
            raise ExperimentIntegrityError("baseline_prompt_differs_from_korvid")
        baseline = prompt_candidate(snapshot["prompt"]["text"])
        summary["baseline_fingerprint"] = baseline.fingerprint
        summary["upstream"] = snapshot
        write_json_artifact(artifact_root / "baseline-snapshot.json", snapshot)
        (artifact_root / "baseline-prompt.txt").write_text(snapshot["prompt"]["text"], encoding="utf-8")
        (artifact_root / "baseline-candidate.yaml").write_text(yaml.safe_dump({
            "schema_version": 1, "candidate_id": baseline.candidate_id,
            "components": baseline.components, "metadata": baseline.metadata,
        }, allow_unicode=True, sort_keys=False), encoding="utf-8")
        with _direct_loopback_requests(), model_session(
            spec, artifact_root / "serving", allow_capacity_changes=allow_capacity_changes,
            budget=budget,
        ) as session:
            summary["endpoint"] = session.base_url
            summary["serving"] = session.evidence
            campaign = spec.campaign(session.base_url)
            checked = inspect_upstream(
                replace(spec.runtime, base_url=session.base_url), spec.model.reference,
                spec.references, seed=spec.evaluation_seed,
            )
            if (
                checked["prompt"]["text"] != snapshot["prompt"]["text"]
                or checked["cases"] != snapshot["cases"]
                or checked["runtime"]["fingerprint"] != snapshot["runtime"]["fingerprint"]
            ):
                raise ExperimentIntegrityError("upstream_source_changed")
            summary["upstream"] = checked
            runner = _BudgetedRunner(
                campaign, budget, spec.total_metric_calls, checked["runtime"]["fingerprint"],
                spec.evaluation_seed,
            )
            train_cases = _split_cases(spec, campaign, "train")
            validation_cases = _split_cases(spec, campaign, "validation")
            holdout_cases = _split_cases(spec, campaign, "holdout")
            _tool_preflight(runner, baseline, train_cases, artifact_root / "preflight", spec.evaluation_seed)
            _phase(summary, artifact_root, "baseline")
            before = assess(runner, baseline, validation_cases, artifact_root / "baseline-validation", seed=spec.evaluation_seed)
            summary["baseline_validation"] = before.to_mapping()
            _phase(summary, artifact_root, "search")
            champion, selected = _search(spec, runner, baseline, before, artifact_root, summary, session.verify)
            summary["candidate_fingerprint"] = champion.fingerprint
            summary["selected_validation"] = selected.to_mapping()
            summary["validation_improved"] = selected.improves_success(before)
            write_json_artifact(artifact_root / "best-candidate.json", {
                "schema_version": 1, "candidate_id": champion.candidate_id,
                "components": champion.components, "metadata": champion.metadata,
            })
            if champion.fingerprint != baseline.fingerprint:
                _phase(summary, artifact_root, "frozen-validation")
                session.verify()
                confirmation_seed = spec.evaluation_seed + spec.repetitions
                confirm_before = assess(runner, baseline, validation_cases, artifact_root / "confirmation-baseline", seed=confirmation_seed)
                confirm_after = assess(runner, champion, validation_cases, artifact_root / "confirmation-candidate", seed=confirmation_seed)
                summary["confirmation_baseline"] = confirm_before.to_mapping()
                summary["confirmation_candidate"] = confirm_after.to_mapping()
                _phase(summary, artifact_root, "holdout")
                summary["holdout_used"] = True
                session.verify()
                holdout_before = assess(runner, baseline, holdout_cases, artifact_root / "holdout-baseline", seed=spec.evaluation_seed)
                holdout_after = assess(runner, champion, holdout_cases, artifact_root / "holdout-candidate", seed=spec.evaluation_seed)
                summary["holdout_baseline"] = holdout_before.to_mapping()
                summary["holdout_candidate"] = holdout_after.to_mapping()
                summary["prompt_improved"] = (
                    selected.improves_success(before) and confirm_after.improves_success(confirm_before)
                    and not holdout_after.compared_score(holdout_before).core_regression
                    and holdout_after.score.hard_safety_failures == 0
                )
                _phase(summary, artifact_root, "application-verification")
                review = artifact_root / "candidate-for-review"
                prompt_path = export_upstream_prompt(
                    snapshot, champion, runner.bounded_campaign(), review, seed=spec.evaluation_seed,
                )
                runner.prompt_path = prompt_path
                application = runner.run(
                    champion, train_cases[0],
                    artifact_root / "application-check", seed=spec.evaluation_seed,
                )
                summary["prompt_override_verified"] = True
                summary["export_check_success"] = application.success
                verified_source = next(
                    item for item in snapshot["cases"]
                    if item["reference"] == spec.case_splits["train"][0]
                )
                receipt_path = write_json_artifact(review / "application-verification.json", {
                    "schema_version": 1,
                    "prompt_override_verified": True,
                    "product_application_verified": False,
                    "candidate_fingerprint": champion.fingerprint,
                    "prompt_sha256": hashlib.sha256(tier_pack_from_candidate(champion).encode()).hexdigest(),
                    "korvid_revision": spec.runtime.korvid_revision,
                    "reference": verified_source["reference"],
                    "source_sha256": verified_source["source_sha256"],
                    "seed": spec.evaluation_seed,
                    "upstream_success": summary["export_check_success"],
                    "qualification": "see_experiment_summary",
                })
                receipt_path.chmod(0o444)
                summary["qualified"] = (
                    selected.qualifies(before) and confirm_after.qualifies(confirm_before)
                    and holdout_after.qualifies(holdout_before)
                    and summary["prompt_override_verified"] and summary["export_check_success"]
                )
                summary["deployment"] = {
                    "original_prompt": "candidate-for-review/original-prompt.txt",
                    "optimized_prompt": "candidate-for-review/optimized-prompt.txt",
                    "diff": "candidate-for-review/prompt.diff",
                    "reload_verification": "candidate-for-review/application-verification.json",
                    "product_application_status": "requires_reviewed_korvid_prompt_pack_update",
                    "model_reference": spec.model.reference, "model_digest": spec.model.digest,
                    "options": {**dict(spec.model.options), "seed": spec.evaluation_seed},
                }
            else:
                summary["stop_reason"] = "no_distinct_improving_candidate"
            session.verify()
            if lab_identity() != summary["lab"]:
                raise ExperimentIntegrityError("prompt_lab_changed")
            budget.check()
        summary["pipeline_completed"] = True
        summary["status"] = "QUALIFIED" if summary["qualified"] else "NOT_CONVERGED"
        _phase(summary, artifact_root, "finished")
    except ToolPreflightInconclusive as exc:
        summary.update(status="PREFLIGHT_INCONCLUSIVE", stop_reason=str(exc))
    except BudgetExhausted as exc:
        summary.update(
            status="NOT_CONVERGED", stop_reason=exc.reason,
            qualified=False, prompt_improved=False,
        )
    finally:
        failure = sys.exc_info()[0]
        if failure is not None:
            summary.update(
                status="CANCELLED" if issubclass(failure, KeyboardInterrupt) else "SYSTEM_ERROR",
                error_type=failure.__name__, pipeline_completed=False, qualified=False,
                prompt_improved=False,
            )
            if isinstance(sys.exc_info()[1], ExperimentIntegrityError):
                summary["stop_reason"] = str(sys.exc_info()[1])
        summary.update(_coverage(artifact_root, baseline))
        summary.update(
            evaluations=budget.evaluations, proposal_attempts=budget.proposals,
            search_metric_calls=runner.search_calls if runner is not None else 0,
            elapsed_seconds=budget.elapsed_seconds,
        )
        write_json_artifact(artifact_root / "experiment-summary.json", summary)
    return summary
