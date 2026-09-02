from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]
from korvid.evals.scenario import bundled_scenarios_dir, load_scenario

from .aks import (
    AKSMissingToolError,
    AKSPortForward,
    AKSPortForwardError,
    AKSPreflightTransientError,
)
from .artifacts import write_json_artifact
from .baseline import PROFILE_NAMES, build_baseline_candidate, write_baseline_candidate
from .bridge_worker import EXECUTION_MODE_LIVE
from .config import load_campaign, load_candidate
from .contracts import (
    AKSPortForwardServing,
    Campaign,
    Candidate,
    EvalCase,
    KorvidReadonlyServing,
)
from .korvid_readonly import KorvidReadonlyRunner
from .optimize import (
    DEFAULT_OPTIMIZATION_SEED,
    OptimizationArtifacts,
    optimize_campaign,
)
from .publish import DEFAULT_MINIMUM_MODEL_IMPROVEMENT, publish_bundle
from .runner import BridgeSystemError, KorvidProcessRunner, KorvidRunner
from .scoring import RepetitionOutcome, pass_hat_k, result_passed, score_result
from .stable_candidates import build_structured_candidates
from .stable_proposer import BoundedAppendProposer
from .stable_rollover import (
    PriorCampaignEvidence,
    load_prior_campaign_evidence,
    write_rollover_lineage,
    write_rollover_winner,
)
from .stable_rollover_candidates import build_rollover_candidates
from .stable_scenarios import (
    FreshHoldoutExhaustedError,
    RolloverScenarioManifest,
    ScenarioManifest,
    build_rollover_scenario_manifest,
    build_scenario_manifest,
)
from .stable_search import (
    StableSearchConfig,
    StableSearchExtension,
    StableSearchSystemError,
    run_stable_search,
)

_STABLE_SEARCH_CAMPAIGN_ID = "stable-search-korvid-small"
_STABLE_SEARCH_MODEL = "qwen3:0.6b"
_STABLE_SEARCH_PROFILE = "small"
_STABLE_SEARCH_TIMEOUT_SECONDS = 160.0
_STABLE_SEARCH_REPETITIONS = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="korvid-prompt-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate candidate and campaign inputs."
    )
    _add_candidate_campaign_arguments(validate_parser)
    validate_parser.set_defaults(func=command_validate)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run bridge evaluation and emit a summary artifact."
    )
    _add_candidate_campaign_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts") / "evaluate",
        help="Directory for request, response, and summary artifacts.",
    )
    evaluate_parser.add_argument(
        "--case-id",
        dest="case_ids",
        action="append",
        default=[],
        help="Limit evaluation to one or more case_id values.",
    )
    evaluate_parser.add_argument(
        "--train-case-id",
        dest="train_case_ids",
        action="append",
        default=[],
        help="Case ids that form the train split recorded in the evaluation summary.",
    )
    evaluate_parser.add_argument(
        "--validation-case-id",
        dest="validation_case_ids",
        action="append",
        default=[],
        help="Case ids that form the validation split recorded in the evaluation summary.",
    )
    evaluate_parser.add_argument(
        "--milestone-case-id",
        dest="milestone_case_ids",
        action="append",
        default=[],
        help="Case ids that form the milestone pack recorded in the evaluation summary.",
    )
    evaluate_parser.add_argument(
        "--bundle-kind",
        choices=("common", "model-specific"),
        default="common",
        help="Bundle kind recorded in the evaluation summary.",
    )
    evaluate_parser.add_argument(
        "--json",
        action="store_true",
        help="Write the evaluation summary JSON to stdout.",
    )
    evaluate_parser.set_defaults(func=command_evaluate)

    optimize_parser = subparsers.add_parser(
        "optimize", help="Run GEPA optimization with DSPy reflection."
    )
    _add_candidate_campaign_arguments(optimize_parser)
    optimize_parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts") / "optimize",
        help="Directory for optimization artifacts.",
    )
    optimize_parser.add_argument(
        "--max-metric-calls",
        type=int,
        required=True,
        help="Maximum GEPA metric calls.",
    )
    optimize_parser.add_argument(
        "--reflection-model",
        help="DSPy LM spec used for reflection proposals, for example openai/gpt-4.1-mini.",
    )
    optimize_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_OPTIMIZATION_SEED,
        help=(
            "Non-negative GEPA search seed; part of the immutable run identity, so a new seed "
            f"always starts a fresh invocation directory (default {DEFAULT_OPTIMIZATION_SEED})."
        ),
    )
    optimize_parser.add_argument(
        "--train-case-id",
        dest="train_case_ids",
        action="append",
        default=[],
        help="Case ids used for the train split; required and disjoint from the validation split.",
    )
    optimize_parser.add_argument(
        "--validation-case-id",
        dest="validation_case_ids",
        action="append",
        default=[],
        help="Case ids used for the validation split; required and disjoint from the train split.",
    )
    optimize_parser.set_defaults(func=command_optimize)

    aks_parser = subparsers.add_parser(
        "aks-check", help="Perform a read-only AKS serving preflight."
    )
    aks_parser.add_argument(
        "--campaign", type=Path, required=True, help="Path to a campaign YAML file."
    )
    aks_parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts") / "aks-check",
        help="Directory for temporary kubeconfig artifacts.",
    )
    aks_parser.set_defaults(func=command_aks_check)

    publish_parser = subparsers.add_parser(
        "publish", help="Publish a prompt bundle into a registry."
    )
    _add_candidate_campaign_arguments(publish_parser)
    publish_parser.add_argument(
        "--model-metadata",
        type=Path,
        required=True,
        help="Path to model metadata JSON.",
    )
    publish_parser.add_argument(
        "--evaluation-summary",
        type=Path,
        required=True,
        help="Path to an evaluation summary JSON artifact.",
    )
    publish_parser.add_argument(
        "--registry-root",
        type=Path,
        default=Path("registry"),
        help="Target directory for published registry files.",
    )
    publish_parser.add_argument(
        "--minimum-model-improvement",
        type=float,
        default=DEFAULT_MINIMUM_MODEL_IMPROVEMENT,
        help=(
            "Minimum effective-score improvement a model-specific bundle must strictly exceed "
            f"over its common baseline (default {DEFAULT_MINIMUM_MODEL_IMPROVEMENT})."
        ),
    )
    publish_parser.set_defaults(func=command_publish)

    baseline_parser = subparsers.add_parser(
        "korvid-baseline",
        help="Materialize the installed Korvid profile's shipped system prompt as a seed candidate.",
    )
    baseline_parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        required=True,
        help="Installed Korvid agent profile to materialize (small or full).",
    )
    baseline_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the immutable baseline candidate YAML to; must not already exist.",
    )
    baseline_parser.set_defaults(func=command_korvid_baseline)

    stable_search_parser = subparsers.add_parser(
        "stable-search",
        help="Run the bounded 306-call stable-search campaign against installed Korvid read-only evals.",
    )
    stable_search_parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Fresh directory for immutable stable-search artifacts; must not already exist.",
    )
    stable_search_parser.add_argument(
        "--target-per-split",
        type=int,
        default=6,
        help="Requested scenario count per train/validation/milestone split before fail-closed reduction.",
    )
    stable_search_parser.add_argument(
        "--reflection-model",
        help="DSPy LM spec for the optional bounded proposer, for example ollama_chat/qwen3:4b.",
    )
    stable_search_parser.add_argument(
        "--enable-bounded-proposer",
        action="store_true",
        help="Attempt at most one bounded proposer candidate from the strongest eligible Stage B structured finalist.",
    )
    stable_search_parser.add_argument(
        "--json",
        action="store_true",
        help="Write the stable-search summary JSON to stdout.",
    )
    stable_search_parser.set_defaults(func=command_stable_search)

    rollover_parser = subparsers.add_parser(
        "stable-search-rollover",
        help="Run the bounded v3 rollover campaign from prior no-winner stable-search evidence.",
    )
    rollover_parser.add_argument(
        "--prior-artifact-root",
        type=Path,
        required=True,
        help="Prior immutable stable-search artifact root whose decision must be no_stable_winner.",
    )
    rollover_parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Fresh directory for immutable rollover artifacts; must not already exist.",
    )
    rollover_parser.add_argument(
        "--winner-output",
        type=Path,
        help="Optional path to write the exact winning append candidate YAML when rollout qualifies.",
    )
    rollover_parser.add_argument(
        "--json",
        action="store_true",
        help="Write the stable-search rollover summary JSON to stdout.",
    )
    rollover_parser.set_defaults(func=command_stable_search_rollover)

    return parser


def _add_candidate_campaign_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidate", type=Path, required=True, help="Path to a candidate YAML file."
    )
    parser.add_argument(
        "--campaign", type=Path, required=True, help="Path to a campaign YAML file."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "func", None)
    if command is None:
        parser.error("a subcommand is required")
    return int(command(args))


def command_validate(args: argparse.Namespace) -> int:
    try:
        candidate, campaign = _load_candidate_campaign(args.candidate, args.campaign)
    except (OSError, ValueError) as exc:
        print(f"validation failed: {exc}", file=_stderr())
        return 2

    print(
        "validated candidate={candidate_id} fingerprint={fingerprint} campaign={campaign_id} cases={case_count} models={models}".format(
            candidate_id=candidate.candidate_id,
            fingerprint=candidate.fingerprint,
            campaign_id=campaign.campaign_id,
            case_count=len(campaign.cases),
            models=",".join(campaign.models),
        )
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    try:
        candidate, campaign = _load_candidate_campaign(args.candidate, args.campaign)
        selected_cases = _select_cases(campaign.cases, args.case_ids)
        case_sets = _resolve_case_sets(
            args,
            [case.case_id for case in selected_cases],
            [case.case_id for case in campaign.cases],
        )
    except (OSError, ValueError) as exc:
        print(f"evaluation failed: {exc}", file=_stderr())
        return 2

    artifact_root = Path(args.artifact_root)
    try:
        with _serving_session(campaign, artifact_root) as model_endpoint:
            runner = _build_runner(campaign, model_endpoint=model_endpoint)
            summary = _evaluate_campaign(
                candidate=candidate,
                campaign=campaign,
                selected_cases=selected_cases,
                runner=runner,
                artifact_root=artifact_root,
                bundle_kind=args.bundle_kind,
                case_sets=case_sets,
                reproduction_command=_evaluate_reproduction_command(args),
            )
    except ValueError as exc:
        print(f"evaluation failed: {exc}", file=_stderr())
        return 2
    except AKSPortForwardError as exc:
        print(f"evaluation failed: AKS serving error: {exc}", file=_stderr())
        return 1
    except BridgeSystemError as exc:
        print(f"evaluation failed: systemic bridge error: {exc}", file=_stderr())
        return 1
    except OSError as exc:
        print(f"evaluation failed: {exc}", file=_stderr())
        return 1

    summary_path = write_json_artifact(
        artifact_root / "evaluation-summary.json", summary
    )
    summary["artifact_refs"] = _collect_artifact_refs(artifact_root)
    summary_path = write_json_artifact(summary_path, summary)

    unsafe = summary["hard_safety_failures"] > 0
    exit_code = 1 if unsafe else 0

    if args.json:
        print(json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2))
    elif exit_code == 0:
        print(
            "evaluated candidate={candidate_id} campaign={campaign_id} aggregate={aggregate:.3f} pass^3={pass3} pass^5={pass5} summary={summary_path}".format(
                candidate_id=candidate.candidate_id,
                campaign_id=campaign.campaign_id,
                aggregate=summary["aggregate_score"],
                pass3=_format_pass_hat_k(summary["pass_at_3"]),
                pass5=_format_pass_hat_k(summary["pass_at_5"]),
                summary_path=summary_path,
            )
        )

    if unsafe:
        print(
            "evaluation failed: one or more runs produced hard safety failures",
            file=_stderr(),
        )
    return exit_code


def command_optimize(args: argparse.Namespace) -> int:
    if not args.reflection_model:
        print("optimization failed: --reflection-model is required", file=_stderr())
        return 2

    try:
        candidate, campaign = _load_candidate_campaign(args.candidate, args.campaign)
        _require_non_negative_seed(args.seed)
        _require_case_selection("--train-case-id", args.train_case_ids)
        _require_case_selection("--validation-case-id", args.validation_case_ids)
        _require_disjoint_case_selections(args.train_case_ids, args.validation_case_ids)
        train_cases = _expand_cases(_select_cases(campaign.cases, args.train_case_ids))
        validation_cases = _expand_cases(
            _select_cases(campaign.cases, args.validation_case_ids)
        )
    except (OSError, ValueError) as exc:
        print(f"optimization failed: {exc}", file=_stderr())
        return 2

    try:
        with _serving_session(campaign, Path(args.artifact_root)) as model_endpoint:
            artifacts = optimize_campaign(
                runner=_build_runner(campaign, model_endpoint=model_endpoint),
                seed_candidate=candidate,
                train_cases=train_cases,
                validation_cases=validation_cases,
                artifact_root=args.artifact_root,
                max_metric_calls=args.max_metric_calls,
                seed=args.seed,
                reflection_lm=_build_reflection_lm(args.reflection_model),
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"optimization failed: {exc}", file=_stderr())
        return 1

    _print_optimization_summary(artifacts)
    return 0


def command_aks_check(args: argparse.Namespace) -> int:
    try:
        campaign = load_campaign(args.campaign)
    except (OSError, ValueError) as exc:
        print(f"aks-check failed: {exc}", file=_stderr())
        return 2

    serving = campaign.serving
    if not isinstance(serving, AKSPortForwardServing):
        print(
            "aks-check failed: campaign does not use aks_port_forward serving",
            file=_stderr(),
        )
        return 2

    try:
        with AKSPortForward(serving, workspace_dir=args.artifact_root) as forward:
            print(
                f"aks preflight passed campaign={campaign.campaign_id} model={serving.model} endpoint={forward.base_url}"
            )
    except AKSMissingToolError as exc:
        print(f"aks-check failed: {exc}", file=_stderr())
        return 1
    except AKSPreflightTransientError as exc:
        print(f"aks-check failed (transient): {exc}", file=_stderr())
        return 75
    except AKSPortForwardError as exc:
        print(f"aks-check failed: {exc}", file=_stderr())
        return 1
    return 0


def command_publish(args: argparse.Namespace) -> int:
    try:
        candidate, campaign = _load_candidate_campaign(args.candidate, args.campaign)
        model_metadata = _load_json_file(args.model_metadata)
        evaluation_summary = _load_json_file(args.evaluation_summary)
        if not isinstance(model_metadata, dict):
            raise ValueError("model metadata must be a JSON object")  # noqa: TRY004 - preserve validation API
        _validate_publish_summary(
            evaluation_summary, candidate, campaign, model_metadata
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"publish failed: {exc}", file=_stderr())
        return 2

    try:
        decision = publish_bundle(
            candidate=candidate,
            campaign=campaign,
            model_metadata=model_metadata,
            evaluation_summary=evaluation_summary,
            registry_root=args.registry_root,
            minimum_model_improvement=args.minimum_model_improvement,
        )
    except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
        print(f"publish failed: {exc}", file=_stderr())
        return 1

    if not decision.published or decision.bundle is None:
        print(f"publish blocked: {decision.reason}", file=_stderr())
        return 1

    print(
        f"published version={decision.bundle.version} kind={decision.bundle.bundle_kind} effective_score={decision.effective_score:.3f} registry={args.registry_root}"
    )
    return 0


def command_korvid_baseline(args: argparse.Namespace) -> int:
    try:
        candidate = build_baseline_candidate(args.profile)
        write_baseline_candidate(candidate, args.output)
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"korvid-baseline failed: {exc}", file=_stderr())
        return 2

    print(
        f"wrote baseline candidate={candidate.candidate_id} fingerprint={candidate.fingerprint} "
        f"profile={args.profile} output={args.output}"
    )
    return 0


def command_stable_search(args: argparse.Namespace) -> int:
    if Path(args.artifact_root).exists() or Path(args.artifact_root).is_symlink():
        print(
            f"stable-search failed: stable search artifact root already exists: {args.artifact_root}",
            file=_stderr(),
        )
        return 2
    if args.enable_bounded_proposer and not args.reflection_model:
        print(
            "stable-search failed: --enable-bounded-proposer requires --reflection-model",
            file=_stderr(),
        )
        return 2
    if args.reflection_model and not args.enable_bounded_proposer:
        print(
            "stable-search failed: --reflection-model requires --enable-bounded-proposer",
            file=_stderr(),
        )
        return 2

    try:
        baseline = build_baseline_candidate(_STABLE_SEARCH_PROFILE)
        candidates = build_structured_candidates(baseline)
        manifest = build_scenario_manifest(target_per_split=args.target_per_split)
        campaign = _build_stable_search_campaign(manifest)
        runner = KorvidReadonlyRunner(campaign=campaign)
        extension = (
            StableSearchExtension(
                bounded_append_proposer=BoundedAppendProposer(
                    _build_reflection_lm(args.reflection_model)
                )
            )
            if args.enable_bounded_proposer
            else None
        )
        artifacts = run_stable_search(
            runner=runner,
            baseline=baseline,
            candidates=candidates,
            manifest=manifest,
            artifact_root=args.artifact_root,
            extension=extension,
        )
    except ValueError as exc:
        print(f"stable-search failed: {exc}", file=_stderr())
        return 2
    except BridgeSystemError as exc:
        error_label = _stable_search_system_error_label(exc)
        if args.json:
            print(
                json.dumps(
                    {"status": "system_error", "error_label": error_label},
                    sort_keys=True,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(
                f"stable-search failed: systemic bridge error: {error_label}",
                file=_stderr(),
            )
        return 1
    except OSError as exc:
        print(f"stable-search failed: {exc}", file=_stderr())
        return 1

    summary = _load_json_file(artifacts.summary_path)
    if args.json:
        print(json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2))
    else:
        decision = summary["decision"]
        print(
            "stable-search status={status} candidate={candidate_id} summary={summary_path}".format(
                status=decision["status"],
                candidate_id=decision["candidate_id"] or "none",
                summary_path=artifacts.summary_path,
            )
        )
    return 0


def command_stable_search_rollover(args: argparse.Namespace) -> int:
    artifact_root = Path(args.artifact_root)
    if artifact_root.exists() or artifact_root.is_symlink():
        print(
            f"stable-search-rollover failed: stable search artifact root already exists: {artifact_root}",
            file=_stderr(),
        )
        return 2

    winner_output = Path(args.winner_output) if args.winner_output is not None else None
    if winner_output is not None and (winner_output.exists() or winner_output.is_symlink()):
        print(
            f"stable-search-rollover failed: rollover winner output already exists: {winner_output}",
            file=_stderr(),
        )
        return 2

    prior: PriorCampaignEvidence | None = None
    rollover: RolloverScenarioManifest | None = None
    lineage_draft_path: Path | None = None

    try:
        prior = load_prior_campaign_evidence(args.prior_artifact_root)
        baseline = build_baseline_candidate(_STABLE_SEARCH_PROFILE)
        candidates = build_rollover_candidates(baseline, prior)
        rollover = build_rollover_scenario_manifest(prior.consumed_assignments)
        campaign = _build_stable_search_campaign(rollover.manifest)
        runner = KorvidReadonlyRunner(campaign=campaign)
        config = StableSearchConfig()
        lineage_draft_path = _rollover_lineage_draft_path(artifact_root)
        if lineage_draft_path.exists() or lineage_draft_path.is_symlink():
            raise ValueError(f"rollover lineage draft already exists: {lineage_draft_path}")
        write_rollover_lineage(lineage_draft_path, prior, rollover)
        artifacts = run_stable_search(
            runner=runner,
            baseline=baseline,
            candidates=candidates,
            manifest=rollover.manifest,
            artifact_root=artifact_root,
            config=config,
        )
    except FreshHoldoutExhaustedError:
        _cleanup_rollover_lineage_draft(lineage_draft_path)
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "no_stable_winner",
                        "terminal_reason": "fresh_holdout_exhausted",
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(
                "stable-search-rollover status=no_stable_winner terminal_reason=fresh_holdout_exhausted"
            )
        return 0
    except ValueError as exc:
        _cleanup_rollover_lineage_draft(lineage_draft_path)
        print(f"stable-search-rollover failed: {exc}", file=_stderr())
        return 2
    except BridgeSystemError as exc:
        error_label = _stable_search_system_error_label(exc)
        if prior is not None and rollover is not None:
            _materialize_rollover_lineage(
                artifact_root=artifact_root,
                draft_path=lineage_draft_path,
                prior=prior,
                rollover=rollover,
                terminal_reason=error_label,
            )
        _cleanup_rollover_lineage_draft(lineage_draft_path)
        if args.json:
            print(
                json.dumps(
                    {"status": "system_error", "error_label": error_label},
                    sort_keys=True,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(
                f"stable-search-rollover failed: systemic bridge error: {error_label}",
                file=_stderr(),
            )
        return 1
    except OSError as exc:
        _cleanup_rollover_lineage_draft(lineage_draft_path)
        print(f"stable-search-rollover failed: {exc}", file=_stderr())
        return 1

    summary = _load_json_file(artifacts.summary_path)
    terminal_reason = _rollover_terminal_reason(summary)
    assert prior is not None
    assert rollover is not None
    _materialize_rollover_lineage(
        artifact_root=artifact_root,
        draft_path=lineage_draft_path,
        prior=prior,
        rollover=rollover,
        terminal_reason=terminal_reason,
    )
    _cleanup_rollover_lineage_draft(lineage_draft_path)

    winning_candidate = _rollover_winner_candidate(summary, candidates)
    if winner_output is not None and winning_candidate is not None:
        write_rollover_winner(winner_output, winning_candidate)

    if args.json:
        print(json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2))
    else:
        decision = summary["decision"]
        print(
            "stable-search-rollover status={status} candidate={candidate_id} summary={summary_path}".format(
                status=decision["status"],
                candidate_id=decision["candidate_id"] or "none",
                summary_path=artifacts.summary_path,
            )
        )
    return 0


def _stable_search_system_error_label(exc: BridgeSystemError) -> str:
    if isinstance(exc, StableSearchSystemError):
        return exc.error_label
    return re.sub(r"(?<!^)(?=[A-Z])", "_", type(exc).__name__).lower()


def _rollover_lineage_draft_path(artifact_root: Path) -> Path:
    return artifact_root.parent / f".{artifact_root.name}.rollover-lineage.json"


def _rollover_lineage_path(artifact_root: Path) -> Path:
    return artifact_root / "rollover-lineage.json"


def _materialize_rollover_lineage(
    *,
    artifact_root: Path,
    draft_path: Path | None,
    prior: PriorCampaignEvidence,
    rollover: RolloverScenarioManifest,
    terminal_reason: str | None,
) -> Path:
    artifact_root.mkdir(parents=True, exist_ok=True)
    lineage_path = _rollover_lineage_path(artifact_root)
    return write_rollover_lineage(lineage_path, prior, rollover, terminal_reason=terminal_reason)


def _cleanup_rollover_lineage_draft(draft_path: Path | None) -> None:
    if draft_path is not None and draft_path.exists():
        draft_path.unlink()


def _rollover_terminal_reason(summary: Mapping[str, Any]) -> str | None:
    status = _rollover_decision_status(summary)
    if status == "promote":
        return "stable_winner"
    return status


def _rollover_winner_candidate(
    summary: Mapping[str, Any],
    candidates: Sequence[Any],
) -> Candidate | None:
    status = _rollover_decision_status(summary)
    if status not in {"promote", "stable_winner"}:
        return None
    decision = summary["decision"]
    if not isinstance(decision, Mapping):
        raise ValueError("stable-search summary decision must be a mapping")  # noqa: TRY004 - preserve validation API
    candidate_id = decision.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("stable-search summary winner must provide candidate_id")
    candidate_index = {
        structured.candidate.candidate_id: structured.candidate
        for structured in candidates
    }
    try:
        return candidate_index[candidate_id]
    except KeyError as exc:
        raise ValueError(f"stable-search summary winner is not a rollover candidate: {candidate_id}") from exc


def _rollover_decision_status(summary: Mapping[str, Any]) -> str:
    decision = summary.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("stable-search summary decision must be a mapping")  # noqa: TRY004 - preserve validation API
    status = decision.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ValueError("stable-search summary decision.status must be a non-empty string")
    return status


def _stderr() -> Any:
    import sys

    return sys.stderr


@contextmanager
def _serving_session(campaign: Campaign, workspace_dir: Path) -> Iterator[str | None]:
    """Hold the exact serving resources a campaign needs for a whole run."""
    serving = campaign.serving
    if not isinstance(serving, AKSPortForwardServing):
        yield None
        return
    with AKSPortForward(serving, workspace_dir=workspace_dir) as forward:
        yield forward.base_url


def _build_runner(campaign: Campaign, *, model_endpoint: str | None) -> KorvidRunner:
    """Select the evidence-producing runner by campaign.serving.backend.

    korvid_readonly campaigns run entirely against the installed korvid.evals
    CLI and take no model_endpoint; process/aks_port_forward campaigns keep
    their existing KorvidProcessRunner construction unchanged.
    """
    if isinstance(campaign.serving, KorvidReadonlyServing):
        return KorvidReadonlyRunner(campaign=campaign)
    return KorvidProcessRunner(
        campaign=campaign,
        timeout_seconds=campaign.bridge_timeout_seconds,
        model_endpoint=model_endpoint,
    )


def _build_stable_search_campaign(manifest: ScenarioManifest) -> Campaign:
    return Campaign(
        schema_version=1,
        campaign_id=_STABLE_SEARCH_CAMPAIGN_ID,
        repetitions=_STABLE_SEARCH_REPETITIONS,
        models=(_STABLE_SEARCH_MODEL,),
        cases=_stable_search_cases(manifest),
        serving=KorvidReadonlyServing(
            backend="korvid_readonly",
            provider="ollama",
            base_url=_require_env_url("KORVID_READONLY_BASE_URL"),
            profile=_STABLE_SEARCH_PROFILE,
            timeout_seconds=_STABLE_SEARCH_TIMEOUT_SECONDS,
        ),
    )


def _stable_search_cases(manifest: ScenarioManifest) -> tuple[EvalCase, ...]:
    catalog = _bundled_eval_cases()
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for assignment in manifest.assignments:
        if assignment.scenario_id in seen:
            continue
        try:
            scenario = catalog[assignment.scenario_id]
        except KeyError as exc:
            raise ValueError(
                f"installed Korvid scenario catalog is missing {assignment.scenario_id!r}"
            ) from exc
        cases.append(
            EvalCase(
                case_id=assignment.scenario_id,
                template_id=f"stable-search-{assignment.split}",
                prompt=scenario.question,
                models=(_STABLE_SEARCH_MODEL,),
            )
        )
        seen.add(assignment.scenario_id)
    if not cases:
        raise ValueError("stable-search scenario manifest must contain at least one case")
    return tuple(cases)


def _bundled_eval_cases() -> dict[str, Any]:
    directory = bundled_scenarios_dir()
    if not directory.is_dir():
        raise ValueError(f"korvid bundled scenarios directory not found: {directory}")
    catalog: dict[str, Any] = {}
    for path in sorted(directory.glob("*.yaml")):
        scenario = load_scenario(path)
        if scenario.id in catalog:
            raise ValueError(
                f"installed Korvid scenario catalog contains duplicate id: {scenario.id}"
            )
        catalog[scenario.id] = scenario
    return catalog


def _require_env_url(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"missing environment variable {name}")
    return value


def _load_candidate_campaign(
    candidate_path: Path, campaign_path: Path
) -> tuple[Candidate, Campaign]:
    return load_candidate(candidate_path), load_campaign(campaign_path)


def _load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_cases(
    cases: Sequence[EvalCase], requested_case_ids: Sequence[str]
) -> tuple[EvalCase, ...]:
    if not requested_case_ids:
        return tuple(cases)

    requested = list(dict.fromkeys(requested_case_ids))
    selected = [case for case in cases if case.case_id in requested]
    missing = [
        case_id
        for case_id in requested
        if case_id not in {case.case_id for case in selected}
    ]
    if missing:
        raise ValueError(f"unknown case_id value(s): {', '.join(missing)}")
    return tuple(selected)


def _expand_cases(cases: Sequence[EvalCase]) -> tuple[EvalCase, ...]:
    expanded: list[EvalCase] = []
    for case in cases:
        for model in case.models:
            expanded.append(
                EvalCase(
                    case_id=case.case_id,
                    template_id=case.template_id,
                    prompt=case.prompt,
                    models=(model,),
                )
            )
    if not expanded:
        raise ValueError("at least one case must be selected")
    return tuple(expanded)


def _require_case_selection(flag: str, case_ids: Sequence[str]) -> list[str]:
    selection = list(dict.fromkeys(case_ids))
    if not selection:
        raise ValueError(f"{flag} is required and must name at least one case")
    return selection


def _require_disjoint_case_selections(
    train_case_ids: Sequence[str], validation_case_ids: Sequence[str]
) -> None:
    overlap = sorted(set(train_case_ids) & set(validation_case_ids))
    if overlap:
        raise ValueError(
            f"train and validation case sets must be disjoint: {', '.join(overlap)}"
        )


def _require_non_negative_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _resolve_case_sets(
    args: argparse.Namespace,
    evaluated_case_ids: Sequence[str],
    campaign_case_ids: Sequence[str],
) -> dict[str, list[str]]:
    train_case_ids = _require_case_selection("--train-case-id", args.train_case_ids)
    validation_case_ids = _require_case_selection(
        "--validation-case-id", args.validation_case_ids
    )
    _require_disjoint_case_selections(train_case_ids, validation_case_ids)
    milestone_case_ids = list(dict.fromkeys(args.milestone_case_ids))

    del evaluated_case_ids
    campaign = set(campaign_case_ids)
    for label, case_ids in (
        ("train", train_case_ids),
        ("validation", validation_case_ids),
        ("milestone", milestone_case_ids),
    ):
        missing = [case_id for case_id in case_ids if case_id not in campaign]
        if missing:
            raise ValueError(
                f"{label} case set must be drawn from the campaign cases: {', '.join(missing)}"
            )

    return {
        "train": train_case_ids,
        "validation": validation_case_ids,
        "milestone": milestone_case_ids,
    }


def _evaluate_campaign(
    *,
    candidate: Candidate,
    campaign: Campaign,
    selected_cases: Sequence[EvalCase],
    runner: KorvidRunner,
    artifact_root: Path,
    bundle_kind: str,
    case_sets: Mapping[str, Sequence[str]],
    reproduction_command: Sequence[str],
) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    executed_cases = _expand_cases(selected_cases)
    run_records: list[dict[str, Any]] = []
    repetition_outcomes: list[RepetitionOutcome] = []
    scores: list[float] = []
    hard_safety_failures = 0

    for case in executed_cases:
        for repetition in range(1, campaign.repetitions + 1):
            run_dir = artifact_root / "runs" / _case_run_slug(case, repetition)
            result = runner.run(
                candidate, case, run_dir, repetition=repetition, seed=repetition - 1
            )
            scored = score_result(result)
            run_records.append(
                {
                    "case_id": case.case_id,
                    "template_id": case.template_id,
                    "model": case.models[0],
                    "repetition": repetition,
                    "accepted": scored.accepted,
                    "unsafe": scored.unsafe,
                    "score": scored.score,
                    "status": result.status,
                    "execution_mode": result.execution_mode,
                    "hard_failures": list(result.grade.hard_failures)
                    if result.grade is not None
                    else [],
                }
            )
            repetition_outcomes.append(
                RepetitionOutcome(
                    case_id=case.case_id,
                    model=case.models[0],
                    repetition=repetition,
                    passed=result_passed(scored),
                )
            )
            scores.append(scored.score)
            hard_safety_failures += (
                len(result.grade.hard_failures) if result.grade is not None else 0
            )

    campaign_case_ids = [case.case_id for case in campaign.cases]
    evaluated_case_ids = list(dict.fromkeys(case.case_id for case in selected_cases))
    evaluated_models = sorted({case.models[0] for case in executed_cases})
    campaign_case_model_pairs = _case_model_pairs(campaign.cases)
    evaluated_case_model_pairs = _case_model_pairs(executed_cases)
    required_milestone_case_ids: list[str] = []
    full_milestone_pack_executed = False
    if bundle_kind == "model-specific" and len(evaluated_models) == 1:
        target_model = evaluated_models[0]
        expected_target_pairs = [
            _encode_case_model_pair(case.case_id, model)
            for case in campaign.cases
            for model in case.models
            if model == target_model
        ]
        required_milestone_case_ids = list(
            dict.fromkeys(
                case_id
                for case_id, _ in (
                    _decode_case_model_pair(pair, "campaign_case_model_pairs")
                    for pair in expected_target_pairs
                )
            )
        )
        full_milestone_pack_executed = (
            evaluated_case_ids == required_milestone_case_ids
            and set(evaluated_case_model_pairs) == set(expected_target_pairs)
        )
    elif bundle_kind == "common":
        required_milestone_case_ids = list(campaign_case_ids)
        full_milestone_pack_executed = evaluated_case_ids == campaign_case_ids

    requested_milestone_case_ids = list(dict.fromkeys(case_sets.get("milestone", ())))
    if requested_milestone_case_ids:
        milestone_case_ids = requested_milestone_case_ids
        milestone_covers_required_pack = set(milestone_case_ids) == set(
            required_milestone_case_ids
        )
    else:
        milestone_case_ids = (
            list(required_milestone_case_ids) if full_milestone_pack_executed else []
        )
        milestone_covers_required_pack = full_milestone_pack_executed
    summary: dict[str, Any] = {
        "bundle_kind": bundle_kind,
        "candidate_id": candidate.candidate_id,
        "candidate_fingerprint": candidate.fingerprint,
        "campaign_id": campaign.campaign_id,
        "campaign_case_ids": campaign_case_ids,
        "evaluated_case_ids": evaluated_case_ids,
        "evaluated_models": evaluated_models,
        "campaign_case_model_pairs": campaign_case_model_pairs,
        "evaluated_case_model_pairs": evaluated_case_model_pairs,
        "aggregate_score": sum(scores) / len(scores) if scores else 0.0,
        "model_scores": _model_scores(run_records),
        # How every graded run got its evidence. `scripted` grades never contacted a
        # model, so publication refuses anything that is not wholly `live`.
        "execution_modes": _execution_modes(run_records),
        "run_execution_modes": _run_execution_modes(run_records),
        "repetitions_per_case": campaign.repetitions,
        "pass_at_3": pass_hat_k(repetition_outcomes, 3),
        "pass_at_5": pass_hat_k(repetition_outcomes, 5),
        "hard_safety_failures": hard_safety_failures,
        "systemic_failures": 0,
        "milestone_passed": (
            hard_safety_failures == 0
            and full_milestone_pack_executed
            and milestone_covers_required_pack
        ),
        "case_sets": {
            "train": list(case_sets["train"]),
            "validation": list(case_sets["validation"]),
            "milestone": milestone_case_ids,
        },
        "artifact_refs": [],
        "reproduction_command": list(reproduction_command),
    }
    return summary


def _case_run_slug(case: EvalCase, repetition: int) -> str:
    normalized_case_id = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in case.case_id
    ).strip("-")
    normalized_model = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in case.models[0]
    ).strip("-")
    return f"{normalized_case_id or 'case'}-{normalized_model or 'model'}-r{repetition:02d}"


def _format_pass_hat_k(value: float | None) -> str:
    if value is None:
        return "insufficient-evidence"
    return f"{value:.3f}"


def _model_scores(run_records: Sequence[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for record in run_records:
        model = str(record["model"])
        totals[model] = totals.get(model, 0.0) + float(record["score"])
        counts[model] = counts.get(model, 0) + 1
    return {model: totals[model] / counts[model] for model in sorted(totals)}


def _execution_modes(run_records: Sequence[dict[str, Any]]) -> list[str]:
    return sorted({str(record["execution_mode"]) for record in run_records})


def _run_execution_modes(run_records: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Name the execution mode of every case/model pair, refusing a pair that disagrees.

    A pair whose repetitions were graded different ways is not one experiment, so it
    can never be summarized into a single claim.
    """
    modes: dict[str, str] = {}
    for record in run_records:
        pair = _encode_case_model_pair(str(record["case_id"]), str(record["model"]))
        mode = str(record["execution_mode"])
        previous = modes.setdefault(pair, mode)
        if previous != mode:
            raise ValueError(
                f"case/model pair {pair} mixed bridge execution modes: {previous} and {mode}"
            )
    return {pair: modes[pair] for pair in sorted(modes)}


def _case_model_pairs(cases: Sequence[EvalCase]) -> list[str]:
    pairs: list[str] = []
    for case in cases:
        for model in case.models:
            pairs.append(_encode_case_model_pair(case.case_id, model))
    return list(dict.fromkeys(pairs))


def _collect_artifact_refs(artifact_root: Path) -> list[str]:
    refs = [
        str(path.relative_to(artifact_root))
        for path in sorted(artifact_root.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}
    ]
    return refs


def _evaluate_reproduction_command(args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "--python",
        "3.12",
        "korvid-prompt-lab",
        "evaluate",
        "--candidate",
        str(args.candidate),
        "--campaign",
        str(args.campaign),
        "--artifact-root",
        str(args.artifact_root),
        "--bundle-kind",
        str(args.bundle_kind),
    ]
    for case_id in args.case_ids:
        command.extend(["--case-id", str(case_id)])
    for case_id in args.train_case_ids:
        command.extend(["--train-case-id", str(case_id)])
    for case_id in args.validation_case_ids:
        command.extend(["--validation-case-id", str(case_id)])
    for case_id in args.milestone_case_ids:
        command.extend(["--milestone-case-id", str(case_id)])
    return command


def _validate_publish_summary(
    summary: Any,
    candidate: Candidate,
    campaign: Campaign,
    model_metadata: dict[str, Any],
) -> None:
    if not isinstance(summary, dict):
        raise ValueError("evaluation summary must be a JSON object")  # noqa: TRY004 - preserve validation API

    candidate_id = _require_string_field(summary, "candidate_id")
    candidate_fingerprint = _require_string_field(summary, "candidate_fingerprint")
    campaign_id = _require_string_field(summary, "campaign_id")
    campaign_case_ids = _require_unique_string_list(
        summary.get("campaign_case_ids"), "campaign_case_ids"
    )
    evaluated_case_ids = _require_unique_string_list(
        summary.get("evaluated_case_ids"), "evaluated_case_ids"
    )
    evaluated_models = _require_unique_string_list(
        summary.get("evaluated_models"), "evaluated_models"
    )
    _require_live_execution_modes(summary)
    campaign_case_model_pairs = _require_unique_string_list(
        summary.get("campaign_case_model_pairs"), "campaign_case_model_pairs"
    )
    evaluated_case_model_pairs = _require_unique_string_list(
        summary.get("evaluated_case_model_pairs"), "evaluated_case_model_pairs"
    )

    expected_campaign_case_ids = [case.case_id for case in campaign.cases]
    expected_campaign_case_id_set = set(expected_campaign_case_ids)
    expected_models = set(campaign.models)
    expected_case_model_pairs = _case_model_pairs(campaign.cases)
    expected_case_model_pair_set = set(expected_case_model_pairs)
    target_model = _require_string_field(model_metadata, "model_family")

    if candidate_id != candidate.candidate_id:
        raise ValueError(
            "evaluation summary candidate_id does not match the candidate file"
        )
    if candidate_fingerprint != candidate.fingerprint:
        raise ValueError(
            "evaluation summary candidate_fingerprint does not match the candidate file"
        )
    if campaign_id != campaign.campaign_id:
        raise ValueError(
            "evaluation summary campaign_id does not match the campaign file"
        )
    if set(campaign_case_ids) != expected_campaign_case_id_set:
        raise ValueError(
            "evaluation summary campaign_case_ids do not match the campaign file"
        )
    if set(campaign_case_model_pairs) != expected_case_model_pair_set:
        raise ValueError(
            "evaluation summary campaign_case_model_pairs do not match the campaign file"
        )
    if any(
        case_id not in expected_campaign_case_id_set for case_id in evaluated_case_ids
    ):
        raise ValueError(
            "evaluation summary evaluated_case_ids must be drawn from the campaign"
        )
    if any(model not in expected_models for model in evaluated_models):
        raise ValueError(
            "evaluation summary evaluated_models must be drawn from the campaign"
        )
    if target_model not in expected_models or target_model not in set(evaluated_models):
        raise ValueError(
            "model metadata model_family must be present in the evaluated campaign models"
        )
    if any(
        pair not in expected_case_model_pair_set for pair in evaluated_case_model_pairs
    ):
        raise ValueError(
            "evaluation summary evaluated_case_model_pairs must be drawn from the campaign"
        )

    evaluated_pair_components = [
        _decode_case_model_pair(pair, "evaluated_case_model_pairs")
        for pair in evaluated_case_model_pairs
    ]
    if {case_id for case_id, _ in evaluated_pair_components} != set(evaluated_case_ids):
        raise ValueError(
            "evaluation summary evaluated_case_ids must match evaluated_case_model_pairs"
        )
    if {model for _, model in evaluated_pair_components} != set(evaluated_models):
        raise ValueError(
            "evaluation summary evaluated_models must match evaluated_case_model_pairs"
        )

    bundle_kind = summary.get("bundle_kind")
    if bundle_kind == "common" and (
        set(evaluated_case_ids) != expected_campaign_case_id_set
        or set(evaluated_models) != expected_models
        or set(evaluated_case_model_pairs) != expected_case_model_pair_set
    ):
        raise ValueError(
            "common publication requires the full campaign case pack and case-model matrix"
        )

    if bundle_kind == "model-specific":
        case_sets = summary.get("case_sets")
        if not isinstance(case_sets, dict):
            raise ValueError("evaluation summary case_sets must be an object")
        milestone_case_ids = _require_unique_string_list(
            case_sets.get("milestone"), "case_sets.milestone"
        )
        expected_target_pairs = [
            pair
            for pair in expected_case_model_pairs
            if _decode_case_model_pair(pair, "campaign_case_model_pairs")[1]
            == target_model
        ]
        expected_target_case_ids = list(
            dict.fromkeys(
                case_id
                for case_id, _ in (
                    _decode_case_model_pair(pair, "campaign_case_model_pairs")
                    for pair in expected_target_pairs
                )
            )
        )
        if not expected_target_pairs:
            raise ValueError(
                "model-specific publication target model is not present in the campaign"
            )
        if set(evaluated_models) != {target_model}:
            raise ValueError(
                "model-specific publication must be bound to the target model only"
            )
        if set(milestone_case_ids) != set(expected_target_case_ids):
            raise ValueError(
                "model-specific publication requires the full milestone case pack"
            )
        if set(evaluated_case_ids) != set(expected_target_case_ids) or set(
            evaluated_case_model_pairs
        ) != set(expected_target_pairs):
            raise ValueError(
                "model-specific publication requires the full target-model case pack"
            )


def _require_live_execution_modes(summary: dict[str, Any]) -> list[str]:
    """Refuse, before anything is written, a summary that is not wholly live evidence.

    ``scripted`` bridge runs replace the model with Korvid's deterministic operation
    scripts, so their grades are model-free. Publishing them would advertise a score
    the model never earned, and a mixed summary is no better: part of the evidence
    would still be model-free.
    """
    modes = _require_unique_string_list(
        summary.get("execution_modes"), "execution_modes"
    )
    if sorted(modes) != [EXECUTION_MODE_LIVE]:
        raise ValueError(
            "evaluation summary execution_modes must be exactly ['live']; scripted bridge"
            f" evidence never contacted a model: {sorted(modes)}"
        )
    return modes


def _require_string_field(summary: dict[str, Any], field_name: str) -> str:
    value = summary.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"evaluation summary {field_name} must be a non-empty string")
    return value


def _require_unique_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"evaluation summary {field_name} must be a non-empty list of strings"
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"evaluation summary {field_name} must be a non-empty list of strings"
            )
        if item in seen:
            raise ValueError(
                f"evaluation summary {field_name} must not contain duplicates"
            )
        normalized.append(item)
        seen.add(item)
    return normalized


def _encode_case_model_pair(case_id: str, model: str) -> str:
    return f"{case_id}::{model}"


def _decode_case_model_pair(value: str, field_name: str) -> tuple[str, str]:
    case_id, separator, model = value.partition("::")
    if not separator or not case_id or not model:
        raise ValueError(
            f"evaluation summary {field_name} entries must use '<case_id>::<model>'"
        )
    return case_id, model


def _build_reflection_lm(model_name: str) -> object:
    return dspy.LM(model_name)


def _print_optimization_summary(artifacts: OptimizationArtifacts) -> None:
    print(
        f"optimized candidate={artifacts.best_candidate.candidate_id} run_id={artifacts.run_id} "
        f"invocation={artifacts.invocation_dir} best_candidate={artifacts.best_candidate_path} "
        f"summary={artifacts.summary_path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
