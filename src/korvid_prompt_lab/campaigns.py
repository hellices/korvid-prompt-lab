from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .contracts import (
    Campaign,
    _require_mapping,
    _require_string,
    _require_unique_string_items,
)

_CANONICAL_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_LIVE_HEX_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class SearchStage:
    name: str
    metric_calls: int
    seeds: tuple[int, ...]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, index: int) -> SearchStage:
        _ensure_exact_keys(mapping, {"name", "metric_calls", "seeds"}, f"stages[{index}]")
        return cls(
            name=_require_string(mapping.get("name"), f"stages[{index}].name"),
            metric_calls=_require_positive_int(
                mapping.get("metric_calls"), f"stages[{index}].metric_calls"
            ),
            seeds=_require_non_negative_int_items(mapping.get("seeds"), f"stages[{index}].seeds"),
        )


@dataclass(frozen=True, slots=True)
class ModelTier:
    name: str
    model: str
    digest: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, index: int) -> ModelTier:
        _ensure_exact_keys(mapping, {"name", "model", "digest"}, f"model_tiers[{index}]")
        digest = _require_canonical_digest(
            mapping.get("digest"), f"model_tiers[{index}].digest"
        )
        return cls(
            name=_require_string(mapping.get("name"), f"model_tiers[{index}].name"),
            model=_require_string(mapping.get("model"), f"model_tiers[{index}].model"),
            digest=digest,
        )


@dataclass(frozen=True, slots=True)
class OptimizationCampaign:
    schema_version: int
    campaign_id: str
    evaluation_campaign: str
    initial_candidate: str
    train_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
    milestone_case_ids: tuple[str, ...]
    stages: tuple[SearchStage, ...]
    model_tiers: tuple[ModelTier, ...]
    total_metric_call_limit: int
    wall_clock_limit_seconds: int
    infrastructure_retry_limit: int
    stagnation_attempt_limit: int
    confirmation_runs: int

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any], *, evaluation_campaign: Campaign
    ) -> OptimizationCampaign:
        _ensure_exact_keys(
            mapping,
            {
                "schema_version",
                "campaign_id",
                "evaluation_campaign",
                "initial_candidate",
                "train_case_ids",
                "validation_case_ids",
                "milestone_case_ids",
                "stages",
                "model_tiers",
                "total_metric_call_limit",
                "wall_clock_limit_seconds",
                "infrastructure_retry_limit",
                "stagnation_attempt_limit",
                "confirmation_runs",
            },
            "optimization campaign",
        )
        if mapping.get("schema_version") != 1:
            raise ValueError("optimization campaign schema_version must be 1")

        evaluation_campaign_id = _require_string(
            mapping.get("evaluation_campaign"), "evaluation_campaign"
        )
        if evaluation_campaign_id != evaluation_campaign.campaign_id:
            raise ValueError("evaluation_campaign must match the evaluation campaign file")

        train_case_ids = _require_unique_string_items(mapping.get("train_case_ids"), "train_case_ids")
        validation_case_ids = _require_unique_string_items(
            mapping.get("validation_case_ids"), "validation_case_ids"
        )
        milestone_case_ids = _require_unique_string_items(
            mapping.get("milestone_case_ids"), "milestone_case_ids"
        )
        _validate_case_sets(
            evaluation_campaign=evaluation_campaign,
            train_case_ids=train_case_ids,
            validation_case_ids=validation_case_ids,
            milestone_case_ids=milestone_case_ids,
        )

        raw_stages = mapping.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("stages must be a non-empty list")
        stages = tuple(
            SearchStage.from_mapping(_require_mapping(item, f"stages[{index}]"), index=index)
            for index, item in enumerate(raw_stages)
        )
        _validate_stages(stages)

        raw_model_tiers = mapping.get("model_tiers")
        if not isinstance(raw_model_tiers, list) or not raw_model_tiers:
            raise ValueError("model_tiers must be a non-empty list")
        model_tiers = tuple(
            ModelTier.from_mapping(_require_mapping(item, f"model_tiers[{index}]"), index=index)
            for index, item in enumerate(raw_model_tiers)
        )
        _validate_model_tiers(model_tiers)

        total_metric_call_limit = _require_positive_int(
            mapping.get("total_metric_call_limit"), "total_metric_call_limit"
        )
        planned_metric_calls = sum(stage.metric_calls * len(stage.seeds) for stage in stages)
        if planned_metric_calls > total_metric_call_limit:
            raise ValueError("total_metric_call_limit must cover every staged search attempt")

        return cls(
            schema_version=1,
            campaign_id=_require_string(mapping.get("campaign_id"), "campaign_id"),
            evaluation_campaign=evaluation_campaign_id,
            initial_candidate=_require_string(mapping.get("initial_candidate"), "initial_candidate"),
            train_case_ids=train_case_ids,
            validation_case_ids=validation_case_ids,
            milestone_case_ids=milestone_case_ids,
            stages=stages,
            model_tiers=model_tiers,
            total_metric_call_limit=total_metric_call_limit,
            wall_clock_limit_seconds=_require_positive_int(
                mapping.get("wall_clock_limit_seconds"), "wall_clock_limit_seconds"
            ),
            infrastructure_retry_limit=_require_positive_int(
                mapping.get("infrastructure_retry_limit"), "infrastructure_retry_limit"
            ),
            stagnation_attempt_limit=_require_positive_int(
                mapping.get("stagnation_attempt_limit"), "stagnation_attempt_limit"
            ),
            confirmation_runs=_require_positive_int(
                mapping.get("confirmation_runs"), "confirmation_runs"
            ),
        )


def _load_yaml(path: Path | str) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_optimization_campaign(path: Path | str, evaluation_campaign: Campaign) -> OptimizationCampaign:
    return OptimizationCampaign.from_mapping(
        _require_mapping(_load_yaml(path), "optimization campaign"),
        evaluation_campaign=evaluation_campaign,
    )


def _http_get_json(url: str) -> Mapping[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return _require_mapping(json.load(response), "model tags probe")


def validate_model_tier_digests(
    campaign: OptimizationCampaign,
    model_endpoint: str,
    *,
    http_get_json: Callable[[str], Mapping[str, Any]] = _http_get_json,
) -> None:
    """Compare canonical manifest digests to live Ollama `/api/tags` digests.

    `load_optimization_campaign()` is intentionally offline: it validates only
    static manifest structure. Callers that already hold a live model endpoint
    must invoke this function before allocating AKS experiment capacity.

    Manifest digests are always canonical `sha256:<64 lowercase hex>`. Live
    `/api/tags` currently returns either bare `64`-hex SHA-256 bytes or the
    same value with the `sha256:` prefix; this function canonicalizes the live
    wire value before comparing it to the manifest.
    """
    payload = _require_mapping(
        http_get_json(f"{model_endpoint.rstrip('/')}/api/tags"), "model tags probe"
    )
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("model tags probe returned an invalid payload")  # noqa: TRY004 - preserve validation API

    digests_by_model: dict[str, list[str]] = {}
    for index, item in enumerate(raw_models):
        model = _require_mapping(item, f"model tags.models[{index}]")
        name = _require_string(model.get("name"), f"model tags.models[{index}].name")
        digest = _canonicalize_live_digest(
            model.get("digest"), f"model tags.models[{index}].digest"
        )
        digests_by_model.setdefault(name, []).append(digest)

    for tier in campaign.model_tiers:
        live_digests = digests_by_model.get(tier.model, [])
        if not live_digests:
            raise ValueError(f"live /api/tags did not advertise model {tier.model}")
        if len(live_digests) != 1:
            raise ValueError(f"live /api/tags advertised duplicate digests for model {tier.model}")
        if live_digests[0] != tier.digest:
            raise ValueError(f"live /api/tags digest mismatch for model {tier.model}")


def _ensure_exact_keys(mapping: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - set(mapping))
    unknown = sorted(set(mapping) - required)
    if missing:
        raise ValueError(f"{context} is missing required field(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _require_canonical_digest(value: Any, context: str) -> str:
    digest = _require_string(value, context)
    if _CANONICAL_DIGEST_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{context} must be sha256:<64 lowercase hex>")
    return digest


def _canonicalize_live_digest(value: Any, context: str) -> str:
    digest = _require_string(value, context)
    if _CANONICAL_DIGEST_PATTERN.fullmatch(digest) is not None:
        return digest
    if _LIVE_HEX_DIGEST_PATTERN.fullmatch(digest) is not None:
        return f"sha256:{digest}"
    raise ValueError(
        f"{context} must be sha256:<64 lowercase hex> or 64 lowercase hex bytes"
    )


def _require_positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _require_non_negative_int_items(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of integers")  # noqa: TRY004 - preserve validation API
    if not value:
        raise ValueError(f"{context} must not be empty")
    items: list[int] = []
    seen: set[int] = set()
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{context}[{index}] must be a non-negative integer")
        if item in seen:
            raise ValueError(f"{context} contains duplicate values")
        seen.add(item)
        items.append(item)
    return tuple(items)


def _validate_case_sets(
    *,
    evaluation_campaign: Campaign,
    train_case_ids: tuple[str, ...],
    validation_case_ids: tuple[str, ...],
    milestone_case_ids: tuple[str, ...],
) -> None:
    overlaps = (
        set(train_case_ids) & set(validation_case_ids)
        or set(train_case_ids) & set(milestone_case_ids)
        or set(validation_case_ids) & set(milestone_case_ids)
    )
    if overlaps:
        raise ValueError("train_case_ids, validation_case_ids, and milestone_case_ids must be pairwise disjoint")

    evaluation_case_ids = {case.case_id for case in evaluation_campaign.cases}
    configured_case_ids = set(train_case_ids) | set(validation_case_ids) | set(milestone_case_ids)
    unknown_case_ids = sorted(configured_case_ids - evaluation_case_ids)
    if unknown_case_ids:
        raise ValueError(f"unknown case_id value(s): {', '.join(unknown_case_ids)}")
    missing_case_ids = sorted(evaluation_case_ids - configured_case_ids)
    if missing_case_ids:
        raise ValueError("train/validation/milestone case sets must cover the evaluation campaign exactly")


def _validate_stages(stages: tuple[SearchStage, ...]) -> None:
    seen_names: set[str] = set()
    seen_seeds: set[int] = set()
    for stage in stages:
        if stage.name in seen_names:
            raise ValueError(f"stages contain duplicate stage name {stage.name}")
        seen_names.add(stage.name)
        duplicate_seeds = sorted(set(stage.seeds) & seen_seeds)
        if duplicate_seeds:
            raise ValueError(f"stages contain duplicate seed values: {', '.join(str(seed) for seed in duplicate_seeds)}")
        seen_seeds.update(stage.seeds)


def _validate_model_tiers(model_tiers: tuple[ModelTier, ...]) -> None:
    seen_names: set[str] = set()
    seen_models: set[str] = set()
    seen_digests: set[str] = set()
    for tier in model_tiers:
        if tier.name in seen_names:
            raise ValueError(f"model_tiers contain duplicate tier name {tier.name}")
        if tier.model in seen_models:
            raise ValueError(f"model_tiers contain duplicate model {tier.model}")
        if tier.digest in seen_digests:
            raise ValueError(f"model_tiers contain duplicate digest {tier.digest}")
        seen_names.add(tier.name)
        seen_models.add(tier.model)
        seen_digests.add(tier.digest)


# ---------------------------------------------------------------------------
# Task 3: Deterministic Campaign State Machine
# ---------------------------------------------------------------------------


class CampaignStatus(StrEnum):
    RUNNING = "running"
    QUALIFIED = "qualified"
    NOT_CONVERGED = "not_converged"
    SYSTEM_ERROR = "system_error"


class ActionKind(StrEnum):
    SEARCH = "search"
    MILESTONE = "milestone"
    CONFIRM = "confirm"


@dataclass(frozen=True, slots=True)
class CampaignScore:
    fingerprint: str
    aggregate: float
    hard_safety_failures: int
    core_regression: bool
    pass_at_3: float = 1.0
    pass_at_5: float = 1.0


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    kind: str  # "evidence" | "system_error" | "config_error"
    score: CampaignScore | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class CampaignAction:
    action_id: str
    kind: ActionKind
    expected_state_hash: str
    stage_index: int = 0
    seed_index: int = 0
    tier_index: int = 0
    metric_calls: int = 0


@dataclass(frozen=True, slots=True)
class TierResult:
    tier_index: int
    champion_fingerprint: str
    champion_score: CampaignScore
    status: CampaignStatus


@dataclass(frozen=True, slots=True)
class CampaignState:
    schema_version: int
    campaign_id: str
    prompt_lab_revision: str
    korvid_revision: str
    status: CampaignStatus
    tier_index: int
    stage_index: int
    seed_index: int
    champion_fingerprint: str
    champion_score: CampaignScore
    metric_calls_used: int
    elapsed_seconds: float
    stagnation_attempts: int
    retries_used: int
    started_at: str
    pending_action_id: str | None = None
    milestone_passed: bool = False
    confirmations_passed: int = 0
    stop_reason: str | None = None
    tier_results: tuple[TierResult, ...] = ()


def _score_rank_key_no_fp(score: CampaignScore) -> tuple[Any, ...]:
    """Core ranking dimensions without fingerprint tie-break.

    Order: systemic zero, core regression, hard-safety presence,
    hard-safety count, aggregate (neg), pass@3 (neg), pass@5 (neg).
    """
    is_systemic_zero = score.aggregate == 0.0 and score.hard_safety_failures > 0
    return (
        is_systemic_zero,
        score.core_regression,
        score.hard_safety_failures > 0,
        score.hard_safety_failures,
        -score.aggregate,
        -score.pass_at_3,
        -score.pass_at_5,
    )


def _is_strictly_better(candidate: CampaignScore, champion: CampaignScore) -> bool:
    """Candidate must be strictly better on core dimensions to promote.

    Equal core scores never promote — fingerprint tie-break only applies
    among candidates that already have at least one core improvement.
    """
    return _score_rank_key_no_fp(candidate) < _score_rank_key_no_fp(champion)


def state_hash(state: CampaignState) -> str:
    """Deterministic SHA-256 hash of campaign state."""
    tier_results_serial = [
        {
            "tier_index": tr.tier_index,
            "champion_fingerprint": tr.champion_fingerprint,
            "status": tr.status.value,
            "score_aggregate": tr.champion_score.aggregate,
            "score_hard": tr.champion_score.hard_safety_failures,
            "score_core_reg": tr.champion_score.core_regression,
            "score_p3": tr.champion_score.pass_at_3,
            "score_p5": tr.champion_score.pass_at_5,
        }
        for tr in state.tier_results
    ]
    mapping: dict[str, Any] = {
        "schema_version": state.schema_version,
        "campaign_id": state.campaign_id,
        "prompt_lab_revision": state.prompt_lab_revision,
        "korvid_revision": state.korvid_revision,
        "status": state.status.value,
        "tier_index": state.tier_index,
        "stage_index": state.stage_index,
        "seed_index": state.seed_index,
        "champion_fingerprint": state.champion_fingerprint,
        "champion_score_fingerprint": state.champion_score.fingerprint,
        "champion_score_aggregate": state.champion_score.aggregate,
        "champion_score_hard_safety_failures": state.champion_score.hard_safety_failures,
        "champion_score_core_regression": state.champion_score.core_regression,
        "champion_score_pass_at_3": state.champion_score.pass_at_3,
        "champion_score_pass_at_5": state.champion_score.pass_at_5,
        "metric_calls_used": state.metric_calls_used,
        "elapsed_seconds": state.elapsed_seconds,
        "stagnation_attempts": state.stagnation_attempts,
        "retries_used": state.retries_used,
        "started_at": state.started_at,
        "pending_action_id": state.pending_action_id,
        "milestone_passed": state.milestone_passed,
        "confirmations_passed": state.confirmations_passed,
        "stop_reason": state.stop_reason,
        "tier_results": tier_results_serial,
    }
    serialized = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    return f"sha256:{digest}"


def initial_state(
    control: OptimizationCampaign,
    prompt_lab_revision: str,
    korvid_revision: str,
    started_at: datetime,
) -> CampaignState:
    """Create the initial campaign state."""
    initial_score = CampaignScore(
        fingerprint=control.initial_candidate,
        aggregate=0.0,
        hard_safety_failures=0,
        core_regression=False,
        pass_at_3=0.0,
        pass_at_5=0.0,
    )
    return CampaignState(
        schema_version=1,
        campaign_id=control.campaign_id,
        prompt_lab_revision=prompt_lab_revision,
        korvid_revision=korvid_revision,
        status=CampaignStatus.RUNNING,
        tier_index=0,
        stage_index=0,
        seed_index=0,
        champion_fingerprint=control.initial_candidate,
        champion_score=initial_score,
        metric_calls_used=0,
        elapsed_seconds=0.0,
        stagnation_attempts=0,
        retries_used=0,
        started_at=started_at.isoformat(),
    )


def _is_terminal(state: CampaignState) -> bool:
    return state.status in (
        CampaignStatus.QUALIFIED,
        CampaignStatus.NOT_CONVERGED,
        CampaignStatus.SYSTEM_ERROR,
    )


def _seeds_exhausted(control: OptimizationCampaign, state: CampaignState) -> bool:
    """All stage seeds have been consumed."""
    if state.stage_index >= len(control.stages):
        return True
    if state.stage_index == len(control.stages) - 1:
        return state.seed_index >= len(control.stages[state.stage_index].seeds)
    return False


def _budget_exceeded(control: OptimizationCampaign, state: CampaignState) -> bool:
    if state.metric_calls_used >= control.total_metric_call_limit:
        return True
    if state.elapsed_seconds > control.wall_clock_limit_seconds:
        return True
    return state.stagnation_attempts >= control.stagnation_attempt_limit


def next_action(
    control: OptimizationCampaign,
    state: CampaignState,
    now: datetime,
) -> CampaignAction | None:
    """Determine next action or None if terminal/budget-exceeded."""
    if _is_terminal(state):
        return None

    if _budget_exceeded(control, state):
        return None

    sh = state_hash(state)
    action_id = str(uuid.uuid5(uuid.NAMESPACE_URL, sh))

    # If stages not exhausted, SEARCH
    if not _seeds_exhausted(control, state):
        stage = control.stages[state.stage_index]
        return CampaignAction(
            action_id=action_id,
            kind=ActionKind.SEARCH,
            expected_state_hash=sh,
            stage_index=state.stage_index,
            seed_index=state.seed_index,
            tier_index=state.tier_index,
            metric_calls=stage.metric_calls,
        )

    # Stages exhausted: milestone then confirm(s)
    if not state.milestone_passed:
        return CampaignAction(
            action_id=action_id,
            kind=ActionKind.MILESTONE,
            expected_state_hash=sh,
            tier_index=state.tier_index,
            metric_calls=0,
        )

    if state.confirmations_passed < control.confirmation_runs:
        return CampaignAction(
            action_id=action_id,
            kind=ActionKind.CONFIRM,
            expected_state_hash=sh,
            tier_index=state.tier_index,
            metric_calls=0,
        )

    return None


def _validate_action(
    control: OptimizationCampaign,
    state: CampaignState,
    action: CampaignAction,
) -> None:
    """Validate action matches the controller-planned next action exactly."""
    current_hash = state_hash(state)
    if action.expected_state_hash != current_hash:
        raise ValueError(
            f"stale action: expected state_hash {action.expected_state_hash}, "
            f"got {current_hash}"
        )

    # Compute what the planned action should be
    planned = next_action(control, state, datetime.fromisoformat(state.started_at))
    if planned is None:
        raise ValueError("no valid action for current state")

    if action.action_id != planned.action_id:
        raise ValueError(
            f"invalid action_id: expected {planned.action_id}, got {action.action_id}"
        )
    if action.kind != planned.kind:
        raise ValueError(
            f"invalid action kind: expected {planned.kind}, got {action.kind}"
        )
    if action.tier_index != planned.tier_index:
        raise ValueError("action tier_index mismatch")
    if action.stage_index != planned.stage_index:
        raise ValueError("action stage_index mismatch")
    if action.seed_index != planned.seed_index:
        raise ValueError("action seed_index mismatch")
    if action.metric_calls != planned.metric_calls:
        raise ValueError("action metric_calls mismatch")


def advance_state(
    control: OptimizationCampaign,
    state: CampaignState,
    action: CampaignAction,
    outcome: AttemptOutcome,
    now: datetime,
) -> CampaignState:
    """Apply action+outcome to state, returning new immutable state."""
    _validate_action(control, state, action)

    started = datetime.fromisoformat(state.started_at)
    elapsed = (now - started).total_seconds()

    # --- CONFIG_ERROR outcome: terminal ---
    if outcome.kind == "config_error":
        return CampaignState(
            schema_version=state.schema_version,
            campaign_id=state.campaign_id,
            prompt_lab_revision=state.prompt_lab_revision,
            korvid_revision=state.korvid_revision,
            status=CampaignStatus.SYSTEM_ERROR,
            tier_index=state.tier_index,
            stage_index=state.stage_index,
            seed_index=state.seed_index,
            champion_fingerprint=state.champion_fingerprint,
            champion_score=state.champion_score,
            metric_calls_used=state.metric_calls_used,
            elapsed_seconds=elapsed,
            stagnation_attempts=state.stagnation_attempts,
            retries_used=state.retries_used,
            started_at=state.started_at,
            milestone_passed=state.milestone_passed,
            confirmations_passed=state.confirmations_passed,
            stop_reason=f"config_error: {outcome.error_message or 'unknown'}",
            tier_results=state.tier_results,
        )

    # --- SYSTEM_ERROR outcome ---
    if outcome.kind == "system_error":
        new_retries = state.retries_used + 1
        # Check if wall-clock exceeded
        wall_clock_exceeded = elapsed > control.wall_clock_limit_seconds
        if new_retries >= control.infrastructure_retry_limit or wall_clock_exceeded:
            stop = "wall_clock_limit_exceeded" if wall_clock_exceeded else "infrastructure_retry_limit_exhausted"
            return CampaignState(
                schema_version=state.schema_version,
                campaign_id=state.campaign_id,
                prompt_lab_revision=state.prompt_lab_revision,
                korvid_revision=state.korvid_revision,
                status=CampaignStatus.SYSTEM_ERROR,
                tier_index=state.tier_index,
                stage_index=state.stage_index,
                seed_index=state.seed_index,
                champion_fingerprint=state.champion_fingerprint,
                champion_score=state.champion_score,
                metric_calls_used=state.metric_calls_used,
                elapsed_seconds=elapsed,
                stagnation_attempts=state.stagnation_attempts,
                retries_used=new_retries,
                started_at=state.started_at,
                milestone_passed=state.milestone_passed,
                confirmations_passed=state.confirmations_passed,
                stop_reason=stop,
                tier_results=state.tier_results,
            )
        return CampaignState(
            schema_version=state.schema_version,
            campaign_id=state.campaign_id,
            prompt_lab_revision=state.prompt_lab_revision,
            korvid_revision=state.korvid_revision,
            status=CampaignStatus.RUNNING,
            tier_index=state.tier_index,
            stage_index=state.stage_index,
            seed_index=state.seed_index,
            champion_fingerprint=state.champion_fingerprint,
            champion_score=state.champion_score,
            metric_calls_used=state.metric_calls_used,
            elapsed_seconds=elapsed,
            stagnation_attempts=state.stagnation_attempts,
            retries_used=new_retries,
            started_at=state.started_at,
            milestone_passed=state.milestone_passed,
            confirmations_passed=state.confirmations_passed,
            tier_results=state.tier_results,
        )

    # --- EVIDENCE outcome ---
    if outcome.kind != "evidence" or outcome.score is None:
        raise ValueError(f"unexpected outcome kind: {outcome.kind}")

    if action.kind is ActionKind.SEARCH:
        new_metric_calls = state.metric_calls_used + action.metric_calls
        candidate_score = outcome.score

        # Promotion: must be different fingerprint AND strictly better on core dims
        if (
            candidate_score.fingerprint != state.champion_fingerprint
            and _is_strictly_better(candidate_score, state.champion_score)
        ):
            new_champion_fp = candidate_score.fingerprint
            new_champion_score = candidate_score
            new_stagnation = 0
        else:
            new_champion_fp = state.champion_fingerprint
            new_champion_score = state.champion_score
            new_stagnation = state.stagnation_attempts + 1

        # Advance seed cursor
        new_stage_index = state.stage_index
        new_seed_index = state.seed_index + 1
        if new_stage_index < len(control.stages) and new_seed_index >= len(
            control.stages[new_stage_index].seeds
        ):
            new_stage_index += 1
            new_seed_index = 0

        # Determine status
        new_status = CampaignStatus.RUNNING
        stop_reason: str | None = None
        if new_metric_calls >= control.total_metric_call_limit:
            new_status = CampaignStatus.NOT_CONVERGED
            stop_reason = "total_metric_call_limit"
        elif elapsed > control.wall_clock_limit_seconds:
            new_status = CampaignStatus.NOT_CONVERGED
            stop_reason = "wall_clock_limit_exceeded"
        elif new_stagnation >= control.stagnation_attempt_limit:
            new_status = CampaignStatus.NOT_CONVERGED
            stop_reason = "stagnation_limit"

        return CampaignState(
            schema_version=state.schema_version,
            campaign_id=state.campaign_id,
            prompt_lab_revision=state.prompt_lab_revision,
            korvid_revision=state.korvid_revision,
            status=new_status,
            tier_index=state.tier_index,
            stage_index=new_stage_index,
            seed_index=new_seed_index,
            champion_fingerprint=new_champion_fp,
            champion_score=new_champion_score,
            metric_calls_used=new_metric_calls,
            elapsed_seconds=elapsed,
            stagnation_attempts=new_stagnation,
            retries_used=state.retries_used,
            started_at=state.started_at,
            milestone_passed=state.milestone_passed,
            confirmations_passed=state.confirmations_passed,
            stop_reason=stop_reason,
            tier_results=state.tier_results,
        )

    if action.kind is ActionKind.MILESTONE:
        return CampaignState(
            schema_version=state.schema_version,
            campaign_id=state.campaign_id,
            prompt_lab_revision=state.prompt_lab_revision,
            korvid_revision=state.korvid_revision,
            status=CampaignStatus.RUNNING,
            tier_index=state.tier_index,
            stage_index=state.stage_index,
            seed_index=state.seed_index,
            champion_fingerprint=state.champion_fingerprint,
            champion_score=state.champion_score,
            metric_calls_used=state.metric_calls_used,
            elapsed_seconds=elapsed,
            stagnation_attempts=state.stagnation_attempts,
            retries_used=state.retries_used,
            started_at=state.started_at,
            milestone_passed=True,
            confirmations_passed=state.confirmations_passed,
            tier_results=state.tier_results,
        )

    if action.kind is ActionKind.CONFIRM:
        # Confirmation: score fingerprint must match champion
        if outcome.score.fingerprint != state.champion_fingerprint:
            # Confirmation failed — NOT_CONVERGED
            return CampaignState(
                schema_version=state.schema_version,
                campaign_id=state.campaign_id,
                prompt_lab_revision=state.prompt_lab_revision,
                korvid_revision=state.korvid_revision,
                status=CampaignStatus.NOT_CONVERGED,
                tier_index=state.tier_index,
                stage_index=state.stage_index,
                seed_index=state.seed_index,
                champion_fingerprint=state.champion_fingerprint,
                champion_score=state.champion_score,
                metric_calls_used=state.metric_calls_used,
                elapsed_seconds=elapsed,
                stagnation_attempts=state.stagnation_attempts,
                retries_used=state.retries_used,
                started_at=state.started_at,
                milestone_passed=state.milestone_passed,
                confirmations_passed=state.confirmations_passed,
                stop_reason="confirmation_failed",
                tier_results=state.tier_results,
            )

        new_confirmations = state.confirmations_passed + 1
        if new_confirmations >= control.confirmation_runs:
            # All confirmations passed — check tier rollover
            if state.tier_index < len(control.model_tiers) - 1:
                # Tier qualified — record result, roll over to next tier
                tier_result = TierResult(
                    tier_index=state.tier_index,
                    champion_fingerprint=state.champion_fingerprint,
                    champion_score=state.champion_score,
                    status=CampaignStatus.QUALIFIED,
                )
                next_tier = state.tier_index + 1
                fresh_score = CampaignScore(
                    fingerprint=control.initial_candidate,
                    aggregate=0.0,
                    hard_safety_failures=0,
                    core_regression=False,
                    pass_at_3=0.0,
                    pass_at_5=0.0,
                )
                return CampaignState(
                    schema_version=state.schema_version,
                    campaign_id=state.campaign_id,
                    prompt_lab_revision=state.prompt_lab_revision,
                    korvid_revision=state.korvid_revision,
                    status=CampaignStatus.RUNNING,
                    tier_index=next_tier,
                    stage_index=0,
                    seed_index=0,
                    champion_fingerprint=control.initial_candidate,
                    champion_score=fresh_score,
                    metric_calls_used=state.metric_calls_used,
                    elapsed_seconds=elapsed,
                    stagnation_attempts=0,
                    retries_used=0,
                    started_at=state.started_at,
                    milestone_passed=False,
                    confirmations_passed=0,
                    tier_results=state.tier_results + (tier_result,),
                )
            # Final tier qualified
            return CampaignState(
                schema_version=state.schema_version,
                campaign_id=state.campaign_id,
                prompt_lab_revision=state.prompt_lab_revision,
                korvid_revision=state.korvid_revision,
                status=CampaignStatus.QUALIFIED,
                tier_index=state.tier_index,
                stage_index=state.stage_index,
                seed_index=state.seed_index,
                champion_fingerprint=state.champion_fingerprint,
                champion_score=state.champion_score,
                metric_calls_used=state.metric_calls_used,
                elapsed_seconds=elapsed,
                stagnation_attempts=state.stagnation_attempts,
                retries_used=state.retries_used,
                started_at=state.started_at,
                milestone_passed=state.milestone_passed,
                confirmations_passed=new_confirmations,
                tier_results=state.tier_results,
            )

        # More confirmations needed
        return CampaignState(
            schema_version=state.schema_version,
            campaign_id=state.campaign_id,
            prompt_lab_revision=state.prompt_lab_revision,
            korvid_revision=state.korvid_revision,
            status=CampaignStatus.RUNNING,
            tier_index=state.tier_index,
            stage_index=state.stage_index,
            seed_index=state.seed_index,
            champion_fingerprint=state.champion_fingerprint,
            champion_score=state.champion_score,
            metric_calls_used=state.metric_calls_used,
            elapsed_seconds=elapsed,
            stagnation_attempts=state.stagnation_attempts,
            retries_used=state.retries_used,
            started_at=state.started_at,
            milestone_passed=state.milestone_passed,
            confirmations_passed=new_confirmations,
            tier_results=state.tier_results,
        )

    raise ValueError(f"unknown action kind: {action.kind}")  # pragma: no cover
