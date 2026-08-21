from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .artifacts import write_json_artifact
from .contracts import Campaign, Candidate, _require_mapping, _require_string

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

# A model-specific override must beat the common bundle by a visible margin, not by
# stochastic noise, so the common-first rollout stays the default.
DEFAULT_MINIMUM_MODEL_IMPROVEMENT = 0.02


@dataclass(frozen=True, slots=True)
class PromptBundle:
    schema_version: int
    bundle_kind: str
    version: str
    model_family: str
    model_digest: str
    candidate_id: str
    aggregate_score: float
    effective_score: float
    pass_at_3: float
    pass_at_5: float
    hard_safety_failures: int
    bundle_dir: Path
    prompt_bundle_path: Path
    evaluation_summary_path: Path


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    published: bool
    reason: str
    effective_score: float
    bundle: PromptBundle | None


def publish_bundle(
    *,
    candidate: Candidate,
    campaign: Campaign,
    model_metadata: Mapping[str, Any],
    evaluation_summary: Mapping[str, Any],
    registry_root: Path | str,
    minimum_model_improvement: float = DEFAULT_MINIMUM_MODEL_IMPROVEMENT,
) -> PromotionDecision:
    if isinstance(minimum_model_improvement, bool) or not isinstance(minimum_model_improvement, (int, float)):
        raise ValueError("minimum_model_improvement must be numeric")  # noqa: TRY004 - preserve validation API
    if float(minimum_model_improvement) < 0.0:
        raise ValueError("minimum_model_improvement must be non-negative")

    model = _normalize_model_metadata(model_metadata)
    summary = _normalize_evaluation_summary(evaluation_summary)
    _require_usable_case_sets(summary["case_sets"], campaign)
    provenance = _extract_evaluation_provenance(evaluation_summary)
    target_model_score = _resolve_target_model_score(evaluation_summary, model["model_family"], summary["aggregate_score"])

    if summary["systemic_failures"]:
        raise RuntimeError("systemic bridge failures must abort publication")

    effective_score = 0.0 if summary["hard_safety_failures"] else target_model_score
    if summary["hard_safety_failures"]:
        return PromotionDecision(
            published=False,
            reason="hard safety failures invalidate the candidate aggregate score",
            effective_score=effective_score,
            bundle=None,
        )

    registry_path = Path(registry_root)
    existing_entries = _load_index_entries(registry_path)
    baseline_entry: Mapping[str, Any] | None = None
    if summary["bundle_kind"] == "model-specific":
        baseline_entry = _find_common_baseline(
            existing_entries,
            campaign_id=campaign.campaign_id,
            model_family=model["model_family"],
            model_digest=model["model_digest"],
        )
        if baseline_entry is None:
            return PromotionDecision(
                published=False,
                reason="common baseline must be published first for model-specific promotion",
                effective_score=effective_score,
                bundle=None,
            )
        if not summary["milestone_passed"]:
            return PromotionDecision(
                published=False,
                reason="model-specific promotion requires a passing milestone evaluation",
                effective_score=effective_score,
                bundle=None,
            )
        improvement = effective_score - float(baseline_entry["effective_score"])
        if improvement <= float(minimum_model_improvement):
            return PromotionDecision(
                published=False,
                reason=(
                    "model-specific promotion requires improvement strictly greater than the configured "
                    "minimum over the common baseline"
                ),
                effective_score=effective_score,
                bundle=None,
            )

    bundle_payload = _build_bundle_payload(
        candidate=candidate,
        campaign=campaign,
        model=model,
        summary=summary,
        effective_score=effective_score,
        baseline_entry=baseline_entry,
        provenance=provenance,
        target_model_score=target_model_score,
    )
    version = _compute_bundle_version(bundle_payload)
    bundle_dir = registry_path / "bundles" / model["model_family"] / version
    prompt_bundle_path = bundle_dir / "prompt-bundle.yaml"
    evaluation_summary_path = bundle_dir / "evaluation-summary.json"

    bundle_payload["version"] = version
    summary_payload = dict(bundle_payload["evaluation"])

    _write_immutable_text(prompt_bundle_path, yaml.safe_dump(bundle_payload, sort_keys=False, allow_unicode=True))
    _write_immutable_json(evaluation_summary_path, summary_payload)

    bundle = PromptBundle(
        schema_version=1,
        bundle_kind=summary["bundle_kind"],
        version=version,
        model_family=model["model_family"],
        model_digest=model["model_digest"],
        candidate_id=candidate.candidate_id,
        aggregate_score=summary["aggregate_score"],
        effective_score=effective_score,
        pass_at_3=summary["pass_at_3"],
        pass_at_5=summary["pass_at_5"],
        hard_safety_failures=summary["hard_safety_failures"],
        bundle_dir=bundle_dir,
        prompt_bundle_path=prompt_bundle_path,
        evaluation_summary_path=evaluation_summary_path,
    )

    updated_entries = _upsert_index_entry(existing_entries, _bundle_index_entry(bundle, candidate, campaign, model))
    write_json_artifact(registry_path / "index.json", {"schema_version": 1, "bundles": updated_entries})
    _write_text_artifact(registry_path / "scoreboard.md", render_scoreboard(updated_entries))

    return PromotionDecision(
        published=True,
        reason="published",
        effective_score=effective_score,
        bundle=bundle,
    )


def render_scoreboard(bundles: Sequence[PromptBundle | Mapping[str, Any]]) -> str:
    rows = sorted((_scoreboard_record(bundle) for bundle in bundles), key=_registry_sort_key)
    lines = [
        "# Prompt Registry Scoreboard",
        "",
        "| Model family | Model digest | Bundle kind | Candidate | Aggregate | pass^3 | pass^5 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model_family} | {model_digest} | {bundle_kind} | {candidate_id} | {aggregate_score:.3f} | {pass_at_3:.3f} | {pass_at_5:.3f} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def _build_bundle_payload(
    *,
    candidate: Candidate,
    campaign: Campaign,
    model: Mapping[str, Any],
    summary: Mapping[str, Any],
    effective_score: float,
    baseline_entry: Mapping[str, Any] | None,
    provenance: Mapping[str, Any],
    target_model_score: float,
) -> dict[str, Any]:
    evaluation_payload = dict(summary)
    evaluation_payload["effective_score"] = effective_score
    evaluation_payload["target_model_score"] = target_model_score
    evaluation_payload.update(provenance)
    if baseline_entry is not None:
        evaluation_payload["common_baseline"] = {
            "version": baseline_entry["version"],
            "effective_score": baseline_entry["effective_score"],
        }

    return {
        "schema_version": 1,
        "bundle_kind": summary["bundle_kind"],
        "version": "",
        "candidate": {
            "schema_version": candidate.schema_version,
            "candidate_id": candidate.candidate_id,
            "fingerprint": candidate.fingerprint,
            "components": _sorted_mapping(candidate.components),
            "metadata": _sorted_mapping(candidate.metadata),
        },
        "campaign": {
            "schema_version": campaign.schema_version,
            "campaign_id": campaign.campaign_id,
            "repetitions": campaign.repetitions,
            "models": list(campaign.models),
            "case_ids": [case.case_id for case in campaign.cases],
            "serving_backend": campaign.serving.backend,
        },
        "model": dict(model),
        "evaluation": evaluation_payload,
    }


def _extract_evaluation_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    summary = _require_mapping(value, "evaluation_summary")
    provenance: dict[str, Any] = {}

    string_fields = ("candidate_id", "candidate_fingerprint", "campaign_id")
    for field_name in string_fields:
        field_value = summary.get(field_name)
        if field_value is not None:
            provenance[field_name] = _require_string(field_value, field_name)

    for field_name in (
        "campaign_case_ids",
        "evaluated_case_ids",
        "evaluated_models",
        "campaign_case_model_pairs",
        "evaluated_case_model_pairs",
    ):
        field_value = summary.get(field_name)
        if field_value is not None:
            provenance[field_name] = _normalize_unordered_string_list(field_value, field_name)

    model_scores = summary.get("model_scores")
    if model_scores is not None:
        provenance["model_scores"] = _normalize_model_scores(model_scores)

    return provenance


def _resolve_target_model_score(summary: Mapping[str, Any], target_model: str, aggregate_score: float) -> float:
    raw_model_scores = summary.get("model_scores")
    if raw_model_scores is None:
        return aggregate_score
    model_scores = _normalize_model_scores(raw_model_scores)
    if target_model not in model_scores:
        raise ValueError("model_scores must include the target model family")
    return model_scores[target_model]


def _sorted_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {key: mapping[key] for key in sorted(mapping)}


def _compute_bundle_version(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"pb-{hashlib.sha256(canonical).hexdigest()[:16]}"


def _normalize_model_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    model = _require_mapping(value, "model_metadata")
    required = {
        "model_family",
        "model_name",
        "model_digest",
        "quantization",
        "context_length",
        "serving_engine",
    }
    missing = sorted(required - set(model))
    if missing:
        raise ValueError(f"model_metadata missing field(s): {', '.join(missing)}")

    model_family = _require_string(model["model_family"], "model_family")
    model_name = _require_string(model["model_name"], "model_name")
    model_digest = _require_string(model["model_digest"], "model_digest")
    if not _DIGEST_PATTERN.fullmatch(model_digest):
        raise ValueError("model_digest must be an exact sha256 digest")

    context_length = model["context_length"]
    if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("context_length must be a positive integer")

    return {
        "model_family": model_family,
        "model_name": model_name,
        "model_digest": model_digest,
        "quantization": _require_string(model["quantization"], "quantization"),
        "context_length": context_length,
        "serving_engine": _require_string(model["serving_engine"], "serving_engine"),
    }


def _normalize_evaluation_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    summary = _require_mapping(value, "evaluation_summary")

    bundle_kind = _require_string(summary.get("bundle_kind"), "bundle_kind")
    if bundle_kind not in {"common", "model-specific"}:
        raise ValueError("bundle_kind must be common or model-specific")

    aggregate_score = _require_float(summary.get("aggregate_score"), "aggregate_score")
    pass_at_3 = _require_pass_hat_k(summary.get("pass_at_3"), "pass_at_3", 3)
    pass_at_5 = _require_pass_hat_k(summary.get("pass_at_5"), "pass_at_5", 5)
    hard_safety_failures = _require_non_negative_int(summary.get("hard_safety_failures"), "hard_safety_failures")
    systemic_failures = _require_non_negative_int(summary.get("systemic_failures"), "systemic_failures")
    milestone_passed = summary.get("milestone_passed")
    if not isinstance(milestone_passed, bool):
        raise ValueError("milestone_passed must be a boolean")  # noqa: TRY004 - preserve validation API

    case_sets = _normalize_case_sets(summary.get("case_sets"))
    artifact_refs = _normalize_unordered_string_list(summary.get("artifact_refs"), "artifact_refs")
    reproduction_command = _normalize_string_list(summary.get("reproduction_command"), "reproduction_command")

    return {
        "bundle_kind": bundle_kind,
        "aggregate_score": aggregate_score,
        "pass_at_3": pass_at_3,
        "pass_at_5": pass_at_5,
        "hard_safety_failures": hard_safety_failures,
        "systemic_failures": systemic_failures,
        "milestone_passed": milestone_passed,
        "case_sets": case_sets,
        "artifact_refs": artifact_refs,
        "reproduction_command": reproduction_command,
    }


def _require_usable_case_sets(case_sets: Mapping[str, Sequence[str]], campaign: Campaign) -> None:
    """Reject vacuous or overlapping train/validation evidence before publication."""
    for split in ("train", "validation"):
        if not case_sets.get(split):
            raise ValueError(f"case_sets must record {split} cases")

    train_case_ids = list(case_sets["train"])
    validation_case_ids = list(case_sets["validation"])
    overlap = sorted(set(train_case_ids) & set(validation_case_ids))
    if overlap:
        raise ValueError(f"case_sets train and validation must be disjoint: {', '.join(overlap)}")

    campaign_case_ids = {case.case_id for case in campaign.cases}
    unknown = sorted(set(train_case_ids + validation_case_ids) - campaign_case_ids)
    if unknown:
        raise ValueError(f"case_sets must be drawn from the campaign cases: {', '.join(unknown)}")


def _normalize_case_sets(value: Any) -> dict[str, list[str]]:
    mapping = _require_mapping(value, "case_sets")
    normalized: dict[str, list[str]] = {}
    for key, entries in sorted(mapping.items()):
        normalized[_require_string(key, "case_sets key")] = _normalize_unordered_string_list(
            entries, f"case_sets[{key}]"
        )
    return normalized


def _normalize_string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list of strings")
    return [_require_string(item, context) for item in value]


def _normalize_unordered_string_list(value: Any, context: str) -> list[str]:
    return sorted(_normalize_string_list(value, context))


def _normalize_model_scores(value: Any) -> dict[str, float]:
    mapping = _require_mapping(value, "model_scores")
    normalized: dict[str, float] = {}
    for model_name, score in sorted(mapping.items()):
        normalized[_require_string(model_name, "model_scores key")] = _require_float(score, "model_scores value")
    if not normalized:
        raise ValueError("model_scores must not be empty")
    return normalized


def _require_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")  # noqa: TRY004 - preserve validation API
    return float(value)


def _require_pass_hat_k(value: Any, context: str, k: int) -> float:
    """Reject publication evidence that never observed k repetitions per case."""
    if value is None:
        raise ValueError(
            f"{context} requires {k} recorded repetitions per case before publication"
        )
    score = _require_float(value, context)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{context} must be between 0.0 and 1.0")
    return score


def _require_non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _load_index_entries(registry_root: Path) -> list[dict[str, Any]]:
    index_path = registry_root / "index.json"
    if not index_path.exists():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    index = _require_mapping(payload, "registry index")
    bundles = index.get("bundles")
    if not isinstance(bundles, list):
        raise ValueError("registry index bundles must be a list")  # noqa: TRY004 - preserve validation API
    return [dict(_require_mapping(entry, "registry bundle entry")) for entry in bundles]


def _find_common_baseline(
    entries: Sequence[Mapping[str, Any]], *, campaign_id: str, model_family: str, model_digest: str
) -> Mapping[str, Any] | None:
    matches = [
        entry
        for entry in entries
        if (
            entry.get("bundle_kind") == "common"
            and entry.get("campaign_id") == campaign_id
            and entry.get("model_family") == model_family
            and entry.get("model_digest") == model_digest
        )
    ]
    if not matches:
        return None
    return max(matches, key=lambda entry: (float(entry["effective_score"]), str(entry["version"])))


def _bundle_index_entry(
    bundle: PromptBundle,
    candidate: Candidate,
    campaign: Campaign,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "version": bundle.version,
        "bundle_kind": bundle.bundle_kind,
        "candidate_id": candidate.candidate_id,
        "candidate_fingerprint": candidate.fingerprint,
        "campaign_id": campaign.campaign_id,
        "model_family": bundle.model_family,
        "model_name": model["model_name"],
        "model_digest": bundle.model_digest,
        "quantization": model["quantization"],
        "context_length": model["context_length"],
        "serving_engine": model["serving_engine"],
        "aggregate_score": bundle.aggregate_score,
        "effective_score": bundle.effective_score,
        "pass_at_3": bundle.pass_at_3,
        "pass_at_5": bundle.pass_at_5,
        "hard_safety_failures": bundle.hard_safety_failures,
        "path": f"bundles/{bundle.model_family}/{bundle.version}",
    }


def _upsert_index_entry(entries: Sequence[Mapping[str, Any]], entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for current in entries:
        merged[(str(current["model_family"]), str(current["version"]))] = dict(current)
    merged[(str(entry["model_family"]), str(entry["version"]))] = dict(entry)
    return [merged[key] for key in sorted(merged, key=lambda item: _registry_sort_key(merged[item]))]


def _registry_sort_key(entry: Mapping[str, Any]) -> tuple[str, str, int, str, str]:
    bundle_kind = str(entry["bundle_kind"])
    bundle_rank = 0 if bundle_kind == "common" else 1
    return (
        str(entry["model_family"]),
        str(entry["model_digest"]),
        bundle_rank,
        str(entry["candidate_id"]),
        str(entry["version"]),
    )


def _scoreboard_record(bundle: PromptBundle | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(bundle, PromptBundle):
        return {
            "model_family": bundle.model_family,
            "model_digest": bundle.model_digest,
            "bundle_kind": bundle.bundle_kind,
            "candidate_id": bundle.candidate_id,
            "aggregate_score": bundle.aggregate_score,
            "pass_at_3": bundle.pass_at_3,
            "pass_at_5": bundle.pass_at_5,
            "version": bundle.version,
        }
    return bundle


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    _write_immutable_text(path, encoded)


def _write_immutable_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise FileExistsError(f"refusing to mutate immutable bundle artifact: {path}")
        return
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def _write_text_artifact(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)
