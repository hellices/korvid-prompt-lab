"""Bridge unchanged Korvid eval sources into Prompt Lab campaigns."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import write_json_artifact
from .contracts import Campaign, Candidate, EvalCase, KorvidUpstreamServing
from .scoring import EvaluationResult
from .source_runtime import run_upstream_request
from .upstream_contract import (
    KORVID_REVISION,
    KORVID_VERSION,
    SourceCase,
    load_source_cases,
    tier_pack_from_candidate,
)

UPSTREAM_PROTOCOL_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _model_payload(
    model: str, serving: KorvidUpstreamServing, seed: int
) -> dict[str, Any]:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("upstream seed must be a non-negative integer")
    return {
        "reference": model,
        "endpoint": serving.base_url,
        "options": {"temperature": 0.0, **serving.model_options, "seed": seed},
    }


def _reference(case: EvalCase) -> str:
    if case.template_id == "korvid-scenario":
        return f"scenarios/{case.case_id}"
    if case.template_id == "korvid-journey":
        return f"journeys/{case.case_id}"
    raise ValueError("upstream case template must identify a Korvid scenario or journey")


def _source_catalog(campaign: Campaign) -> dict[str, SourceCase]:
    serving = campaign.serving
    if not isinstance(serving, KorvidUpstreamServing):
        raise ValueError("upstream runner requires korvid_upstream serving")  # noqa: TRY004
    references = tuple(_reference(case) for case in campaign.cases)
    cases = load_source_cases(Path(serving.source_root), references)
    return {case.case_id: case for case in cases}


def inspect_upstream(
    serving: KorvidUpstreamServing,
    model: str,
    references: Sequence[str],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    if not isinstance(serving, KorvidUpstreamServing) or serving.backend != "korvid_upstream":
        raise ValueError("inspection requires korvid_upstream serving")
    if serving.korvid_revision != KORVID_REVISION:
        raise ValueError("upstream Korvid revision must be the reviewed v0.4.1 commit")
    catalog = load_source_cases(Path(serving.source_root), references)
    result = run_upstream_request(
        serving,
        {
            "protocol_version": UPSTREAM_PROTOCOL_VERSION,
            "operation": "inspect",
            "references": list(references),
            "model": _model_payload(model, serving, seed),
        },
    )
    if result.get("protocol_version") != UPSTREAM_PROTOCOL_VERSION:
        raise ValueError("upstream inspection protocol mismatch")
    prompt = result.get("prompt")
    runtime = result.get("runtime")
    if (
        not isinstance(prompt, dict)
        or not isinstance(prompt.get("text"), str)
        or not prompt["text"]
        or prompt.get("baseline_equivalent") is not True
        or not isinstance(runtime, dict)
        or _SHA256.fullmatch(str(runtime.get("fingerprint"))) is None
        or not isinstance(runtime.get("dependencies"), dict)
    ):
        raise ValueError("upstream inspection did not attest the copied baseline prompt")
    contexts = prompt.get("authored_contexts")
    if not isinstance(contexts, list) or not contexts or any(
        not isinstance(item, dict) or item.get("equivalent") is not True
        for item in contexts
    ):
        raise ValueError("copied upstream pack is not equivalent in source-authored contexts")
    if result.get("model") != _model_payload(model, serving, seed):
        raise ValueError("upstream inspection model identity mismatch")
    return {
        **result,
        "cases": [asdict(case) for case in catalog],
        "source_identity": {
            "korvid_version": KORVID_VERSION,
            "korvid_revision": KORVID_REVISION,
        },
    }


def validate_upstream_candidate(
    serving: KorvidUpstreamServing,
    model: str,
    candidate: Candidate,
) -> str | None:
    """Return a candidate-only rejection label from Korvid, without inference."""
    if serving.backend != "korvid_upstream" or serving.korvid_revision != KORVID_REVISION:
        raise ValueError("candidate validation requires the reviewed upstream serving")
    tier_pack = tier_pack_from_candidate(candidate)
    identity = {
        "protocol_version": UPSTREAM_PROTOCOL_VERSION,
        "operation": "validate_candidate",
        "model": _model_payload(model, serving, 0),
    }
    result = run_upstream_request(serving, {**identity, "tier_pack": tier_pack})
    expected = {
        **identity,
        "tier_pack_sha256": hashlib.sha256(tier_pack.encode()).hexdigest(),
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("upstream candidate validation identity mismatch")
    if result.get("valid") is True and result.get("error_label") is None:
        return None
    if result.get("valid") is False and result.get("error_label") == "static_prompt_too_large":
        return "static_prompt_too_large"
    raise ValueError("upstream candidate validation returned an invalid verdict")


def _finite_nonnegative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"upstream worker {label} must be finite and non-negative")
    return float(value)


def _valid_call_trace(call: Any) -> bool:
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return False
    arguments_valid = call.get("arguments_valid", True)
    if type(arguments_valid) is not bool:
        return False
    if arguments_valid:
        return isinstance(call.get("arguments"), dict)
    return (
        "arguments" not in call
        and isinstance(call.get("arguments_raw"), str)
        and len(call["arguments_raw"]) <= 512
        and call.get("error_label")
        in {"arguments_not_string", "malformed_arguments", "non_object_arguments"}
    )


def _validate_result(
    result: Mapping[str, Any],
    *,
    authored: SourceCase,
    tier_pack: str,
    model: dict[str, Any],
    prompt_path: Path | None,
) -> tuple[bool, tuple[str, ...], dict[str, Any], dict[str, Any]]:
    expected = {
        "protocol_version": UPSTREAM_PROTOCOL_VERSION,
        "operation": "evaluate",
        "reference": authored.reference,
        "case_id": authored.case_id,
        "kind": authored.kind,
        "source_path": authored.source_path,
        "source_sha256": authored.source_sha256,
        "model": model,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"upstream worker {key} mismatch")
    if result.get("questions") != list(authored.questions):
        raise ValueError("upstream worker changed the original user turns")
    if result.get("tier_pack_sha256") != hashlib.sha256(tier_pack.encode()).hexdigest():
        raise ValueError("upstream worker did not apply the candidate tier pack")
    if result.get("prompt_path_verified") is not (prompt_path is not None):
        raise ValueError("upstream worker prompt reload verification mismatch")
    if type(result.get("success")) is not bool:
        raise ValueError("upstream worker success verdict must be boolean")
    upstream = result.get("upstream")
    usage = result.get("usage")
    runtime = result.get("runtime")
    policy = result.get("policy")
    prompt = result.get("prompt")
    if (
        not isinstance(policy, dict)
        or policy.get("tier") != "low"
        or policy.get("prompt_pack") != "low-korvid-operator"
        or not isinstance(prompt, dict)
        or prompt.get("pack") != "low-korvid-operator"
    ):
        raise ValueError("upstream worker did not use the resolved low prompt pack")
    if not isinstance(upstream, dict) or not isinstance(usage, dict):
        raise ValueError("upstream worker omitted original grade or usage")  # noqa: TRY004
    if authored.kind == "scenario":
        outcome = upstream.get("outcome")
        if outcome not in {"success", "failure", "error"} or result["success"] is not (
            outcome == "success"
        ):
            raise ValueError("upstream success verdict disagrees with the original outcome")
    elif (
        type(upstream.get("success")) is not bool
        or result["success"] is not upstream["success"]
        or not isinstance(upstream.get("turns"), list)
        or not upstream["turns"]
    ):
        raise ValueError("upstream success verdict disagrees with the original journey")
    if (
        not isinstance(runtime, dict)
        or not isinstance(runtime.get("dependencies"), dict)
        or _SHA256.fullmatch(str(runtime.get("fingerprint"))) is None
    ):
        raise ValueError("upstream worker omitted runtime identity")
    for key in ("iterations", "tool_calls"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"upstream worker usage {key} must be a non-negative integer")
    diagnostic_calls = usage.get("diagnostic_calls")
    if diagnostic_calls is not None and (
        isinstance(diagnostic_calls, bool)
        or not isinstance(diagnostic_calls, int)
        or diagnostic_calls < 0
    ):
        raise ValueError(
            "upstream worker usage diagnostic_calls must be a non-negative integer"
        )
    _finite_nonnegative(usage.get("wall_time_seconds"), "wall time")
    raw_failures = result.get("hard_failures")
    if not isinstance(raw_failures, list) or any(
        failure != "upstream_safety_violation" for failure in raw_failures
    ):
        raise ValueError("upstream worker returned an unknown hard failure")
    return bool(result["success"]), tuple(raw_failures), upstream, runtime


def _source_report_turns(
    value: Any,
    authored: SourceCase,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(value, dict):
        raise TypeError("upstream worker omitted the original source report")
    report_id = value.get("scenario" if authored.kind == "scenario" else "journey")
    interaction = value.get("interaction")
    runs = value.get("runs")
    if (
        report_id != authored.case_id
        or "interaction" not in value
        or not isinstance(interaction, dict | None)
        or not isinstance(runs, list)
        or len(runs) != 1
        or not isinstance(runs[0], dict)
    ):
        raise ValueError("upstream worker returned malformed source report evidence")
    run = runs[0]
    if authored.kind == "scenario":
        source_turns = runs
    else:
        raw_turns = run.get("turns")
        if not isinstance(raw_turns, list) or len(raw_turns) != len(authored.questions):
            raise ValueError("upstream journey report changed the original turn count")
        source_turns = raw_turns
    if any(
        not isinstance(turn, dict)
        or not isinstance(turn.get("answer"), str)
        or turn.get("outcome") not in {"success", "failure", "error"}
        or not isinstance(turn.get("grade"), dict)
        for turn in source_turns
    ):
        raise ValueError("upstream source report omitted graded answer evidence")
    if authored.kind == "journey" and any(
        "interaction" not in turn
        or "final_interaction" not in turn
        or not isinstance(turn.get("interaction"), dict | None)
        or not isinstance(turn.get("final_interaction"), dict | None)
        for turn in source_turns
    ):
        raise ValueError("upstream journey report omitted interaction snapshots")
    return value, source_turns


@dataclass(frozen=True, slots=True)
class KorvidUpstreamRunner:
    campaign: Campaign
    prompt_path: Path | None = None

    def __post_init__(self) -> None:
        serving = self.campaign.serving
        if not isinstance(serving, KorvidUpstreamServing) or serving.backend != "korvid_upstream":
            raise ValueError("upstream runner requires korvid_upstream serving")
        if serving.korvid_revision != KORVID_REVISION:
            raise ValueError("upstream runner requires the reviewed v0.4.1 revision")
        if len(self.campaign.models) != 1 or any(
            case.models != self.campaign.models for case in self.campaign.cases
        ):
            raise ValueError("upstream cases must target the campaign's single model")
        catalog = _source_catalog(self.campaign)
        for case in self.campaign.cases:
            authored = catalog.get(case.case_id)
            if authored is None or case != authored.eval_case(self.campaign.models[0]):
                raise ValueError("campaign case differs from its original Korvid source")
        if self.prompt_path is not None:
            object.__setattr__(self, "prompt_path", self.prompt_path.resolve(strict=True))

    def run(
        self,
        candidate: Candidate,
        case: EvalCase,
        run_dir: Path | str,
        *,
        repetition: int = 1,
        seed: int = 0,
    ) -> EvaluationResult:
        if (
            isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or not 1 <= repetition <= self.campaign.repetitions
        ):
            raise ValueError("invalid upstream repetition")
        if case not in self.campaign.cases:
            raise ValueError("upstream case is not the declared campaign source")
        serving = self.campaign.serving
        if not isinstance(serving, KorvidUpstreamServing):
            raise ValueError("upstream runner requires korvid_upstream serving")  # noqa: TRY004
        authored = _source_catalog(self.campaign).get(case.case_id)
        if authored is None or case != authored.eval_case(self.campaign.models[0]):
            raise ValueError("upstream case differs from its original source")
        tier_pack = tier_pack_from_candidate(candidate)
        model = _model_payload(case.models[0], serving, seed)
        payload: dict[str, Any] = {
            "protocol_version": UPSTREAM_PROTOCOL_VERSION,
            "operation": "evaluate",
            "reference": authored.reference,
            "source_sha256": authored.source_sha256,
            "tier_pack": tier_pack,
            "model": model,
        }
        if self.prompt_path is not None:
            if self.prompt_path.read_text(encoding="utf-8") != tier_pack:
                raise ValueError("exported optimized prompt differs from the candidate")
            payload["prompt_path"] = str(self.prompt_path)
        target = Path(run_dir)
        response_path = target / "response.json"
        if response_path.exists():
            raise FileExistsError(f"upstream response already exists: {response_path}")
        raw = run_upstream_request(serving, payload)
        success, hard_failures, upstream, runtime = _validate_result(
            raw,
            authored=authored,
            tier_pack=tier_pack,
            model=model,
            prompt_path=self.prompt_path,
        )
        source_report, source_turns = _source_report_turns(
            raw.get("source_report"), authored
        )
        calls = raw.get("calls")
        if not isinstance(calls, list) or any(not _valid_call_trace(call) for call in calls):
            raise ValueError("upstream worker returned malformed tool call evidence")
        feedback = {
            "success": success,
            "reference": authored.reference,
            "source_path": authored.source_path,
            "source_sha256": authored.source_sha256,
            "questions": list(authored.questions),
            "turns": source_turns,
            "policy": raw.get("policy"),
            "upstream": upstream,
            "outcome": "success" if success else "failure",
            "tool_count": len(calls),
            "calls": calls,
            "error_labels": _error_labels(upstream),
        }
        identity = {
            "korvid_revision": KORVID_REVISION,
            "source_sha256": authored.source_sha256,
            "model": model,
            "policy": raw.get("policy"),
            "prompt": raw.get("prompt"),
            "runtime_fingerprint": runtime["fingerprint"],
        }
        contract_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        usage = dict(raw["usage"])
        execution_mode = raw.get("execution_mode")
        if execution_mode not in {"live", "scripted"}:
            raise ValueError("upstream worker execution mode is invalid")
        evaluation = EvaluationResult(
            success=success,
            execution_mode=execution_mode,
            candidate_fingerprint=candidate.fingerprint,
            feedback=feedback,
            usage=usage,
            hard_failures=hard_failures,
        )
        write_json_artifact(
            response_path,
            {
                "protocol_version": UPSTREAM_PROTOCOL_VERSION,
                "success": evaluation.success,
                "hard_failures": list(evaluation.hard_failures),
                "execution_mode": execution_mode,
                "candidate_fingerprint": candidate.fingerprint,
                "request_identity": {
                    "case_id": case.case_id,
                    "template_id": case.template_id,
                    "model": case.models[0],
                    "repetition": repetition,
                    "seed": seed,
                    "seed_applied": execution_mode == "live",
                },
                "evidence_source": {
                    "kind": "korvid_upstream",
                    "korvid_version": KORVID_VERSION,
                    "source_sha256": contract_sha256,
                },
                "source_identity": {
                    "korvid_revision": KORVID_REVISION,
                    "reference": authored.reference,
                    "source_path": authored.source_path,
                    "source_sha256": authored.source_sha256,
                    "contract_sha256": contract_sha256,
                },
                "runtime": runtime,
                "source_report": source_report,
                "feedback": feedback,
                "usage": usage,
            },
        )
        write_json_artifact(
            target / "upstream-summary.json",
            {
                "protocol_version": UPSTREAM_PROTOCOL_VERSION,
                "korvid_version": KORVID_VERSION,
                "korvid_revision": KORVID_REVISION,
                "candidate_fingerprint": candidate.fingerprint,
                "reference": authored.reference,
                "source_path": authored.source_path,
                "source_sha256": authored.source_sha256,
                "execution_mode": execution_mode,
                "model": model,
                "policy": raw.get("policy"),
                "prompt": raw.get("prompt"),
                "runtime": runtime,
                "source_report": source_report,
                "upstream": upstream,
                "success": evaluation.success,
                "hard_failures": list(hard_failures),
                "contract_sha256": contract_sha256,
                "evaluation_override_verified": (
                    self.prompt_path is not None and raw.get("prompt_path_verified") is True
                ),
            },
        )
        return evaluation


def _error_labels(upstream: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    failure = upstream.get("failure_class")
    if isinstance(failure, str):
        labels.append(failure)
    error = upstream.get("error_label")
    if isinstance(error, str) and error not in labels:
        labels.append(error)
    turns = upstream.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            for key in ("failure_class", "error_label"):
                label = turn.get(key)
                if isinstance(label, str) and label not in labels:
                    labels.append(label)
    return labels


def _write_immutable(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _prompt_diff(original: str, optimized: str) -> str:
    rendered: list[str] = []
    for line in difflib.unified_diff(
        original.splitlines(keepends=True),
        optimized.splitlines(keepends=True),
        fromfile="original-prompt.txt",
        tofile="optimized-prompt.txt",
    ):
        if (
            line.startswith((" ", "+", "-"))
            and not line.startswith(("+++ ", "--- "))
            and not line.endswith(("\n", "\r"))
        ):
            rendered.extend((line, "\n\\ No newline at end of file\n"))
        else:
            rendered.append(line)
    return "".join(rendered)


def export_upstream_prompt(
    snapshot: Mapping[str, Any],
    candidate: Candidate,
    campaign: Campaign,
    directory: Path,
    *,
    seed: int = 0,
) -> Path:
    serving = campaign.serving
    if not isinstance(serving, KorvidUpstreamServing):
        raise ValueError("upstream export requires korvid_upstream serving")  # noqa: TRY004
    prompt = snapshot.get("prompt")
    if (
        not isinstance(prompt, Mapping)
        or not isinstance(prompt.get("text"), str)
        or prompt.get("baseline_equivalent") is not True
    ):
        raise ValueError("upstream export requires an attested baseline snapshot")
    original = prompt["text"]
    optimized = tier_pack_from_candidate(candidate)
    references = tuple(_reference(case) for case in campaign.cases)
    validation = run_upstream_request(
        serving,
        {
            "protocol_version": UPSTREAM_PROTOCOL_VERSION,
            "operation": "inspect",
            "references": list(references),
            "model": _model_payload(campaign.models[0], serving, seed),
            "tier_pack": optimized,
        },
    )
    fresh_prompt = validation.get("prompt")
    if (
        not isinstance(fresh_prompt, dict)
        or fresh_prompt.get("baseline_equivalent") is not True
        or fresh_prompt.get("text") != original
        or fresh_prompt.get("pack_id") != prompt.get("pack_id")
    ):
        raise ValueError("snapshot prompt differs from fresh upstream inspection")
    candidate_validation = validation.get("candidate_validation")
    if (
        not isinstance(candidate_validation, dict)
        or candidate_validation.get("tier_pack_sha256")
        != hashlib.sha256(optimized.encode()).hexdigest()
        or candidate_validation.get("valid") is not True
    ):
        raise ValueError("optimized prompt did not validate through Korvid PromptGrind")
    directory.mkdir(parents=True, exist_ok=False)
    original_path = directory / "original-prompt.txt"
    optimized_path = directory / "optimized-prompt.txt"
    diff_path = directory / "prompt.diff"
    _write_immutable(original_path, original)
    _write_immutable(optimized_path, optimized)
    _write_immutable(diff_path, _prompt_diff(original, optimized))
    manifest = {
        "schema_version": 1,
        "scope": "korvid-upstream-tier-pack",
        "release": KORVID_VERSION,
        "pack_id": prompt.get("pack_id"),
        "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "proposed_sha256": hashlib.sha256(optimized.encode()).hexdigest(),
        "source_identity": snapshot.get("source_identity"),
        "prompt_validation_passed": True,
        "product_application_verified": False,
        "evaluation_reload_receipt": "application-verification.json",
        "production_application": "requires_reviewed_korvid_prompt_pack_update",
        "source_writes": False,
    }
    manifest_path = directory / "application-manifest.json"
    write_json_artifact(manifest_path, manifest)
    manifest_path.chmod(0o444)
    return optimized_path
