"""Native low-tier UI evaluation and additive rule candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .artifacts import write_json_artifact
from .contracts import Campaign, Candidate, EvalCase, KorvidNativeServing
from .native_contract import (
    NATIVE_KORVID_REVISION,
    NATIVE_KORVID_VERSION,
    NATIVE_PROTOCOL_VERSION,
    find_native_case,
    native_cases,
    require_native_pack,
    rules_candidate,
    rules_from_candidate,
)
from .native_source import run_native_request
from .runner import BridgeInvocationError
from .scoring import BridgeResult, OperationGrade


def initialize_native(directory: Path, *, model: str, repetitions: int = 3) -> tuple[Path, Path]:
    if not model.startswith("ollama/") or not model.removeprefix("ollama/").strip() or model != model.strip():
        raise ValueError("native model must be an explicit ollama/<model> reference")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    directory.mkdir(parents=True, exist_ok=False)
    candidate_path, campaign_path = directory / "candidate.yaml", directory / "campaign.yaml"
    candidate = rules_candidate([])
    candidate_path.write_text(yaml.safe_dump({
        "schema_version": 1, "candidate_id": candidate.candidate_id,
        "components": candidate.components, "metadata": candidate.metadata,
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    campaign_path.write_text(yaml.safe_dump({
        "schema_version": 1, "campaign_id": "korvid-native-low-v1",
        "repetitions": repetitions, "models": [model],
        "cases": [
            {"case_id": case.case_id, "template_id": f"native-{case.split}",
             "prompt": case.prompt, "models": [model]}
            for case in native_cases()
        ],
        "serving": {
            "backend": "korvid_native", "source_root": "env:KORVID_NATIVE_SOURCE_ROOT",
            "base_url": "env:KORVID_NATIVE_MODEL_URL", "korvid_revision": NATIVE_KORVID_REVISION,
            "timeout_seconds": 240,
        },
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return candidate_path, campaign_path


def native_model(model: str, serving: KorvidNativeServing, seed: int) -> dict[str, Any]:
    if not model.startswith("ollama/"):
        raise ValueError("native UI campaign currently supports explicit local ollama/<model> references")
    return {
        "reference": model, "endpoint": serving.base_url,
        "options": {"temperature": 0.0, "seed": seed},
    }


def validate_native_policy(value: Any) -> None:
    # Attest the reviewed release's policy; never use this declaration to arm
    # tools. The unchanged native ModelRouter alone constructs the tool surface.
    expected = {
        "tier": "low", "prompt_pack_id": "low-korvid-operator",
        "max_iterations": 6, "max_history_chars": 24000,
        "max_result_chars": 3000, "max_tool_calls_per_iteration": 1,
        "tools": [
            "diagnose_pod", "diagnose_pvc", "diagnose_service", "diagnose_workload",
            "get_events", "get_logs", "get_resource", "helm_list_releases",
            "list_operators", "list_resources", "open_describe", "open_logs",
        ],
    }
    if not isinstance(value, dict) or value != expected:
        raise ValueError("native policy does not match the reviewed v0.4.1 low-tier contract")
    if any(type(value[key]) is not int for key in expected if key.startswith("max_")):
        raise ValueError("native policy budgets must be integers")


def validate_native_prompt_identity(payload: Mapping[str, Any]) -> None:
    validate_native_policy(payload.get("policy"))
    fingerprint = payload.get("prompt_fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch("[0-9a-f]{64}", fingerprint) is None:
        raise ValueError("native worker prompt fingerprint is missing")


def _validate_worker_result(
    payload: Mapping[str, Any], *, rules: list[str], case_id: str, execution_mode: str,
) -> tuple[float, int]:
    expected = {
        "protocol_version": NATIVE_PROTOCOL_VERSION, "korvid_version": NATIVE_KORVID_VERSION,
        "rules": rules, "case_id": case_id, "execution_mode": execution_mode,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"native worker {name} mismatch")
    if payload.get("rules_applied") is not True or payload.get("ui_follow") is not True:
        raise ValueError("native worker did not verify production rules and follow")
    validate_native_prompt_identity(payload)
    wall_time = payload.get("wall_time_seconds")
    if (
        isinstance(wall_time, bool)
        or not isinstance(wall_time, (int, float))
        or not math.isfinite(wall_time)
        or wall_time < 0
    ):
        raise ValueError("native worker wall_time_seconds must be finite and non-negative")
    iterations = payload.get("iterations")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("native worker iterations must be a positive integer")
    return float(wall_time), iterations


@dataclass(frozen=True)
class KorvidNativeRunner:
    campaign: Campaign
    script_factory: Callable[[EvalCase], list[list[dict[str, Any]]]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.campaign.serving, KorvidNativeServing):
            raise ValueError("native runner requires korvid_native serving")  # noqa: TRY004
        require_native_pack([case.case_id for case in self.campaign.cases])
        if len(self.campaign.models) != 1 or any(case.models != self.campaign.models for case in self.campaign.cases):
            raise ValueError("native cases and export must target exactly the same single model")

    def run(
        self, candidate: Candidate, case: EvalCase, run_dir: Path | str, *,
        repetition: int = 1, seed: int = 0,
    ) -> BridgeResult:
        serving = self.campaign.serving
        if not isinstance(serving, KorvidNativeServing):
            raise ValueError("native runner requires korvid_native serving")  # noqa: TRY004
        if isinstance(repetition, bool) or not isinstance(repetition, int) or not 1 <= repetition <= self.campaign.repetitions:
            raise ValueError("invalid native repetition")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("invalid native seed")
        if case not in self.campaign.cases or case.models != self.campaign.models:
            raise ValueError("native case and model must match the declared campaign entry")
        authored = find_native_case(case.case_id)
        if case.prompt != authored.prompt or case.template_id != f"native-{authored.split}" or len(case.models) != 1:
            raise ValueError("native case does not match its authored identity")
        rules = rules_from_candidate(candidate)
        path = Path(run_dir) / "response.json"
        if path.exists():
            raise FileExistsError(f"native response already exists: {path}")
        script = self.script_factory(case) if self.script_factory is not None else None
        mode = "scripted" if script is not None else "live"
        result = run_native_request(serving, {
            "protocol_version": NATIVE_PROTOCOL_VERSION, "operation": "evaluate",
            "case_id": case.case_id, "rules": rules,
            "model": native_model(case.models[0], serving, seed), "script": script,
        })
        wall_time, iterations = _validate_worker_result(
            result, rules=rules, case_id=case.case_id, execution_mode=mode
        )
        for key in ("initial", "expected", "observed"):
            if not isinstance(result.get(key), dict):
                raise ValueError(f"native worker {key} must be an object")  # noqa: TRY004
        for key in ("missing_postconditions", "calls", "errors"):
            if not isinstance(result.get(key), list):
                raise ValueError(f"native worker {key} must be a list")  # noqa: TRY004
        if result["expected"] != dict(authored.expected):
            raise ValueError("native worker expected state differs from authored case")
        missing = [
            key for key, expected in authored.expected
            if result["observed"].get(key) != expected
        ]
        calls = result["calls"]
        for call in calls:
            if (
                not isinstance(call, dict) or not isinstance(call.get("name"), str)
                or not isinstance(call.get("arguments"), dict) or type(call.get("ok")) is not bool
                or not isinstance(call.get("result"), str)
            ):
                raise ValueError("native worker returned a malformed action")
        errors = result["errors"]
        if any(not isinstance(error, str) for error in errors):
            raise ValueError("native errors must be bounded labels")
        armed = result["policy"].get("tools")
        if not isinstance(armed, list) or not armed or any(not isinstance(name, str) for name in armed):
            raise ValueError("native worker omitted armed tool identities")
        forbidden = [call["name"] for call in calls if call["name"] not in armed]
        hard_failures = ("unarmed_tool_attempt",) if forbidden else ()
        complete = bool(calls) and not (missing or errors or hard_failures) and all(call["ok"] for call in calls)
        if result.get("runtime_failed"):
            raise BridgeInvocationError("native model/runtime failed; no prompt score recorded")
        grade = OperationGrade(
            completion=float(complete),
            verification=(len(authored.expected) - len(missing)) / len(authored.expected) if calls else 0.0,
            efficiency=1.0 / len(calls) if calls else 0.0,
            hard_failures=hard_failures,
        )
        feedback = {
            "prompt": case.prompt, "initial": result["initial"], "expected": result["expected"],
            "observed": result["observed"], "missing_postconditions": missing,
            "calls": calls, "errors": errors, "tools": result["policy"],
        }
        runtime = result.get("runtime")
        if (
            not isinstance(runtime, dict) or not isinstance(runtime.get("dependencies"), dict)
            or not runtime["dependencies"]
            or not isinstance(runtime.get("fingerprint"), str)
            or re.fullmatch("[0-9a-f]{64}", runtime["fingerprint"]) is None
        ):
            raise ValueError("native worker did not record its actual runtime dependency identity")
        identity = {
            "korvid_revision": NATIVE_KORVID_REVISION, "scenario_sha256": authored.fingerprint,
            "policy": result["policy"], "model": native_model(case.models[0], serving, seed),
            "runtime_fingerprint": runtime["fingerprint"],
        }
        # No endpoint or runtime paths in publication metadata; the contract digest
        # still detects differing model parameters, policy, source, and dependencies.
        contract_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        usage = {
            "tool_calls": len(calls),
            "iterations": iterations,
            "wall_time_seconds": wall_time,
        }
        bridge = BridgeResult(
            protocol_version=2, status="completed", execution_mode=mode,
            candidate_fingerprint=candidate.fingerprint, grade=grade, answer="",
            journal={"navigation_feedback": feedback}, usage=usage, error=None,
        )
        write_json_artifact(path, {
            "protocol_version": 2, "status": "completed", "execution_mode": mode,
            "candidate_fingerprint": candidate.fingerprint,
            "request_identity": {
                "case_id": case.case_id, "template_id": case.template_id, "model": case.models[0],
                "repetition": repetition, "seed": seed, "seed_applied": mode == "live",
            },
            "evidence_source": {
                "kind": "korvid_native", "korvid_version": NATIVE_KORVID_VERSION,
                "scenario_sha256": contract_digest,
            },
            "grade": asdict(grade), "answer": "", "error": None,
            "journal": {
                "journey_id": case.case_id, "checkpoints": [], "missing_checkpoints": [],
                "checkpoint_counts": {}, "journal_event_count": len(calls),
                "audit_record_count": 0, "hard_failure_count": len(hard_failures),
            },
            "usage": usage,
        })
        write_json_artifact(Path(run_dir) / "native-summary.json", {
            "protocol_version": 1, "korvid_revision": NATIVE_KORVID_REVISION,
            "candidate_fingerprint": candidate.fingerprint, "case_id": case.case_id,
            "execution_mode": mode, "policy": result["policy"],
            "prompt_fingerprint": result["prompt_fingerprint"],
            "runtime": runtime,
            "rules_applied": True, "contract_sha256": contract_digest,
            "state_verified": complete,
        })
        return bridge
