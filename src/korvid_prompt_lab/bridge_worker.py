"""One-shot Korvid operation-journey bridge worker.

This module is launched by ``korvid-bridge`` inside the Korvid source
checkout's own ``uv`` environment, so Korvid and its Textual dependency
resolve without this repository ever vendoring them. It therefore imports
nothing from ``korvid_prompt_lab`` and nothing from Korvid at import time:
every Korvid symbol is resolved lazily inside :func:`run_bridge`, which keeps
the request/response protocol unit-testable in this repository's own
environment.

The process is deliberately one-shot. Prompt overrides are injected by
rebinding ``build_profile`` in Korvid's ``tests.evals.operation_app`` module,
which is safe precisely because the rebinding dies with this process and can
never leak into another candidate's run.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

PROTOCOL_VERSION = 2
EXIT_SYSTEMIC_FAILURE = 2
MAX_ANSWER_CHARS = 4000
MAX_ERROR_CHARS = 300
DEFAULT_PROFILE = "small"
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 5.0
DEFAULT_TURN_TIMEOUT_SECONDS = 120.0
AUDIT_FILENAME = "korvid-audit.jsonl"

#: A grade produced by contacting a real model provider over the request's endpoint.
EXECUTION_MODE_LIVE = "live"
#: A grade produced from Korvid's deterministic operation scripts — no model was asked.
EXECUTION_MODE_SCRIPTED = "scripted"
#: The closed vocabulary of :data:`PROTOCOL_VERSION` 2's ``execution_mode`` field.
#:
#: Protocol 1 had no such field, so a version-1 peer could never prove that a grade
#: came from a model. Both sides moved to 2 together and version 1 is refused rather
#: than migrated: assuming "live" for a silent peer is exactly the failure this
#: field exists to prevent.
EXECUTION_MODES: frozenset[str] = frozenset({EXECUTION_MODE_LIVE, EXECUTION_MODE_SCRIPTED})

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

#: Offline mirror of Korvid's ``LIFECYCLE_CHECKPOINTS``. It exists so the safe
#: response projection can be tested without the Korvid checkout; the live run
#: always projects against Korvid's own tuple, which stays authoritative.
LIFECYCLE_FALLBACK: tuple[str, ...] = (
    "goal_received",
    "target_resolved",
    "precondition_read",
    "write_requested",
    "approval_observed",
    "mutation_started",
    "mutation_finished",
    "postcondition_read",
    "outcome_reported",
)

#: Environment variables whose values must never reach a response or a log.
SENSITIVE_ENV_NAMES: tuple[str, ...] = (
    "KORVID_EVAL_API_KEY",
    "KORVID_BRIDGE_API_KEY",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)

_CREDENTIAL_PHRASE = re.compile(
    r"(?i)\b(?:authorization|bearer|api[-_]?key|apikey|token|secret|password)\b"
    r"(?:[\s:=]*\b(?:authorization|bearer|api[-_]?key|apikey|token|secret|password)\b)*"
    r"[\s:=]*\S+"
)


class WorkerConfigurationError(Exception):
    """A systemic failure: bad config, bad request, or an unusable checkout.

    Never graded. The worker exits non-zero so the runner reports a systemic
    bridge failure instead of scoring a run that never happened.
    """


class WorkerModelFailure(Exception):
    """Korvid produced no executable result because the model itself failed."""


# --- small strict validators ---------------------------------------------------


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkerConfigurationError(f"{context} must be a mapping")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerConfigurationError(f"{context} must be a non-empty string")
    return value


def _require_int(value: Any, context: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerConfigurationError(f"{context} must be an integer")
    if value < minimum:
        raise WorkerConfigurationError(f"{context} must be at least {minimum}")
    return value


def _require_keys(
    mapping: Mapping[str, Any],
    required: set[str],
    context: str,
    *,
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise WorkerConfigurationError(f"{context} has unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(mapping))
    if missing:
        raise WorkerConfigurationError(f"{context} is missing field(s): {', '.join(missing)}")


def _require_loopback_endpoint(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerConfigurationError("runtime.model_endpoint must be a loopback http URL or null")
    parts = urlsplit(value)
    if parts.scheme != "http" or parts.hostname not in _LOOPBACK_HOSTS or parts.port is None:
        raise WorkerConfigurationError(
            "runtime.model_endpoint must be a loopback http URL with an explicit port,"
            " for example http://127.0.0.1:41001"
        )
    if parts.path not in {"", "/"} or parts.query or parts.fragment:
        raise WorkerConfigurationError(
            "runtime.model_endpoint must be a loopback base URL without path, query, or fragment"
        )
    return value


def _validate_component_key(key: Any) -> str:
    if not isinstance(key, str):
        raise WorkerConfigurationError("candidate component key must be a string")
    if key in {"system", "append"}:
        return key
    if key.startswith("tool.") and len(key) > len("tool."):
        return key
    raise WorkerConfigurationError(
        f"candidate component key must be system, append, or tool.<tool-name>: {key!r}"
    )


def candidate_fingerprint(candidate: Mapping[str, Any]) -> str:
    """The candidate identity the control plane signs its request with."""
    payload = {
        "schema_version": candidate.get("schema_version"),
        "candidate_id": candidate.get("candidate_id"),
        "components": dict(_require_mapping(candidate.get("components"), "candidate.components")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# --- request ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BridgeRequest:
    """The validated one-shot request this worker is allowed to act on."""

    candidate_fingerprint: str
    candidate_id: str
    _components: tuple[tuple[str, str], ...]
    case_id: str
    template_id: str
    prompt: str
    model: str
    repetition: int
    seed: int
    campaign_id: str
    repetitions: int
    artifact_dir: str
    model_endpoint: str | None

    @property
    def components(self) -> dict[str, str]:
        return dict(self._components)

    @property
    def request_identity(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "template_id": self.template_id,
            "model": self.model,
            "repetition": self.repetition,
            "seed": self.seed,
        }


def parse_request(payload: Any) -> BridgeRequest:
    """Validate a bridge request; every rejection is systemic, never graded."""
    request = _require_mapping(payload, "request")
    _require_keys(
        request,
        {"protocol_version", "candidate_fingerprint", "candidate", "case", "runtime"},
        "request",
    )

    protocol_version = request["protocol_version"]
    if isinstance(protocol_version, bool) or protocol_version != PROTOCOL_VERSION:
        raise WorkerConfigurationError(f"request protocol_version must be {PROTOCOL_VERSION}")

    candidate = _require_mapping(request["candidate"], "candidate")
    _require_keys(
        candidate,
        {"schema_version", "candidate_id", "components"},
        "candidate",
        optional={"metadata"},
    )
    if candidate.get("schema_version") != 1:
        raise WorkerConfigurationError("candidate schema_version must be 1")
    candidate_id = _require_string(candidate.get("candidate_id"), "candidate.candidate_id")

    raw_components = _require_mapping(candidate.get("components"), "candidate.components")
    if not raw_components:
        raise WorkerConfigurationError("candidate components must not be empty")
    components: list[tuple[str, str]] = []
    for key, value in raw_components.items():
        component_key = _validate_component_key(key)
        components.append((component_key, _require_string(value, f"candidate component {component_key}")))

    declared_fingerprint = _require_string(request.get("candidate_fingerprint"), "candidate_fingerprint")
    if declared_fingerprint != candidate_fingerprint(candidate):
        raise WorkerConfigurationError("candidate_fingerprint does not match the candidate in the request")

    case = _require_mapping(request["case"], "case")
    _require_keys(case, {"case_id", "template_id", "prompt", "model", "repetition", "seed"}, "case")

    runtime = _require_mapping(request["runtime"], "runtime")
    _require_keys(runtime, {"campaign_id", "repetitions", "artifact_dir", "model_endpoint"}, "runtime")

    endpoint = runtime["model_endpoint"]
    model_endpoint = None if endpoint is None else _require_loopback_endpoint(endpoint)

    return BridgeRequest(
        candidate_fingerprint=declared_fingerprint,
        candidate_id=candidate_id,
        _components=tuple(components),
        case_id=_require_string(case.get("case_id"), "case.case_id"),
        template_id=_require_string(case.get("template_id"), "case.template_id"),
        prompt=_require_string(case.get("prompt"), "case.prompt"),
        model=_require_string(case.get("model"), "case.model"),
        repetition=_require_int(case.get("repetition"), "case.repetition", minimum=1),
        seed=_require_int(case.get("seed"), "case.seed", minimum=0),
        campaign_id=_require_string(runtime.get("campaign_id"), "runtime.campaign_id"),
        repetitions=_require_int(runtime.get("repetitions"), "runtime.repetitions", minimum=1),
        artifact_dir=_require_string(runtime.get("artifact_dir"), "runtime.artifact_dir"),
        model_endpoint=model_endpoint,
    )


def load_request(path: Path | str) -> BridgeRequest:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkerConfigurationError(f"request artifact could not be read: {path}") from exc
    except UnicodeDecodeError as exc:
        raise WorkerConfigurationError("request artifact is not valid UTF-8 JSON") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WorkerConfigurationError("request artifact is not valid JSON") from exc
    return parse_request(payload)


# --- execution mode -------------------------------------------------------------


def require_execution_mode(value: Any) -> str:
    """Return *value* only when it names a mode in the closed vocabulary."""
    if not isinstance(value, str) or value not in EXECUTION_MODES:
        raise WorkerConfigurationError(
            f"execution_mode must be one of {', '.join(sorted(EXECUTION_MODES))}"
        )
    return value


def resolve_execution_mode(request: BridgeRequest, *, scripted: bool) -> str:
    """Decide, fail-closed, which execution mode this run is allowed to claim.

    ``--scripted`` replaces the model with Korvid's canned operation scripts, so a
    scripted run grades prompt plumbing, never a model. A request that carries a
    ``runtime.model_endpoint`` is a live campaign by construction, and letting it
    return scripted evidence would publish a model-free score as if a model had
    earned it. That combination is systemic, not gradeable.
    """
    if not scripted:
        return EXECUTION_MODE_LIVE
    if request.model_endpoint is not None:
        raise WorkerConfigurationError(
            "scripted mode is refused for a request that carries runtime.model_endpoint:"
            " a live campaign must never be graded from Korvid's deterministic scripts"
        )
    return EXECUTION_MODE_SCRIPTED


# --- candidate -> Korvid prompt overrides --------------------------------------


def map_components_to_overrides(components: Mapping[str, str]) -> dict[str, Any]:
    """Map candidate components onto Korvid's ``PromptOverrides`` slots.

    ``system`` and ``append`` fill the role statement and its suffix;
    ``tool.<name>`` fills one per-tool description. An absent slot stays
    ``None`` so Korvid keeps its own shipped wording.
    """
    system: str | None = None
    append: str | None = None
    tool_descriptions: dict[str, str] = {}

    for key, value in components.items():
        component_key = _validate_component_key(key)
        text = _require_string(value, f"candidate component {component_key}")
        if component_key == "system":
            system = text
        elif component_key == "append":
            append = text
        else:
            tool_descriptions[component_key[len("tool.") :]] = text

    return {"system": system, "append": append, "tool_descriptions": tool_descriptions}


def install_prompt_overrides(module: Any, overrides: Any) -> None:
    """Bind *overrides* into *module*'s ``build_profile`` for this process only."""
    build_profile = getattr(module, "build_profile", None)
    if not callable(build_profile):
        raise WorkerConfigurationError(
            "korvid operation harness does not expose a patchable build_profile"
        )
    module.build_profile = functools.partial(build_profile, overrides=overrides)


# --- journey selection ----------------------------------------------------------


def select_journey(journeys: Sequence[Any], template_id: str) -> Any:
    """Return exactly the bundled operation journey named by ``template_id``."""
    matches = [journey for journey in journeys if getattr(journey, "id", None) == template_id]
    if len(matches) != 1:
        raise WorkerConfigurationError(
            f"campaign template_id must name exactly one bundled operation journey: {template_id!r}"
        )
    return matches[0]


def require_prompt_matches_journey(prompt: str, journey: Any) -> None:
    """Refuse a campaign prompt that is not the journey's own first turn."""
    turns = tuple(getattr(journey, "turns", ()) or ())
    if not turns:
        raise WorkerConfigurationError(
            f"operation journey {getattr(journey, 'id', '?')!r} has no first turn to compare the prompt against"
        )
    if prompt != turns[0]:
        raise WorkerConfigurationError(
            f"campaign prompt does not match the first turn of operation journey"
            f" {getattr(journey, 'id', '?')!r}"
        )


# --- safe response projection ---------------------------------------------------


def project_journal(run: Any, lifecycle_checkpoints: Sequence[str]) -> dict[str, Any]:
    """Project a graded run into a reflection-safe journal summary.

    Only checkpoint names from Korvid's own lifecycle vocabulary and integer
    counts leave this process. Raw journal payloads, audit records, manifests,
    credentials, and tool output never do.
    """
    grade = run.grade
    lifecycle = tuple(lifecycle_checkpoints)
    known = set(lifecycle)

    counts: dict[str, int] = {}
    journal = tuple(getattr(run, "journal", ()) or ())
    for entry in journal:
        if not isinstance(entry, Mapping):
            continue
        event = entry.get("event")
        if isinstance(event, str) and event in known:
            counts[event] = counts.get(event, 0) + 1

    reached = {str(name) for name in getattr(grade, "checkpoints", ())}
    missing = {str(name) for name in getattr(grade, "missing_checkpoints", ())}
    audit = tuple(getattr(run, "audit", ()) or ())

    return {
        "journey_id": str(getattr(run, "journey_id", "")),
        "checkpoints": [name for name in lifecycle if name in reached],
        "missing_checkpoints": [name for name in lifecycle if name in missing],
        "checkpoint_counts": dict(sorted(counts.items())),
        "journal_event_count": len(journal),
        "audit_record_count": len(audit),
        "hard_failure_count": len(tuple(getattr(grade, "hard_failures", ()))),
    }


def _metric(value: Any, field_name: str) -> float:
    """Korvid grades completion/verification as booleans; scoring wants 0..1."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(value):
        return min(1.0, max(0.0, float(value)))
    raise WorkerConfigurationError(f"korvid grade {field_name} is not a numeric signal")


def _bounded_answer(answer: Any) -> str:
    text = answer if isinstance(answer, str) else ""
    if len(text) <= MAX_ANSWER_CHARS:
        return text
    return f"{text[:MAX_ANSWER_CHARS]} [truncated]"


def _hard_failures(grade: Any) -> list[str]:
    failures: list[str] = []
    for item in tuple(getattr(grade, "hard_failures", ())):
        name = str(item).strip()
        failures.append(name or "unspecified_hard_failure")
    return failures


def build_completed_response(
    request: BridgeRequest,
    run: Any,
    lifecycle_checkpoints: Sequence[str],
    *,
    execution_mode: str,
) -> dict[str, Any]:
    """Build the strict response for a run Korvid actually executed and graded.

    A graded-but-incomplete run is still ``completed``: it earned a low grade,
    which is evidence, not a systemic failure. ``execution_mode`` travels with the
    grade so no downstream consumer has to guess whether a model produced it.
    """
    grade = run.grade
    wall_time = getattr(run, "wall_time_s", 0.0)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed",
        "execution_mode": require_execution_mode(execution_mode),
        "candidate_fingerprint": request.candidate_fingerprint,
        "request_identity": request.request_identity,
        "grade": {
            "completion": _metric(getattr(grade, "completion", None), "completion"),
            "verification": _metric(getattr(grade, "verification", None), "verification"),
            "efficiency": _metric(getattr(grade, "efficiency", None), "efficiency"),
            "hard_failures": _hard_failures(grade),
        },
        "answer": _bounded_answer(getattr(run, "answer", "")),
        "journal": project_journal(run, lifecycle_checkpoints),
        "usage": {
            "tool_calls": int(getattr(grade, "tool_calls", 0)),
            "iterations": int(getattr(grade, "iterations", 0)),
            "wall_time_seconds": round(float(wall_time), 3),
        },
        "error": None,
    }


def build_model_failure_response(
    request: BridgeRequest,
    error: BaseException | str,
    *,
    execution_mode: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the response for a run Korvid could not execute because of the model."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "model_failure",
        "execution_mode": require_execution_mode(execution_mode),
        "candidate_fingerprint": request.candidate_fingerprint,
        "request_identity": request.request_identity,
        "grade": None,
        "answer": "",
        "journal": {"checkpoints": [], "checkpoint_counts": {}},
        "usage": {},
        "error": sanitize_error(error, env),
    }


# --- error sanitization ----------------------------------------------------------


def sanitize_error(error: BaseException | str, env: Mapping[str, str] | None = None) -> str:
    """Return a bounded, credential-free description of *error*."""
    environment = os.environ if env is None else env
    text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    text = _CREDENTIAL_PHRASE.sub("[redacted-credential]", text)
    for name in SENSITIVE_ENV_NAMES:
        secret = environment.get(name, "")
        if secret and secret.strip():
            text = text.replace(secret, "***")
    text = " ".join(text.split())
    if len(text) > MAX_ERROR_CHARS:
        text = f"{text[:MAX_ERROR_CHARS]} [truncated]"
    return text or "unspecified bridge failure"


# --- atomic response write --------------------------------------------------------


def write_response(path: Path | str, payload: Mapping[str, Any]) -> Path:
    """Serialize first, then replace atomically, so a reader never sees a partial file."""
    try:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    except (TypeError, ValueError) as exc:
        raise WorkerConfigurationError("bridge response is not serializable protocol JSON") from exc

    response_path = Path(path)
    temp_path = response_path.with_name(f"{response_path.name}.tmp")
    try:
        response_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(encoded, encoding="utf-8")
        os.replace(temp_path, response_path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise WorkerConfigurationError(f"bridge response could not be written: {response_path}") from exc
    return response_path


# --- failure taxonomy ---------------------------------------------------------------


class TurnPhase:
    """Records whether Korvid ever got as far as asking the model for a turn.

    Korvid's harness raises the same wait timeout for two very different events:
    a Textual selection or approval step that never settled (before the model was
    ever consulted), and a turn the model failed to finish. Only the second is the
    model's fault. Instead of guessing from the message, the worker observes the
    provider: the phase flips the first time a completion is actually requested.
    """

    __slots__ = ("_started",)

    def __init__(self) -> None:
        self._started = False

    @property
    def model_turn_started(self) -> bool:
        return self._started

    def mark_model_turn_started(self) -> None:
        self._started = True


class _TurnObservingProvider:
    """Wraps a Korvid provider so the worker sees the first real model turn.

    Everything except ``complete`` is forwarded untouched, including the ``aclose``
    hook Korvid looks up on the object the factory returned.
    """

    __slots__ = ("_phase", "_provider")

    def __init__(self, provider: Any, phase: TurnPhase) -> None:
        self._provider = provider
        self._phase = phase

    def complete(self, messages: Any, tools: Any, *, stream: bool = True) -> Any:
        self._phase.mark_model_turn_started()
        return self._provider.complete(messages, tools, stream=stream)

    def __getattr__(self, name: str) -> Any:
        if name in _TurnObservingProvider.__slots__:  # pragma: no cover - recursion guard
            raise AttributeError(name)
        return getattr(self._provider, name)


def observe_model_turns(provider: Any, phase: TurnPhase) -> Any:
    """Return *provider* wrapped so *phase* learns when the model is first asked."""
    return _TurnObservingProvider(provider, phase)


def classify_run_failure(
    error: BaseException,
    phase: TurnPhase,
    *,
    provider_errors: tuple[type[BaseException], ...] = (),
    env: Mapping[str, str] | None = None,
) -> None:
    """Raise the right kind of failure for an exception that escaped the journey.

    A provider or transport error is always the model's: only the provider raises it.
    A timeout is the model's only once the model has actually been asked for a turn;
    a timeout before that is the Textual harness failing to reach the model at all,
    which is systemic — grading it ``model_failure`` would score a broken harness
    ``0.0`` and let an optimization run to completion against no evidence.
    """
    if provider_errors and isinstance(error, provider_errors):
        raise WorkerModelFailure(sanitize_error(error, env)) from error
    if phase.model_turn_started:
        raise WorkerModelFailure(sanitize_error(error, env)) from error
    raise WorkerConfigurationError(
        "korvid harness failed before the model was asked for a turn:"
        f" {sanitize_error(error, env)}"
    ) from error


# --- Korvid execution --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Korvid:
    operation_app: ModuleType
    run_operation_journey: Any
    approval_timeout_for: Any
    load_operation_journeys: Any
    bundled_operations_dir: Any
    lifecycle_checkpoints: tuple[str, ...]
    prompt_overrides: Any
    scripted_provider: Any
    operation_scripts: Mapping[str, Any]
    live_provider: Any
    static_credentials: Any
    #: Errors only a provider or its transport can raise — always the model's fault.
    provider_errors: tuple[type[BaseException], ...]
    #: Errors whose blame depends on whether a model turn had started.
    turn_timeout_errors: tuple[type[BaseException], ...]


def _import_korvid() -> _Korvid:
    # Every symbol below lives in the checkout named by KORVID_SOURCE_ROOT, so it
    # is resolvable only in the worker's uv environment, never in this one.
    try:
        import httpx
        from korvid.agent.profiles import PromptOverrides
        from korvid.evals.operation import (
            LIFECYCLE_CHECKPOINTS,
            bundled_operations_dir,
            load_operation_journeys,
        )
        from korvid.evals.scripted import ScriptedProvider
        from korvid.providers.openai_compat import OpenAICompatProvider, ProviderError
        from korvid.providers.static_creds import StaticHeaderSource
        from tests.evals import operation_app
        from tests.evals.operation_campaign import approval_timeout_for
        from tests.evals.operation_scripts import OPERATION_SCRIPTS
        from tests.ui.waits import WaitTimeout
    except ImportError as exc:
        raise WorkerConfigurationError(
            "korvid operation harness is not importable; KORVID_SOURCE_ROOT must point at a"
            " Korvid source checkout whose uv environment is installed"
        ) from exc

    return _Korvid(
        operation_app=operation_app,
        run_operation_journey=operation_app.run_operation_journey,
        approval_timeout_for=approval_timeout_for,
        load_operation_journeys=load_operation_journeys,
        bundled_operations_dir=bundled_operations_dir,
        lifecycle_checkpoints=tuple(LIFECYCLE_CHECKPOINTS),
        prompt_overrides=PromptOverrides,
        scripted_provider=ScriptedProvider,
        operation_scripts=OPERATION_SCRIPTS,
        live_provider=OpenAICompatProvider,
        static_credentials=StaticHeaderSource,
        provider_errors=(ProviderError, httpx.HTTPError),
        turn_timeout_errors=(WaitTimeout, TimeoutError),
    )


def _build_provider_factory(
    korvid: _Korvid,
    request: BridgeRequest,
    *,
    scripted: bool,
    turn_timeout: float,
    env: Mapping[str, str],
    phase: TurnPhase,
) -> Any:
    if scripted:
        script = korvid.operation_scripts.get(request.template_id)
        if script is None:
            raise WorkerConfigurationError(
                f"scripted mode has no deterministic script for operation journey {request.template_id!r}"
            )
        return lambda: observe_model_turns(korvid.scripted_provider(script), phase)

    if request.model_endpoint is None:
        raise WorkerConfigurationError(
            "live mode requires runtime.model_endpoint; run with --scripted for the deterministic self-test"
        )

    base_url = f"{request.model_endpoint.rstrip('/')}/v1"
    api_key = env.get("KORVID_EVAL_API_KEY", "").strip()

    def factory() -> Any:
        credentials = korvid.static_credentials(api_key) if api_key else None
        return observe_model_turns(
            korvid.live_provider(
                base_url,
                request.model,
                credentials=credentials,
                timeout_seconds=turn_timeout,
            ),
            phase,
        )

    return factory


def run_bridge(
    request: BridgeRequest,
    *,
    execution_mode: str,
    profile: str = DEFAULT_PROFILE,
    approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    turn_timeout: float = DEFAULT_TURN_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run one graded Korvid operation journey and return the strict response."""
    environment = os.environ if env is None else env
    scripted = require_execution_mode(execution_mode) == EXECUTION_MODE_SCRIPTED
    phase = TurnPhase()
    korvid = _import_korvid()

    journeys = korvid.load_operation_journeys(korvid.bundled_operations_dir())
    journey = select_journey(journeys, request.template_id)
    require_prompt_matches_journey(request.prompt, journey)

    overrides = korvid.prompt_overrides(**map_components_to_overrides(request.components))
    install_prompt_overrides(korvid.operation_app, overrides)

    provider_factory = _build_provider_factory(
        korvid,
        request,
        scripted=scripted,
        turn_timeout=turn_timeout,
        env=environment,
        phase=phase,
    )

    audit_path = Path(request.artifact_dir) / AUDIT_FILENAME
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        # Korvid's AuditLog appends and the grader re-reads the whole file, and the
        # control plane reuses a deterministic run directory. A leftover audit from a
        # previous invocation would let a stale intent satisfy this run's audit probe
        # and mask a `write_without_audit_intent` hard failure, so start from empty.
        audit_path.unlink(missing_ok=True)
    except OSError as exc:
        raise WorkerConfigurationError(f"audit log could not be prepared: {audit_path}") from exc

    try:
        run = asyncio.run(
            korvid.run_operation_journey(
                journey,
                audit_path=audit_path,
                provider_factory=provider_factory,
                profile_name=profile,
                approval_timeout_seconds=korvid.approval_timeout_for(journey, approval_timeout),
                turn_timeout=turn_timeout,
            )
        )
    except (*korvid.provider_errors, *korvid.turn_timeout_errors) as exc:
        classify_run_failure(exc, phase, provider_errors=korvid.provider_errors, env=environment)
        raise  # pragma: no cover - classify_run_failure always raises

    return build_completed_response(
        request, run, korvid.lifecycle_checkpoints, execution_mode=execution_mode
    )


# --- entry point --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="korvid-bridge-worker",
        description="Run one graded Korvid operation journey for a prompt candidate.",
    )
    parser.add_argument("--request", required=True, type=Path, help="Path to the bridge request JSON.")
    parser.add_argument("--response", required=True, type=Path, help="Path to write the bridge response JSON.")
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Use Korvid's deterministic operation scripts instead of the live model endpoint.",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Korvid agent profile to arm.")
    parser.add_argument(
        "--approval-timeout",
        type=float,
        default=DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        help="Approval window injected into the Korvid app.",
    )
    parser.add_argument(
        "--turn-timeout",
        type=float,
        default=DEFAULT_TURN_TIMEOUT_SECONDS,
        help="Upper bound on one turn reaching a dialog or ending.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    request: BridgeRequest | None = None
    execution_mode: str | None = None
    try:
        request = load_request(args.request)
        # Fail closed before Korvid is even imported: a live request may never be
        # answered with scripted evidence.
        execution_mode = resolve_execution_mode(request, scripted=args.scripted)
        payload = run_bridge(
            request,
            execution_mode=execution_mode,
            profile=args.profile,
            approval_timeout=args.approval_timeout,
            turn_timeout=args.turn_timeout,
        )
    except WorkerModelFailure as exc:
        if request is None or execution_mode is None:  # pragma: no cover - defensive; the model cannot fail before parsing
            print(f"korvid-bridge-worker: {sanitize_error(exc)}", file=sys.stderr)
            return EXIT_SYSTEMIC_FAILURE
        payload = build_model_failure_response(request, exc, execution_mode=execution_mode)
    except WorkerConfigurationError as exc:
        print(f"korvid-bridge-worker: {sanitize_error(exc)}", file=sys.stderr)
        return EXIT_SYSTEMIC_FAILURE
    except Exception as exc:  # noqa: BLE001 - a systemic failure must never be graded
        print(f"korvid-bridge-worker: {sanitize_error(exc)}", file=sys.stderr)
        return EXIT_SYSTEMIC_FAILURE

    try:
        write_response(args.response, payload)
    except WorkerConfigurationError as exc:
        print(f"korvid-bridge-worker: {sanitize_error(exc)}", file=sys.stderr)
        return EXIT_SYSTEMIC_FAILURE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
