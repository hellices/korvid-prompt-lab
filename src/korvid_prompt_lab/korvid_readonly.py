"""Runner for the ``korvid_readonly`` campaign serving backend.

Executes exactly one installed Korvid read-only scenario per call, through
the real ``python -m korvid.evals`` CLI shipped with the installed
``korvid[agent]`` wheel, and normalizes its JSON into the shared
:class:`~korvid_prompt_lab.scoring.BridgeResult` contract so existing
campaign machinery (comparison, publish, GEPA) can treat it like any other
runner.

Korvid's bundled scenario pack remains the single source of truth: this
module never vendors or forks scenario content. For each call it looks the
requested scenario up by id in the installed wheel via
``korvid.evals.scenario.load_scenario``, verifies the authored question
matches the campaign's ``EvalCase.prompt`` exactly, copies only that one
scenario file into a private temporary pack, and deletes the pack again
once the run finishes (success or failure).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from korvid.agent.profiles import build_profile
from korvid.evals.scenario import Scenario, bundled_scenarios_dir, load_scenario

from .bridge_worker import EXECUTION_MODE_LIVE, PROTOCOL_VERSION
from .contracts import (
    Campaign,
    Candidate,
    EvalCase,
    KorvidReadonlyServing,
    _ensure_keys,
    _require_bridge_timeout,
    _require_mapping,
    _require_string,
)
from .runner import (
    BridgeArtifactError,
    BridgeIdentityMismatchError,
    BridgeInvocationError,
    BridgeMalformedOutputError,
    BridgeMissingOutputError,
    BridgeProcessExitError,
    BridgeTimeoutError,
    _decode_process_output,
)
from .scoring import BridgeResult, OperationGrade

#: Overridable for tests: the argv prefix used to invoke korvid's live eval
#: CLI. Production code always uses the interpreter running this process so
#: the same installed wheel that built the baseline also serves it.
_KORVID_EVALS_COMMAND: tuple[str, ...] = (sys.executable, "-m", "korvid.evals")

#: Share of the outer subprocess.run(timeout=...) budget reserved for
#: korvid.evals' own non-HTTP overhead (interpreter startup, importing the
#: installed wheel, writing the requested --json artifact, ...) before the
#: remainder is divided across the installed profile's max_iterations. See
#: :func:`_eval_request_timeout_seconds`.
_EVAL_TIMEOUT_OVERHEAD_FRACTION = 0.1
#: Upper bound on that reservation, so a long outer budget does not donate an
#: unreasonably large share to process overhead.
_EVAL_TIMEOUT_OVERHEAD_MAX_SECONDS = 10.0
#: Lower bound on that reservation, so a short outer budget still keeps a
#: usable floor of overhead reserved.
_EVAL_TIMEOUT_OVERHEAD_MIN_SECONDS = 1.0

#: Candidate component keys this runner knows how to project onto the
#: installed CLI's ``--system-prompt-file``/``--prompt-append-file`` flags.
#: Per-tool overrides (``tool.<name>``) have no equivalent flag on
#: ``korvid.evals`` and must fail closed rather than be silently dropped.
_SUPPORTED_COMPONENTS = frozenset({"system", "append"})

#: Fields the installed ``korvid.evals --json`` output must carry on one run
#: entry. Any deviation (missing or unexpected field) is malformed output.
_RUN_FIELDS = frozenset(
    {
        "grade",
        "citations",
        "answer",
        "iterations",
        "tool_calls",
        "resolvable_tool_calls",
        "on_target_tool_calls",
        "malformed_tool_calls",
        "write_attempts",
        "safety_violations",
        "input_tokens",
        "output_tokens",
        "tokens_estimated",
        "wall_time_s",
        "error",
    }
)

_GRADE_FIELDS = frozenset(
    {
        "diagnosis_success",
        "evidence_fetched",
        "missing_mentions",
        "forbidden_mentions",
        "missing_evidence",
    }
)

_CITATION_FIELDS = frozenset(
    {"cited", "unsupported", "uncited_evidence", "coverage", "precision"}
)


@dataclass(frozen=True, slots=True)
class KorvidReadonlyRunner:
    """Executes one ``korvid_readonly`` campaign case per :meth:`run` call."""

    campaign: Campaign
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        serving = self.campaign.serving
        if not isinstance(serving, KorvidReadonlyServing):
            raise ValueError("KorvidReadonlyRunner requires korvid_readonly serving")  # noqa: TRY004 - preserve validation API
        if self.timeout_seconds is None:
            # Runtime policy belongs to the campaign's serving config, never
            # to a candidate or the optimizer.
            object.__setattr__(self, "timeout_seconds", serving.timeout_seconds)
        _require_bridge_timeout(self.timeout_seconds, "timeout_seconds")

    def run(
        self,
        candidate: Candidate,
        case: EvalCase,
        run_dir: Path | str,
        *,
        repetition: int = 1,
        seed: int = 0,
    ) -> BridgeResult:
        serving = self.campaign.serving
        if not isinstance(serving, KorvidReadonlyServing):
            raise ValueError("KorvidReadonlyRunner requires korvid_readonly serving")  # noqa: TRY004 - preserve validation API

        if (
            isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or repetition <= 0
        ):
            raise ValueError("repetition must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if repetition > self.campaign.repetitions:
            raise ValueError("repetition must not exceed campaign.repetitions")
        if len(case.models) != 1:
            raise ValueError("KorvidReadonlyRunner requires exactly one model per case")

        _require_supported_components(candidate)
        system_prompt = candidate.components["system"]
        append_prompt = candidate.components.get("append")

        scenario, scenario_path = _locate_bundled_scenario(case.case_id)
        if scenario.question != case.prompt:
            raise ValueError(
                f"korvid bundled scenario {case.case_id!r} question does not match the case prompt"
            )

        run_path = Path(run_dir)
        try:
            run_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BridgeArtifactError(
                f"runner could not prepare run directory: {run_path}"
            ) from exc

        output_path = run_path / "korvid-eval-output.json"
        try:
            if output_path.exists():
                output_path.unlink()
        except OSError as exc:
            raise BridgeArtifactError(
                f"runner could not clean previous output artifact: {output_path}"
            ) from exc

        try:
            pack_dir = Path(tempfile.mkdtemp(prefix="korvid-readonly-pack-"))
        except OSError as exc:
            raise BridgeArtifactError(
                "runner could not create a private scenario pack"
            ) from exc

        try:
            command = self._build_command(
                serving,
                pack_dir,
                scenario_path,
                system_prompt,
                append_prompt,
                output_path,
            )
            env = self._build_environment(serving, case)
            completed = self._invoke(command, env)

            if not output_path.exists():
                # A real invocation failure (crash, unreachable provider, ...)
                # never produces the requested artifact; a nonzero exit here
                # is unambiguous and must stay a process-exit error rather
                # than the generic missing-output error below, which is
                # reserved for a zero exit that still forgot to write output.
                if completed.returncode != 0:
                    raise BridgeProcessExitError(_process_exit_message(completed))
                raise BridgeMissingOutputError(
                    "korvid.evals did not create the requested --json output artifact"
                )

            run_payload = _load_single_run(output_path, case)

            if completed.returncode != 0 and not _has_model_failure_error(run_payload):
                # The installed Korvid CLI exits nonzero for a genuine model
                # failure too, but *only* alongside valid JSON whose run
                # carries a non-blank ``error``. Any other nonzero exit
                # (a successful-looking run, corrupted process output that
                # still happened to parse, ...) is a real contract violation
                # and must fail closed as a process error, never be
                # normalized into a completed/model_failure result.
                raise BridgeProcessExitError(_process_exit_message(completed))
        finally:
            shutil.rmtree(pack_dir, ignore_errors=True)

        return _to_bridge_result(candidate, run_payload)

    def _build_command(
        self,
        serving: KorvidReadonlyServing,
        pack_dir: Path,
        scenario_path: Path,
        system_prompt: str,
        append_prompt: str | None,
        output_path: Path,
    ) -> tuple[str, ...]:
        try:
            shutil.copyfile(scenario_path, pack_dir / scenario_path.name)
            system_path = pack_dir / "system-prompt.txt"
            system_path.write_text(system_prompt, encoding="utf-8")
            append_path: Path | None = None
            if append_prompt is not None:
                append_path = pack_dir / "prompt-append.txt"
                append_path.write_text(append_prompt, encoding="utf-8")
        except OSError as exc:
            raise BridgeArtifactError(
                "runner could not prepare a private prompt pack"
            ) from exc

        command: list[str] = [
            *_KORVID_EVALS_COMMAND,
            "--scenarios",
            str(pack_dir),
            "--reps",
            "1",
            "--profile",
            serving.profile,
            "--system-prompt-file",
            str(system_path),
        ]
        if append_path is not None:
            command += ["--prompt-append-file", str(append_path)]
        command += ["--json", str(output_path)]
        return tuple(command)

    def _build_environment(
        self, serving: KorvidReadonlyServing, case: EvalCase
    ) -> dict[str, str]:
        # Only the documented KORVID_EVAL_* values this runner is responsible
        # for are set explicitly; anything else (e.g. KORVID_EVAL_API_KEY)
        # flows through from the inherited environment untouched, so
        # credentials are never read from or written into campaign config.
        outer_timeout = self.timeout_seconds
        # Guaranteed non-None by __post_init__, which always resolves it from
        # either the constructor override or serving.timeout_seconds before
        # validating it with _require_bridge_timeout.
        assert outer_timeout is not None
        return {
            **os.environ,
            "KORVID_EVAL_BASE_URL": _effective_base_url(serving),
            "KORVID_EVAL_MODEL": case.models[0],
            # This runner's own subprocess.run(timeout=...) budget is
            # self.timeout_seconds (the effective outer budget, honoring a
            # constructor override over serving.timeout_seconds); the
            # per-HTTP-request value handed to Korvid must be derived from
            # that same effective budget, never from serving.timeout_seconds
            # directly, so a runner-level override is never silently ignored
            # here. See _eval_request_timeout_seconds.
            "KORVID_EVAL_TIMEOUT_SECONDS": repr(
                _eval_request_timeout_seconds(outer_timeout, serving.profile)
            ),
        }

    def _invoke(
        self, command: tuple[str, ...], env: dict[str, str]
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                env=env,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise BridgeInvocationError(
                f"korvid.evals could not be launched: {command[0]}"
            ) from exc
        except OSError as exc:
            raise BridgeInvocationError(
                f"korvid.evals could not be executed: {command[0]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BridgeTimeoutError(
                f"korvid.evals timed out after {self.timeout_seconds} seconds"
            ) from exc

        # Deliberately not raised here: a nonzero exit is only classifiable
        # once the caller has looked at whether valid, exactly-one-run JSON
        # with a populated ``error`` was also written (a legitimate model
        # failure per the installed Korvid CLI's own contract).
        return completed


def _eval_request_timeout_seconds(timeout_seconds: float, profile: str) -> float:
    """Derive the per-HTTP-request timeout handed to the installed Korvid CLI

    as ``KORVID_EVAL_TIMEOUT_SECONDS``.

    ``KORVID_EVAL_TIMEOUT_SECONDS`` bounds one HTTP request inside Korvid's
    own agent loop, while *timeout_seconds* is this runner's *whole-process*
    wall-clock budget (:func:`subprocess.run`'s own ``timeout``). The
    installed profile's agent loop can issue up to its own ``max_iterations``
    sequential requests before finishing -- successfully, or with a genuine
    model failure it reports via a populated ``run.error`` -- so setting the
    two timeouts equal lets the outer subprocess kill preempt Korvid
    mid-iteration, before it can ever write that JSON. That turns a per-case
    model failure into a systemic process error instead of the model_failure
    status it should be.

    A bounded share of *timeout_seconds* is reserved for the CLI's own
    non-HTTP overhead (interpreter startup, importing the installed wheel,
    writing the requested ``--json`` artifact, ...); the remainder is divided
    across the installed profile's ``max_iterations`` -- read from the
    installed Korvid wheel via ``korvid.agent.profiles.build_profile`` (the
    same API :mod:`korvid_prompt_lab.baseline` already uses), never
    hard-coded in this repository -- so the derived value stays strictly
    below *timeout_seconds* regardless of profile or timeout configuration.
    """
    try:
        agent_profile = build_profile(profile, readonly=True, resize_supported=False)
    except ValueError as exc:
        raise ValueError(
            f"installed Korvid rejected profile {profile!r}: {exc}"
        ) from exc

    overhead = min(
        max(
            timeout_seconds * _EVAL_TIMEOUT_OVERHEAD_FRACTION,
            _EVAL_TIMEOUT_OVERHEAD_MIN_SECONDS,
        ),
        _EVAL_TIMEOUT_OVERHEAD_MAX_SECONDS,
        # Never reserve more than half the budget: a very short outer
        # timeout must still leave a usable, strictly-positive remainder to
        # divide across iterations.
        timeout_seconds * 0.5,
    )
    budget = timeout_seconds - overhead
    return _require_bridge_timeout(
        budget / agent_profile.max_iterations, "derived KORVID_EVAL_TIMEOUT_SECONDS"
    )


def _process_exit_message(completed: subprocess.CompletedProcess[bytes]) -> str:
    detail = _decode_process_output(completed.stderr) or _decode_process_output(
        completed.stdout
    )
    detail = detail or f"exit code {completed.returncode}"
    return f"korvid.evals exited non-zero: {detail}"


def _has_model_failure_error(run_payload: Mapping[str, Any]) -> bool:
    """Whether *run_payload* carries the shape the installed Korvid CLI uses

    for a genuine model failure: a non-blank ``error`` string. Deliberately
    lenient (only peeks at ``error``, not the full run-field contract) so the
    caller can route to the existing, stricter validation in
    :func:`_to_bridge_result` either way; this only decides whether a nonzero
    exit is *allowed* to reach that validation instead of failing closed as a
    process error first.
    """
    error = run_payload.get("error")
    return isinstance(error, str) and bool(error.strip())


def _effective_base_url(serving: KorvidReadonlyServing) -> str:
    """The OpenAI-compatible URL to hand `korvid.evals` as ``KORVID_EVAL_BASE_URL``.

    ``korvid.evals`` only ever talks OpenAI-compatible HTTP: it has no native
    ollama transport. When ``provider`` is ``ollama``, ``base_url`` is the
    server's native root (the shape every other ollama-aware tool in this
    repo uses), so this appends the ``/v1`` compatibility suffix ollama
    serves it under. An ``openai-compat`` base URL is already in the shape
    the CLI expects and is used verbatim.
    """
    base = serving.base_url.rstrip("/")
    if serving.provider == "ollama" and not base.endswith("/v1"):
        return f"{base}/v1"
    return base


def _require_supported_components(candidate: Candidate) -> None:
    components = candidate.components
    unsupported = sorted(set(components) - _SUPPORTED_COMPONENTS)
    if unsupported:
        raise ValueError(
            "korvid_readonly runner does not support component(s): "
            + ", ".join(unsupported)
        )
    if "system" not in components:
        raise ValueError("korvid_readonly runner requires a 'system' component")


def _locate_bundled_scenario(scenario_id: str) -> tuple[Scenario, Path]:
    """Find the one installed scenario file whose id matches *scenario_id*.

    Reads only the installed wheel's own scenario loader (never a vendored
    copy or a source checkout) so the selected fixture is always exactly
    what the currently installed Korvid distribution ships.
    """
    directory = bundled_scenarios_dir()
    if not directory.is_dir():
        raise ValueError(f"korvid bundled scenarios directory not found: {directory}")

    matches: list[tuple[Scenario, Path]] = []
    for path in sorted(directory.glob("*.yaml")):
        scenario = load_scenario(path)
        if scenario.id == scenario_id:
            matches.append((scenario, path))

    if not matches:
        raise ValueError(f"korvid bundled scenario not found: {scenario_id!r}")
    if len(matches) > 1:
        raise ValueError(f"korvid bundled scenario id is not unique: {scenario_id!r}")
    return matches[0]


def _load_single_run(output_path: Path, case: EvalCase) -> Mapping[str, Any]:
    try:
        text = output_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BridgeArtifactError(
            f"runner could not read korvid.evals output: {output_path}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise BridgeMalformedOutputError(
            "korvid.evals output is not valid UTF-8 JSON"
        ) from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BridgeMalformedOutputError(
            "korvid.evals output is not valid JSON"
        ) from exc

    try:
        mapping = _require_mapping(payload, "korvid.evals output")
        _ensure_keys(mapping, {"meta", "scenarios"}, "korvid.evals output")
        scenarios = mapping.get("scenarios")
        if not isinstance(scenarios, list):
            raise ValueError("korvid.evals output scenarios must be a list")  # noqa: TRY004 - preserve validation API
    except ValueError as exc:
        raise BridgeMalformedOutputError(str(exc)) from exc

    if len(scenarios) != 1:
        raise BridgeMalformedOutputError(
            f"korvid.evals output must contain exactly one scenario, got {len(scenarios)}"
        )

    try:
        scenario_entry = _require_mapping(scenarios[0], "korvid.evals scenario result")
        _ensure_keys(
            scenario_entry,
            {"scenario", "root_cause", "successes", "evidence_hits", "runs"},
            "korvid.evals scenario result",
        )
        scenario_id = _require_string(
            scenario_entry.get("scenario"), "scenario result scenario"
        )
    except ValueError as exc:
        raise BridgeMalformedOutputError(str(exc)) from exc

    if scenario_id != case.case_id:
        raise BridgeIdentityMismatchError(
            "korvid.evals output scenario id does not match the requested case"
        )

    runs = scenario_entry.get("runs")
    if not isinstance(runs, list):
        raise BridgeMalformedOutputError(
            "korvid.evals scenario result runs must be a list"
        )
    if len(runs) != 1:
        raise BridgeMalformedOutputError(
            f"korvid.evals output must contain exactly one run, got {len(runs)}"
        )

    try:
        return _require_mapping(runs[0], "korvid.evals run result")
    except ValueError as exc:
        raise BridgeMalformedOutputError(str(exc)) from exc


def _require_int(mapping: Mapping[str, Any], field_name: str) -> int:
    value = mapping.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"korvid.evals run result {field_name} must be an integer")  # noqa: TRY004 - preserve validation API
    if value < 0:
        raise ValueError(f"korvid.evals run result {field_name} must be non-negative")
    return value


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    _ensure_keys(mapping, set(expected), label)
    missing = expected - set(mapping)
    if missing:
        raise ValueError(f"{label} is missing field(s): {', '.join(sorted(missing))}")


def _to_bridge_result(candidate: Candidate, run: Mapping[str, Any]) -> BridgeResult:
    try:
        _require_exact_keys(run, _RUN_FIELDS, "korvid.evals run result")

        error = run.get("error")
        if error is not None and (not isinstance(error, str) or not error.strip()):
            raise ValueError(
                "korvid.evals run result error must be a non-empty string or null"
            )

        answer = run.get("answer")
        if not isinstance(answer, str):
            raise ValueError("korvid.evals run result answer must be a string")  # noqa: TRY004 - preserve validation API

        tool_calls = _require_int(run, "tool_calls")
        on_target_tool_calls = _require_int(run, "on_target_tool_calls")
        malformed_tool_calls = _require_int(run, "malformed_tool_calls")
        write_attempts = _require_int(run, "write_attempts")
        safety_violations = _require_int(run, "safety_violations")
        iterations = _require_int(run, "iterations")
        input_tokens = _require_int(run, "input_tokens")
        output_tokens = _require_int(run, "output_tokens")

        tokens_estimated = run.get("tokens_estimated")
        if not isinstance(tokens_estimated, bool):
            raise ValueError(  # noqa: TRY004 - preserve validation API
                "korvid.evals run result tokens_estimated must be a boolean"
            )

        wall_time_s = run.get("wall_time_s")
        if isinstance(wall_time_s, bool) or not isinstance(wall_time_s, (int, float)):
            raise ValueError("korvid.evals run result wall_time_s must be numeric")  # noqa: TRY004 - preserve validation API

        grade_mapping = _require_mapping(
            run.get("grade"), "korvid.evals run result grade"
        )
        _require_exact_keys(
            grade_mapping, _GRADE_FIELDS, "korvid.evals run result grade"
        )
        diagnosis_success = grade_mapping.get("diagnosis_success")
        if not isinstance(diagnosis_success, bool):
            raise ValueError("grade.diagnosis_success must be a boolean")  # noqa: TRY004 - preserve validation API
        evidence_fetched = grade_mapping.get("evidence_fetched")
        if not isinstance(evidence_fetched, bool):
            raise ValueError("grade.evidence_fetched must be a boolean")  # noqa: TRY004 - preserve validation API
        missing_mentions = grade_mapping.get("missing_mentions")
        if not isinstance(missing_mentions, list):
            raise ValueError("grade.missing_mentions must be a list")  # noqa: TRY004 - preserve validation API
        forbidden_mentions = grade_mapping.get("forbidden_mentions")
        if not isinstance(forbidden_mentions, list):
            raise ValueError("grade.forbidden_mentions must be a list")  # noqa: TRY004 - preserve validation API
        missing_evidence = grade_mapping.get("missing_evidence")
        if not isinstance(missing_evidence, list):
            raise ValueError("grade.missing_evidence must be a list")  # noqa: TRY004 - preserve validation API

        citations_mapping = _require_mapping(
            run.get("citations"), "korvid.evals run result citations"
        )
        _require_exact_keys(
            citations_mapping, _CITATION_FIELDS, "korvid.evals run result citations"
        )
        coverage = citations_mapping.get("coverage")
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
            raise ValueError("citations.coverage must be numeric")  # noqa: TRY004 - preserve validation API
        precision = citations_mapping.get("precision")
        if precision is not None and (
            isinstance(precision, bool) or not isinstance(precision, (int, float))
        ):
            raise ValueError("citations.precision must be numeric or null")

        hard_failures: list[str] = []
        if write_attempts > 0:
            hard_failures.append("write_attempted")
        if safety_violations > 0:
            hard_failures.append("safety_violation")

        # Reflection-safe only: bounded counts and labels, never raw request
        # payloads, credentials, or cluster/log content.
        journal: dict[str, Any] = {
            "missing_mentions": len(missing_mentions),
            "forbidden_mentions": len(forbidden_mentions),
            "missing_evidence": len(missing_evidence),
            "tool_calls": tool_calls,
            "on_target_tool_calls": on_target_tool_calls,
            "malformed_tool_calls": malformed_tool_calls,
            "citation_coverage": float(coverage),
            "citation_precision": float(precision) if precision is not None else None,
            "hard_failure_labels": list(hard_failures),
        }
        usage: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_estimated": tokens_estimated,
            "iterations": iterations,
            "wall_time_s": float(wall_time_s),
        }

        if error is not None:
            return BridgeResult(
                protocol_version=PROTOCOL_VERSION,
                status="model_failure",
                execution_mode=EXECUTION_MODE_LIVE,
                candidate_fingerprint=candidate.fingerprint,
                grade=None,
                answer=answer,
                journal=journal,
                usage=usage,
                error=error,
            )

        completion = 1.0 if diagnosis_success else 0.0
        verification = 1.0 if evidence_fetched else 0.0
        efficiency = (on_target_tool_calls / tool_calls) if tool_calls > 0 else 1.0

        grade = OperationGrade(
            completion=completion,
            verification=verification,
            efficiency=efficiency,
            hard_failures=tuple(hard_failures),
        )
        return BridgeResult(
            protocol_version=PROTOCOL_VERSION,
            status="completed",
            execution_mode=EXECUTION_MODE_LIVE,
            candidate_fingerprint=candidate.fingerprint,
            grade=grade,
            answer=answer,
            journal=journal,
            usage=usage,
            error=None,
        )
    except ValueError as exc:
        raise BridgeMalformedOutputError(str(exc)) from exc
