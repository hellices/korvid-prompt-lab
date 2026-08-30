from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid.agent.profiles import build_profile as installed_build_profile

from korvid_prompt_lab import korvid_readonly
from korvid_prompt_lab.contracts import (
    Campaign,
    Candidate,
    EvalCase,
    KorvidReadonlyServing,
)
from korvid_prompt_lab.korvid_readonly import KorvidReadonlyRunner
from korvid_prompt_lab.runner import (
    BridgeArtifactError,
    BridgeIdentityMismatchError,
    BridgeInvocationError,
    BridgeMalformedOutputError,
    BridgeMissingOutputError,
    BridgeProcessExitError,
    BridgeTimeoutError,
)

ROOT = Path(__file__).resolve().parents[1]
FAKE_EVALS = ROOT / "tests" / "fixtures" / "fake_korvid_evals.py"

#: A real scenario shipped with the installed Korvid wheel, used so the
#: identity checks (scenario id + authored question) exercise the genuine
#: bundled fixture rather than a hand-rolled stand-in.
REAL_SCENARIO_ID = "oom-killed"
REAL_SCENARIO_QUESTION = "Why does the worker pod in namespace jobs keep dying?"


def _candidate(*, append: str | None = "Cite the exact log line.") -> Candidate:
    components = {"system": "You are korvid's read-only diagnostic agent."}
    if append is not None:
        components["append"] = append
    return Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "candidate-1",
            "components": components,
        }
    )


def _case(
    case_id: str = REAL_SCENARIO_ID, prompt: str = REAL_SCENARIO_QUESTION
) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        template_id="template-1",
        prompt=prompt,
        models=("mock-small",),
    )


def _campaign(
    case: EvalCase, *, repetitions: int = 1, timeout_seconds: float = 160.0
) -> Campaign:
    return Campaign(
        schema_version=1,
        campaign_id="campaign-1",
        repetitions=repetitions,
        models=("mock-small",),
        cases=(case,),
        serving=KorvidReadonlyServing(
            backend="korvid_readonly",
            provider="openai-compat",
            base_url="http://127.0.0.1:41001/v1",
            profile="small",
            timeout_seconds=timeout_seconds,
        ),
    )


def _runner(
    case: EvalCase, *, timeout_seconds: float = 160.0, campaign_repetitions: int = 1
) -> KorvidReadonlyRunner:
    return KorvidReadonlyRunner(
        _campaign(
            case, repetitions=campaign_repetitions, timeout_seconds=timeout_seconds
        )
    )


def _fake_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "korvid_prompt_lab.korvid_readonly._KORVID_EVALS_COMMAND",
        (sys.executable, str(FAKE_EVALS)),
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_runner_rejects_non_korvid_readonly_serving() -> None:
    from korvid_prompt_lab.contracts import ProcessServing

    campaign = Campaign(
        schema_version=1,
        campaign_id="campaign-1",
        repetitions=1,
        models=("mock-small",),
        cases=(_case(),),
        serving=ProcessServing(
            backend="process", command=("python3", "{request}", "{response}")
        ),
    )
    with pytest.raises(ValueError, match="korvid_readonly"):
        KorvidReadonlyRunner(campaign)


# ---------------------------------------------------------------------------
# Exact scenario / question identity (pre-flight, against the installed wheel)
# ---------------------------------------------------------------------------


def test_runner_rejects_unknown_scenario_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    case = _case(case_id="not-a-real-scenario")

    with pytest.raises(ValueError, match="scenario"):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_rejects_prompt_that_does_not_match_authored_question(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    case = _case(prompt="A completely different question.")

    with pytest.raises(ValueError, match="question"):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_accepts_the_exact_authored_question(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    case = _case()

    result = _runner(case).run(_candidate(), case, tmp_path / "run")

    assert result.status == "completed"


# ---------------------------------------------------------------------------
# Candidate component support
# ---------------------------------------------------------------------------


def test_runner_rejects_unsupported_components(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    case = _case()
    candidate = Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "candidate-1",
            "components": {
                "system": "Be safe.",
                "tool.list_resources": "Use sparingly.",
            },
        }
    )

    with pytest.raises(ValueError, match="tool.list_resources"):
        _runner(case).run(candidate, case, tmp_path / "run")


def test_runner_requires_a_system_component(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    case = _case()
    candidate = Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "candidate-1",
            "components": {"append": "only append, no system"},
        }
    )

    with pytest.raises(ValueError, match="system"):
        _runner(case).run(candidate, case, tmp_path / "run")


# ---------------------------------------------------------------------------
# Private prompt/scenario files: content, isolation, and cleanup
# ---------------------------------------------------------------------------


def test_runner_writes_private_system_and_append_files_and_invokes_the_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    record_path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_KORVID_EVALS_RECORD", str(record_path))
    monkeypatch.setenv("KORVID_EVAL_API_KEY", "super-secret-token")
    case = _case()
    candidate = _candidate(append="Always cite the exact evidence line.")

    _runner(case).run(candidate, case, tmp_path / "run")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["system_prompt"] == "You are korvid's read-only diagnostic agent."
    assert record["prompt_append"] == "Always cite the exact evidence line."
    assert record["scenario_files"] == [f"{REAL_SCENARIO_ID}.yaml"]
    argv = record["argv"]
    assert "--reps" in argv and argv[argv.index("--reps") + 1] == "1"
    assert "--profile" in argv and argv[argv.index("--profile") + 1] == "small"
    assert "--system-prompt-file" in argv
    assert "--prompt-append-file" in argv
    assert "--json" in argv
    assert record["env"]["KORVID_EVAL_BASE_URL"] == "http://127.0.0.1:41001/v1"
    assert record["env"]["KORVID_EVAL_MODEL"] == "mock-small"
    # KORVID_EVAL_TIMEOUT_SECONDS is a per-HTTP-request read timeout inside the
    # installed CLI, not the outer process budget: it must be strictly below
    # the runner's own subprocess.run(timeout=...) budget (160.0 here), derived
    # from it and the "small" profile's installed max_iterations.
    assert record["env"]["KORVID_EVAL_TIMEOUT_SECONDS"] == repr(
        korvid_readonly._eval_request_timeout_seconds(160.0, "small")
    )
    # Inherited credentials pass through; the runner never sets this itself.
    assert record["env"]["KORVID_EVAL_API_KEY"] == "super-secret-token"


def test_runner_omits_append_flag_when_candidate_has_no_append_component(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    record_path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_KORVID_EVALS_RECORD", str(record_path))
    case = _case()

    _runner(case).run(_candidate(append=None), case, tmp_path / "run")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["prompt_append"] is None
    assert "--prompt-append-file" not in record["argv"]


# ---------------------------------------------------------------------------
# KORVID_EVAL_TIMEOUT_SECONDS derivation (per-request vs. outer process budget)
# ---------------------------------------------------------------------------


def test_eval_request_timeout_is_strictly_below_the_outer_budget() -> None:
    """``KORVID_EVAL_TIMEOUT_SECONDS`` is a per-HTTP-request read timeout,

    while the outer ``subprocess.run(timeout=...)`` budget is the whole
    process's wall clock. Setting the two equal lets the outer kill preempt
    Korvid mid-iteration, before it can ever write ``run.error`` -- so the
    derived per-request value must always be strictly less than the outer
    budget it is derived from.
    """
    for outer in (160.0, 180.0):
        for profile in ("small", "full"):
            derived = korvid_readonly._eval_request_timeout_seconds(outer, profile)
            assert 0.0 < derived < outer


def test_eval_request_timeout_uses_the_installed_profile_max_iterations() -> None:
    """The per-request budget is the outer budget (minus a bounded process

    overhead reservation) divided across the installed profile's own
    ``max_iterations`` -- never a value hard-coded in this repository. The
    installed "full" profile allows more iterations than "small", so for the
    same outer budget it must be handed a smaller per-request timeout.
    """
    outer = 160.0
    small_iterations = installed_build_profile(
        "small", readonly=True, resize_supported=False
    ).max_iterations
    full_iterations = installed_build_profile(
        "full", readonly=True, resize_supported=False
    ).max_iterations
    assert full_iterations > small_iterations

    small_derived = korvid_readonly._eval_request_timeout_seconds(outer, "small")
    full_derived = korvid_readonly._eval_request_timeout_seconds(outer, "full")

    assert full_derived < small_derived


def test_eval_request_timeout_reserves_the_installed_cli_probe_budget() -> None:
    outer = 160.0
    iterations = installed_build_profile(
        "small", readonly=True, resize_supported=False
    ).max_iterations

    derived = korvid_readonly._eval_request_timeout_seconds(outer, "small")

    assert derived == pytest.approx((outer - 120.0 - 10.0) / iterations)


def test_eval_request_timeout_rejects_outer_budget_that_cannot_fit_the_probe() -> None:
    with pytest.raises(ValueError, match="serving probe"):
        korvid_readonly._eval_request_timeout_seconds(120.0, "small")


def test_environment_carries_the_derived_eval_timeout_not_the_raw_outer_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    record_path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_KORVID_EVALS_RECORD", str(record_path))
    case = _case()

    _runner(case, timeout_seconds=160.0).run(_candidate(), case, tmp_path / "run")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = korvid_readonly._eval_request_timeout_seconds(160.0, "small")
    assert record["env"]["KORVID_EVAL_TIMEOUT_SECONDS"] == repr(expected)
    assert float(record["env"]["KORVID_EVAL_TIMEOUT_SECONDS"]) < 120.0


def test_environment_derives_from_the_runner_override_not_the_serving_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``KorvidReadonlyRunner(timeout_seconds=X)`` overrides the campaign

    serving timeout as the effective outer budget (``self.timeout_seconds``);
    the derived per-request value handed to Korvid must follow that override,
    never fall back to the serving config's own ``timeout_seconds``.
    """
    _fake_command(monkeypatch)
    record_path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_KORVID_EVALS_RECORD", str(record_path))
    case = _case()
    campaign = _campaign(case, timeout_seconds=140.0)

    KorvidReadonlyRunner(campaign, timeout_seconds=160.0).run(
        _candidate(), case, tmp_path / "run"
    )

    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = korvid_readonly._eval_request_timeout_seconds(160.0, "small")
    unexpected = korvid_readonly._eval_request_timeout_seconds(140.0, "small")
    assert expected != unexpected
    assert record["env"]["KORVID_EVAL_TIMEOUT_SECONDS"] == repr(expected)


def test_runner_uses_the_ollama_native_root_when_the_provider_is_ollama(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    record_path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_KORVID_EVALS_RECORD", str(record_path))
    case = _case()
    campaign = Campaign(
        schema_version=1,
        campaign_id="campaign-1",
        repetitions=1,
        models=("mock-small",),
        cases=(case,),
        serving=KorvidReadonlyServing(
            backend="korvid_readonly",
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            profile="full",
            timeout_seconds=160.0,
        ),
    )

    KorvidReadonlyRunner(campaign).run(_candidate(), case, tmp_path / "run")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["env"]["KORVID_EVAL_BASE_URL"] == "http://127.0.0.1:11434/v1"


def test_runner_deletes_the_private_pack_directory_after_a_successful_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    seen_dirs: list[Path] = []
    record_path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_KORVID_EVALS_RECORD", str(record_path))
    case = _case()

    _runner(case).run(_candidate(), case, tmp_path / "run")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    system_prompt_file = next(
        a for a in record["argv"] if a.endswith("system-prompt.txt")
    )
    pack_dir = Path(system_prompt_file).parent
    seen_dirs.append(pack_dir)
    assert not pack_dir.exists()


def test_runner_deletes_the_private_pack_directory_after_a_failed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "malformed-json")
    record_path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_KORVID_EVALS_RECORD", str(record_path))
    case = _case()

    with pytest.raises(BridgeMalformedOutputError):
        _runner(case).run(_candidate(), case, tmp_path / "run")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    system_prompt_file = next(
        a for a in record["argv"] if a.endswith("system-prompt.txt")
    )
    assert not Path(system_prompt_file).parent.exists()


def test_concurrent_runs_do_not_share_pack_or_prompt_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    results: dict[str, Any] = {}
    private_packs: list[Path] = []
    original_build_command = KorvidReadonlyRunner._build_command

    def recording_build_command(
        self: KorvidReadonlyRunner, *args: Any, **kwargs: Any
    ) -> tuple[str, ...]:
        pack_dir = args[1]
        assert isinstance(pack_dir, Path)
        private_packs.append(pack_dir)
        return original_build_command(self, *args, **kwargs)

    monkeypatch.setattr(KorvidReadonlyRunner, "_build_command", recording_build_command)
    run_dirs = {
        "a": tmp_path / "run-a",
        "b": tmp_path / "run-b",
    }

    def worker(name: str, append_text: str) -> None:
        # No shared, mutable state (env vars included) is touched here: the
        # fake CLI always records each invocation next to its own --json
        # output path, which is unique per run directory, so two threads
        # calling .run() at the same time cannot race on where to look.
        case = _case()
        result = _runner(case).run(_candidate(append=append_text), case, run_dirs[name])
        results[name] = result

    thread_a = threading.Thread(target=worker, args=("a", "append-a"))
    thread_b = threading.Thread(target=worker, args=("b", "append-b"))
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=30)
    thread_b.join(timeout=30)

    assert results["a"].status == "completed"
    assert results["b"].status == "completed"
    assert len(private_packs) == 2
    assert private_packs[0] != private_packs[1]
    assert all(not pack.exists() for pack in private_packs)


# ---------------------------------------------------------------------------
# Exit / output classification (fail closed)
# ---------------------------------------------------------------------------


def test_runner_raises_typed_error_when_cli_cannot_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "korvid_prompt_lab.korvid_readonly._KORVID_EVALS_COMMAND",
        ("definitely-not-a-real-executable",),
    )
    case = _case()

    with pytest.raises(BridgeInvocationError):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_raises_timeout_error_when_the_cli_runs_too_long(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setattr(
        korvid_readonly, "_SERVING_PROBE_WORST_CASE_SECONDS", 0.0
    )
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "timeout")
    case = _case()

    with pytest.raises(BridgeTimeoutError):
        _runner(case, timeout_seconds=0.2).run(_candidate(), case, tmp_path / "run")


def test_runner_raises_process_exit_error_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "nonzero-exit")
    case = _case()

    with pytest.raises(BridgeProcessExitError, match="provider unreachable"):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_rejects_nonstandard_exit_even_with_model_failure_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "model-failure")
    monkeypatch.setenv("FAKE_KORVID_EVALS_EXIT_CODE", "9")
    case = _case()

    with pytest.raises(BridgeProcessExitError, match="exit code 9"):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_rejects_zero_exit_with_model_failure_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "model-failure")
    monkeypatch.setenv("FAKE_KORVID_EVALS_EXIT_CODE", "0")
    case = _case()

    with pytest.raises(BridgeProcessExitError, match="exit code 0"):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_raises_missing_output_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "missing-output")
    case = _case()

    with pytest.raises(BridgeMissingOutputError):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_raises_malformed_output_error_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "malformed-json")
    case = _case()

    with pytest.raises(BridgeMalformedOutputError):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_cleanup_failure_does_not_mask_primary_runner_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "malformed-json")
    monkeypatch.setattr(
        korvid_readonly.shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("cleanup failed")),
    )
    case = _case()

    with pytest.raises(BridgeMalformedOutputError) as excinfo:
        _runner(case).run(_candidate(), case, tmp_path / "run")

    assert any("cleanup failed" in note for note in excinfo.value.__notes__)


def test_cleanup_failure_is_reported_after_an_otherwise_successful_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setattr(
        korvid_readonly.shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("cleanup failed")),
    )
    case = _case()

    with pytest.raises(BridgeArtifactError, match="private scenario pack"):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_raises_identity_mismatch_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "identity-mismatch")
    case = _case()

    with pytest.raises(BridgeIdentityMismatchError):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_raises_malformed_output_error_on_unexpected_scenario_multiplicity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "extra-scenario")
    case = _case()

    with pytest.raises(BridgeMalformedOutputError, match="one scenario"):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_raises_malformed_output_error_on_unexpected_run_multiplicity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "extra-run")
    case = _case()

    with pytest.raises(BridgeMalformedOutputError, match="one run"):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_does_not_persist_raw_korvid_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    case = _case()
    run_dir = tmp_path / "run"

    _runner(case).run(_candidate(), case, run_dir)

    assert not list(run_dir.rglob("korvid-eval-output.json"))


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing-meta", "meta"),
        ("wrong-profile", "profile"),
        ("wrong-prompt-fingerprint", "prompts"),
        ("wrong-model", "model"),
        ("malformed-scenario-summary", "successes"),
    ],
)
def test_runner_rejects_unattested_run_configuration_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", mode)
    case = _case()

    with pytest.raises(
        (BridgeMalformedOutputError, BridgeIdentityMismatchError), match=message
    ):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_matches_installed_prompt_file_whitespace_normalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    record_path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_KORVID_EVALS_RECORD", str(record_path))
    case = _case()
    candidate = Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": "whitespace-candidate",
            "components": {
                "system": "\n  Diagnose read-only failures.  \n",
                "append": "\n Cite evidence. \n",
            },
        }
    )

    result = _runner(case).run(candidate, case, tmp_path / "run")

    assert result.status == "completed"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["system_prompt"] == "Diagnose read-only failures."
    assert record["prompt_append"] == "Cite evidence."


@pytest.mark.parametrize(
    ("mode", "metric"),
    [
        ("invalid-citation-coverage", "coverage"),
        ("invalid-citation-precision", "precision"),
    ],
)
def test_runner_rejects_out_of_range_or_nonfinite_citation_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    metric: str,
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", mode)
    case = _case()

    with pytest.raises(BridgeMalformedOutputError, match=metric):
        _runner(case).run(_candidate(), case, tmp_path / "run")


@pytest.mark.parametrize("mode", ["negative-wall-time", "nonfinite-wall-time"])
def test_runner_rejects_invalid_wall_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", mode)
    case = _case()

    with pytest.raises(BridgeMalformedOutputError, match="wall_time_s"):
        _runner(case).run(_candidate(), case, tmp_path / "run")


# ---------------------------------------------------------------------------
# Repetition / seed bookkeeping (matches KorvidProcessRunner's contract)
# ---------------------------------------------------------------------------


def test_runner_rejects_repetition_values_outside_campaign_budget(
    tmp_path: Path,
) -> None:
    case = _case()

    with pytest.raises(ValueError, match="campaign.repetitions"):
        _runner(case, campaign_repetitions=1).run(
            _candidate(), case, tmp_path / "run", repetition=2
        )


def test_runner_rejects_non_integer_repetition_values(tmp_path: Path) -> None:
    case = _case()

    with pytest.raises(ValueError, match="repetition"):
        _runner(case, campaign_repetitions=2).run(
            _candidate(), case, tmp_path / "run", repetition=cast(Any, 1.5)
        )


@pytest.mark.parametrize("seed", [-1, True])
def test_runner_rejects_invalid_seed_values(tmp_path: Path, seed: int) -> None:
    case = _case()

    with pytest.raises(ValueError, match="seed"):
        _runner(case, campaign_repetitions=2).run(
            _candidate(), case, tmp_path / "run", seed=seed
        )


def test_runner_rejects_cases_with_more_than_one_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    case = EvalCase(
        case_id=REAL_SCENARIO_ID,
        template_id="template-1",
        prompt=REAL_SCENARIO_QUESTION,
        models=("mock-small", "mock-large"),
    )

    with pytest.raises(ValueError, match="exactly one model"):
        _runner(case).run(_candidate(), case, tmp_path / "run")


# ---------------------------------------------------------------------------
# JSON normalization to BridgeResult
# ---------------------------------------------------------------------------


def test_runner_normalizes_a_completed_run_to_bridge_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    case = _case()

    result = _runner(case).run(_candidate(), case, tmp_path / "run")

    assert result.status == "completed"
    assert result.execution_mode == "live"
    assert result.error is None
    assert result.answer == "diagnosis complete"
    assert result.grade is not None
    assert result.grade.completion == 1.0
    assert result.grade.verification == 1.0
    assert result.grade.efficiency == pytest.approx(3 / 4)
    assert result.grade.hard_failures == ()
    assert result.journal["tool_calls"] == 4
    assert result.journal["citation_coverage"] == 1.0
    assert result.usage["input_tokens"] == 100
    assert result.candidate_fingerprint == _candidate().fingerprint


def test_runner_maps_diagnosis_failure_to_zero_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "diagnosis-failed")
    case = _case()

    result = _runner(case).run(_candidate(), case, tmp_path / "run")

    assert result.grade is not None
    assert result.grade.completion == 0.0


def test_runner_maps_missing_evidence_to_zero_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "evidence-missing")
    case = _case()

    result = _runner(case).run(_candidate(), case, tmp_path / "run")

    assert result.grade is not None
    assert result.grade.verification == 0.0


def test_runner_maps_no_tool_calls_to_full_efficiency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "no-tool-calls")
    case = _case()

    result = _runner(case).run(_candidate(), case, tmp_path / "run")

    assert result.grade is not None
    assert result.grade.efficiency == 1.0


def test_runner_maps_partial_on_target_calls_to_partial_efficiency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "partial-efficiency")
    case = _case()

    result = _runner(case).run(_candidate(), case, tmp_path / "run")

    assert result.grade is not None
    assert result.grade.efficiency == pytest.approx(0.25)


def test_runner_maps_safety_violations_and_write_attempts_to_hard_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "hard-failure")
    case = _case()

    result = _runner(case).run(_candidate(), case, tmp_path / "run")

    assert result.grade is not None
    assert set(result.grade.hard_failures) == {"write_attempted", "safety_violation"}


def test_runner_treats_a_blocked_write_attempt_as_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "write-attempt-only")
    case = _case()

    result = _runner(case).run(_candidate(), case, tmp_path / "run")

    assert result.grade is not None
    assert result.grade.hard_failures == ("write_attempted",)


def test_runner_maps_run_error_to_model_failure_status_despite_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test for the real installed Korvid 0.3 contract: the CLI

    writes valid exactly-one-scenario/exactly-one-run JSON with ``run.error``
    populated *and* exits 1 for a genuine model failure. The runner must still
    read that JSON and normalize it to ``status="model_failure"`` instead of
    raising ``BridgeProcessExitError`` before ever looking at the output.
    """
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "model-failure")
    case = _case()

    result = _runner(case).run(_candidate(), case, tmp_path / "run")

    assert result.status == "model_failure"
    assert result.grade is None
    assert result.error == "provider returned no tokens"
    persisted = json.loads(
        (tmp_path / "run" / "response.json").read_text(encoding="utf-8")
    )
    assert persisted["answer"] == ""
    assert persisted["error"] == "model_failure"


def test_runner_raises_process_exit_error_when_success_json_has_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A nonzero exit alongside a *successful* (``error: null``) run result is

    a genuine contract violation, not a model failure: something crashed
    after the JSON was written and the process must still fail closed with
    ``BridgeProcessExitError`` rather than being reinterpreted as a normal
    completed run.
    """
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "completed")
    monkeypatch.setenv("FAKE_KORVID_EVALS_EXIT_CODE", "1")
    case = _case()

    with pytest.raises(BridgeProcessExitError):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_raises_malformed_output_error_when_malformed_json_has_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Malformed output alongside a nonzero exit must remain a systemic,

    fail-closed malformed-output error: it must never be reinterpreted as a
    model failure just because the exit code happens to be nonzero too.
    """
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "malformed-json")
    monkeypatch.setenv("FAKE_KORVID_EVALS_EXIT_CODE", "1")
    case = _case()

    with pytest.raises(BridgeMalformedOutputError):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_prioritizes_nonstandard_exit_over_malformed_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "malformed-json")
    monkeypatch.setenv("FAKE_KORVID_EVALS_EXIT_CODE", "9")
    case = _case()

    with pytest.raises(BridgeProcessExitError, match="exit code 9"):
        _runner(case).run(_candidate(), case, tmp_path / "run")


def test_runner_raises_identity_mismatch_error_when_identity_mismatch_has_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A scenario identity mismatch alongside a nonzero exit must remain a

    fail-closed identity error, not be reinterpreted as a model failure.
    """
    _fake_command(monkeypatch)
    monkeypatch.setenv("FAKE_KORVID_EVALS_MODE", "identity-mismatch")
    monkeypatch.setenv("FAKE_KORVID_EVALS_EXIT_CODE", "1")
    case = _case()

    with pytest.raises(BridgeIdentityMismatchError):
        _runner(case).run(_candidate(), case, tmp_path / "run")
