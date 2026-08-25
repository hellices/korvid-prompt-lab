from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import textwrap
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from korvid_prompt_lab.campaign_artifacts import _serialize_state
from korvid_prompt_lab.campaigns import (
    CampaignScore,
    CampaignState,
    initial_state,
    load_optimization_campaign,
    state_hash,
)
from korvid_prompt_lab.config import load_campaign, load_candidate

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "run-optimization-campaign-step.sh"
_CONTROL = _ROOT / "examples/optimization-campaigns/qwen3-small-operator.yaml"
_EVALUATION = _ROOT / "examples/campaigns/aks-small-operator-qualification.yaml"
_CANDIDATE = _ROOT / "examples/candidates/shipped-small.yaml"


def _state(kind: str) -> CampaignState:
    os.environ.setdefault("KORVID_AKS_MODEL", "qwen3:0.6b")
    os.environ.setdefault("KORVID_AKS_NAMESPACE", "ollama")
    os.environ.setdefault("KORVID_AKS_SERVICE", "ollama")
    control = load_optimization_campaign(_CONTROL, load_campaign(_EVALUATION))
    state = initial_state(
        control,
        prompt_lab_revision="a" * 40,
        korvid_revision="b" * 40,
        started_at=datetime.now(tz=UTC),
    )
    fingerprint = load_candidate(_CANDIDATE).fingerprint
    state = replace(
        state,
        champion_fingerprint=fingerprint,
        champion_score=CampaignScore(
            fingerprint=fingerprint,
            aggregate=0.5,
            hard_safety_failures=0,
            core_regression=False,
            systemic_failures=0,
            pass_at_3=0.0,
            pass_at_5=0.0,
        ),
    )
    if kind in {"milestone", "confirm"}:
        state = replace(
            state,
            stage_index=len(control.stages) - 1,
            seed_index=len(control.stages[-1].seeds),
        )
    if kind == "confirm":
        state = replace(state, milestone_passed=True)
    return state


def _fake_round(
    path: Path,
    calls: Path,
    *,
    exit_code: int,
    config: bool,
    evidence_mode: str,
) -> None:
    marker = (
        'printf "config_error\\n" > "$GROUNDING_ARTIFACT_ROOT/outcome-kind"'
        if config
        else ":"
    )
    evidence = {
        "none": ":",
        "malformed": (
            'mkdir -p "$GROUNDING_ARTIFACT_ROOT/safe-evidence"; '
            'printf "{not-json\\n" > '
            '"$GROUNDING_ARTIFACT_ROOT/safe-evidence/round-summary.json"'
        ),
        "contradictory": (
            'mkdir -p "$GROUNDING_ARTIFACT_ROOT/safe-evidence"; '
            'touch "$GROUNDING_ARTIFACT_ROOT/safe-evidence/contradictory"'
        ),
    }[evidence_mode]
    path.write_text(
        "#!/usr/bin/env bash\n"
        + textwrap.dedent(
            f"""\
            set -Eeuo pipefail
            python3 - "{calls}" <<'PY'
            import json
            import os
            import sys
            from pathlib import Path
            keys = (
                "GROUNDING_ACTION_KIND",
                "GROUNDING_ROUND_TYPE",
                "GROUNDING_MODEL",
                "KORVID_AKS_MODEL",
                "GROUNDING_MAX_METRIC_CALLS",
                "GROUNDING_SEED",
                "GROUNDING_TRAIN_CASE_IDS",
                "GROUNDING_VALIDATION_CASE_IDS",
                "GROUNDING_MILESTONE_CASE_IDS",
                "GROUNDING_EVALUATION_CASE_IDS",
                "GROUNDING_CAMPAIGN_ACTION_ID",
            )
            target = Path(sys.argv[1])
            entries = json.loads(target.read_text()) if target.exists() else []
            entries.append({{key: os.environ.get(key) for key in keys}})
            target.write_text(json.dumps(entries))
            PY
            mkdir -p "$GROUNDING_ARTIFACT_ROOT"
            {marker}
            {evidence}
            exit {exit_code}
            """
        ),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _grounding_process_fakes(root: Path, calls: Path) -> Path:
    fake_bin = root / "grounding-bin"
    fake_bin.mkdir()

    def write(name: str, body: str) -> None:
        path = fake_bin / name
        path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    write(
        "az",
        'if [[ "$*" == *"nodepool show"* ]]; then echo 0; fi\n',
    )
    write("kubectl", "exit 0\n")
    write("kubelogin", "exit 0\n")
    write("uv", "exit 0\n")
    write("korvid-grounding-report", "exit 0\n")
    prompt_lab = fake_bin / "korvid-prompt-lab"
    prompt_lab.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"path = Path({str(calls)!r})\n"
        "entries = json.loads(path.read_text()) if path.exists() else []\n"
        "entries.append(sys.argv[1:])\n"
        "path.write_text(json.dumps(entries))\n"
        "raise SystemExit(70 if sys.argv[1] == 'evaluate' else 0)\n",
        encoding="utf-8",
    )
    prompt_lab.chmod(prompt_lab.stat().st_mode | stat.S_IXUSR)
    return fake_bin


def run_step(
    tmp_path: Path,
    *,
    kind: str = "search",
    round_exit: int = 70,
    config: bool = False,
    reflection_configured: bool = True,
    evidence_mode: str = "none",
    record_campaign_calls: bool = False,
    real_round: bool = False,
    campaign_command_mode: str = "delegate",
    expected_hash: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, str | None]], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    state = _state(kind)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(_serialize_state(state)), encoding="utf-8")
    calls = tmp_path / "round-calls.json"
    fake_round = tmp_path / "fake-grounding-round.sh"
    if not real_round:
        _fake_round(
            fake_round,
            calls,
            exit_code=round_exit,
            config=config,
            evidence_mode=evidence_mode,
        )
    output = tmp_path / "campaign-output"
    env = dict(os.environ)
    env.update(
        {
            "CAMPAIGN_CONTROL": str(_CONTROL),
            "CAMPAIGN_STATE": str(state_path),
            "CAMPAIGN_CANDIDATE": str(_CANDIDATE),
            "CAMPAIGN_OUTPUT_ROOT": str(output),
            "CAMPAIGN_EXPECTED_PRIOR_HASH": expected_hash or state_hash(state),
            "GROUNDING_CAMPAIGN": str(_EVALUATION.relative_to(_ROOT)),
            "GROUNDING_REFLECTION_MODEL": "openai/gpt-4.1-mini",
            "GROUNDING_REFLECTION_CREDENTIAL": "test-token",
            "KORVID_AKS_NAMESPACE": "ollama",
            "KORVID_AKS_SERVICE": "ollama",
            "KORVID_SOURCE_ROOT": "/fake/korvid",
            "WORKFLOW_RUN_URL": "https://example.test/run/1",
            "PROMPT_LAB_REVISION": "a" * 40,
            "KORVID_REVISION": "b" * 40,
        }
    )
    if real_round:
        fake_bin = _grounding_process_fakes(tmp_path, calls)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
    else:
        env["GROUNDING_ROUND_SCRIPT"] = str(fake_round)
    if record_campaign_calls:
        real_campaign = shutil.which("korvid-campaign")
        assert real_campaign is not None
        campaign_calls = tmp_path / "campaign-calls.txt"
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        shim = fake_bin / "korvid-campaign"
        shim.write_text(
            "#!/usr/bin/env bash\n"
            'printf "%s\\n" "$1" >> "$CAMPAIGN_CALLS"\n'
            'if [[ "$CAMPAIGN_COMMAND_MODE:$1" == '
            '"evidence-advance-fails:validate-evidence" ]]; then exit 0; fi\n'
            'if [[ "$CAMPAIGN_COMMAND_MODE:$1" == '
            '"evidence-advance-fails:advance" ]]; then exit 1; fi\n'
            'if [[ "$CAMPAIGN_COMMAND_MODE:$1" == '
            '"reject-contradictory:validate-evidence" ]]; then exit 1; fi\n'
            'exec "$REAL_KORVID_CAMPAIGN" "$@"\n',
            encoding="utf-8",
        )
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["CAMPAIGN_CALLS"] = str(campaign_calls)
        env["REAL_KORVID_CAMPAIGN"] = real_campaign
        env["CAMPAIGN_COMMAND_MODE"] = campaign_command_mode
    if not reflection_configured:
        env.pop("GROUNDING_REFLECTION_MODEL", None)
        env.pop("GROUNDING_REFLECTION_CREDENTIAL", None)
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    recorded = json.loads(calls.read_text()) if calls.exists() else []
    if record_campaign_calls:
        recorded.append(
            {
                "_campaign_calls": (
                    (tmp_path / "campaign-calls.txt").read_text().splitlines()
                )
            }
        )
    return result, recorded, output


def test_search_exports_exact_splits_and_validation_scope(tmp_path: Path) -> None:
    result, calls, output = run_step(tmp_path)

    assert result.returncode == 70, result.stderr
    assert len(calls) == 1
    call = calls[0]
    control = load_optimization_campaign(_CONTROL, load_campaign(_EVALUATION))
    assert call["GROUNDING_ACTION_KIND"] == "SEARCH"
    assert call["GROUNDING_ROUND_TYPE"] == "optimize-evaluate"
    assert call["GROUNDING_TRAIN_CASE_IDS"].splitlines() == list(
        control.train_case_ids
    )
    assert call["GROUNDING_VALIDATION_CASE_IDS"].splitlines() == list(
        control.validation_case_ids
    )
    assert call["GROUNDING_EVALUATION_CASE_IDS"].splitlines() == list(
        control.validation_case_ids
    )
    state = json.loads((output / "campaign-state.json").read_text())
    assert state["metric_calls_used"] == 0
    assert state["retries_used"] == 1


def test_milestone_and_confirm_are_evaluate_only_on_holdout(
    tmp_path: Path,
) -> None:
    _state("search")
    control = load_optimization_campaign(_CONTROL, load_campaign(_EVALUATION))
    for kind in ("milestone", "confirm"):
        result, calls, _ = run_step(tmp_path / kind, kind=kind)
        assert result.returncode == 70, result.stderr
        assert len(calls) == 1
        assert calls[0]["GROUNDING_ACTION_KIND"] == kind.upper()
        assert calls[0]["GROUNDING_ROUND_TYPE"] == "evaluate"
        assert calls[0]["GROUNDING_MAX_METRIC_CALLS"] is None
        assert calls[0]["GROUNDING_EVALUATION_CASE_IDS"].splitlines() == list(
            control.milestone_case_ids
        )


def test_real_milestone_and_confirm_reach_only_exact_holdout_evaluation(
    tmp_path: Path,
) -> None:
    _state("search")
    control = load_optimization_campaign(_CONTROL, load_campaign(_EVALUATION))
    for kind in ("milestone", "confirm"):
        result, calls, _ = run_step(
            tmp_path / kind,
            kind=kind,
            real_round=True,
        )

        assert result.returncode == 70, result.stderr
        evaluate_calls = [call for call in calls if call[0] == "evaluate"]
        assert len(evaluate_calls) == 1, result.stderr
        argv = evaluate_calls[0]
        selected = [
            argv[index + 1]
            for index, value in enumerate(argv)
            if value == "--case-id"
        ]
        assert selected == list(control.milestone_case_ids)
        assert not any(call[0] == "optimize" for call in calls)


def test_digest_config_error_consumes_no_budget_or_retry(tmp_path: Path) -> None:
    result, calls, output = run_step(tmp_path, config=True)

    assert result.returncode == 70
    assert len(calls) == 1
    state = json.loads((output / "campaign-state.json").read_text())
    assert state["status"] == "system_error"
    assert state["metric_calls_used"] == 0
    assert state["retries_used"] == 0
    assert state["stop_reason"].startswith("config_error:")


def test_reflection_config_error_advances_without_running_round(
    tmp_path: Path,
) -> None:
    result, calls, output = run_step(tmp_path, reflection_configured=False)

    assert result.returncode == 70
    assert calls == []
    state = json.loads((output / "campaign-state.json").read_text())
    assert state["status"] == "system_error"
    assert state["metric_calls_used"] == 0
    assert state["retries_used"] == 0
    assert state["stop_reason"].startswith("config_error:")


def test_stale_expected_hash_never_runs_expensive_action(tmp_path: Path) -> None:
    result, calls, output = run_step(
        tmp_path,
        expected_hash="sha256:" + "f" * 64,
    )

    assert result.returncode == 70
    assert calls == []
    assert not output.exists()


def test_missing_safe_evidence_advances_system_error_once(tmp_path: Path) -> None:
    prior = _state("search")
    result, calls, output = run_step(
        tmp_path,
        round_exit=0,
        record_campaign_calls=True,
    )

    assert result.returncode == 70
    campaign_calls = calls[-1]["_campaign_calls"]
    assert campaign_calls.count("validate-evidence") == 1
    assert campaign_calls.count("advance") == 1
    state = json.loads((output / "campaign-state.json").read_text())
    assert state["retries_used"] == prior.retries_used + 1
    assert state["metric_calls_used"] == prior.metric_calls_used
    assert state["champion_score"] == _serialize_state(prior)["champion_score"]


def test_grounding_exit_70_directly_advances_system_error_once(
    tmp_path: Path,
) -> None:
    result, calls, output = run_step(
        tmp_path,
        round_exit=70,
        record_campaign_calls=True,
    )

    assert result.returncode == 70
    campaign_calls = calls[-1]["_campaign_calls"]
    assert "validate-evidence" not in campaign_calls
    assert campaign_calls.count("advance") == 1
    state = json.loads((output / "campaign-state.json").read_text())
    assert state["metric_calls_used"] == 0
    assert state["retries_used"] == 1


def test_malformed_safe_evidence_advances_system_error_once(
    tmp_path: Path,
) -> None:
    prior = _state("search")
    result, calls, output = run_step(
        tmp_path,
        round_exit=1,
        evidence_mode="malformed",
        record_campaign_calls=True,
    )

    assert result.returncode == 70
    campaign_calls = calls[-1]["_campaign_calls"]
    assert campaign_calls.count("validate-evidence") == 1
    assert campaign_calls.count("advance") == 1
    state = json.loads((output / "campaign-state.json").read_text())
    assert state["retries_used"] == prior.retries_used + 1
    assert state["metric_calls_used"] == prior.metric_calls_used
    assert state["champion_score"] == _serialize_state(prior)["champion_score"]


def test_contradictory_safe_evidence_advances_system_error_once(
    tmp_path: Path,
) -> None:
    prior = _state("search")
    result, calls, output = run_step(
        tmp_path,
        round_exit=0,
        evidence_mode="contradictory",
        record_campaign_calls=True,
        campaign_command_mode="reject-contradictory",
    )

    assert result.returncode == 70
    campaign_calls = calls[-1]["_campaign_calls"]
    assert campaign_calls.count("validate-evidence") == 1
    assert campaign_calls.count("advance") == 1
    state = json.loads((output / "campaign-state.json").read_text())
    assert state["retries_used"] == prior.retries_used + 1
    assert state["metric_calls_used"] == prior.metric_calls_used
    assert state["champion_score"] == _serialize_state(prior)["champion_score"]


def test_ambiguous_evidence_advance_failure_never_falls_back(
    tmp_path: Path,
) -> None:
    result, calls, output = run_step(
        tmp_path,
        round_exit=0,
        record_campaign_calls=True,
        campaign_command_mode="evidence-advance-fails",
    )

    assert result.returncode == 70
    campaign_calls = calls[-1]["_campaign_calls"]
    assert campaign_calls.count("validate-evidence") == 1
    assert campaign_calls.count("advance") == 1
    assert "render" not in campaign_calls
    assert not output.exists()
