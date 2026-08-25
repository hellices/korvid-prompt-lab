from __future__ import annotations

import json
import os
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


def _fake_round(path: Path, calls: Path, *, exit_code: int, config: bool) -> None:
    marker = (
        'printf "config_error\\n" > "$GROUNDING_ARTIFACT_ROOT/outcome-kind"'
        if config
        else ":"
    )
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
            exit {exit_code}
            """
        ),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_step(
    tmp_path: Path,
    *,
    kind: str = "search",
    round_exit: int = 70,
    config: bool = False,
    reflection_configured: bool = True,
    expected_hash: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, str | None]], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    state = _state(kind)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(_serialize_state(state)), encoding="utf-8")
    calls = tmp_path / "round-calls.json"
    fake_round = tmp_path / "fake-grounding-round.sh"
    _fake_round(fake_round, calls, exit_code=round_exit, config=config)
    output = tmp_path / "campaign-output"
    env = dict(os.environ)
    env.update(
        {
            "CAMPAIGN_CONTROL": str(_CONTROL),
            "CAMPAIGN_STATE": str(state_path),
            "CAMPAIGN_CANDIDATE": str(_CANDIDATE),
            "CAMPAIGN_OUTPUT_ROOT": str(output),
            "CAMPAIGN_EXPECTED_PRIOR_HASH": expected_hash or state_hash(state),
            "GROUNDING_CAMPAIGN": str(_EVALUATION),
            "GROUNDING_ROUND_SCRIPT": str(fake_round),
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
    control = load_optimization_campaign(_CONTROL, load_campaign(_EVALUATION))
    for kind in ("milestone", "confirm"):
        result, calls, _ = run_step(tmp_path / kind, kind=kind)
        assert result.returncode == 70, result.stderr
        assert len(calls) == 1
        assert calls[0]["GROUNDING_ACTION_KIND"] == kind.upper()
        assert calls[0]["GROUNDING_ROUND_TYPE"] == "evaluate"
        assert calls[0]["GROUNDING_MAX_METRIC_CALLS"] == "0"
        assert calls[0]["GROUNDING_EVALUATION_CASE_IDS"].splitlines() == list(
            control.milestone_case_ids
        )


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
