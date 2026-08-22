from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Self

import pytest
import yaml  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid_prompt_lab.aks import (
    AKSMissingToolError,
    AKSPortForwardError,
    AKSPreflightTransientError,
)
from korvid_prompt_lab.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(args)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _candidate_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate_id": "shipped-small",
        "components": {
            "system": "You are korvid's bounded Kubernetes operator.",
            "append": "Verify the postcondition before reporting completion.",
            "tool.scale_resource": "Request an approval-gated replica-count change.",
        },
        "metadata": {"source": "shipped"},
    }


def _process_campaign_payload(
    case_id: str = "smoke-happy", *, extra_case_ids: tuple[str, ...] = ()
) -> dict[str, Any]:
    cases = [
        {
            "case_id": case_id,
            "template_id": "smoke-template",
            "prompt": "Confirm the control loop stays bounded.",
            "models": ["mock-small"],
        }
    ]
    for extra_case_id in extra_case_ids:
        cases.append(
            {
                "case_id": extra_case_id,
                "template_id": "smoke-template",
                "prompt": f"Confirm {extra_case_id}.",
                "models": ["mock-small"],
            }
        )
    return {
        "schema_version": 1,
        "campaign_id": "local-smoke",
        "repetitions": 1,
        "models": ["mock-small"],
        "cases": cases,
        "serving": {
            "backend": "process",
            "command": [
                sys.executable,
                str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
                "--request",
                "{request}",
                "--response",
                "{response}",
            ],
        },
    }


def _aks_campaign_payload(
    *,
    command: list[str] | None = None,
    case_ids: tuple[str, ...] = ("aks-case",),
    repetitions: int = 1,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": "aks-smoke",
        "repetitions": repetitions,
        "models": ["qwen3-4b"],
        "cases": [
            {
                "case_id": case_id,
                "template_id": "aks-template",
                "prompt": "Confirm the shared runner service is reachable.",
                "models": ["qwen3-4b"],
            }
            for case_id in case_ids
        ],
        "serving": {
            "backend": "aks_port_forward",
            "resource_group": "rg-pension-guard",
            "cluster_name": "aks-shared-runners",
            "namespace": "korvid",
            "service": "korvid-api",
            "model": "qwen3-4b",
            "command": command
            or [
                sys.executable,
                str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
                "--request",
                "{request}",
                "--response",
                "{response}",
            ],
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _case_model_pair(case_id: str, model: str) -> str:
    return f"{case_id}::{model}"


def test_main_help_lists_available_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "validate" in captured.out
    assert "evaluate" in captured.out
    assert "optimize" in captured.out
    assert "aks-check" in captured.out
    assert "publish" in captured.out


def test_validate_accepts_example_candidate_and_campaign() -> None:
    exit_code, stdout, stderr = _run_cli(
        [
            "validate",
            "--candidate",
            str(ROOT / "examples" / "candidates" / "shipped-small.yaml"),
            "--campaign",
            str(ROOT / "examples" / "campaigns" / "local-smoke.yaml"),
        ]
    )

    assert exit_code == 0
    assert "shipped-small" in stdout
    assert "local-smoke" in stdout
    assert stderr == ""


def test_validate_returns_exit_2_for_malformed_candidate(tmp_path: Path) -> None:
    candidate_path = _write_yaml(
        tmp_path / "candidate.yaml",
        {
            "schema_version": 1,
            "candidate_id": "broken-candidate",
            "components": {},
        },
    )
    campaign_path = _write_yaml(tmp_path / "campaign.yaml", _process_campaign_payload())

    exit_code, stdout, stderr = _run_cli(
        [
            "validate",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "components must not be empty" in stderr


def test_evaluate_runs_fake_bridge_and_emits_json_summary(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"

    exit_code, stdout, stderr = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(ROOT / "examples" / "candidates" / "shipped-small.yaml"),
            "--campaign",
            str(ROOT / "examples" / "campaigns" / "local-smoke.yaml"),
            "--artifact-root",
            str(artifact_root),
            "--train-case-id",
            "smoke-happy",
            "--validation-case-id",
            "smoke-guardrail",
            "--json",
        ]
    )

    assert exit_code == 0
    assert stderr == ""

    summary = json.loads(stdout)
    assert summary["bundle_kind"] == "common"
    assert summary["aggregate_score"] == pytest.approx(0.91)
    assert summary["hard_safety_failures"] == 0
    assert summary["systemic_failures"] == 0
    assert summary["milestone_passed"] is True
    assert summary["repetitions_per_case"] == 5
    assert summary["pass_at_3"] == pytest.approx(1.0)
    assert summary["pass_at_5"] == pytest.approx(1.0)
    assert summary["case_sets"]["train"] == ["smoke-happy"]
    assert summary["case_sets"]["validation"] == ["smoke-guardrail"]
    assert summary["campaign_case_model_pairs"] == [
        _case_model_pair("smoke-happy", "mock-small"),
        _case_model_pair("smoke-guardrail", "mock-small"),
    ]
    assert summary["evaluated_case_model_pairs"] == [
        _case_model_pair("smoke-happy", "mock-small"),
        _case_model_pair("smoke-guardrail", "mock-small"),
    ]
    assert any(
        ref.endswith("evaluation-summary.json") for ref in summary["artifact_refs"]
    )
    assert (artifact_root / "evaluation-summary.json").is_file()


def test_evaluate_summary_records_how_every_run_produced_its_grade(
    tmp_path: Path,
) -> None:
    # The synthetic bridge is scripted evidence by default; ``live-mode`` is the
    # explicit opt-in a mode-aggregation test needs to see both modes at once.
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload(
            "smoke-live[live-mode]", extra_case_ids=("smoke-scripted",)
        ),
    )

    exit_code, stdout, _ = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--train-case-id",
            "smoke-live[live-mode]",
            "--validation-case-id",
            "smoke-scripted",
            "--json",
        ]
    )

    assert exit_code == 0
    summary = json.loads(stdout)
    assert summary["execution_modes"] == ["live", "scripted"]
    assert summary["run_execution_modes"] == {
        "smoke-live[live-mode]::mock-small": "live",
        "smoke-scripted::mock-small": "scripted",
    }


def test_evaluate_summary_of_a_wholly_live_campaign_declares_live_only(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload(
            "smoke-happy[live-mode]", extra_case_ids=("smoke-guardrail[live-mode]",)
        ),
    )

    exit_code, stdout, _ = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--train-case-id",
            "smoke-happy[live-mode]",
            "--validation-case-id",
            "smoke-guardrail[live-mode]",
            "--json",
        ]
    )

    assert exit_code == 0
    assert json.loads(stdout)["execution_modes"] == ["live"]


def test_evaluate_returns_exit_1_for_systemic_bridge_failure(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload(
            "bridge-outage[systemic-status]", extra_case_ids=("bridge-standby",)
        ),
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--train-case-id",
            "bridge-outage[systemic-status]",
            "--validation-case-id",
            "bridge-standby",
        ]
    )

    assert exit_code == 1
    assert stdout == ""
    assert "systemic" in stderr


def test_evaluate_partial_model_specific_run_does_not_claim_milestone_passed(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-a", extra_case_ids=("smoke-b", "smoke-c")),
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--bundle-kind",
            "model-specific",
            "--case-id",
            "smoke-a",
            "--case-id",
            "smoke-b",
            "--train-case-id",
            "smoke-a",
            "--validation-case-id",
            "smoke-b",
            "--json",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    summary = json.loads(stdout)
    assert summary["milestone_passed"] is False
    assert summary["campaign_case_ids"] == ["smoke-a", "smoke-b", "smoke-c"]
    assert summary["evaluated_case_ids"] == ["smoke-a", "smoke-b"]
    assert summary["campaign_case_model_pairs"] == [
        _case_model_pair("smoke-a", "mock-small"),
        _case_model_pair("smoke-b", "mock-small"),
        _case_model_pair("smoke-c", "mock-small"),
    ]
    assert summary["evaluated_case_model_pairs"] == [
        _case_model_pair("smoke-a", "mock-small"),
        _case_model_pair("smoke-b", "mock-small"),
    ]
    assert summary["case_sets"]["milestone"] == []


def test_evaluate_model_specific_target_model_pack_can_pass_without_full_campaign(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        {
            "schema_version": 1,
            "campaign_id": "asymmetric-model-pack",
            "repetitions": 1,
            "models": ["mock-small", "mock-large"],
            "cases": [
                {
                    "case_id": "small-case",
                    "template_id": "smoke-template",
                    "prompt": "Confirm small case.",
                    "models": ["mock-small"],
                },
                {
                    "case_id": "large-case",
                    "template_id": "smoke-template",
                    "prompt": "Confirm large case.",
                    "models": ["mock-large"],
                },
                {
                    "case_id": "large-guardrail",
                    "template_id": "smoke-template",
                    "prompt": "Confirm the large guardrail.",
                    "models": ["mock-large"],
                },
            ],
            "serving": {
                "backend": "process",
                "command": [
                    sys.executable,
                    str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
                    "--request",
                    "{request}",
                    "--response",
                    "{response}",
                ],
            },
        },
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--bundle-kind",
            "model-specific",
            "--case-id",
            "large-case",
            "--case-id",
            "large-guardrail",
            "--train-case-id",
            "large-case",
            "--validation-case-id",
            "large-guardrail",
            "--json",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    summary = json.loads(stdout)
    assert summary["milestone_passed"] is True
    assert summary["evaluated_models"] == ["mock-large"]
    assert summary["campaign_case_ids"] == [
        "small-case",
        "large-case",
        "large-guardrail",
    ]
    assert summary["evaluated_case_ids"] == ["large-case", "large-guardrail"]
    assert summary["case_sets"]["milestone"] == ["large-case", "large-guardrail"]
    assert summary["case_sets"]["train"] == ["large-case"]
    assert summary["case_sets"]["validation"] == ["large-guardrail"]
    assert summary["campaign_case_model_pairs"] == [
        _case_model_pair("small-case", "mock-small"),
        _case_model_pair("large-case", "mock-large"),
        _case_model_pair("large-guardrail", "mock-large"),
    ]
    assert summary["evaluated_case_model_pairs"] == [
        _case_model_pair("large-case", "mock-large"),
        _case_model_pair("large-guardrail", "mock-large"),
    ]


def test_evaluate_model_specific_multi_model_run_does_not_claim_milestone_passed(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        {
            "schema_version": 1,
            "campaign_id": "multi-model-specific",
            "repetitions": 1,
            "models": ["mock-small", "mock-large"],
            "cases": [
                {
                    "case_id": "shared-case",
                    "template_id": "smoke-template",
                    "prompt": "Confirm shared case.",
                    "models": ["mock-small", "mock-large"],
                },
                {
                    "case_id": "shared-guardrail",
                    "template_id": "smoke-template",
                    "prompt": "Confirm the shared guardrail.",
                    "models": ["mock-small", "mock-large"],
                },
            ],
            "serving": {
                "backend": "process",
                "command": [
                    sys.executable,
                    str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
                    "--request",
                    "{request}",
                    "--response",
                    "{response}",
                ],
            },
        },
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--bundle-kind",
            "model-specific",
            "--train-case-id",
            "shared-case",
            "--validation-case-id",
            "shared-guardrail",
            "--json",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    summary = json.loads(stdout)
    assert summary["milestone_passed"] is False
    assert summary["case_sets"]["milestone"] == []


def test_optimize_requires_reflection_model_and_invokes_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}

    def fake_build_reflection_lm(model: str) -> object:
        calls["reflection_model"] = model
        return {"model": model}

    def fake_optimize_campaign(**kwargs: Any) -> object:
        calls["optimize"] = kwargs
        summary_path = tmp_path / "optimization-summary.json"
        best_candidate_path = tmp_path / "best-candidate.yaml"
        summary_path.write_text("{}", encoding="utf-8")
        best_candidate_path.write_text(
            "schema_version: 1\ncandidate_id: shipped-small\ncomponents:\n  system: ok\n",
            encoding="utf-8",
        )
        return type(
            "Artifacts",
            (),
            {
                "best_candidate": kwargs["seed_candidate"],
                "best_candidate_path": best_candidate_path,
                "summary_path": summary_path,
                "run_id": "1111222233334444",
                "invocation_dir": tmp_path,
            },
        )()

    monkeypatch.setattr(
        "korvid_prompt_lab.cli._build_reflection_lm", fake_build_reflection_lm
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.optimize_campaign", fake_optimize_campaign
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "optimize",
            "--candidate",
            str(ROOT / "examples" / "candidates" / "shipped-small.yaml"),
            "--campaign",
            str(ROOT / "examples" / "campaigns" / "local-smoke.yaml"),
            "--artifact-root",
            str(tmp_path / "optimization"),
            "--max-metric-calls",
            "2",
            "--reflection-model",
            "openai/gpt-4.1-mini",
            "--train-case-id",
            "smoke-happy",
            "--validation-case-id",
            "smoke-guardrail",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert "best-candidate.yaml" in stdout
    assert calls["reflection_model"] == "openai/gpt-4.1-mini"
    assert calls["optimize"]["max_metric_calls"] == 2

    missing_code, missing_stdout, missing_stderr = _run_cli(
        [
            "optimize",
            "--candidate",
            str(ROOT / "examples" / "candidates" / "shipped-small.yaml"),
            "--campaign",
            str(ROOT / "examples" / "campaigns" / "local-smoke.yaml"),
            "--artifact-root",
            str(tmp_path / "optimization-missing"),
            "--max-metric-calls",
            "2",
            "--train-case-id",
            "smoke-happy",
            "--validation-case-id",
            "smoke-guardrail",
        ]
    )

    assert missing_code == 2
    assert missing_stdout == ""
    assert "reflection-model" in missing_stderr


def test_aks_check_performs_read_only_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    class FakeAKSPortForward:
        def __init__(self, serving: object, workspace_dir: Path | str) -> None:
            calls["serving"] = serving
            calls["workspace_dir"] = Path(workspace_dir)
            self.base_url = "http://127.0.0.1:41001"

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            pass

    monkeypatch.setattr("korvid_prompt_lab.cli.AKSPortForward", FakeAKSPortForward)
    campaign_path = _write_yaml(tmp_path / "aks-campaign.yaml", _aks_campaign_payload())

    exit_code, stdout, stderr = _run_cli(
        [
            "aks-check",
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "aks-check"),
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert "http://127.0.0.1:41001" in stdout
    assert calls["workspace_dir"] == tmp_path / "aks-check"


def test_aks_check_returns_75_for_transient_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Transient AKS errors (endpoints not ready) must return EX_TEMPFAIL 75."""

    class FailingTransient:
        def __init__(self, serving: object, workspace_dir: Path | str) -> None:
            pass

        def __enter__(self) -> Self:
            raise AKSPreflightTransientError("AKS Service must expose Ready endpoints")

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            pass

    monkeypatch.setattr("korvid_prompt_lab.cli.AKSPortForward", FailingTransient)
    campaign_path = _write_yaml(tmp_path / "aks-campaign.yaml", _aks_campaign_payload())

    exit_code, _stdout, stderr = _run_cli(
        [
            "aks-check",
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "aks-check"),
        ]
    )

    assert exit_code == 75
    assert "(transient)" in stderr


def test_aks_check_returns_1_for_permanent_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Permanent AKS errors (cluster identity) must return exit 1."""

    class FailingPermanent:
        def __init__(self, serving: object, workspace_dir: Path | str) -> None:
            pass

        def __enter__(self) -> Self:
            raise AKSPortForwardError("AKS cluster lookup failed")

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            pass

    monkeypatch.setattr("korvid_prompt_lab.cli.AKSPortForward", FailingPermanent)
    campaign_path = _write_yaml(tmp_path / "aks-campaign.yaml", _aks_campaign_payload())

    exit_code, _stdout, stderr = _run_cli(
        [
            "aks-check",
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "aks-check"),
        ]
    )

    assert exit_code == 1
    assert "(transient)" not in stderr


def test_aks_check_returns_1_for_missing_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing kubectl/az raises AKSMissingToolError which returns exit 1 (permanent)."""

    class FailingMissingTool:
        def __init__(self, serving: object, workspace_dir: Path | str) -> None:
            pass

        def __enter__(self) -> Self:
            raise AKSMissingToolError("kubectl not found")

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            pass

    monkeypatch.setattr("korvid_prompt_lab.cli.AKSPortForward", FailingMissingTool)
    campaign_path = _write_yaml(tmp_path / "aks-campaign.yaml", _aks_campaign_payload())

    exit_code, _stdout, stderr = _run_cli(
        [
            "aks-check",
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "aks-check"),
        ]
    )

    assert exit_code == 1
    assert "kubectl not found" in stderr
    assert "Traceback" not in stderr


def test_publish_reads_inputs_and_writes_registry(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )
    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-small",
            "model_name": "mock-small@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    evaluation_summary_path = _write_json(
        tmp_path / "evaluation-summary.json",
        {
            "bundle_kind": "common",
            "candidate_id": "shipped-small",
            "candidate_fingerprint": "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7",
            "campaign_id": "local-smoke",
            "campaign_case_ids": ["smoke-happy", "smoke-guardrail"],
            "evaluated_case_ids": ["smoke-happy", "smoke-guardrail"],
            "evaluated_models": ["mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-guardrail", "mock-small"),
            ],
            "evaluated_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-guardrail", "mock-small"),
            ],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "execution_modes": ["live"],
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-happy"],
                "validation": ["smoke-guardrail"],
                "milestone": ["smoke-happy", "smoke-guardrail"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": [
                "uv",
                "run",
                "--python",
                "3.12",
                "korvid-prompt-lab",
                "evaluate",
                "--candidate",
                "candidate.yaml",
                "--campaign",
                "campaign.yaml",
            ],
        },
    )
    registry_root = tmp_path / "registry"

    exit_code, stdout, stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(evaluation_summary_path),
            "--registry-root",
            str(registry_root),
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert "published" in stdout
    assert (registry_root / "index.json").is_file()
    assert (registry_root / "scoreboard.md").is_file()


def test_publish_rejects_mismatched_candidate_identity_in_summary(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )
    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-small",
            "model_name": "mock-small@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    evaluation_summary_path = _write_json(
        tmp_path / "evaluation-summary.json",
        {
            "bundle_kind": "common",
            "candidate_id": "wrong-candidate",
            "candidate_fingerprint": "deadbeef",
            "campaign_id": "local-smoke",
            "campaign_case_ids": ["smoke-happy", "smoke-guardrail"],
            "evaluated_case_ids": ["smoke-happy", "smoke-guardrail"],
            "evaluated_models": ["mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-guardrail", "mock-small"),
            ],
            "evaluated_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-guardrail", "mock-small"),
            ],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "execution_modes": ["live"],
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-happy"],
                "validation": ["smoke-guardrail"],
                "milestone": ["smoke-happy", "smoke-guardrail"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": [
                "uv",
                "run",
                "--python",
                "3.12",
                "korvid-prompt-lab",
                "evaluate",
            ],
        },
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(evaluation_summary_path),
            "--registry-root",
            str(tmp_path / "registry"),
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "candidate_id" in stderr or "candidate_fingerprint" in stderr


def test_publish_rejects_model_specific_summary_without_full_milestone_pack(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-a", extra_case_ids=("smoke-b",)),
    )
    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-small",
            "model_name": "mock-small@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    registry_root = tmp_path / "registry"

    baseline_summary_path = _write_json(
        tmp_path / "baseline-summary.json",
        {
            "bundle_kind": "common",
            "candidate_id": "shipped-small",
            "candidate_fingerprint": "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7",
            "campaign_id": "local-smoke",
            "campaign_case_ids": ["smoke-a", "smoke-b"],
            "evaluated_case_ids": ["smoke-a", "smoke-b"],
            "evaluated_models": ["mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("smoke-a", "mock-small"),
                _case_model_pair("smoke-b", "mock-small"),
            ],
            "evaluated_case_model_pairs": [
                _case_model_pair("smoke-a", "mock-small"),
                _case_model_pair("smoke-b", "mock-small"),
            ],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "execution_modes": ["live"],
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-a"],
                "validation": ["smoke-b"],
                "milestone": ["smoke-a", "smoke-b"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": [
                "uv",
                "run",
                "--python",
                "3.12",
                "korvid-prompt-lab",
                "evaluate",
            ],
        },
    )
    baseline_code, _, baseline_stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(baseline_summary_path),
            "--registry-root",
            str(registry_root),
        ]
    )
    assert baseline_code == 0, baseline_stderr

    incomplete_model_summary_path = _write_json(
        tmp_path / "model-summary.json",
        {
            "bundle_kind": "model-specific",
            "candidate_id": "shipped-small",
            "candidate_fingerprint": "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7",
            "campaign_id": "local-smoke",
            "campaign_case_ids": ["smoke-a", "smoke-b"],
            "evaluated_case_ids": ["smoke-a"],
            "evaluated_models": ["mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("smoke-a", "mock-small"),
                _case_model_pair("smoke-b", "mock-small"),
            ],
            "evaluated_case_model_pairs": [_case_model_pair("smoke-a", "mock-small")],
            "aggregate_score": 0.95,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "execution_modes": ["live"],
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-a"],
                "validation": ["smoke-a"],
                "milestone": ["smoke-a"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": [
                "uv",
                "run",
                "--python",
                "3.12",
                "korvid-prompt-lab",
                "evaluate",
            ],
        },
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(incomplete_model_summary_path),
            "--registry-root",
            str(registry_root),
            "--minimum-model-improvement",
            "0.01",
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "milestone" in stderr
    assert (
        len(
            json.loads((registry_root / "index.json").read_text(encoding="utf-8"))[
                "bundles"
            ]
        )
        == 1
    )


def test_publish_rejects_common_summary_without_full_case_model_matrix(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        {
            "schema_version": 1,
            "campaign_id": "matrix-smoke",
            "repetitions": 1,
            "models": ["mock-small", "mock-large"],
            "cases": [
                {
                    "case_id": "smoke-a",
                    "template_id": "smoke-template",
                    "prompt": "Confirm smoke-a.",
                    "models": ["mock-small", "mock-large"],
                },
                {
                    "case_id": "smoke-b",
                    "template_id": "smoke-template",
                    "prompt": "Confirm smoke-b.",
                    "models": ["mock-small", "mock-large"],
                },
            ],
            "serving": {
                "backend": "process",
                "command": [
                    sys.executable,
                    str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
                    "--request",
                    "{request}",
                    "--response",
                    "{response}",
                ],
            },
        },
    )
    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-small",
            "model_name": "mock-small@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    evaluation_summary_path = _write_json(
        tmp_path / "evaluation-summary.json",
        {
            "bundle_kind": "common",
            "candidate_id": "shipped-small",
            "candidate_fingerprint": "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7",
            "campaign_id": "matrix-smoke",
            "campaign_case_ids": ["smoke-a", "smoke-b"],
            "evaluated_case_ids": ["smoke-a", "smoke-b"],
            "evaluated_models": ["mock-large", "mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("smoke-a", "mock-small"),
                _case_model_pair("smoke-a", "mock-large"),
                _case_model_pair("smoke-b", "mock-small"),
                _case_model_pair("smoke-b", "mock-large"),
            ],
            "evaluated_case_model_pairs": [
                _case_model_pair("smoke-a", "mock-small"),
                _case_model_pair("smoke-a", "mock-large"),
                _case_model_pair("smoke-b", "mock-small"),
            ],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "execution_modes": ["live"],
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-a"],
                "validation": ["smoke-b"],
                "milestone": ["smoke-a", "smoke-b"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": [
                "uv",
                "run",
                "--python",
                "3.12",
                "korvid-prompt-lab",
                "evaluate",
            ],
        },
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(evaluation_summary_path),
            "--registry-root",
            str(tmp_path / "registry"),
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "case_model" in stderr or "common" in stderr


def test_publish_accepts_common_summary_with_full_model_matrix_despite_model_order(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        {
            "schema_version": 1,
            "campaign_id": "matrix-smoke",
            "repetitions": 1,
            "models": ["mock-small", "mock-large"],
            "cases": [
                {
                    "case_id": "smoke-happy",
                    "template_id": "smoke-template",
                    "prompt": "Confirm the control loop stays bounded.",
                    "models": ["mock-small", "mock-large"],
                },
                {
                    "case_id": "smoke-guardrail",
                    "template_id": "smoke-template",
                    "prompt": "Confirm the approval guardrail still blocks unbounded mutation.",
                    "models": ["mock-small", "mock-large"],
                },
            ],
            "serving": {
                "backend": "process",
                "command": [
                    sys.executable,
                    str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
                    "--request",
                    "{request}",
                    "--response",
                    "{response}",
                ],
            },
        },
    )
    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-small",
            "model_name": "mock-small@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    evaluation_summary_path = _write_json(
        tmp_path / "evaluation-summary.json",
        {
            "bundle_kind": "common",
            "candidate_id": "shipped-small",
            "candidate_fingerprint": "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7",
            "campaign_id": "matrix-smoke",
            "campaign_case_ids": ["smoke-happy", "smoke-guardrail"],
            "evaluated_case_ids": ["smoke-happy", "smoke-guardrail"],
            "evaluated_models": ["mock-large", "mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-happy", "mock-large"),
                _case_model_pair("smoke-guardrail", "mock-small"),
                _case_model_pair("smoke-guardrail", "mock-large"),
            ],
            "evaluated_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-large"),
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-guardrail", "mock-large"),
                _case_model_pair("smoke-guardrail", "mock-small"),
            ],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "execution_modes": ["live"],
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-happy"],
                "validation": ["smoke-guardrail"],
                "milestone": ["smoke-happy", "smoke-guardrail"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": [
                "uv",
                "run",
                "--python",
                "3.12",
                "korvid-prompt-lab",
                "evaluate",
            ],
        },
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(evaluation_summary_path),
            "--registry-root",
            str(tmp_path / "registry"),
        ]
    )

    assert exit_code == 0
    assert "published" in stdout
    assert stderr == ""


def test_publish_accepts_common_summary_with_sorted_case_lists(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        {
            "schema_version": 1,
            "campaign_id": "ordered-cases",
            "repetitions": 1,
            "models": ["mock-small"],
            "cases": [
                {
                    "case_id": "b-case",
                    "template_id": "smoke-template",
                    "prompt": "Confirm b-case.",
                    "models": ["mock-small"],
                },
                {
                    "case_id": "a-case",
                    "template_id": "smoke-template",
                    "prompt": "Confirm a-case.",
                    "models": ["mock-small"],
                },
            ],
            "serving": {
                "backend": "process",
                "command": [
                    sys.executable,
                    str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
                    "--request",
                    "{request}",
                    "--response",
                    "{response}",
                ],
            },
        },
    )
    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-small",
            "model_name": "mock-small@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    evaluation_summary_path = _write_json(
        tmp_path / "evaluation-summary.json",
        {
            "bundle_kind": "common",
            "candidate_id": "shipped-small",
            "candidate_fingerprint": "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7",
            "campaign_id": "ordered-cases",
            "campaign_case_ids": ["a-case", "b-case"],
            "evaluated_case_ids": ["a-case", "b-case"],
            "evaluated_models": ["mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("a-case", "mock-small"),
                _case_model_pair("b-case", "mock-small"),
            ],
            "evaluated_case_model_pairs": [
                _case_model_pair("a-case", "mock-small"),
                _case_model_pair("b-case", "mock-small"),
            ],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "execution_modes": ["live"],
            "milestone_passed": True,
            "case_sets": {
                "train": ["a-case"],
                "validation": ["b-case"],
                "milestone": ["a-case", "b-case"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": [
                "uv",
                "run",
                "--python",
                "3.12",
                "korvid-prompt-lab",
                "evaluate",
            ],
        },
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(evaluation_summary_path),
            "--registry-root",
            str(tmp_path / "registry"),
        ]
    )

    assert exit_code == 0
    assert "published" in stdout
    assert stderr == ""


def test_publish_rejects_common_summary_with_unrelated_model_metadata(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )
    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-unrelated",
            "model_name": "mock-unrelated@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    evaluation_summary_path = _write_json(
        tmp_path / "evaluation-summary.json",
        {
            "bundle_kind": "common",
            "candidate_id": "shipped-small",
            "candidate_fingerprint": "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7",
            "campaign_id": "local-smoke",
            "campaign_case_ids": ["smoke-happy", "smoke-guardrail"],
            "evaluated_case_ids": ["smoke-happy", "smoke-guardrail"],
            "evaluated_models": ["mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-guardrail", "mock-small"),
            ],
            "evaluated_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-guardrail", "mock-small"),
            ],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "execution_modes": ["live"],
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-happy"],
                "validation": ["smoke-guardrail"],
                "milestone": ["smoke-happy", "smoke-guardrail"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": [
                "uv",
                "run",
                "--python",
                "3.12",
                "korvid-prompt-lab",
                "evaluate",
            ],
        },
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(evaluation_summary_path),
            "--registry-root",
            str(tmp_path / "registry"),
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "model_family" in stderr or "target model" in stderr


def test_publish_rejects_model_specific_summary_not_bound_to_target_model(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        {
            "schema_version": 1,
            "campaign_id": "matrix-smoke",
            "repetitions": 1,
            "models": ["mock-small", "mock-large"],
            "cases": [
                {
                    "case_id": "smoke-happy",
                    "template_id": "smoke-template",
                    "prompt": "Confirm the control loop stays bounded.",
                    "models": ["mock-small", "mock-large"],
                },
                {
                    "case_id": "smoke-guardrail",
                    "template_id": "smoke-template",
                    "prompt": "Confirm the approval guardrail still blocks unbounded mutation.",
                    "models": ["mock-small", "mock-large"],
                },
            ],
            "serving": {
                "backend": "process",
                "command": [
                    sys.executable,
                    str(ROOT / "tests" / "fixtures" / "fake_korvid_bridge.py"),
                    "--request",
                    "{request}",
                    "--response",
                    "{response}",
                ],
            },
        },
    )
    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-small",
            "model_name": "mock-small@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    registry_root = tmp_path / "registry"

    baseline_summary_path = _write_json(
        tmp_path / "baseline-summary.json",
        {
            "bundle_kind": "common",
            "candidate_id": "shipped-small",
            "candidate_fingerprint": "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7",
            "campaign_id": "matrix-smoke",
            "campaign_case_ids": ["smoke-happy", "smoke-guardrail"],
            "evaluated_case_ids": ["smoke-happy", "smoke-guardrail"],
            "evaluated_models": ["mock-large", "mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-happy", "mock-large"),
                _case_model_pair("smoke-guardrail", "mock-small"),
                _case_model_pair("smoke-guardrail", "mock-large"),
            ],
            "evaluated_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-happy", "mock-large"),
                _case_model_pair("smoke-guardrail", "mock-small"),
                _case_model_pair("smoke-guardrail", "mock-large"),
            ],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "execution_modes": ["live"],
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-happy"],
                "validation": ["smoke-guardrail"],
                "milestone": ["smoke-happy", "smoke-guardrail"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": [
                "uv",
                "run",
                "--python",
                "3.12",
                "korvid-prompt-lab",
                "evaluate",
            ],
        },
    )
    baseline_code, _, baseline_stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(baseline_summary_path),
            "--registry-root",
            str(registry_root),
        ]
    )
    assert baseline_code == 0, baseline_stderr

    model_summary_path = _write_json(
        tmp_path / "model-summary.json",
        {
            "bundle_kind": "model-specific",
            "candidate_id": "shipped-small",
            "candidate_fingerprint": "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7",
            "campaign_id": "matrix-smoke",
            "campaign_case_ids": ["smoke-happy", "smoke-guardrail"],
            "evaluated_case_ids": ["smoke-happy", "smoke-guardrail"],
            "evaluated_models": ["mock-large", "mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-happy", "mock-large"),
                _case_model_pair("smoke-guardrail", "mock-small"),
                _case_model_pair("smoke-guardrail", "mock-large"),
            ],
            "evaluated_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-happy", "mock-large"),
                _case_model_pair("smoke-guardrail", "mock-small"),
                _case_model_pair("smoke-guardrail", "mock-large"),
            ],
            "aggregate_score": 0.95,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "execution_modes": ["live"],
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-happy"],
                "validation": ["smoke-guardrail"],
                "milestone": ["smoke-happy", "smoke-guardrail"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": [
                "uv",
                "run",
                "--python",
                "3.12",
                "korvid-prompt-lab",
                "evaluate",
            ],
        },
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(model_summary_path),
            "--registry-root",
            str(registry_root),
            "--minimum-model-improvement",
            "0.01",
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "model-specific" in stderr or "target model" in stderr


def _recording_bridge(path: Path, events_path: Path) -> list[str]:
    path.write_text(
        """
import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[sys.argv.index("--request") + 1]).read_text(encoding="utf-8"))
response_path = Path(sys.argv[sys.argv.index("--response") + 1])
events_path = Path(sys.argv[sys.argv.index("--events") + 1])
endpoint = request["runtime"]["model_endpoint"]
with events_path.open("a", encoding="utf-8") as handle:
    handle.write("request:{0}\\n".format(endpoint))
response_path.write_text(
    json.dumps(
        {
            "protocol_version": 2,
            "status": "completed",
            "execution_mode": "live",
            "candidate_fingerprint": request["candidate_fingerprint"],
            "request_identity": {
                "case_id": request["case"]["case_id"],
                "template_id": request["case"]["template_id"],
                "model": request["case"]["model"],
                "repetition": request["case"]["repetition"],
                "seed": request["case"]["seed"],
            },
            "grade": {
                "completion": 0.9,
                "verification": 0.8,
                "efficiency": 0.7,
                "hard_failures": [],
            },
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"], "model_endpoint": endpoint},
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    ),
    encoding="utf-8",
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return [
        sys.executable,
        str(path),
        "--request",
        "{request}",
        "--response",
        "{response}",
        "--events",
        str(events_path),
    ]


def _fake_port_forward_class(
    events_path: Path,
    calls: dict[str, Any],
    *,
    base_url: str = "http://127.0.0.1:41001",
    enter_error: Exception | None = None,
) -> type:
    class FakeAKSPortForward:
        def __init__(self, serving: object, workspace_dir: Path | str) -> None:
            calls.setdefault("instances", 0)
            calls["instances"] += 1
            calls["serving"] = serving
            calls["workspace_dir"] = Path(workspace_dir)
            self.base_url = base_url

        def __enter__(self) -> Self:
            if enter_error is not None:
                raise enter_error
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write("enter\n")
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write("exit\n")

    return FakeAKSPortForward


def test_evaluate_keeps_one_loopback_forward_open_for_the_whole_aks_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events_path = tmp_path / "events.log"
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.AKSPortForward",
        _fake_port_forward_class(events_path, calls),
    )
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "aks-campaign.yaml",
        _aks_campaign_payload(
            command=_recording_bridge(tmp_path / "recording_bridge.py", events_path),
            case_ids=("aks-case", "aks-guardrail"),
            repetitions=2,
        ),
    )
    artifact_root = tmp_path / "artifacts"

    exit_code, _stdout, stderr = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(artifact_root),
            "--train-case-id",
            "aks-case",
            "--validation-case-id",
            "aks-guardrail",
        ]
    )

    assert exit_code == 0, stderr
    assert calls["instances"] == 1
    assert calls["workspace_dir"] == artifact_root

    events = events_path.read_text(encoding="utf-8").split()
    assert events[0] == "enter"
    assert events[-1] == "exit"
    assert events.count("enter") == 1
    assert events.count("exit") == 1
    assert events[1:-1] == ["request:http://127.0.0.1:41001"] * 4

    requests = sorted((artifact_root / "runs").rglob("request.json"))
    assert len(requests) == 4
    for request_path in requests:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        assert payload["runtime"]["model_endpoint"] == "http://127.0.0.1:41001"


def test_evaluate_closes_the_aks_forward_when_the_bridge_fails_systemically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events_path = tmp_path / "events.log"
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.AKSPortForward",
        _fake_port_forward_class(events_path, calls),
    )
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "aks-campaign.yaml",
        _aks_campaign_payload(case_ids=("outage[systemic-status]", "aks-guardrail")),
    )

    exit_code, _stdout, stderr = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--train-case-id",
            "outage[systemic-status]",
            "--validation-case-id",
            "aks-guardrail",
        ]
    )

    assert exit_code == 1
    assert "systemic" in stderr
    assert events_path.read_text(encoding="utf-8").split() == ["enter", "exit"]


def test_evaluate_returns_exit_1_when_the_aks_preflight_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events_path = tmp_path / "events.log"
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.AKSPortForward",
        _fake_port_forward_class(
            events_path,
            calls,
            enter_error=AKSPortForwardError("AKS Service must expose Ready endpoints"),
        ),
    )
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "aks-campaign.yaml",
        _aks_campaign_payload(case_ids=("aks-case", "aks-guardrail")),
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--train-case-id",
            "aks-case",
            "--validation-case-id",
            "aks-guardrail",
        ]
    )

    assert exit_code == 1
    assert stdout == ""
    assert "Ready endpoints" in stderr


def test_optimize_runs_the_whole_search_inside_one_aks_forward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events_path = tmp_path / "events.log"
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.AKSPortForward",
        _fake_port_forward_class(events_path, calls),
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.cli._build_reflection_lm", lambda model: {"model": model}
    )

    def fake_optimize_campaign(**kwargs: Any) -> object:
        calls["runner"] = kwargs["runner"]
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write("optimize\n")
        summary_path = tmp_path / "optimization-summary.json"
        best_candidate_path = tmp_path / "best-candidate.yaml"
        summary_path.write_text("{}", encoding="utf-8")
        best_candidate_path.write_text(
            "schema_version: 1\ncandidate_id: shipped-small\ncomponents:\n  system: ok\n",
            encoding="utf-8",
        )
        return type(
            "Artifacts",
            (),
            {
                "best_candidate": kwargs["seed_candidate"],
                "best_candidate_path": best_candidate_path,
                "summary_path": summary_path,
                "run_id": "5555666677778888",
                "invocation_dir": tmp_path,
            },
        )()

    monkeypatch.setattr(
        "korvid_prompt_lab.cli.optimize_campaign", fake_optimize_campaign
    )
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "aks-campaign.yaml",
        _aks_campaign_payload(case_ids=("aks-train", "aks-val")),
    )

    exit_code, _stdout, stderr = _run_cli(
        [
            "optimize",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "optimization"),
            "--max-metric-calls",
            "2",
            "--reflection-model",
            "openai/gpt-4.1-mini",
            "--train-case-id",
            "aks-train",
            "--validation-case-id",
            "aks-val",
        ]
    )

    assert exit_code == 0, stderr
    assert events_path.read_text(encoding="utf-8").split() == [
        "enter",
        "optimize",
        "exit",
    ]
    assert calls["instances"] == 1
    assert calls["workspace_dir"] == tmp_path / "optimization"
    assert calls["runner"].model_endpoint == "http://127.0.0.1:41001"


def test_evaluate_pass_hat_k_requires_full_operation_completion_not_a_positive_score(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_payload = _process_campaign_payload(
        "smoke-happy", extra_case_ids=("smoke-partial[partial-completion]",)
    )
    campaign_payload["repetitions"] = 3
    campaign_path = _write_yaml(tmp_path / "campaign.yaml", campaign_payload)
    artifact_root = tmp_path / "artifacts"

    exit_code, stdout, stderr = _run_cli(
        _evaluate_args(
            candidate_path,
            campaign_path,
            artifact_root,
            validation=("smoke-partial[partial-completion]",),
            extra=("--json",),
        )
    )

    assert exit_code == 0, stderr
    summary = json.loads(stdout)

    partial_responses = [
        json.loads((run_dir / "response.json").read_text(encoding="utf-8"))
        for run_dir in sorted((artifact_root / "runs").iterdir())
        if run_dir.name.startswith("smoke-partial")
    ]
    assert len(partial_responses) == 3
    for response in partial_responses:
        assert response["status"] == "completed"
        assert response["grade"]["hard_failures"] == []
        assert response["grade"]["completion"] == 0.0
        assert response["grade"]["verification"] > 0.0
        assert response["grade"]["efficiency"] > 0.0

    assert summary["aggregate_score"] > 0.0
    assert summary["hard_safety_failures"] == 0
    assert summary["pass_at_3"] == pytest.approx(0.5)


def test_evaluate_reports_insufficient_pass_hat_k_evidence_for_short_campaigns(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_payload = _process_campaign_payload(
        "smoke-happy", extra_case_ids=("smoke-guardrail",)
    )
    campaign_payload["repetitions"] = 1
    campaign_path = _write_yaml(tmp_path / "campaign.yaml", campaign_payload)

    exit_code, stdout, stderr = _run_cli(
        _evaluate_args(
            candidate_path, campaign_path, tmp_path / "artifacts", extra=("--json",)
        )
    )

    assert exit_code == 0, stderr
    summary = json.loads(stdout)
    assert summary["repetitions_per_case"] == 1
    assert summary["pass_at_3"] is None
    assert summary["pass_at_5"] is None


def test_evaluate_pass_hat_k_requires_every_repetition_to_pass(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_payload = _process_campaign_payload(
        "smoke-happy", extra_case_ids=("smoke-flaky[flaky-after-2]",)
    )
    campaign_payload["repetitions"] = 5
    campaign_path = _write_yaml(tmp_path / "campaign.yaml", campaign_payload)

    exit_code, stdout, stderr = _run_cli(
        _evaluate_args(
            candidate_path,
            campaign_path,
            tmp_path / "artifacts",
            validation=("smoke-flaky[flaky-after-2]",),
            extra=("--json",),
        )
    )

    assert exit_code == 0, stderr
    summary = json.loads(stdout)
    assert summary["repetitions_per_case"] == 5
    assert summary["pass_at_3"] == pytest.approx(0.5)
    assert summary["pass_at_5"] == pytest.approx(0.5)


def test_evaluate_prints_insufficient_pass_hat_k_evidence_in_the_text_summary(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_payload = _process_campaign_payload(
        "smoke-happy", extra_case_ids=("smoke-guardrail",)
    )
    campaign_payload["repetitions"] = 1
    campaign_path = _write_yaml(tmp_path / "campaign.yaml", campaign_payload)

    exit_code, stdout, stderr = _run_cli(
        _evaluate_args(candidate_path, campaign_path, tmp_path / "artifacts")
    )

    assert exit_code == 0, stderr
    assert "pass^3=insufficient-evidence" in stdout
    assert "pass^5=insufficient-evidence" in stdout


def _evaluate_args(
    candidate_path: Path,
    campaign_path: Path,
    artifact_root: Path,
    *,
    train: tuple[str, ...] = ("smoke-happy",),
    validation: tuple[str, ...] = ("smoke-guardrail",),
    milestone: tuple[str, ...] = (),
    extra: tuple[str, ...] = (),
) -> list[str]:
    args = [
        "evaluate",
        "--candidate",
        str(candidate_path),
        "--campaign",
        str(campaign_path),
        "--artifact-root",
        str(artifact_root),
    ]
    for case_id in train:
        args.extend(["--train-case-id", case_id])
    for case_id in validation:
        args.extend(["--validation-case-id", case_id])
    for case_id in milestone:
        args.extend(["--milestone-case-id", case_id])
    args.extend(extra)
    return args


def test_evaluate_records_the_explicit_disjoint_case_sets(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )

    exit_code, stdout, stderr = _run_cli(
        _evaluate_args(
            candidate_path,
            campaign_path,
            tmp_path / "artifacts",
            milestone=("smoke-happy", "smoke-guardrail"),
            extra=("--json",),
        )
    )

    assert exit_code == 0, stderr
    summary = json.loads(stdout)
    assert summary["case_sets"]["train"] == ["smoke-happy"]
    assert summary["case_sets"]["validation"] == ["smoke-guardrail"]
    assert summary["case_sets"]["milestone"] == ["smoke-happy", "smoke-guardrail"]
    assert summary["milestone_passed"] is True


@pytest.mark.parametrize(
    ("train", "validation", "message"),
    [
        ((), ("smoke-guardrail",), "--train-case-id"),
        (("smoke-happy",), (), "--validation-case-id"),
        (("smoke-happy",), ("smoke-happy",), "disjoint"),
        (("smoke-happy", "smoke-guardrail"), ("smoke-guardrail",), "disjoint"),
    ],
)
def test_evaluate_requires_explicit_disjoint_case_splits(
    tmp_path: Path, train: tuple[str, ...], validation: tuple[str, ...], message: str
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )

    exit_code, stdout, stderr = _run_cli(
        _evaluate_args(
            candidate_path,
            campaign_path,
            tmp_path / "artifacts",
            train=train,
            validation=validation,
        )
    )

    assert exit_code == 2
    assert stdout == ""
    assert message in stderr


def test_evaluate_rejects_case_sets_that_were_not_evaluated(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )

    exit_code, stdout, stderr = _run_cli(
        _evaluate_args(
            candidate_path,
            campaign_path,
            tmp_path / "artifacts",
            train=("smoke-happy",),
            validation=("smoke-guardrail",),
            extra=("--case-id", "smoke-happy"),
        )
    )

    assert exit_code == 2
    assert stdout == ""
    assert "evaluated" in stderr


def test_evaluate_rejects_milestone_cases_that_were_not_evaluated(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )

    exit_code, stdout, stderr = _run_cli(
        _evaluate_args(
            candidate_path,
            campaign_path,
            tmp_path / "artifacts",
            milestone=("smoke-missing",),
        )
    )

    assert exit_code == 2
    assert stdout == ""
    assert "evaluated" in stderr


@pytest.mark.parametrize(
    ("split_args", "message"),
    [
        ((), "--train-case-id"),
        (("--train-case-id", "smoke-happy"), "--validation-case-id"),
        (
            ("--train-case-id", "smoke-happy", "--validation-case-id", "smoke-happy"),
            "disjoint",
        ),
    ],
)
def test_optimize_requires_explicit_disjoint_case_splits(
    tmp_path: Path, split_args: tuple[str, ...], message: str
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "optimize",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "optimization"),
            "--max-metric-calls",
            "2",
            "--reflection-model",
            "openai/gpt-4.1-mini",
            *split_args,
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert message in stderr


def test_optimize_threads_an_explicit_seed_and_reports_the_invocation_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        "korvid_prompt_lab.cli._build_reflection_lm", lambda model: {"model": model}
    )

    def fake_optimize_campaign(**kwargs: Any) -> object:
        calls["optimize"] = kwargs
        invocation_dir = tmp_path / "optimization" / "invocations" / "abc123def4567890"
        invocation_dir.mkdir(parents=True)
        summary_path = invocation_dir / "optimization-summary.json"
        best_candidate_path = invocation_dir / "best-candidate.yaml"
        summary_path.write_text("{}", encoding="utf-8")
        best_candidate_path.write_text(
            "schema_version: 1\ncandidate_id: shipped-small\ncomponents:\n  system: ok\n",
            encoding="utf-8",
        )
        return type(
            "Artifacts",
            (),
            {
                "best_candidate": kwargs["seed_candidate"],
                "best_candidate_path": best_candidate_path,
                "summary_path": summary_path,
                "run_id": "abc123def4567890",
                "invocation_dir": invocation_dir,
            },
        )()

    monkeypatch.setattr(
        "korvid_prompt_lab.cli.optimize_campaign", fake_optimize_campaign
    )
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "optimize",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "optimization"),
            "--max-metric-calls",
            "2",
            "--reflection-model",
            "openai/gpt-4.1-mini",
            "--seed",
            "13",
            "--train-case-id",
            "smoke-happy",
            "--validation-case-id",
            "smoke-guardrail",
        ]
    )

    assert exit_code == 0, stderr
    assert calls["optimize"]["seed"] == 13
    assert "run_id=abc123def4567890" in stdout
    assert str(tmp_path / "optimization" / "invocations" / "abc123def4567890") in stdout


def test_optimize_defaults_to_a_zero_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        "korvid_prompt_lab.cli._build_reflection_lm", lambda model: {"model": model}
    )

    def fake_optimize_campaign(**kwargs: Any) -> object:
        calls["optimize"] = kwargs
        invocation_dir = tmp_path / "invocation"
        invocation_dir.mkdir(parents=True, exist_ok=True)
        summary_path = invocation_dir / "optimization-summary.json"
        best_candidate_path = invocation_dir / "best-candidate.yaml"
        summary_path.write_text("{}", encoding="utf-8")
        best_candidate_path.write_text(
            "schema_version: 1\ncandidate_id: shipped-small\ncomponents:\n  system: ok\n",
            encoding="utf-8",
        )
        return type(
            "Artifacts",
            (),
            {
                "best_candidate": kwargs["seed_candidate"],
                "best_candidate_path": best_candidate_path,
                "summary_path": summary_path,
                "run_id": "0000000000000000",
                "invocation_dir": invocation_dir,
            },
        )()

    monkeypatch.setattr(
        "korvid_prompt_lab.cli.optimize_campaign", fake_optimize_campaign
    )
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )

    exit_code, _stdout, stderr = _run_cli(
        [
            "optimize",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "optimization"),
            "--max-metric-calls",
            "2",
            "--reflection-model",
            "openai/gpt-4.1-mini",
            "--train-case-id",
            "smoke-happy",
            "--validation-case-id",
            "smoke-guardrail",
        ]
    )

    assert exit_code == 0, stderr
    assert calls["optimize"]["seed"] == 0


def test_optimize_rejects_a_negative_seed(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )

    exit_code, stdout, stderr = _run_cli(
        [
            "optimize",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "optimization"),
            "--max-metric-calls",
            "2",
            "--reflection-model",
            "openai/gpt-4.1-mini",
            "--seed",
            "-1",
            "--train-case-id",
            "smoke-happy",
            "--validation-case-id",
            "smoke-guardrail",
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "seed must be a non-negative integer" in stderr


def test_evaluate_and_optimize_use_the_campaign_bridge_timeout_for_every_bridge_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from korvid_prompt_lab import cli as cli_module

    real_runner_class = cli_module.KorvidProcessRunner
    created: list[Any] = []

    def recording_runner(**kwargs: Any) -> Any:
        runner = real_runner_class(**kwargs)
        created.append(runner)
        return runner

    monkeypatch.setattr("korvid_prompt_lab.cli.KorvidProcessRunner", recording_runner)
    monkeypatch.setattr(
        "korvid_prompt_lab.cli._build_reflection_lm", lambda model: {"model": model}
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.cli.optimize_campaign",
        lambda **kwargs: _fake_optimization_artifacts(tmp_path / "invocation", kwargs),
    )

    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_payload = _process_campaign_payload(
        "smoke-happy", extra_case_ids=("smoke-guardrail",)
    )
    campaign_payload["bridge_timeout_seconds"] = 12.5
    campaign_path = _write_yaml(tmp_path / "campaign.yaml", campaign_payload)

    evaluate_code, _stdout, evaluate_stderr = _run_cli(
        _evaluate_args(candidate_path, campaign_path, tmp_path / "artifacts")
    )
    assert evaluate_code == 0, evaluate_stderr

    optimize_code, _optimize_stdout, optimize_stderr = _run_cli(
        [
            "optimize",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "optimization"),
            "--max-metric-calls",
            "2",
            "--reflection-model",
            "openai/gpt-4.1-mini",
            "--train-case-id",
            "smoke-happy",
            "--validation-case-id",
            "smoke-guardrail",
        ]
    )
    assert optimize_code == 0, optimize_stderr

    assert len(created) == 2
    assert [runner.timeout_seconds for runner in created] == [
        pytest.approx(12.5),
        pytest.approx(12.5),
    ]


def test_evaluate_aborts_when_the_bridge_exceeds_the_campaign_bridge_timeout(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_payload = _process_campaign_payload(
        "smoke-slow[timeout]", extra_case_ids=("smoke-guardrail",)
    )
    campaign_payload["bridge_timeout_seconds"] = 0.1
    campaign_path = _write_yaml(tmp_path / "campaign.yaml", campaign_payload)

    exit_code, stdout, stderr = _run_cli(
        _evaluate_args(
            candidate_path,
            campaign_path,
            tmp_path / "artifacts",
            train=("smoke-slow[timeout]",),
            validation=("smoke-guardrail",),
        )
    )

    assert exit_code == 1
    assert stdout == ""
    assert "bridge timed out after 0.1 seconds" in stderr


def test_evaluate_rejects_a_non_positive_campaign_bridge_timeout(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_payload = _process_campaign_payload(
        "smoke-happy", extra_case_ids=("smoke-guardrail",)
    )
    campaign_payload["bridge_timeout_seconds"] = 0
    campaign_path = _write_yaml(tmp_path / "campaign.yaml", campaign_payload)

    exit_code, stdout, stderr = _run_cli(
        _evaluate_args(candidate_path, campaign_path, tmp_path / "artifacts")
    )

    assert exit_code == 2
    assert stdout == ""
    assert "bridge_timeout_seconds must be a positive number" in stderr


def _fake_optimization_artifacts(invocation_dir: Path, kwargs: dict[str, Any]) -> Any:
    invocation_dir.mkdir(parents=True, exist_ok=True)
    summary_path = invocation_dir / "optimization-summary.json"
    best_candidate_path = invocation_dir / "best-candidate.yaml"
    summary_path.write_text("{}", encoding="utf-8")
    best_candidate_path.write_text(
        "schema_version: 1\ncandidate_id: shipped-small\ncomponents:\n  system: ok\n",
        encoding="utf-8",
    )
    return type(
        "Artifacts",
        (),
        {
            "best_candidate": kwargs["seed_candidate"],
            "best_candidate_path": best_candidate_path,
            "summary_path": summary_path,
            "run_id": "9999888877776666",
            "invocation_dir": invocation_dir,
        },
    )()


def _publish_summary_payload(
    *, bundle_kind: str, aggregate_score: float
) -> dict[str, Any]:
    return {
        "bundle_kind": bundle_kind,
        "candidate_id": "shipped-small",
        "candidate_fingerprint": "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7",
        "campaign_id": "local-smoke",
        "campaign_case_ids": ["smoke-happy", "smoke-guardrail"],
        "evaluated_case_ids": ["smoke-happy", "smoke-guardrail"],
        "evaluated_models": ["mock-small"],
        "campaign_case_model_pairs": [
            _case_model_pair("smoke-happy", "mock-small"),
            _case_model_pair("smoke-guardrail", "mock-small"),
        ],
        "evaluated_case_model_pairs": [
            _case_model_pair("smoke-happy", "mock-small"),
            _case_model_pair("smoke-guardrail", "mock-small"),
        ],
        "aggregate_score": aggregate_score,
        "repetitions_per_case": 5,
        "pass_at_3": 1.0,
        "pass_at_5": 1.0,
        "hard_safety_failures": 0,
        "systemic_failures": 0,
        "execution_modes": ["live"],
        "milestone_passed": True,
        "case_sets": {
            "train": ["smoke-happy"],
            "validation": ["smoke-guardrail"],
            "milestone": ["smoke-happy", "smoke-guardrail"],
        },
        "artifact_refs": ["artifacts/evaluation-summary.json"],
        "reproduction_command": [
            "uv",
            "run",
            "--python",
            "3.12",
            "korvid-prompt-lab",
            "evaluate",
        ],
    }


def test_publish_blocks_a_marginal_model_specific_override_with_the_default_threshold(
    tmp_path: Path,
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )
    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-small",
            "model_name": "mock-small@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    registry_root = tmp_path / "registry"

    def publish(summary_path: Path) -> tuple[int, str, str]:
        return _run_cli(
            [
                "publish",
                "--candidate",
                str(candidate_path),
                "--campaign",
                str(campaign_path),
                "--model-metadata",
                str(model_metadata_path),
                "--evaluation-summary",
                str(summary_path),
                "--registry-root",
                str(registry_root),
            ]
        )

    common_code, _, common_stderr = publish(
        _write_json(
            tmp_path / "common-summary.json",
            _publish_summary_payload(bundle_kind="common", aggregate_score=0.5),
        )
    )
    marginal_code, marginal_stdout, marginal_stderr = publish(
        _write_json(
            tmp_path / "marginal-summary.json",
            _publish_summary_payload(
                bundle_kind="model-specific", aggregate_score=0.5078125
            ),
        )
    )
    clear_code, clear_stdout, clear_stderr = publish(
        _write_json(
            tmp_path / "clear-summary.json",
            _publish_summary_payload(
                bundle_kind="model-specific", aggregate_score=0.75
            ),
        )
    )

    assert common_code == 0, common_stderr
    assert marginal_code == 1
    assert marginal_stdout == ""
    assert "improvement" in marginal_stderr
    assert clear_code == 0, clear_stderr
    assert "published" in clear_stdout


def test_publish_rejects_insufficient_pass_hat_k_evidence(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )
    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-small",
            "model_name": "mock-small@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    summary_payload = _publish_summary_payload(
        bundle_kind="common", aggregate_score=0.85
    )
    summary_payload["repetitions_per_case"] = 1
    summary_payload["pass_at_3"] = None
    summary_payload["pass_at_5"] = None
    summary_path = _write_json(tmp_path / "evaluation-summary.json", summary_payload)

    exit_code, stdout, stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(summary_path),
            "--registry-root",
            str(tmp_path / "registry"),
        ]
    )

    assert exit_code == 1
    assert stdout == ""
    assert "recorded repetitions" in stderr


@pytest.mark.parametrize(
    "execution_modes",
    [None, ["scripted"], ["live", "scripted"], "live", []],
)
def test_publish_refuses_a_summary_that_is_not_wholly_live(
    tmp_path: Path, execution_modes: Any
) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )
    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-small",
            "model_name": "mock-small@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    summary = _publish_summary_payload(bundle_kind="common", aggregate_score=0.85)
    summary["candidate_fingerprint"] = (
        "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7"
    )
    if execution_modes is None:
        del summary["execution_modes"]
    else:
        summary["execution_modes"] = execution_modes
    evaluation_summary_path = _write_json(tmp_path / "evaluation-summary.json", summary)
    registry_root = tmp_path / "registry"

    exit_code, stdout, stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(evaluation_summary_path),
            "--registry-root",
            str(registry_root),
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "execution_modes" in stderr
    assert not registry_root.exists()


def test_publish_accepts_a_summary_whose_runs_were_all_live(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-happy", extra_case_ids=("smoke-guardrail",)),
    )
    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-small",
            "model_name": "mock-small@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    summary = _publish_summary_payload(bundle_kind="common", aggregate_score=0.85)
    summary["candidate_fingerprint"] = (
        "05386c2e97901414449c3e2356ff736f1def9c9ca5172e28d7b63fa120a335d7"
    )
    evaluation_summary_path = _write_json(tmp_path / "evaluation-summary.json", summary)
    registry_root = tmp_path / "registry"

    exit_code, stdout, stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(evaluation_summary_path),
            "--registry-root",
            str(registry_root),
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert "published" in stdout


def test_shipped_smoke_evidence_can_never_be_published_as_live(tmp_path: Path) -> None:
    """The documented local smoke path must not be able to mint publishable evidence.

    ``examples/campaigns/local-smoke.yaml`` runs the bundled synthetic bridge, which
    never contacts a model. If that run can produce a summary publication accepts, a
    perfect 0.91 / pass^k 1.0 bundle can be minted with no model in the loop at all.
    """
    artifact_root = tmp_path / "artifacts"
    evaluate_code, _stdout, evaluate_stderr = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(ROOT / "examples" / "candidates" / "shipped-small.yaml"),
            "--campaign",
            str(ROOT / "examples" / "campaigns" / "local-smoke.yaml"),
            "--artifact-root",
            str(artifact_root),
            "--train-case-id",
            "smoke-happy",
            "--validation-case-id",
            "smoke-guardrail",
        ]
    )

    assert evaluate_code == 0, evaluate_stderr
    summary_path = artifact_root / "evaluation-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["execution_modes"] == ["scripted"]

    model_metadata_path = _write_json(
        tmp_path / "model-metadata.json",
        {
            "model_family": "mock-small",
            "model_name": "mock-small@2026-08-21",
            "model_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "quantization": "fp16",
            "context_length": 8192,
            "serving_engine": "korvid-process",
        },
    )
    registry_root = tmp_path / "registry"

    publish_code, publish_stdout, publish_stderr = _run_cli(
        [
            "publish",
            "--candidate",
            str(ROOT / "examples" / "candidates" / "shipped-small.yaml"),
            "--campaign",
            str(ROOT / "examples" / "campaigns" / "local-smoke.yaml"),
            "--model-metadata",
            str(model_metadata_path),
            "--evaluation-summary",
            str(summary_path),
            "--registry-root",
            str(registry_root),
        ]
    )

    assert publish_code == 2
    assert publish_stdout == ""
    assert "execution_modes" in publish_stderr
    assert not registry_root.exists()
