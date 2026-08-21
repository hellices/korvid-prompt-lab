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


def _process_campaign_payload(case_id: str = "smoke-happy", *, extra_case_ids: tuple[str, ...] = ()) -> dict[str, Any]:
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


def _aks_campaign_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": "aks-smoke",
        "repetitions": 1,
        "models": ["qwen3-4b"],
        "cases": [
            {
                "case_id": "aks-case",
                "template_id": "aks-template",
                "prompt": "Confirm the shared runner service is reachable.",
                "models": ["qwen3-4b"],
            }
        ],
        "serving": {
            "backend": "aks_port_forward",
            "resource_group": "rg-pension-guard",
            "cluster_name": "aks-shared-runners",
            "namespace": "korvid",
            "service": "korvid-api",
            "model": "qwen3-4b",
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
            "--json",
        ]
    )

    assert exit_code == 0
    assert stderr == ""

    summary = json.loads(stdout)
    assert summary["bundle_kind"] == "common"
    assert summary["aggregate_score"] == pytest.approx(0.85)
    assert summary["hard_safety_failures"] == 0
    assert summary["systemic_failures"] == 0
    assert summary["milestone_passed"] is True
    assert summary["case_sets"]["validation"] == ["smoke-happy"]
    assert summary["campaign_case_model_pairs"] == [_case_model_pair("smoke-happy", "mock-small")]
    assert summary["evaluated_case_model_pairs"] == [_case_model_pair("smoke-happy", "mock-small")]
    assert any(ref.endswith("evaluation-summary.json") for ref in summary["artifact_refs"])
    assert (artifact_root / "evaluation-summary.json").is_file()


def test_evaluate_returns_exit_1_for_systemic_bridge_failure(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(tmp_path / "campaign.yaml", _process_campaign_payload("bridge-outage[systemic-status]"))

    exit_code, stdout, stderr = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ]
    )

    assert exit_code == 1
    assert stdout == ""
    assert "systemic" in stderr


def test_evaluate_rejects_non_process_serving_campaign_with_exit_2(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(tmp_path / "aks-campaign.yaml", _aks_campaign_payload())

    exit_code, stdout, stderr = _run_cli(
        [
            "evaluate",
            "--candidate",
            str(candidate_path),
            "--campaign",
            str(campaign_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "process serving" in stderr


def test_evaluate_partial_model_specific_run_does_not_claim_milestone_passed(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(
        tmp_path / "campaign.yaml",
        _process_campaign_payload("smoke-a", extra_case_ids=("smoke-b",)),
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
            "--json",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    summary = json.loads(stdout)
    assert summary["milestone_passed"] is False
    assert summary["campaign_case_ids"] == ["smoke-a", "smoke-b"]
    assert summary["evaluated_case_ids"] == ["smoke-a"]
    assert summary["campaign_case_model_pairs"] == [
        _case_model_pair("smoke-a", "mock-small"),
        _case_model_pair("smoke-b", "mock-small"),
    ]
    assert summary["evaluated_case_model_pairs"] == [_case_model_pair("smoke-a", "mock-small")]
    assert summary["case_sets"]["milestone"] == []


def test_evaluate_model_specific_target_model_pack_can_pass_without_full_campaign(tmp_path: Path) -> None:
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
            "--json",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    summary = json.loads(stdout)
    assert summary["milestone_passed"] is True
    assert summary["evaluated_models"] == ["mock-large"]
    assert summary["campaign_case_ids"] == ["small-case", "large-case"]
    assert summary["evaluated_case_ids"] == ["large-case"]
    assert summary["case_sets"]["milestone"] == ["large-case"]
    assert summary["campaign_case_model_pairs"] == [
        _case_model_pair("small-case", "mock-small"),
        _case_model_pair("large-case", "mock-large"),
    ]
    assert summary["evaluated_case_model_pairs"] == [_case_model_pair("large-case", "mock-large")]


def test_evaluate_model_specific_multi_model_run_does_not_claim_milestone_passed(tmp_path: Path) -> None:
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
                }
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
        best_candidate_path.write_text("schema_version: 1\ncandidate_id: shipped-small\ncomponents:\n  system: ok\n", encoding="utf-8")
        return type(
            "Artifacts",
            (),
            {
                "best_candidate": kwargs["seed_candidate"],
                "best_candidate_path": best_candidate_path,
                "summary_path": summary_path,
            },
        )()

    monkeypatch.setattr("korvid_prompt_lab.cli._build_reflection_lm", fake_build_reflection_lm)
    monkeypatch.setattr("korvid_prompt_lab.cli.optimize_campaign", fake_optimize_campaign)

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
        ]
    )

    assert missing_code == 2
    assert missing_stdout == ""
    assert "reflection-model" in missing_stderr


def test_aks_check_performs_read_only_preflight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def test_publish_reads_inputs_and_writes_registry(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(tmp_path / "campaign.yaml", _process_campaign_payload())
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
            "campaign_case_ids": ["smoke-happy"],
            "evaluated_case_ids": ["smoke-happy"],
            "evaluated_models": ["mock-small"],
            "campaign_case_model_pairs": [_case_model_pair("smoke-happy", "mock-small")],
            "evaluated_case_model_pairs": [_case_model_pair("smoke-happy", "mock-small")],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-happy"],
                "validation": ["smoke-happy"],
                "milestone": ["smoke-happy"],
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


def test_publish_rejects_mismatched_candidate_identity_in_summary(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(tmp_path / "campaign.yaml", _process_campaign_payload())
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
            "campaign_case_ids": ["smoke-happy"],
            "evaluated_case_ids": ["smoke-happy"],
            "evaluated_models": ["mock-small"],
            "campaign_case_model_pairs": [_case_model_pair("smoke-happy", "mock-small")],
            "evaluated_case_model_pairs": [_case_model_pair("smoke-happy", "mock-small")],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-happy"],
                "validation": ["smoke-happy"],
                "milestone": ["smoke-happy"],
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


def test_publish_rejects_model_specific_summary_without_full_milestone_pack(tmp_path: Path) -> None:
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
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-a", "smoke-b"],
                "validation": ["smoke-a", "smoke-b"],
                "milestone": ["smoke-a", "smoke-b"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": ["uv", "run", "--python", "3.12", "korvid-prompt-lab", "evaluate"],
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
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-a"],
                "validation": ["smoke-a"],
                "milestone": ["smoke-a"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": ["uv", "run", "--python", "3.12", "korvid-prompt-lab", "evaluate"],
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
    assert len(json.loads((registry_root / "index.json").read_text(encoding="utf-8"))["bundles"]) == 1


def test_publish_rejects_common_summary_without_full_case_model_matrix(tmp_path: Path) -> None:
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
                }
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
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-a", "smoke-b"],
                "validation": ["smoke-a", "smoke-b"],
                "milestone": ["smoke-a", "smoke-b"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": ["uv", "run", "--python", "3.12", "korvid-prompt-lab", "evaluate"],
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


def test_publish_accepts_common_summary_with_full_model_matrix_despite_model_order(tmp_path: Path) -> None:
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
                }
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
            "campaign_case_ids": ["smoke-happy"],
            "evaluated_case_ids": ["smoke-happy"],
            "evaluated_models": ["mock-large", "mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-happy", "mock-large"),
            ],
            "evaluated_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-large"),
                _case_model_pair("smoke-happy", "mock-small"),
            ],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-happy"],
                "validation": ["smoke-happy"],
                "milestone": ["smoke-happy"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": ["uv", "run", "--python", "3.12", "korvid-prompt-lab", "evaluate"],
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
            "milestone_passed": True,
            "case_sets": {
                "train": ["a-case", "b-case"],
                "validation": ["a-case", "b-case"],
                "milestone": ["a-case", "b-case"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": ["uv", "run", "--python", "3.12", "korvid-prompt-lab", "evaluate"],
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


def test_publish_rejects_common_summary_with_unrelated_model_metadata(tmp_path: Path) -> None:
    candidate_path = _write_yaml(tmp_path / "candidate.yaml", _candidate_payload())
    campaign_path = _write_yaml(tmp_path / "campaign.yaml", _process_campaign_payload())
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
            "campaign_case_ids": ["smoke-happy"],
            "evaluated_case_ids": ["smoke-happy"],
            "evaluated_models": ["mock-small"],
            "campaign_case_model_pairs": [_case_model_pair("smoke-happy", "mock-small")],
            "evaluated_case_model_pairs": [_case_model_pair("smoke-happy", "mock-small")],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-happy"],
                "validation": ["smoke-happy"],
                "milestone": ["smoke-happy"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": ["uv", "run", "--python", "3.12", "korvid-prompt-lab", "evaluate"],
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


def test_publish_rejects_model_specific_summary_not_bound_to_target_model(tmp_path: Path) -> None:
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
                }
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
            "campaign_case_ids": ["smoke-happy"],
            "evaluated_case_ids": ["smoke-happy"],
            "evaluated_models": ["mock-large", "mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-happy", "mock-large"),
            ],
            "evaluated_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-happy", "mock-large"),
            ],
            "aggregate_score": 0.85,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-happy"],
                "validation": ["smoke-happy"],
                "milestone": ["smoke-happy"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": ["uv", "run", "--python", "3.12", "korvid-prompt-lab", "evaluate"],
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
            "campaign_case_ids": ["smoke-happy"],
            "evaluated_case_ids": ["smoke-happy"],
            "evaluated_models": ["mock-large", "mock-small"],
            "campaign_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-happy", "mock-large"),
            ],
            "evaluated_case_model_pairs": [
                _case_model_pair("smoke-happy", "mock-small"),
                _case_model_pair("smoke-happy", "mock-large"),
            ],
            "aggregate_score": 0.95,
            "pass_at_3": 1.0,
            "pass_at_5": 1.0,
            "hard_safety_failures": 0,
            "systemic_failures": 0,
            "milestone_passed": True,
            "case_sets": {
                "train": ["smoke-happy"],
                "validation": ["smoke-happy"],
                "milestone": ["smoke-happy"],
            },
            "artifact_refs": ["artifacts/evaluation-summary.json"],
            "reproduction_command": ["uv", "run", "--python", "3.12", "korvid-prompt-lab", "evaluate"],
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
