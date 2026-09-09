from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from korvid_prompt_lab.contracts import Campaign, KorvidUpstreamServing
from korvid_prompt_lab.upstream import export_upstream_prompt
from korvid_prompt_lab.upstream_contract import load_source_cases, prompt_candidate

SOURCE_ROOT = Path(os.environ.get("KORVID_NATIVE_SOURCE_ROOT", ""))
pytestmark = pytest.mark.skipif(
    not os.environ.get("KORVID_NATIVE_SOURCE_ROOT"),
    reason="set KORVID_NATIVE_SOURCE_ROOT to the Korvid v0.4.1 source checkout",
)


def _campaign() -> Campaign:
    model = "ollama/qwen3:0.6b"
    source = load_source_cases(SOURCE_ROOT, ("scenarios/image-pull-typo",))[0]
    return Campaign(
        campaign_id="upstream-export",
        repetitions=1,
        models=(model,),
        cases=(source.eval_case(model),),
        serving=KorvidUpstreamServing(
            backend="korvid_upstream",
            source_root=str(SOURCE_ROOT),
            base_url="http://127.0.0.1:11434",
            korvid_revision="33c483e041006eb20259a024ed85a9323e52c8f0",
            timeout_seconds=240,
            model_options=MappingProxyType({}),
        ),
    )


def test_export_preserves_original_and_records_prompt_validation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = "original Korvid prompt\n"
    optimized = "optimized Korvid prompt\n"
    snapshot = {
        "prompt": {
            "text": original,
            "pack_id": "low-korvid-operator",
            "baseline_equivalent": True,
        },
        "source_identity": {
            "korvid_version": "0.4.1",
            "korvid_revision": "33c483e041006eb20259a024ed85a9323e52c8f0",
        },
    }
    campaign = _campaign()
    inspected: dict[str, Any] = {}

    def inspect(_serving: KorvidUpstreamServing, payload: dict[str, Any]) -> dict[str, Any]:
        inspected.update(payload)
        text = payload["tier_pack"]
        return {
            "prompt": {
                "text": original,
                "pack_id": "low-korvid-operator",
                "baseline_equivalent": True,
            },
            "candidate_validation": {
                "valid": True,
                "tier_pack_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        }

    monkeypatch.setattr("korvid_prompt_lab.upstream.run_upstream_request", inspect)
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.KorvidUpstreamRunner.run",
        lambda *_args, **_kwargs: pytest.fail(
            "export must not spend the shared evaluation budget"
        ),
    )
    candidate = prompt_candidate(optimized)

    exported = export_upstream_prompt(
        snapshot, candidate, campaign, tmp_path / "release", seed=9
    )

    assert exported.read_text() == optimized
    assert (tmp_path / "release/original-prompt.txt").read_text() == original
    assert "--- original-prompt.txt" in (tmp_path / "release/prompt.diff").read_text()
    assert "+++ optimized-prompt.txt" in (tmp_path / "release/prompt.diff").read_text()
    assert inspected["tier_pack"] == optimized
    assert inspected["model"]["options"]["seed"] == 9
    assert not (tmp_path / "release/evaluation-verification").exists()

    manifest = json.loads(
        (tmp_path / "release/application-manifest.json").read_text()
    )
    assert manifest["scope"] == "korvid-upstream-tier-pack"
    assert manifest["pack_id"] == "low-korvid-operator"
    assert manifest["original_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert manifest["proposed_sha256"] == hashlib.sha256(optimized.encode()).hexdigest()
    assert manifest["prompt_validation_passed"] is True
    assert manifest["product_application_verified"] is False
    assert manifest["evaluation_reload_receipt"] == "application-verification.json"
    assert "prompt_override_verified" not in manifest
    assert "evaluation_override_verified" not in manifest
    assert manifest["production_application"] == (
        "requires_reviewed_korvid_prompt_pack_update"
    )
    assert manifest["source_writes"] is False
    assert "production_applied" not in manifest
    assert exported.stat().st_mode & 0o222 == 0


def test_export_rejects_snapshot_text_not_matching_fresh_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    snapshot = {
        "prompt": {
            "text": "caller-supplied text",
            "pack_id": "low-korvid-operator",
            "baseline_equivalent": True,
        },
        "source_identity": {
            "korvid_version": "0.4.1",
            "korvid_revision": "33c483e041006eb20259a024ed85a9323e52c8f0",
        },
    }
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.run_upstream_request",
        lambda *_args, **_kwargs: {
            "prompt": {
                "text": "actual upstream text",
                "pack_id": "low-korvid-operator",
                "baseline_equivalent": True,
            },
            "candidate_validation": {
                "valid": True,
                "tier_pack_sha256": hashlib.sha256(b"optimized").hexdigest(),
            },
        },
    )

    with pytest.raises(ValueError, match="fresh upstream inspection"):
        export_upstream_prompt(
            snapshot,
            prompt_candidate("optimized"),
            campaign,
            tmp_path / "release",
        )

    assert not (tmp_path / "release").exists()


def test_exported_diff_applies_when_prompts_have_no_trailing_newline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = "original prompt"
    optimized = "optimized prompt"
    campaign = _campaign()
    snapshot = {
        "prompt": {
            "text": original,
            "pack_id": "low-korvid-operator",
            "baseline_equivalent": True,
        },
        "source_identity": {
            "korvid_version": "0.4.1",
            "korvid_revision": "33c483e041006eb20259a024ed85a9323e52c8f0",
        },
    }
    monkeypatch.setattr(
        "korvid_prompt_lab.upstream.run_upstream_request",
        lambda *_args, **_kwargs: {
            "prompt": {
                "text": original,
                "pack_id": "low-korvid-operator",
                "baseline_equivalent": True,
            },
            "candidate_validation": {
                "valid": True,
                "tier_pack_sha256": hashlib.sha256(optimized.encode()).hexdigest(),
            },
        },
    )
    release = tmp_path / "release"
    export_upstream_prompt(
        snapshot,
        prompt_candidate(optimized),
        campaign,
        release,
    )
    assert (release / "original-prompt.txt").read_bytes() == original.encode()
    assert (release / "optimized-prompt.txt").read_bytes() == optimized.encode()
    assert (
        (release / "prompt.diff").read_text().count(
            "\\ No newline at end of file"
        )
        == 2
    )
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "original-prompt.txt").write_bytes(original.encode())
    (checkout / "optimized-prompt.txt").write_bytes(original.encode())

    subprocess.run(
        ["git", "apply", "--check", "-p0", str(release / "prompt.diff")],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "apply", "-p0", str(release / "prompt.diff")],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (checkout / "optimized-prompt.txt").read_bytes() == optimized.encode()
