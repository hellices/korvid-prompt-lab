"""Safe round projection allowlist (final review finding 1).

The workflow must accept the *real* sanitized projections `write_safe_evidence`
produces — including `responses/` and, for a changed candidate,
`before-responses/` — while still rejecting raw artifact roots, transcripts,
audit journals, kubeconfig, credentials, GEPA state, unexpected paths, symlinks
and unsafe files.

These tests exercise the shared allowlist directly and then run the actual
workflow packaging and continuation predicates as subprocesses against real
`write_safe_evidence` output.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_optimization_campaign_workflow import (
    embedded_python,
    load_workflow,
    step,
)
from test_rounds import (
    CHANGED_BEST_CANDIDATE,
    CHANGED_FINGERPRINT,
    FINGERPRINT,
    response,
    write_live_fixture,
)

from korvid_prompt_lab.campaign_artifacts import (
    validate_safe_round_package,
)
from korvid_prompt_lab.rounds import write_safe_evidence

MANIFEST = ROOT / "examples/optimization-campaigns/qwen3-small-operator.yaml"
INITIAL_CANDIDATE = ROOT / "examples/candidates/shipped-small.yaml"


def build_real_safe_round(tmp_path: Path) -> Path:
    """Produce a genuine changed-candidate safe evidence projection."""
    before_root = write_live_fixture(
        tmp_path / "before",
        aggregate_score=0.1,
        responses=[
            response("completed", answer="before raw secret"),
            response(
                "completed",
                run_id="case-a-model-a-r02",
                repetition=2,
                answer="before raw secret",
            ),
        ],
        repetitions_per_case=2,
    )
    after_root = write_live_fixture(
        tmp_path / "after",
        candidate=CHANGED_BEST_CANDIDATE,
        aggregate_score=0.4,
        responses=[
            response(
                "completed",
                candidate_fingerprint=CHANGED_FINGERPRINT,
                answer="after raw secret",
            ),
            response(
                "completed",
                run_id="case-a-model-a-r02",
                repetition=2,
                candidate_fingerprint=CHANGED_FINGERPRINT,
                answer="after raw secret",
            ),
        ],
        repetitions_per_case=2,
        include_optimization=True,
        include_best_candidate=True,
        seed_candidate_fingerprint=FINGERPRINT,
    )
    return write_safe_evidence(
        after_root,
        tmp_path / "safe-evidence",
        before_artifact_root=before_root,
        optimize_artifact_root=after_root,
        prompt_lab_revision="a" * 40,
        korvid_revision="b" * 40,
        workflow_run_url="https://github.example/actions/runs/42",
        campaign_action_id="11111111-1111-5111-8111-111111111111",
    )


class TestSafeRoundAllowlist:
    def test_accepts_real_write_safe_evidence_output(self, tmp_path: Path) -> None:
        package = build_real_safe_round(tmp_path)
        assert (package / "responses").is_dir()
        assert (package / "before-responses").is_dir()
        validate_safe_round_package(package)

    def test_accepts_package_without_comparison(self, tmp_path: Path) -> None:
        root = write_live_fixture(tmp_path / "only")
        package = write_safe_evidence(root, tmp_path / "safe-evidence")
        assert (package / "responses").is_dir()
        assert not (package / "before-responses").exists()
        validate_safe_round_package(package)

    def test_rejects_missing_required_summary(self, tmp_path: Path) -> None:
        package = build_real_safe_round(tmp_path)
        (package / "round-summary.md").unlink()
        with pytest.raises(ValueError, match="missing required file"):
            validate_safe_round_package(package)

    @pytest.mark.parametrize(
        "relative",
        [
            "runs/case-a-model-a-r01/response.json",
            "transcripts/session.log",
            "audit/journal.jsonl",
            "kubeconfig",
            ".kube/config",
            "credentials.json",
            "gepa-state/state.bin",
            "responses/nested/run.json",
            "responses/run.txt",
            "unexpected.json",
        ],
    )
    def test_rejects_unsafe_paths(self, tmp_path: Path, relative: str) -> None:
        package = build_real_safe_round(tmp_path)
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="safe round package"):
            validate_safe_round_package(package)

    def test_rejects_raw_artifact_root(self, tmp_path: Path) -> None:
        raw_root = write_live_fixture(tmp_path / "raw")
        with pytest.raises(ValueError, match="safe round package"):
            validate_safe_round_package(raw_root)

    def test_rejects_symlinked_entry(self, tmp_path: Path) -> None:
        package = build_real_safe_round(tmp_path)
        secret = tmp_path / "secret.json"
        secret.write_text("{}", encoding="utf-8")
        (package / "responses" / "leak.json").symlink_to(secret)
        with pytest.raises(ValueError, match="symlink"):
            validate_safe_round_package(package)

    def test_rejects_symlinked_root(self, tmp_path: Path) -> None:
        package = build_real_safe_round(tmp_path)
        link = tmp_path / "link"
        link.symlink_to(package, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            validate_safe_round_package(link)


def _run_embedded(
    code: str, env_overrides: dict[str, str], output: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-", str(output)],
        input=code,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _github_output_entries(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
    )


@pytest.fixture()
def scratch() -> Iterator[Path]:
    directory = ROOT / "artifacts" / f"safe-round-package-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


class TestWorkflowPackagingAcceptsSafeProjections:
    """The real packaging and continuation predicates, run as processes."""

    def _prepare_initial(self, scratch: Path) -> dict[str, str]:
        workflow = load_workflow()
        code = embedded_python(step(workflow, "campaign", "prepare"))
        output = scratch / "prepare-output"
        campaign_root = scratch / "campaign"
        result = _run_embedded(
            code,
            {
                "MANIFEST": MANIFEST.relative_to(ROOT).as_posix(),
                "CAMPAIGN_ID": "qwen3-small-operator-v3",
                "MANIFEST_SHA256": (
                    "sha256:" + hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
                ),
                "EVALUATION_CAMPAIGN": (
                    "examples/campaigns/aks-small-operator-qualification.yaml"
                ),
                "INITIAL_CANDIDATE": INITIAL_CANDIDATE.relative_to(ROOT).as_posix(),
                "PROMPT_LAB_REF": "a" * 40,
                "KORVID_REF": "b" * 40,
                "PRIOR_RUN_ID": "",
                "EXPECTED_STATE_HASH": "",
                "PRIOR_ROOT": str(scratch / "unused-prior"),
                "CAMPAIGN_ROOT": str(campaign_root),
                "KORVID_AKS_NAMESPACE": "ollama",
                "KORVID_AKS_SERVICE": "ollama",
            },
            output,
        )
        assert result.returncode == 0, result.stderr
        return _github_output_entries(output)

    def _build_next_root(
        self, scratch: Path, prepared: dict[str, str], round_evidence: Path | None,
    ) -> Path:
        next_root = scratch / "next"
        next_root.mkdir()
        shutil.copyfile(prepared["state-path"], next_root / "campaign-state.json")
        (next_root / "campaign-summary.md").write_text(
            "# Campaign\n", encoding="utf-8"
        )
        (next_root / "campaign-action.json").write_text(
            json.dumps({"action_id": "11111111-1111-5111-8111-111111111111"}),
            encoding="utf-8",
        )
        if round_evidence is not None:
            shutil.copytree(round_evidence, next_root / "round-evidence")
        return next_root

    def _run_package(
        self, scratch: Path, prepared: dict[str, str], next_root: Path,
    ) -> subprocess.CompletedProcess[str]:
        workflow = load_workflow()
        code = embedded_python(step(workflow, "campaign", "package"))
        output = scratch / "package-output"
        return _run_embedded(
            code,
            {
                "MANIFEST": MANIFEST.relative_to(ROOT).as_posix(),
                "CAMPAIGN_ID": "qwen3-small-operator-v3",
                "MANIFEST_SHA256": (
                    "sha256:" + hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
                ),
                "PROMPT_LAB_REF": "a" * 40,
                "KORVID_REF": "b" * 40,
                "CURRENT_CANDIDATE": prepared["candidate-path"],
                "SEED_CANDIDATE": INITIAL_CANDIDATE.relative_to(ROOT).as_posix(),
                "SEED_CANDIDATE_FINGERPRINT": prepared["seed-candidate-fingerprint"],
                "NEXT_ROOT": str(next_root),
                "SAFE_UPLOAD_ROOT": str(scratch / "safe-upload"),
                "WRAPPER_EXIT": "0",
            },
            output,
        )

    def test_real_safe_projections_are_packaged_and_resumable(
        self, scratch: Path, tmp_path: Path,
    ) -> None:
        prepared = self._prepare_initial(scratch)
        package = build_real_safe_round(tmp_path)
        next_root = self._build_next_root(scratch, prepared, package)

        result = self._run_package(scratch, prepared, next_root)
        assert result.returncode == 0, result.stderr

        entries = _github_output_entries(scratch / "package-output")
        assert entries["state-hash"] == prepared["prior-state-hash"]
        packaged = (
            scratch
            / "safe-upload"
            / "safe-round"
            / "11111111-1111-5111-8111-111111111111"
        )
        assert (packaged / "responses").is_dir()
        assert (packaged / "before-responses").is_dir()
        assert list((packaged / "responses").glob("*.json"))
        assert list((packaged / "before-responses").glob("*.json"))

        # The uploaded package must still be an acceptable continuation input.
        workflow = load_workflow()
        code = embedded_python(step(workflow, "campaign", "prepare"))
        output = scratch / "resume-output"
        continued = _run_embedded(
            code,
            {
                "MANIFEST": MANIFEST.relative_to(ROOT).as_posix(),
                "CAMPAIGN_ID": "qwen3-small-operator-v3",
                "MANIFEST_SHA256": (
                    "sha256:" + hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
                ),
                "EVALUATION_CAMPAIGN": (
                    "examples/campaigns/aks-small-operator-qualification.yaml"
                ),
                "INITIAL_CANDIDATE": INITIAL_CANDIDATE.relative_to(ROOT).as_posix(),
                "PROMPT_LAB_REF": "a" * 40,
                "KORVID_REF": "b" * 40,
                "PRIOR_RUN_ID": "4242",
                "EXPECTED_STATE_HASH": prepared["prior-state-hash"],
                "PRIOR_ROOT": str(scratch / "safe-upload"),
                "CAMPAIGN_ROOT": str(scratch / "resume-campaign"),
                "KORVID_AKS_NAMESPACE": "ollama",
                "KORVID_AKS_SERVICE": "ollama",
            },
            output,
        )
        assert continued.returncode == 0, continued.stderr
        resumed = _github_output_entries(output)
        assert resumed["prior-state-hash"] == prepared["prior-state-hash"]
        assert resumed["action-kind"] == "search"

    def test_packaging_rejects_raw_evidence_in_round_output(
        self, scratch: Path, tmp_path: Path,
    ) -> None:
        prepared = self._prepare_initial(scratch)
        package = build_real_safe_round(tmp_path)
        (package / "transcripts").mkdir()
        (package / "transcripts" / "session.log").write_text("raw", encoding="utf-8")
        next_root = self._build_next_root(scratch, prepared, package)

        result = self._run_package(scratch, prepared, next_root)
        assert result.returncode != 0
        assert "safe round package" in (result.stderr + result.stdout)
        assert not (scratch / "safe-upload" / "safe-round").exists() or not list(
            (scratch / "safe-upload" / "safe-round").iterdir()
        )
