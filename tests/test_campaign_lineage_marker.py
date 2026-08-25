"""Cross-run campaign lineage marker protocol (final review finding 5).

A local file CAS cannot serialize two runs that each downloaded their own copy
of the same prior state. Before any expensive work, a run must reject a prior
state hash that another run in the same repository already consumed, using a
durable GitHub artifact marker.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_optimization_campaign_workflow import (
    embedded_python,
    index,
    load_workflow,
    step,
)

MARKER_NAME_RE = re.compile(
    r"^campaign-lineage-[a-z0-9][a-z0-9._-]{0,62}-(initial|sha256-[0-9a-f]{64})$"
)
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _run_node(script: str, artifacts: list[dict[str, object]], run_id: str) -> dict:
    harness = f"""
const ARTIFACTS = {json.dumps(artifacts)};
const core = {{
  outputs: {{}},
  failed: null,
  setOutput(key, value) {{ this.outputs[key] = value; }},
  setFailed(message) {{ this.failed = String(message); }},
  info() {{}},
}};
const context = {{ repo: {{ owner: 'octo', repo: 'kube-prompt-grounding' }} }};
const github = {{
  paginate: async (fn, params) => fn(params),
  rest: {{ actions: {{ listArtifactsForRepo: async () => ARTIFACTS }} }},
}};
const scanStep = async () => {{
{script}
}};
(async () => {{
  await scanStep();
  console.log('RESULT ' + JSON.stringify({{ outputs: core.outputs, failed: core.failed }}));
}})();
"""
    env = os.environ.copy()
    env["MARKER_NAME"] = env.get("MARKER_NAME", "")
    env["CURRENT_RUN_ID"] = run_id
    result = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("RESULT ")
    )
    return json.loads(line[len("RESULT ") :])


def _scan_script() -> str:
    return str(step(load_workflow(), "campaign", "lineage-scan")["with"]["script"])


def _artifact(name: str, run_id: int, *, expired: bool = False) -> dict[str, object]:
    return {
        "name": name,
        "expired": expired,
        "workflow_run": {"id": run_id},
    }


class TestLineageScan:
    def _scan(
        self,
        artifacts: list[dict[str, object]],
        *,
        marker: str,
        run_id: str = "9999",
    ) -> dict:
        os.environ["MARKER_NAME"] = marker
        try:
            return _run_node(_scan_script(), artifacts, run_id)
        finally:
            os.environ.pop("MARKER_NAME", None)

    def test_duplicate_prior_input_is_detected(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-" + HASH_A.replace(":", "-")
        result = self._scan([_artifact(marker, 4242)], marker=marker)
        assert result["failed"] is None
        assert result["outputs"]["marker-run-id"] == "4242"

    def test_duplicate_initial_state_is_detected(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan([_artifact(marker, 17)], marker=marker)
        assert result["outputs"]["marker-run-id"] == "17"

    def test_successor_with_a_new_hash_proceeds(self) -> None:
        consumed = "campaign-lineage-qwen3-small-operator-" + HASH_A.replace(":", "-")
        successor = "campaign-lineage-qwen3-small-operator-" + HASH_B.replace(":", "-")
        result = self._scan([_artifact(consumed, 4242)], marker=successor)
        assert result["failed"] is None
        assert result["outputs"]["marker-run-id"] == ""

    def test_own_marker_is_ignored(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan([_artifact(marker, 9999)], marker=marker, run_id="9999")
        assert result["outputs"]["marker-run-id"] == ""

    def test_expired_marker_is_ignored(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan([_artifact(marker, 5, expired=True)], marker=marker)
        assert result["outputs"]["marker-run-id"] == ""

    def test_unrelated_artifacts_are_ignored(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan(
            [_artifact("safe-campaign-evidence-qwen3-small-operator-x", 5)],
            marker=marker,
        )
        assert result["outputs"]["marker-run-id"] == ""

    def test_invalid_marker_name_fails_closed(self) -> None:
        result = self._scan([], marker="../evil")
        assert result["failed"]


@pytest.fixture()
def scratch() -> Iterator[Path]:
    directory = ROOT / "artifacts" / f"lineage-marker-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _marker_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "campaign_id": "qwen3-small-operator",
        "from_state_key": HASH_A.replace(":", "-"),
        "from_state_hash": HASH_A,
        "to_state_hash": HASH_B,
        "producer_run_id": "4242",
        "prompt_lab_revision": "a" * 40,
        "korvid_revision": "b" * 40,
    }
    payload.update(overrides)
    return payload


def _run_reject(scratch: Path, payload: object, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    workflow = load_workflow()
    code = embedded_python(step(workflow, "campaign", "lineage-reject"))
    conflict = scratch / "lineage-conflict"
    conflict.mkdir(parents=True, exist_ok=True)
    (conflict / "lineage-marker.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    env = os.environ.copy()
    env.update(
        {
            "CAMPAIGN_ID": "qwen3-small-operator",
            "LINEAGE_FROM_KEY": HASH_A.replace(":", "-"),
            "EXPECTED_STATE_HASH": HASH_A,
            "MARKER_RUN_ID": "4242",
            "LINEAGE_CONFLICT_ROOT": str(conflict),
        }
    )
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-", str(scratch / "reject-output")],
        input=code,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class TestLineageReject:
    def test_valid_marker_stops_the_run(self, scratch: Path) -> None:
        result = _run_reject(scratch, _marker_payload())
        assert result.returncode != 0
        assert "already consumed" in (result.stderr + result.stdout)

    def test_initial_marker_stops_the_run(self, scratch: Path) -> None:
        result = _run_reject(
            scratch,
            _marker_payload(from_state_key="initial", from_state_hash=""),
            LINEAGE_FROM_KEY="initial",
            EXPECTED_STATE_HASH="",
        )
        assert result.returncode != 0
        assert "already consumed" in (result.stderr + result.stdout)

    @pytest.mark.parametrize(
        "payload",
        [
            _marker_payload(schema_version=2),
            _marker_payload(campaign_id="other"),
            _marker_payload(from_state_hash=HASH_B),
            _marker_payload(to_state_hash="nope"),
            _marker_payload(producer_run_id="9999"),
            _marker_payload(prompt_lab_revision="short"),
            ["not", "a", "mapping"],
        ],
    )
    def test_invalid_marker_fails_closed(
        self, scratch: Path, payload: object,
    ) -> None:
        result = _run_reject(scratch, payload)
        assert result.returncode != 0


class TestLineageWorkflowStructure:
    def test_marker_is_checked_before_any_expensive_work(self) -> None:
        workflow = load_workflow()
        assert index(workflow, "campaign", "prepare") < index(
            workflow, "campaign", "lineage-scan"
        )
        for later in ("lineage-download", "lineage-reject"):
            assert index(workflow, "campaign", "lineage-scan") < index(
                workflow, "campaign", later
            )
        for expensive in ("korvid-token", "azure", "attempt"):
            assert index(workflow, "campaign", "lineage-reject") < index(
                workflow, "campaign", expensive
            )

    def test_marker_is_claimed_after_upload_and_before_dispatch(self) -> None:
        workflow = load_workflow()
        assert index(workflow, "campaign", "upload") < index(
            workflow, "campaign", "lineage-write"
        ) < index(workflow, "campaign", "lineage-claim") < index(
            workflow, "campaign", "dispatch"
        )
        claim = step(workflow, "campaign", "lineage-claim")
        assert claim["with"]["if-no-files-found"] == "error"
        assert claim["with"]["name"] == (
            "${{ steps.prepare.outputs.lineage-marker-name }}"
        )
        write = step(workflow, "campaign", "lineage-write")
        assert write["if"] == "steps.upload.outcome == 'success'"

    def test_marker_upload_failure_prevents_dispatch(self) -> None:
        workflow = load_workflow()
        dispatch = step(workflow, "campaign", "dispatch")
        assert "steps.lineage-claim.outcome == 'success'" in dispatch["if"]
        terminal = next(
            item
            for item in workflow["jobs"]["campaign"]["steps"]
            if item.get("name") == "Enforce terminal and persistence result"
        )
        assert "LINEAGE_CLAIM_OUTCOME" in terminal["env"]
        assert '[[ "$LINEAGE_CLAIM_OUTCOME" == "success" ]]' in str(terminal["run"])

    def test_no_token_is_exposed_in_argv_or_logs(self) -> None:
        workflow = load_workflow()
        for step_id in (
            "lineage-scan",
            "lineage-download",
            "lineage-reject",
            "lineage-write",
            "lineage-claim",
        ):
            item = step(workflow, "campaign", step_id)
            body = str(item.get("run", ""))
            assert "GH_TOKEN" not in body
            assert "github.token" not in body
            assert "secrets." not in body
            for value in (item.get("env") or {}).values():
                assert "secrets." not in str(value)
                assert "github.token" not in str(value)

    def test_prepare_emits_a_deterministic_marker_name(self) -> None:
        workflow = load_workflow()
        prepare = step(workflow, "campaign", "prepare")
        body = embedded_python(prepare)
        assert "lineage-marker-name=" in body
        assert "lineage-from-key=" in body
        assert 'f"campaign-lineage-{control.campaign_id}-{lineage_from_key}"' in body
