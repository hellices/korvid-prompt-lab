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


REPOSITORY = {
    "id": 4242000,
    "full_name": "octo/kube-prompt-grounding",
    "default_branch": "main",
}
WORKFLOW_PATH = ".github/workflows/optimization-campaign.yml"


def trusted_run(run_id: int, **overrides: object) -> dict[str, object]:
    """A workflow run that satisfies the full producer trust predicate."""
    data: dict[str, object] = {
        "id": run_id,
        "path": WORKFLOW_PATH,
        "event": "workflow_dispatch",
        "head_branch": REPOSITORY["default_branch"],
        "conclusion": "success",
        "head_repository": {
            "id": REPOSITORY["id"],
            "full_name": REPOSITORY["full_name"],
        },
        "repository": {"id": REPOSITORY["id"], "full_name": REPOSITORY["full_name"]},
    }
    data.update(overrides)
    return data


def _run_node(
    script: str,
    *,
    pages: list[list[dict[str, object]]],
    runs: dict[str, dict[str, object]],
    run_id: str,
    env: dict[str, str],
    repository: dict[str, object] | None = None,
    paginator: str = "listArtifactsForRepo",
) -> dict:
    harness = f"""
const PAGES = {json.dumps(pages)};
const RUNS = {json.dumps(runs)};
const REPOSITORY = {json.dumps(repository if repository is not None else REPOSITORY)};
const core = {{
  outputs: {{}},
  failed: null,
  setOutput(key, value) {{ this.outputs[key] = value; }},
  setFailed(message) {{ this.failed = String(message); }},
  info() {{}},
  warning() {{}},
}};
const context = {{
  repo: {{ owner: 'octo', repo: 'kube-prompt-grounding' }},
  payload: {{ repository: REPOSITORY }},
}};
const calls = {{ paginate: [] }};
const github = {{
  paginate: async (fn, params) => {{
    calls.paginate.push(params);
    const out = [];
    for (const page of PAGES) {{ out.push(...page); }}
    return out;
  }},
  rest: {{
    actions: {{
      {paginator}: async () => [],
      getWorkflowRun: async ({{ run_id }}) => {{
        const found = RUNS[String(run_id)];
        if (!found) {{ throw new Error('not found'); }}
        return {{ data: found }};
      }},
    }},
  }},
}};
const scanStep = async () => {{
{script}
}};
(async () => {{
  await scanStep();
  console.log('RESULT ' + JSON.stringify({{
    outputs: core.outputs, failed: core.failed, calls,
  }}));
}})();
"""
    child_env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "TMPDIR", "LANG")
        if key in os.environ
    }
    child_env.update(env)
    child_env["CURRENT_RUN_ID"] = run_id
    result = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        env=child_env,
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


def _recovery_scan_script() -> str:
    return str(
        step(load_workflow(), "campaign", "recovery-scan")["with"]["script"]
    )


def _artifact(
    name: str, run_id: int, *, expired: bool = False, **run_overrides: object,
) -> dict[str, object]:
    workflow_run: dict[str, object] = {
        "id": run_id,
        "repository_id": REPOSITORY["id"],
        "head_repository_id": REPOSITORY["id"],
        "head_branch": REPOSITORY["default_branch"],
    }
    workflow_run.update(run_overrides)
    return {"name": name, "expired": expired, "workflow_run": workflow_run}


class TestLineageScan:
    """Producer trust, not just artifact name (wave 2 finding 4)."""

    def _scan(
        self,
        artifacts: list[dict[str, object]],
        *,
        marker: str,
        run_id: str = "9999",
        runs: dict[str, dict[str, object]] | None = None,
        pages: list[list[dict[str, object]]] | None = None,
        repository: dict[str, object] | None = None,
    ) -> dict:
        if pages is None:
            pages = [artifacts]
        if runs is None:
            runs = {}
            for page in pages:
                for artifact in page:
                    producer = artifact["workflow_run"]["id"]  # type: ignore[index]
                    runs.setdefault(str(producer), trusted_run(int(producer)))
        return _run_node(
            _scan_script(),
            pages=pages,
            runs=runs,
            run_id=run_id,
            env={"MARKER_NAME": marker},
            repository=repository,
        )

    def test_duplicate_prior_input_is_detected(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-" + HASH_A.replace(":", "-")
        result = self._scan([_artifact(marker, 4242)], marker=marker)
        assert result["failed"] is None
        assert result["outputs"]["marker-run-id"] == "4242"
        assert result["outputs"]["marker-scope"] == "foreign"

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
        assert result["outputs"]["marker-scope"] == ""

    def test_own_marker_is_reported_as_a_rerun(self) -> None:
        """A rerun must stop, not silently redo the expensive action."""
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan([_artifact(marker, 9999)], marker=marker, run_id="9999")
        assert result["outputs"]["marker-run-id"] == "9999"
        assert result["outputs"]["marker-scope"] == "current"

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

    def test_marker_from_a_foreign_workflow_is_ignored(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan(
            [_artifact(marker, 77)],
            marker=marker,
            runs={"77": trusted_run(77, path=".github/workflows/other.yml")},
        )
        assert result["failed"] is None
        assert result["outputs"]["marker-run-id"] == ""

    def test_marker_from_a_fork_run_is_ignored(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan(
            [_artifact(marker, 78, head_repository_id=999999)],
            marker=marker,
            runs={"78": trusted_run(78)},
        )
        assert result["failed"] is None
        assert result["outputs"]["marker-run-id"] == ""

    def test_marker_whose_run_reports_a_fork_head_repository_is_ignored(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan(
            [_artifact(marker, 79)],
            marker=marker,
            runs={
                "79": trusted_run(
                    79,
                    head_repository={"id": 999999, "full_name": "evil/fork"},
                )
            },
        )
        assert result["failed"] is None
        assert result["outputs"]["marker-run-id"] == ""

    def test_marker_from_a_non_default_branch_is_ignored(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan(
            [_artifact(marker, 80, head_branch="feature/x")],
            marker=marker,
            runs={"80": trusted_run(80, head_branch="feature/x")},
        )
        assert result["failed"] is None
        assert result["outputs"]["marker-run-id"] == ""

    def test_marker_from_an_untrusted_event_is_ignored(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan(
            [_artifact(marker, 81)],
            marker=marker,
            runs={"81": trusted_run(81, event="pull_request")},
        )
        assert result["failed"] is None
        assert result["outputs"]["marker-run-id"] == ""

    def test_marker_whose_run_cannot_be_resolved_is_ignored(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan([_artifact(marker, 82)], marker=marker, runs={})
        assert result["failed"] is None
        assert result["outputs"]["marker-run-id"] == ""

    def test_marker_from_a_foreign_repository_is_ignored(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan(
            [_artifact(marker, 83, repository_id=555)],
            marker=marker,
            runs={"83": trusted_run(83)},
        )
        assert result["failed"] is None
        assert result["outputs"]["marker-run-id"] == ""

    def test_untrusted_artifact_does_not_mask_a_trusted_one(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan(
            [
                _artifact(marker, 90, head_repository_id=999999),
                _artifact(marker, 91),
            ],
            marker=marker,
            runs={"90": trusted_run(90), "91": trusted_run(91)},
        )
        assert result["failed"] is None
        assert result["outputs"]["marker-run-id"] == "91"

    def test_marker_on_a_later_page_is_found(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        pages = [
            [_artifact("unrelated-artifact", 1)],
            [_artifact(marker, 4343)],
        ]
        result = self._scan([], marker=marker, pages=pages)
        assert result["outputs"]["marker-run-id"] == "4343"
        assert result["calls"]["paginate"][0]["per_page"] == 100
        assert result["calls"]["paginate"][0]["name"] == marker

    def test_incomplete_repository_context_fails_closed(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan(
            [_artifact(marker, 4242)],
            marker=marker,
            repository={"id": REPOSITORY["id"]},
        )
        assert result["failed"]


class TestRecoveryScan:
    """Locating the trusted marker a *failed* producer run left behind."""

    def _scan(
        self,
        artifacts: list[dict[str, object]],
        *,
        campaign_id: str = "qwen3-small-operator",
        prior_run_id: str = "4242",
        runs: dict[str, dict[str, object]] | None = None,
    ) -> dict:
        if runs is None:
            runs = {prior_run_id: trusted_run(int(prior_run_id), conclusion="failure")}
        return _run_node(
            _recovery_scan_script(),
            pages=[artifacts],
            runs=runs,
            run_id="9999",
            env={"CAMPAIGN_ID": campaign_id, "PRIOR_RUN_ID": prior_run_id},
            paginator="listWorkflowRunArtifacts",
        )

    def test_finds_the_single_marker_of_the_failed_producer(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan(
            [_artifact("safe-campaign-evidence-x", 4242), _artifact(marker, 4242)],
        )
        assert result["failed"] is None
        assert result["outputs"]["marker-name"] == marker

    def test_no_marker_fails_closed(self) -> None:
        result = self._scan([_artifact("safe-campaign-evidence-x", 4242)])
        assert result["failed"]
        assert result["outputs"].get("marker-name", "") == ""

    def test_multiple_markers_fail_closed(self) -> None:
        a = "campaign-lineage-qwen3-small-operator-initial"
        b = "campaign-lineage-qwen3-small-operator-" + HASH_A.replace(":", "-")
        result = self._scan([_artifact(a, 4242), _artifact(b, 4242)])
        assert result["failed"]

    def test_marker_for_another_campaign_is_ignored(self) -> None:
        result = self._scan([_artifact("campaign-lineage-other-initial", 4242)])
        assert result["failed"]

    def test_untrusted_producer_run_fails_closed(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan(
            [_artifact(marker, 4242)],
            runs={"4242": trusted_run(4242, event="pull_request")},
        )
        assert result["failed"]

    def test_expired_marker_fails_closed(self) -> None:
        marker = "campaign-lineage-qwen3-small-operator-initial"
        result = self._scan([_artifact(marker, 4242, expired=True)])
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


def _recovery_marker(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "campaign_id": "qwen3-small-operator",
        "from_state_key": "initial",
        "from_state_hash": "",
        "to_state_hash": HASH_B,
        "producer_run_id": "4242",
        "prompt_lab_revision": "a" * 40,
        "korvid_revision": "b" * 40,
    }
    payload.update(overrides)
    return payload


def _run_recovery_verify(
    scratch: Path, payload: object, **env_overrides: str,
) -> subprocess.CompletedProcess[str]:
    workflow = load_workflow()
    code = embedded_python(step(workflow, "campaign", "recovery-verify"))
    recovery = scratch / "lineage-recovery"
    recovery.mkdir(parents=True, exist_ok=True)
    (recovery / "lineage-marker.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "TMPDIR", "LANG")
        if key in os.environ
    }
    env.update(
        {
            "CAMPAIGN_ID": "qwen3-small-operator",
            "PRIOR_RUN_ID": "4242",
            "EXPECTED_STATE_HASH": HASH_B,
            "MARKER_NAME": "campaign-lineage-qwen3-small-operator-initial",
            "PROMPT_LAB_REF": "a" * 40,
            "KORVID_REF": "b" * 40,
            "LINEAGE_RECOVERY_ROOT": str(recovery),
        }
    )
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-", str(scratch / "recovery-output")],
        input=code,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class TestLineageRecoveryVerification:
    """A post-upload failure stays recoverable, but only with real proof."""

    def test_matching_marker_authorizes_recovery(self, scratch: Path) -> None:
        result = _run_recovery_verify(scratch, _recovery_marker())
        assert result.returncode == 0, result.stderr
        entries = (scratch / "recovery-output").read_text(encoding="utf-8")
        assert "verified=true" in entries

    def test_marker_for_a_different_produced_state_is_rejected(
        self, scratch: Path,
    ) -> None:
        result = _run_recovery_verify(
            scratch, _recovery_marker(to_state_hash=HASH_A),
        )
        assert result.returncode != 0
        assert "not the requested" in (result.stderr + result.stdout)

    @pytest.mark.parametrize(
        ("payload", "env"),
        [
            (_recovery_marker(campaign_id="other"), {}),
            (_recovery_marker(producer_run_id="9999"), {}),
            (_recovery_marker(prompt_lab_revision="c" * 40), {}),
            (_recovery_marker(korvid_revision="c" * 40), {}),
            (_recovery_marker(schema_version=2), {}),
            (_recovery_marker(from_state_key="sha256-" + "b" * 64), {}),
            (
                _recovery_marker(from_state_hash=HASH_B, from_state_key="initial"),
                {},
            ),
            (["not", "a", "mapping"], {}),
            (_recovery_marker(), {"EXPECTED_STATE_HASH": "nope"}),
        ],
    )
    def test_untrustworthy_recovery_marker_fails_closed(
        self, scratch: Path, payload: object, env: dict[str, str],
    ) -> None:
        result = _run_recovery_verify(scratch, payload, **env)
        assert result.returncode != 0

    def test_extra_files_in_the_recovery_package_fail_closed(
        self, scratch: Path,
    ) -> None:
        recovery = scratch / "lineage-recovery"
        recovery.mkdir(parents=True, exist_ok=True)
        (recovery / "extra.json").write_text("{}", encoding="utf-8")
        result = _run_recovery_verify(scratch, _recovery_marker())
        assert result.returncode != 0
        assert "invalid shape" in (result.stderr + result.stdout)


class TestLineageRecoveryWorkflowStructure:
    def test_failed_prior_run_is_admissible_only_as_a_recovery(self) -> None:
        workflow = load_workflow()
        trust = str(step(workflow, "identity", "trust")["with"]["script"])
        assert "prior.data.conclusion !== 'success'" in trust
        assert "prior.data.conclusion !== 'failure'" in trust
        assert "core.setOutput('prior-run-conclusion', prior.data.conclusion)" in trust
        assert workflow["jobs"]["identity"]["outputs"]["prior-run-conclusion"] == (
            "${{ steps.trust.outputs.prior-run-conclusion }}"
        )

    def test_recovery_proof_precedes_any_expensive_work(self) -> None:
        workflow = load_workflow()
        for later in ("recovery-download", "recovery-verify", "prepare"):
            assert index(workflow, "campaign", "recovery-scan") < index(
                workflow, "campaign", later
            )
        for expensive in ("korvid-token", "azure", "attempt"):
            assert index(workflow, "campaign", "recovery-verify") < index(
                workflow, "campaign", expensive
            )
        assert index(workflow, "campaign", "download") < index(
            workflow, "campaign", "recovery-scan"
        )

    def test_recovery_steps_only_run_for_a_failed_producer(self) -> None:
        workflow = load_workflow()
        scan = step(workflow, "campaign", "recovery-scan")
        assert scan["if"] == (
            "inputs.prior_run_id != '' && "
            "needs.identity.outputs.prior-run-conclusion == 'failure'"
        )
        for step_id in ("recovery-download", "recovery-verify"):
            assert step(workflow, "campaign", step_id)["if"] == (
                "steps.recovery-scan.outputs.marker-name != ''"
            )

    def test_prepare_requires_the_recovery_proof_for_a_failed_producer(self) -> None:
        workflow = load_workflow()
        prepare = step(workflow, "campaign", "prepare")
        assert prepare["env"]["PRIOR_RUN_CONCLUSION"] == (
            "${{ needs.identity.outputs.prior-run-conclusion }}"
        )
        assert prepare["env"]["RECOVERY_VERIFIED"] == (
            "${{ steps.recovery-verify.outputs.verified }}"
        )
        body = embedded_python(prepare)
        assert 'os.environ.get("PRIOR_RUN_CONCLUSION") == "failure"' in body
        assert 'os.environ.get("RECOVERY_VERIFIED") != "true"' in body

    def test_owned_cleanup_failure_is_never_success_shaped(self) -> None:
        """Recovery must not weaken the owned-capacity terminal contract."""
        workflow = load_workflow()
        terminal = next(
            item
            for item in workflow["jobs"]["campaign"]["steps"]
            if item.get("name") == "Enforce terminal and persistence result"
        )
        body = str(terminal["run"])
        assert '[[ "$CLEANUP_OUTCOME" == "success" ]] || exit 70' in body
        assert '[[ "$UPLOAD_OUTCOME" == "success" ]] || exit 70' in body
        assert '[[ "$LINEAGE_CLAIM_OUTCOME" == "success" ]] || exit 70' in body
        cleanup = step(workflow, "campaign", "cleanup")
        assert 'exit "$cleanup_status"' in str(cleanup["run"])

    def test_rerun_stops_with_a_bounded_recovery_instruction(
        self, scratch: Path,
    ) -> None:
        result = _run_reject(
            scratch,
            _marker_payload(producer_run_id="9999"),
            MARKER_SCOPE="current",
            MARKER_RUN_ID="9999",
        )
        combined = result.stderr + result.stdout
        assert result.returncode != 0
        assert "do not re-run this job" in combined
        assert "prior_run_id=9999" in combined
        assert f"expected_state_hash={HASH_B}" in combined

    def test_foreign_duplicate_also_names_the_recovery_dispatch(
        self, scratch: Path,
    ) -> None:
        result = _run_reject(scratch, _marker_payload(), MARKER_SCOPE="foreign")
        combined = result.stderr + result.stdout
        assert result.returncode != 0
        assert "already consumed" in combined
        assert "prior_run_id=4242" in combined


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
