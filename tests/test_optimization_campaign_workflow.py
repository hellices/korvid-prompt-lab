"""Structural contracts for the protected optimization campaign outer loop."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "optimization-campaign.yml"
README_PATH = ROOT / "README.md"

ACTION_PINS = {
    "actions/checkout": "11bd71901bbe5b1630ceea73d27597364c9af683",
    "actions/download-artifact": "d3f86a106a0bac45b974a628896c90dbdf5c8093",
    "actions/setup-python": "a26af69be951a213d495a4c3e4e4022e16d87065",
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
    "actions/create-github-app-token": "df432ceedc7162793a195dd1713ff69aefc7379e",
    "actions/github-script": "60a0d83039c74a4aee543508d2ffcb1c3799cdea",
    "astral-sh/setup-uv": "e92bafb6253dcd438e0484186d7669ea7a8ca1cc",
    "azure/login": "a65d910e8af852a8061c627c456678983e180302",
}


def load_workflow() -> dict[str, Any]:
    raw = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    if True in raw and "on" not in raw:
        raw["on"] = raw.pop(True)
    return raw


def test_frozen_workflows_have_a_tracked_uv_lockfile() -> None:
    workflows = (
        WORKFLOW_PATH,
        ROOT / ".github" / "workflows" / "grounding-round.yml",
    )
    assert all("uv sync --python 3.12 --frozen" in path.read_text(encoding="utf-8") for path in workflows)

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "uv.lock"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0, "uv.lock must be tracked when workflows use uv sync --frozen"


def job(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    return workflow["jobs"][name]


def steps(workflow: dict[str, Any], job_name: str) -> list[dict[str, Any]]:
    return list(job(workflow, job_name)["steps"])


def step(workflow: dict[str, Any], job_name: str, step_id: str) -> dict[str, Any]:
    matches = [item for item in steps(workflow, job_name) if item.get("id") == step_id]
    assert len(matches) == 1, f"expected exactly one {job_name}.{step_id} step"
    return matches[0]


def index(workflow: dict[str, Any], job_name: str, step_id: str) -> int:
    return next(
        i
        for i, item in enumerate(steps(workflow, job_name))
        if item.get("id") == step_id
    )


def all_script_bodies(workflow: dict[str, Any]) -> list[str]:
    bodies: list[str] = []
    for job_name in workflow["jobs"]:
        for item in steps(workflow, job_name):
            if "run" in item:
                bodies.append(str(item["run"]))
            script = dict(item.get("with", {})).get("script")
            if script:
                bodies.append(str(script))
    return bodies


def embedded_python(item: dict[str, Any]) -> str:
    body = str(item["run"])
    marker = "python3 - \"$GITHUB_OUTPUT\" <<'PY'\n"
    assert body.count(marker) == 1
    block = body.split(marker, 1)[1]
    assert block.endswith("PY\n")
    return block[: -len("PY\n")]


def test_trigger_inputs_permissions_and_protected_job_are_exact() -> None:
    workflow = load_workflow()

    assert set(workflow["on"]) == {"workflow_dispatch"}
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert inputs == {
        "manifest": {
            "description": "Optimization campaign manifest in the trusted Prompt Lab revision",
            "required": True,
            "type": "string",
            "default": "examples/optimization-campaigns/qwen3-small-operator.yaml",
        },
        "prompt_lab_ref": {
            "description": "Trusted Prompt Lab commit (exact 40-hex SHA)",
            "required": True,
            "type": "string",
        },
        "korvid_ref": {
            "description": "Trusted Korvid commit (exact 40-hex SHA)",
            "required": True,
            "type": "string",
        },
        "prior_run_id": {
            "description": "Exact prior successful run ID; must be paired with expected_state_hash",
            "required": False,
            "type": "string",
        },
        "expected_state_hash": {
            "description": "Exact sha256 state hash from prior_run_id; must be paired",
            "required": False,
            "type": "string",
        },
    }
    assert workflow["permissions"] == {
        "actions": "write",
        "contents": "read",
        "id-token": "write",
    }
    assert workflow["concurrency"] == {
        "group": "optimization-campaign-dispatch-${{ github.repository }}",
        "cancel-in-progress": False,
    }, "the full run must finish before its dispatched successor verifies it"

    assert set(workflow["jobs"]) == {"identity", "campaign"}
    identity = job(workflow, "identity")
    campaign = job(workflow, "campaign")
    assert identity["runs-on"] == "ubuntu-latest"
    assert campaign["needs"] == "identity"
    assert campaign["runs-on"] == "prompt-lab-runners"
    assert campaign["environment"] == "aks-grounding"
    assert campaign["concurrency"] == {
        "group": "optimization-campaign-${{ needs.identity.outputs.campaign-id }}",
        "cancel-in-progress": False,
    }
    assert identity["outputs"] == {
        "campaign-id": "${{ steps.manifest.outputs.campaign-id }}",
        "manifest-sha256": "${{ steps.manifest.outputs.manifest-sha256 }}",
        "evaluation-campaign": "${{ steps.manifest.outputs.evaluation-campaign }}",
        "initial-candidate": "${{ steps.manifest.outputs.initial-candidate }}",
        "model": "${{ steps.manifest.outputs.model }}",
        "expected-artifact-hash": "${{ steps.validate.outputs.expected-artifact-hash }}",
        "prior-run-conclusion": "${{ steps.trust.outputs.prior-run-conclusion }}",
    }


def test_untrusted_inputs_and_prior_run_are_rejected_before_checkout() -> None:
    workflow = load_workflow()
    validation = step(workflow, "identity", "validate")
    trust = step(workflow, "identity", "trust")

    assert validation["env"] == {
        "DEFAULT_BRANCH": "${{ github.event.repository.default_branch }}",
        "WORKFLOW_REF_NAME": "${{ github.ref_name }}",
        "MANIFEST": "${{ inputs.manifest }}",
        "PROMPT_LAB_REF": "${{ inputs.prompt_lab_ref }}",
        "KORVID_REF": "${{ inputs.korvid_ref }}",
        "PRIOR_RUN_ID": "${{ inputs.prior_run_id }}",
        "EXPECTED_STATE_HASH": "${{ inputs.expected_state_hash }}",
    }
    validation_body = str(validation["run"])
    assert '[[ "$WORKFLOW_REF_NAME" == "$DEFAULT_BRANCH" ]]' in validation_body
    assert validation_body.count("must be an exact 40-hex commit SHA") == 2
    assert '-n "$PRIOR_RUN_ID" && -z "$EXPECTED_STATE_HASH"' in validation_body
    assert '-z "$PRIOR_RUN_ID" && -n "$EXPECTED_STATE_HASH"' in validation_body
    assert re.search(
        r'PRIOR_RUN_ID.*\^\[1-9\]\[0-9\]\{0,19\}\$', validation_body, re.DOTALL
    )
    assert "sha256:[0-9a-f]{64}" in validation_body
    assert "expected-artifact-hash=sha256-${EXPECTED_STATE_HASH#sha256:}" in validation_body
    assert ".." in validation_body and "relative YAML path" in validation_body

    assert trust["uses"] == (
        "actions/github-script@"
        + ACTION_PINS["actions/github-script"]
    )
    assert trust["env"] == {
        "PROMPT_LAB_REF": "${{ inputs.prompt_lab_ref }}",
        "KORVID_REF": "${{ inputs.korvid_ref }}",
        "PRIOR_RUN_ID": "${{ inputs.prior_run_id }}",
        "EXPECTED_REPOSITORY": "${{ github.repository }}",
        "DEFAULT_BRANCH": "${{ github.event.repository.default_branch }}",
        "KORVID_REPOSITORY": "${{ github.repository_owner }}/korvid",
        "CURRENT_RUN_ID": "${{ github.run_id }}",
    }
    trust_script = str(trust["with"]["script"])
    for contract in (
        "compareCommitsWithBasehead",
        "head_repository",
        "getWorkflowRun",
        ".github/workflows/optimization-campaign.yml",
        "workflow_dispatch",
        "conclusion !== 'success'",
        "prior.data.head_branch !== defaultBranch",
    ):
        assert contract in trust_script

    assert index(workflow, "identity", "validate") < index(
        workflow, "identity", "trust"
    ) < index(workflow, "identity", "checkout")


def test_manifest_identity_is_validated_and_drives_concurrency() -> None:
    workflow = load_workflow()
    checkout = step(workflow, "identity", "checkout")
    manifest = step(workflow, "identity", "manifest")

    assert checkout["uses"] == "actions/checkout@" + ACTION_PINS["actions/checkout"]
    assert checkout["with"] == {
        "ref": "${{ inputs.prompt_lab_ref }}",
        "path": "prompt-lab-identity",
        "persist-credentials": False,
    }
    assert manifest["working-directory"] == "prompt-lab-identity"
    assert manifest["env"] == {
        "MANIFEST": "${{ inputs.manifest }}",
        "PROMPT_LAB_REF": "${{ inputs.prompt_lab_ref }}",
        "KORVID_REF": "${{ inputs.korvid_ref }}",
    }
    body = str(manifest["run"])
    assert "load_optimization_campaign" in body
    assert "load_candidate" in body
    assert "hashlib.sha256" in body
    assert "campaign-id=" in body
    assert "manifest-sha256=" in body
    assert "^[a-z0-9][a-z0-9._-]{0,62}$" in body


def test_campaign_preflight_downloads_and_verifies_exact_prior_artifact() -> None:
    workflow = load_workflow()
    campaign_checkout = step(workflow, "campaign", "checkout")
    download = step(workflow, "campaign", "download")
    prepare = step(workflow, "campaign", "prepare")
    app_token = step(workflow, "campaign", "korvid-token")
    azure = step(workflow, "campaign", "azure")

    assert campaign_checkout["with"] == {
        "ref": "${{ inputs.prompt_lab_ref }}",
        "path": "prompt-lab",
        "persist-credentials": False,
    }
    assert download["if"] == "inputs.prior_run_id != ''"
    assert download["uses"] == (
        "actions/download-artifact@" + ACTION_PINS["actions/download-artifact"]
    )
    assert download["with"] == {
        "name": (
            "safe-campaign-evidence-${{ needs.identity.outputs.campaign-id }}-"
            "${{ needs.identity.outputs.expected-artifact-hash }}"
        ),
        "path": "prompt-lab/artifacts/optimization-campaign/prior",
        "github-token": "${{ github.token }}",
        "repository": "${{ github.repository }}",
        "run-id": "${{ inputs.prior_run_id }}",
    }

    prepare_body = str(prepare["run"])
    for contract in (
        "state_hash(state)",
        "expected_state_hash",
        "prompt_lab_revision",
        "korvid_revision",
        "manifest_sha256",
        "campaign_id",
        "champion-candidate.yaml",
        "state.status.value != \"running\"",
        "is_symlink()",
        "initial_state(",
    ):
        assert contract in prepare_body
    assert "campaign_root.mkdir(parents=True, exist_ok=True)" in prepare_body
    assert "prior-state.json already exists" in prepare_body
    assert prepare["env"]["PRIOR_ROOT"].endswith(
        "/artifacts/optimization-campaign/prior"
    )
    assert prepare["env"]["CAMPAIGN_ID"] == (
        "${{ needs.identity.outputs.campaign-id }}"
    )
    assert prepare["env"]["MANIFEST_SHA256"] == (
        "${{ needs.identity.outputs.manifest-sha256 }}"
    )

    assert index(workflow, "campaign", "download") < index(
        workflow, "campaign", "prepare"
    ) < index(workflow, "campaign", "korvid-token")
    assert index(workflow, "campaign", "korvid-imports") < index(
        workflow, "campaign", "azure"
    )
    assert app_token["with"]["permission-contents"] == "read"
    assert azure["uses"] == "azure/login@" + ACTION_PINS["azure/login"]


def test_prepare_initialization_executes_and_writes_github_output() -> None:
    workflow = load_workflow()
    code = embedded_python(step(workflow, "campaign", "prepare"))
    scratch = ROOT / "artifacts" / f"workflow-prepare-test-{uuid.uuid4().hex}"
    output = scratch / "github-output"
    campaign_root = scratch / "campaign"
    manifest = ROOT / "examples/optimization-campaigns/qwen3-small-operator.yaml"
    env = os.environ.copy()
    env.update(
        {
            "MANIFEST": manifest.relative_to(ROOT).as_posix(),
            "CAMPAIGN_ID": "qwen3-small-operator-v4",
            "MANIFEST_SHA256": (
                "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
            ),
            "EVALUATION_CAMPAIGN": (
                "examples/campaigns/aks-small-operator-qualification.yaml"
            ),
            "INITIAL_CANDIDATE": "examples/candidates/shipped-small.yaml",
            "PROMPT_LAB_REF": "a" * 40,
            "KORVID_REF": "b" * 40,
            "PRIOR_RUN_ID": "",
            "EXPECTED_STATE_HASH": "",
            "PRIOR_ROOT": str(scratch / "prior"),
            "CAMPAIGN_ROOT": str(campaign_root),
            "KORVID_AKS_NAMESPACE": "ollama",
            "KORVID_AKS_SERVICE": "ollama",
        }
    )

    scratch.mkdir(parents=True)
    try:
        result = subprocess.run(
            [sys.executable, "-", str(output)],
            input=code,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        entries = dict(
            line.split("=", 1)
            for line in output.read_text(encoding="utf-8").splitlines()
        )
        assert set(entries) == {
            "prior-state-hash",
            "state-path",
            "candidate-path",
            "action-kind",
            "lineage-from-key",
            "lineage-marker-name",
            "seed-candidate-fingerprint",
        }
        assert entries["lineage-from-key"] == "initial"
        assert entries["lineage-marker-name"] == (
            "campaign-lineage-qwen3-small-operator-v4-initial"
        )
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", entries["prior-state-hash"])
        assert entries["action-kind"] == "search"
        assert Path(entries["state-path"]).is_file()
        assert Path(entries["candidate-path"]).is_file()
    finally:
        shutil.rmtree(scratch)


def test_exact_revisions_are_checked_out_and_imported_before_azure() -> None:
    workflow = load_workflow()
    korvid = step(workflow, "campaign", "korvid-checkout")
    imports = step(workflow, "campaign", "korvid-imports")

    assert korvid["with"] == {
        "repository": "${{ github.repository_owner }}/korvid",
        "ref": "${{ inputs.korvid_ref }}",
        "token": "${{ steps.korvid-token.outputs.token }}",
        "path": "korvid",
        "persist-credentials": False,
    }
    assert imports["working-directory"] == "prompt-lab"
    assert imports["env"] == {
        "KORVID_SOURCE_ROOT": "${{ github.workspace }}/korvid",
        "UV_PROJECT_ENVIRONMENT": "${{ runner.temp }}/korvid-uv-env",
    }
    assert "korvid-bridge --check-imports" in imports["run"]


def test_run_executes_one_action_packages_safe_evidence_and_appends_summary() -> None:
    workflow = load_workflow()
    bodies = all_script_bodies(workflow)
    assert sum(body.count("run-optimization-campaign-step.sh") for body in bodies) == 1

    attempt = step(workflow, "campaign", "attempt")
    package = step(workflow, "campaign", "package")
    summary = step(workflow, "campaign", "summary")
    upload = step(workflow, "campaign", "upload")

    assert attempt["timeout-minutes"] == 150
    assert attempt["env"]["CAMPAIGN_EXPECTED_PRIOR_HASH"] == (
        "${{ steps.prepare.outputs.prior-state-hash }}"
    )
    assert attempt["env"]["CAMPAIGN_OUTPUT_ROOT"].endswith(
        "/artifacts/optimization-campaign/next"
    )
    assert "bash scripts/run-optimization-campaign-step.sh" in attempt["run"]

    package_body = str(package["run"])
    assert "safe-campaign" in package_body
    assert "safe-round" in package_body
    assert "champion-candidate.yaml" in package_body
    assert "manifest-identity.json" in package_body
    assert "state_hash(state)" in package_body
    assert "safe-campaign-evidence-" in package_body
    assert 'new_hash.replace(":", "-")' in package_body
    assert "validate_safe_round_package(round_evidence)" in package_body
    assert '{"responses", "raw", "transcripts"}' not in package_body

    assert summary["if"] == (
        "always() && steps.package.outcome == 'success'"
    )
    assert summary["run"] == (
        'cat "$SAFE_UPLOAD_ROOT/safe-campaign/campaign-summary.md" '
        '>> "$GITHUB_STEP_SUMMARY"\n'
    )
    assert upload["uses"] == (
        "actions/upload-artifact@" + ACTION_PINS["actions/upload-artifact"]
    )
    assert upload["if"] == "steps.package.outcome == 'success'"
    assert upload["with"] == {
        "name": "${{ steps.package.outputs.artifact-name }}",
        "path": (
            "prompt-lab/artifacts/optimization-campaign/safe-upload/"
            "safe-campaign/\n"
            "prompt-lab/artifacts/optimization-campaign/safe-upload/"
            "safe-round/\n"
        ),
        "retention-days": 30,
        "if-no-files-found": "error",
    }
    assert index(workflow, "campaign", "attempt") < index(
        workflow, "campaign", "package"
    ) < index(workflow, "campaign", "upload")


def test_cleanup_is_always_owned_idempotent_and_non_destructive() -> None:
    workflow = load_workflow()
    record = step(workflow, "campaign", "modeleval")
    cleanup = step(workflow, "campaign", "cleanup")

    assert "original-count=" in record["run"]
    assert cleanup["if"] == "always()"
    assert cleanup["env"] == {
        "ORIGINAL_NODE_COUNT": "${{ steps.modeleval.outputs.original-count }}",
        "CURRENT_RUNNER_POD": "${{ runner.name }}",
    }
    body = str(cleanup["run"])
    assert "--name modeleval" in body
    assert "--node-count 0" in body
    assert '[[ "$ORIGINAL_NODE_COUNT" == "0" ]]' in body
    assert "arc-runners-prompt-lab" in body
    assert "actions.github.com/scale-set-name=prompt-lab-runners" in body
    assert "CURRENT_RUNNER_POD" in body
    assert "delete" not in body
    assert "kill" not in body
    assert "pkill" not in body


def embedded_arc_python(item: dict[str, Any]) -> str:
    """Extract the ARC observation heredoc from the cleanup step."""
    body = str(item["run"])
    marker = 'RUNNER_JSON="$runner_json" python3 - <<\'PY\''
    _, _, rest = body.partition(marker)
    assert rest, "ARC observation heredoc not found"
    rest = rest.partition("\n")[2]
    code, _, _ = rest.partition("\nPY\n")
    return textwrap.dedent(code) + "\n"


def test_unrelated_arc_runner_pods_are_advisory_only() -> None:
    workflow = load_workflow()
    code = embedded_arc_python(step(workflow, "campaign", "cleanup"))
    runner_json = json.dumps(
        {
            "items": [
                {
                    "metadata": {"name": "prompt-lab-runners-abcde"},
                    "status": {"phase": "Running"},
                },
                {
                    "metadata": {"name": "prompt-lab-runners-unrelated-1"},
                    "status": {"phase": "Succeeded"},
                },
                {
                    "metadata": {
                        "name": "prompt-lab-runners-unrelated-2",
                        "deletionTimestamp": "2026-08-26T00:00:00Z",
                    },
                    "status": {"phase": "Running"},
                },
            ]
        }
    )
    env = os.environ.copy()
    env.update(
        {
            "CURRENT_RUNNER_POD": "prompt-lab-runners-abcde",
            "RUNNER_JSON": runner_json,
        }
    )
    result = subprocess.run(
        [sys.executable, "-"],
        input=code,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "::warning::" in result.stdout
    assert "prompt-lab-runners-unrelated-1" in result.stdout
    assert "prompt-lab-runners-unrelated-2" in result.stdout


def test_clean_arc_observation_reports_success() -> None:
    workflow = load_workflow()
    code = embedded_arc_python(step(workflow, "campaign", "cleanup"))
    env = os.environ.copy()
    env.update(
        {
            "CURRENT_RUNNER_POD": "prompt-lab-runners-abcde",
            "RUNNER_JSON": json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": "prompt-lab-runners-abcde"},
                            "status": {"phase": "Running"},
                        }
                    ]
                }
            ),
        }
    )
    result = subprocess.run(
        [sys.executable, "-"],
        input=code,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "::warning::" not in result.stdout


def test_only_owned_capacity_restoration_can_fail_cleanup() -> None:
    workflow = load_workflow()
    body = str(step(workflow, "campaign", "cleanup")["run"])

    owned, _, observation = body.partition("cleanup_kubeconfig=")
    assert observation, "cleanup step no longer separates observation from restoration"
    # Owned modeleval restoration remains fatal…
    assert owned.count("cleanup_status=$?") >= 1
    # …but nothing in the advisory ARC observation may touch cleanup_status.
    assert "cleanup_status" not in observation.replace(
        'exit "$cleanup_status"', ""
    )
    assert "arc_status" in observation
    assert "::warning::" in observation


def test_dispatch_is_running_only_after_upload_and_cleanup() -> None:
    workflow = load_workflow()
    dispatch = step(workflow, "campaign", "dispatch")

    assert dispatch["if"] == (
        "steps.package.outputs.status == 'running' && "
        "steps.upload.outcome == 'success' && "
        "steps.lineage-claim.outcome == 'success' && "
        "steps.cleanup.outcome == 'success'"
    )
    assert dispatch["env"] == {
        "MANIFEST": "${{ inputs.manifest }}",
        "PROMPT_LAB_REF": "${{ inputs.prompt_lab_ref }}",
        "KORVID_REF": "${{ inputs.korvid_ref }}",
        "PRIOR_RUN_ID": "${{ github.run_id }}",
        "EXPECTED_STATE_HASH": "${{ steps.package.outputs.state-hash }}",
    }
    assert dispatch["uses"] == "actions/github-script@" + ACTION_PINS["actions/github-script"]
    assert dispatch["with"]["github-token"] == "${{ github.token }}"
    body = str(dispatch["with"]["script"])
    assert "github.rest.actions.createWorkflowDispatch" in body
    assert "workflow_id: 'optimization-campaign.yml'" in body
    for field in (
        "manifest",
        "prompt_lab_ref",
        "korvid_ref",
        "prior_run_id",
        "expected_state_hash",
    ):
        assert field in body
    assert "GH_TOKEN" not in body
    assert "github.token" not in body
    required_tools = next(
        item
        for item in steps(workflow, "campaign")
        if item.get("name") == "Verify required tools"
    )
    tools_body = str(required_tools["run"])
    assert "for tool in az kubectl kubelogin uv" in tools_body
    assert "az gh kubectl" not in tools_body
    assert index(workflow, "campaign", "upload") < index(
        workflow, "campaign", "cleanup"
    ) < index(workflow, "campaign", "dispatch")


def test_no_automatic_publish_or_write_capability_is_hidden_in_steps() -> None:
    workflow = load_workflow()
    rendered = "\n".join(all_script_bodies(workflow)).lower()
    uses = [
        str(item.get("uses", ""))
        for job_name in workflow["jobs"]
        for item in steps(workflow, job_name)
    ]

    assert "publish" not in rendered
    assert "pull-requests" not in workflow["permissions"]
    assert "contents: write" not in WORKFLOW_PATH.read_text(encoding="utf-8")
    assert sum("create-github-app-token" in value for value in uses) == 1
    dispatch = step(workflow, "campaign", "dispatch")
    assert "private-key" not in dispatch.get("env", {})


def test_all_third_party_actions_are_exactly_pinned() -> None:
    workflow = load_workflow()
    for job_name in workflow["jobs"]:
        for item in steps(workflow, job_name):
            if "uses" not in item:
                continue
            uses = str(item["uses"])
            name, sha = uses.split("@", 1)
            assert re.fullmatch(r"[0-9a-f]{40}", sha)
            assert ACTION_PINS[name] == sha


def test_readme_distinguishes_canary_qualification_and_terminal_states() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    marker = "## Bounded optimization campaigns"
    assert readme.count(marker) == 1
    section = readme.split(marker, 1)[1].split("\n## ", 1)[0]
    section = " ".join(section.replace("**", "").split())

    for exact_contract in (
        "32761941498",
        "pipeline canary",
        "did not improve the prompt",
        "not qualification evidence",
        "`RUNNING`",
        "`QUALIFIED`",
        "`NOT_CONVERGED`",
        "`SYSTEM_ERROR`",
        "12 calls × 3 seeds",
        "24 calls × 2 seeds",
        "48 calls × 1 seed",
        "240 metric calls",
        "21,600 seconds (6 hours)",
        "1 infrastructure retry",
        "3 consecutive non-promoting attempts",
        "1 independent confirmation",
        "model tiers are independent",
        "explicit publication approval",
    ):
        assert exact_contract in section


def test_attempt_supplies_every_environment_the_wrapper_requires() -> None:
    """The wrapper's required-env preamble must be fully satisfied by `attempt`.

    Wave 2 finding 1: the evaluation campaign resolves `env:KORVID_AKS_MODEL`,
    so the strict control loader used by `korvid-campaign plan` needs it before
    the wrapper can export the planned tier model itself.
    """
    workflow = load_workflow()
    attempt = step(workflow, "campaign", "attempt")
    script = (ROOT / "scripts" / "run-optimization-campaign-step.sh").read_text(
        encoding="utf-8"
    )
    required = set(re.findall(r'^: "\$\{([A-Z0-9_]+):\?', script, re.MULTILINE))

    assert "KORVID_AKS_MODEL" in required
    job_env = set(workflow["jobs"]["campaign"].get("env") or {})
    supplied = set(attempt["env"]) | job_env
    assert required <= supplied, sorted(required - supplied)


def test_attempt_binds_the_validated_identity_model() -> None:
    workflow = load_workflow()
    attempt = step(workflow, "campaign", "attempt")
    assert attempt["env"]["KORVID_AKS_MODEL"] == (
        "${{ needs.identity.outputs.model }}"
    )
    outputs = workflow["jobs"]["identity"]["outputs"]
    assert outputs["model"] == "${{ steps.manifest.outputs.model }}"
    identity = step(workflow, "identity", "manifest")
    body = str(identity["run"])
    assert 'stream.write(f"model={first_tier[\'model\']}\\n")' in body


def test_identity_validation_does_not_require_protected_environment_variables() -> None:
    workflow = load_workflow()
    identity_job = job(workflow, "identity")
    identity = step(workflow, "identity", "manifest")
    environment = identity.get("env") or {}
    body = str(identity["run"])

    assert "environment" not in identity_job
    assert "KORVID_AKS_NAMESPACE" not in environment
    assert "KORVID_AKS_SERVICE" not in environment
    namespace_assignment = 'os.environ["KORVID_AKS_NAMESPACE"] = "identity-validation"'
    service_assignment = 'os.environ["KORVID_AKS_SERVICE"] = "identity-validation"'
    assert namespace_assignment in body
    assert service_assignment in body
    assert body.index(namespace_assignment) < body.index("load_campaign(evaluation_path)")
    assert body.index(service_assignment) < body.index("load_campaign(evaluation_path)")
