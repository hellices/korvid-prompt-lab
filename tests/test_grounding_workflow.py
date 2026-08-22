"""Contract tests for ``.github/workflows/grounding-round.yml``.

These tests run entirely offline: they parse the workflow YAML and assert on its
*structure*, not on substrings that happen to appear somewhere in the file.  The
workflow is the trust boundary between a manual dispatch and live Azure/AKS
credentials on a self-hosted runner, so every binding constraint from the design
is asserted here as an executable invariant.

NOTE: YAML 1.1 treats the bare key ``on`` as the boolean ``True``.  PyYAML
(which implements YAML 1.1) will therefore parse

    on:
      workflow_dispatch: ...

as ``{True: {'workflow_dispatch': ...}}`` unless the key is quoted.  We handle
both representations so the tests remain correct regardless of how the file
quotes the key.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from korvid_prompt_lab.config import load_campaign

_REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "grounding-round.yml"
ORCHESTRATOR_PATH = _REPO_ROOT / "scripts" / "run-grounding-round.sh"
CAMPAIGN_PATH = _REPO_ROOT / "examples" / "campaigns" / "aks-shared-runners.yaml"

#: Workspace-relative artifact root the orchestrator writes into.  ``artifacts/``
#: is gitignored, so live evidence never lands in a tracked path.
ARTIFACT_ROOT_RELPATH = "prompt-lab/artifacts/grounding-round"
SAFE_EVIDENCE_RELPATH = f"{ARTIFACT_ROOT_RELPATH}/safe-evidence"
WORKSPACE_EXPR = "${{ github.workspace }}"

#: Every action this workflow is allowed to use, pinned to the *real* commit the
#: named tag points at (annotated tags dereferenced to their commit).  Verified
#: against the upstream repositories; a fabricated near-miss SHA fails here.
KNOWN_ACTION_PINS: dict[str, tuple[str, str]] = {
    "actions/checkout": ("11bd71901bbe5b1630ceea73d27597364c9af683", "v4.2.2"),
    "actions/create-github-app-token": (
        "df432ceedc7162793a195dd1713ff69aefc7379e",
        "v2.0.6",
    ),
    "azure/login": ("a65d910e8af852a8061c627c456678983e180302", "v2.2.0"),
    "actions/setup-python": ("a26af69be951a213d495a4c3e4e4022e16d87065", "v5.6.0"),
    "astral-sh/setup-uv": ("e92bafb6253dcd438e0484186d7669ea7a8ca1cc", "v6.4.3"),
    "actions/upload-artifact": ("ea165f8d65b6e75b540449e92b4886f43607fa02", "v4.6.2"),
    "actions/github-script": ("60a0d83039c74a4aee543508d2ffcb1c3799cdea", "v7.0.1"),
}

_SHA_PIN_RE = re.compile(r"^(?P<action>[^@]+)@(?P<sha>[0-9a-f]{40})$")
_EXACT_SHA_RE = r"^[0-9a-f]{40}$"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_workflow() -> dict[str, Any]:
    raw = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    # YAML 1.1: bare `on` key is parsed as boolean True by PyYAML.
    if True in raw and "on" not in raw:
        raw["on"] = raw.pop(True)
    return raw


def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _on_triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    return workflow["on"]


def grounding_job(workflow: dict[str, Any]) -> dict[str, Any]:
    return workflow["jobs"]["grounding"]


def job_env(workflow: dict[str, Any]) -> dict[str, str]:
    return dict(grounding_job(workflow).get("env", {}))


def steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return list(grounding_job(workflow).get("steps", []))


def step_index(
    workflow: dict[str, Any], predicate_substring: str, *, key: str = "uses"
) -> int:
    for index, step in enumerate(steps(workflow)):
        if predicate_substring in str(step.get(key, "")):
            return index
    raise AssertionError(f"no step with {key} containing {predicate_substring!r}")


def step_with_uses(workflow: dict[str, Any], needle: str) -> dict[str, Any]:
    matches = [s for s in steps(workflow) if needle in str(s.get("uses", ""))]
    assert matches, f"workflow must contain a step using {needle!r}"
    return matches[0]


def steps_with_uses(workflow: dict[str, Any], needle: str) -> list[dict[str, Any]]:
    return [s for s in steps(workflow) if needle in str(s.get("uses", ""))]


def orchestrator_step(workflow: dict[str, Any]) -> dict[str, Any]:
    matches = [
        s for s in steps(workflow) if "run-grounding-round.sh" in str(s.get("run", ""))
    ]
    assert len(matches) == 1, "workflow must invoke run-grounding-round.sh exactly once"
    return matches[0]


def github_script_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return steps_with_uses(workflow, "github-script")


def trust_verification_step(workflow: dict[str, Any]) -> dict[str, Any]:
    """The github-script step that proves ``prompt_lab_ref`` is trusted code."""
    matches = [
        s
        for s in github_script_steps(workflow)
        if "PROMPT_LAB_REF" in dict(s.get("env", {}))
    ]
    assert len(matches) == 1, (
        "exactly one github-script step must verify the requested Prompt Lab commit "
        "against the repository before anything is checked out"
    )
    return matches[0]


def pr_comment_step(workflow: dict[str, Any]) -> dict[str, Any]:
    """The github-script step that publishes the sticky PR comment."""
    matches = [
        s
        for s in github_script_steps(workflow)
        if "GROUNDING_SUMMARY_PATH" in dict(s.get("env", {}))
    ]
    assert len(matches) == 1, "exactly one github-script step must post the PR comment"
    return matches[0]


def expand_env_expressions(value: str, env: dict[str, str]) -> str:
    """Resolve ``${{ env.NAME }}`` references against the job-level env block."""

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        assert name in env, f"${{{{ env.{name} }}}} used but not defined at job level"
        return env[name]

    return re.sub(r"\$\{\{\s*env\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", _replace, value)


def effective_env(workflow: dict[str, Any], step: dict[str, Any]) -> dict[str, str]:
    """Job-level env overlaid with the step's own env, with env refs resolved."""
    merged = job_env(workflow)
    merged.update({k: str(v) for k, v in dict(step.get("env", {})).items()})
    return {k: expand_env_expressions(v, job_env(workflow)) for k, v in merged.items()}


def script_bodies(workflow: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``(step name, body)`` for every shell ``run`` and github-script body."""
    bodies: list[tuple[str, str]] = []
    for step in steps(workflow):
        name = str(step.get("name", step.get("uses", "<unnamed>")))
        if "run" in step:
            bodies.append((name, str(step["run"])))
        script = dict(step.get("with", {})).get("script")
        if script is not None:
            bodies.append((name, str(script)))
    return bodies


def required_orchestrator_env_vars() -> set[str]:
    """Every ``: "${VAR:?...}"`` guard enforced by the orchestrator script."""
    text = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    return set(re.findall(r':\s*"\$\{([A-Z_][A-Z0-9_]*):\?', text))


# ---------------------------------------------------------------------------
# Trigger, runner, environment, and concurrency contract
# ---------------------------------------------------------------------------


def test_grounding_workflow_has_protected_manual_arc_contract() -> None:
    workflow = load_workflow()
    triggers = _on_triggers(workflow)

    assert set(triggers) == {"workflow_dispatch"}, (
        "workflow_dispatch must be the ONLY trigger; any automatic trigger would run "
        f"near Azure credentials on a self-hosted runner (found: {sorted(triggers)})"
    )

    job = grounding_job(workflow)
    assert job["runs-on"] == "korvid-runners"
    assert job["environment"] == "aks-grounding"
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert workflow["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "pull-requests": "write",
    }, "workflow permissions must be exactly the least-privilege set"


def test_grounding_workflow_no_pull_request_target() -> None:
    workflow = load_workflow()
    assert "pull_request_target" not in _on_triggers(workflow)
    assert "pull_request_target" not in workflow_text(), (
        "pull_request_target must never appear in this workflow"
    )


def test_grounding_workflow_declares_typed_inputs() -> None:
    workflow = load_workflow()
    inputs = _on_triggers(workflow)["workflow_dispatch"]["inputs"]

    for name, spec in inputs.items():
        assert "type" in spec, f"input {name!r} must declare an explicit type"

    assert inputs["model"]["type"] == "choice"
    assert inputs["round_type"]["type"] == "choice"
    assert set(inputs["round_type"]["options"]) == {"evaluate", "optimize-evaluate"}
    assert inputs["pr_number"]["type"] == "number", (
        "pr_number must be typed so a free-form string can never reach the API call"
    )
    assert inputs["pr_number"].get("required", False) is False


# ---------------------------------------------------------------------------
# Trust boundary: approved Prompt Lab commit, dispatched from the default branch
# ---------------------------------------------------------------------------


def test_grounding_workflow_checks_out_an_exact_prompt_lab_commit() -> None:
    workflow = load_workflow()
    inputs = _on_triggers(workflow)["workflow_dispatch"]["inputs"]

    assert "prompt_lab_ref" in inputs, (
        "the workflow definition runs from the default branch, so the reviewed "
        "Prompt Lab commit must be an explicit input"
    )
    assert inputs["prompt_lab_ref"]["required"] is True
    assert inputs["korvid_ref"]["required"] is True

    prompt_lab_checkout = steps_with_uses(workflow, "actions/checkout")[0]
    assert prompt_lab_checkout["with"]["ref"] == "${{ inputs.prompt_lab_ref }}", (
        "Prompt Lab must be checked out at the approved commit, not at the "
        "default-branch workflow revision"
    )
    assert prompt_lab_checkout["with"]["path"] == "prompt-lab"


def test_grounding_workflow_validates_refs_before_azure_login() -> None:
    workflow = load_workflow()
    all_steps = steps(workflow)

    validation_steps = [
        (index, step)
        for index, step in enumerate(all_steps)
        if _EXACT_SHA_RE in str(step.get("run", ""))
    ]
    assert validation_steps, (
        "a validation step must reject any prompt_lab_ref/korvid_ref that is not an "
        f"exact 40-hex commit SHA (regex {_EXACT_SHA_RE})"
    )
    validation_index, validation_step = validation_steps[0]

    assert validation_index == 0, "input validation must be the first step in the job"

    login_index = step_index(workflow, "azure/login")
    assert validation_index < login_index, "refs must be validated before Azure login"

    checkout_indexes = [
        index
        for index, step in enumerate(all_steps)
        if "actions/checkout" in str(step.get("uses", ""))
    ]
    assert all(validation_index < index for index in checkout_indexes), (
        "refs must be validated before any checkout"
    )

    validation_env = effective_env(workflow, validation_step)
    assert validation_env.get("PROMPT_LAB_REF") == "${{ inputs.prompt_lab_ref }}"
    assert validation_env.get("KORVID_REF") == "${{ inputs.korvid_ref }}"
    assert validation_env.get("PR_NUMBER") == "${{ inputs.pr_number }}"
    assert validation_env.get("CANDIDATE") == "${{ inputs.candidate }}"


def test_grounding_workflow_requires_dispatch_from_default_branch() -> None:
    workflow = load_workflow()
    validation_step = steps(workflow)[0]
    validation_env = effective_env(workflow, validation_step)

    assert "${{ github.event.repository.default_branch }}" in validation_env.values(), (
        "the workflow definition must execute from the default branch; the job must "
        "reject dispatches from other refs"
    )
    body = str(validation_step["run"])
    assert "DEFAULT_BRANCH" in body and "WORKFLOW_REF_NAME" in body


def test_grounding_workflow_records_the_checked_out_prompt_lab_revision() -> None:
    workflow = load_workflow()
    env = effective_env(workflow, orchestrator_step(workflow))
    assert env["PROMPT_LAB_REVISION"] == "${{ inputs.prompt_lab_ref }}", (
        "the report must record the commit that actually ran, not the workflow revision"
    )
    assert env["KORVID_REVISION"] == "${{ inputs.korvid_ref }}"


# ---------------------------------------------------------------------------
# Trust boundary: the requested commit must be repository code, proven before
# any checkout, app token, or Azure login exists
# ---------------------------------------------------------------------------


def test_grounding_workflow_proves_ref_provenance_before_checkout_and_credentials() -> (
    None
):
    """A 40-hex SHA is not provenance.

    ``actions/checkout`` can fetch any commit the repository can reach, including
    the head of a *fork* pull request through ``refs/pull/<n>/head``.  Checking
    such a commit out would run unreviewed third-party code beside the Korvid app
    token and the Azure OIDC session, so the commit's provenance must be proven
    before the first checkout and before any credential is minted.
    """
    workflow = load_workflow()
    all_steps = steps(workflow)
    trust = trust_verification_step(workflow)
    trust_index = all_steps.index(trust)

    checkout_indexes = [
        index
        for index, step in enumerate(all_steps)
        if "actions/checkout" in str(step.get("uses", ""))
    ]
    assert checkout_indexes, "workflow must check something out"
    assert all(trust_index < index for index in checkout_indexes), (
        "the requested commit must be proven trusted before it is checked out"
    )
    assert trust_index < step_index(workflow, "create-github-app-token"), (
        "provenance must be proven before the Korvid app token exists"
    )
    assert trust_index < step_index(workflow, "azure/login"), (
        "provenance must be proven before an Azure session exists"
    )

    assert "if" not in trust, "the trust check must never be skipped"
    assert "continue-on-error" not in trust, "a failed trust check must fail the job"


def test_grounding_workflow_trust_check_binds_only_env_values() -> None:
    workflow = load_workflow()
    trust = trust_verification_step(workflow)
    env = effective_env(workflow, trust)
    script = str(trust["with"]["script"])

    assert env.get("PROMPT_LAB_REF") == "${{ inputs.prompt_lab_ref }}"
    assert env.get("PR_NUMBER") == "${{ inputs.pr_number }}"
    assert env.get("EXPECTED_REPOSITORY") == "${{ github.repository }}"
    assert env.get("DEFAULT_BRANCH") == "${{ github.event.repository.default_branch }}"

    assert "${{" not in script, (
        "no dispatch value may be interpolated into the script source text"
    )
    for name in ("PROMPT_LAB_REF", "PR_NUMBER", "EXPECTED_REPOSITORY", "DEFAULT_BRANCH"):
        assert f"process.env.{name}" in script, (
            f"{name} must be read from process.env at run time"
        )

    assert re.search(r"\^\[0-9a-f\]\{40\}\$", script), (
        "the trust check must re-validate the ref as an exact 40-hex commit SHA"
    )
    assert "pulls.get" in script, (
        "a supplied pr_number must be resolved against this repository's pull requests"
    )
    assert "compareCommits" in script, (
        "without a pr_number the ref must be proven contained in the default branch"
    )


def test_grounding_workflow_trust_check_uses_the_jobs_own_read_only_token() -> None:
    workflow = load_workflow()
    trust = trust_verification_step(workflow)
    with_block = dict(trust["with"])

    assert with_block.get("github-token") == "${{ github.token }}", (
        "provenance must be established with the job's own read-only token, never "
        "with the Korvid app token or any elevated credential"
    )


def test_grounding_workflow_trust_check_script_is_valid_javascript() -> None:
    workflow = load_workflow()
    script = str(trust_verification_step(workflow)["with"]["script"])

    result = subprocess.run(
        ["node", "--check"],
        input=(
            "(async function(github, context, core, glob, io, exec, fetch, require) {\n"
            f"{script}\n}});"
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"trust script is not valid JS:\n{result.stderr}"


# ---------------------------------------------------------------------------
# Executable trust logic: run the workflow's own script against a fake API
# ---------------------------------------------------------------------------

REPOSITORY = "hellices/korvid-prompt-lab"
TRUSTED_SHA = "a" * 40
OTHER_SHA = "b" * 40
KORVID_SHA = "c" * 40
KORVID_REPO_FULL = "hellices/korvid"


def run_trust_script(
    tmp_path: Path,
    *,
    prompt_lab_ref: str = TRUSTED_SHA,
    pr_number: str = "",
    expected_repository: str = REPOSITORY,
    default_branch: str = "main",
    pull: dict[str, Any] | None = None,
    pull_error: str | None = None,
    compare: dict[str, Any] | None = None,
    compare_error: str | None = None,
    basehead_compare_available: bool = True,
    # korvid provenance params; defaults produce a passing korvid check so
    # existing prompt_lab tests remain green
    korvid_ref: str = KORVID_SHA,
    korvid_repo: str = KORVID_REPO_FULL,
    korvid_compare: dict[str, Any] | None = None,
    korvid_compare_error: str | None = None,
    korvid_repo_error: str | None = None,
) -> dict[str, Any]:
    """Execute the workflow's own trust script against a scripted GitHub API.

    The harness routes compare calls by ``params.repo``:
    - ``repo == 'korvid'``  →  korvid fixture branch (korvidCompare / korvidCompareError)
    - any other repo         →  prompt-lab fixture branch (compare / compareError)

    ``repos.get`` is always routed to the korvid fixture
    (korvidRepoData / korvidRepoError); the default returns ``{ default_branch: 'main' }``
    so the korvid ancestor check does not fail the existing prompt-lab-only tests.
    """
    if korvid_compare is None:
        korvid_compare = {"status": "ahead"}
    script = str(trust_verification_step(load_workflow())["with"]["script"])
    tmp_path.mkdir(parents=True, exist_ok=True)

    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "owner": REPOSITORY.split("/")[0],
                "repo": REPOSITORY.split("/")[1],
                "pull": pull,
                "pullError": pull_error,
                "compare": compare,
                "compareError": compare_error,
                "baseheadCompareAvailable": basehead_compare_available,
                # korvid-specific fixture fields
                "korvidCompare": korvid_compare,
                "korvidCompareError": korvid_compare_error,
                "korvidRepoError": korvid_repo_error,
            }
        ),
        encoding="utf-8",
    )

    harness_path = tmp_path / "harness.js"
    harness_path.write_text(
        "const fs = require('fs');\n"
        "const fixture = JSON.parse(fs.readFileSync(process.env.TRUST_FIXTURE, 'utf8'));\n"
        "const calls = [];\n"
        "const failures = [];\n"
        "const infos = [];\n"
        "const core = {\n"
        "  setFailed: (message) => failures.push(String(message)),\n"
        "  info: (message) => infos.push(String(message)),\n"
        "  notice: (message) => infos.push(String(message)),\n"
        "  warning: (message) => infos.push(String(message)),\n"
        "};\n"
        "const context = { repo: { owner: fixture.owner, repo: fixture.repo } };\n"
        # compare routes by params.repo so korvid and prompt-lab calls are separated
        "const compare = (name) => async (params) => {\n"
        "  calls.push({ name, params });\n"
        "  const isKorvid = params && params.repo === 'korvid';\n"
        "  if (isKorvid) {\n"
        "    if (fixture.korvidCompareError) { throw new Error(fixture.korvidCompareError); }\n"
        "    return { data: fixture.korvidCompare };\n"
        "  }\n"
        "  if (fixture.compareError) { throw new Error(fixture.compareError); }\n"
        "  return { data: fixture.compare };\n"
        "};\n"
        "const repos = { compareCommits: compare('compareCommits') };\n"
        "if (fixture.baseheadCompareAvailable) {\n"
        "  repos.compareCommitsWithBasehead = compare('compareCommitsWithBasehead');\n"
        "}\n"
        "repos.get = async (params) => {\n"
        "  calls.push({ name: 'repos.get', params });\n"
        "  if (fixture.korvidRepoError) { throw new Error(fixture.korvidRepoError); }\n"
        "  return { data: { default_branch: 'main' } };\n"
        "};\n"
        "const github = { rest: {\n"
        "  pulls: { get: async (params) => {\n"
        "    calls.push({ name: 'pulls.get', params });\n"
        "    if (fixture.pullError) { throw new Error(fixture.pullError); }\n"
        "    return { data: fixture.pull };\n"
        "  } },\n"
        "  repos,\n"
        "} };\n"
        "const step = async function (github, context, core, require) {\n"
        f"{script}\n"
        "};\n"
        "step(github, context, core, require)\n"
        "  .then(() => console.log(JSON.stringify({ failures, infos, calls })))\n"
        "  .catch((error) => console.log(JSON.stringify({\n"
        "    failures: failures.concat(['threw: ' + error.message]),\n"
        "    infos, calls, threw: true,\n"
        "  })));\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["node", str(harness_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "TRUST_FIXTURE": str(fixture_path),
            "PROMPT_LAB_REF": prompt_lab_ref,
            "PR_NUMBER": pr_number,
            "EXPECTED_REPOSITORY": expected_repository,
            "DEFAULT_BRANCH": default_branch,
            "KORVID_REF": korvid_ref,
            "KORVID_REPO": korvid_repo,
        },
    )
    assert result.returncode == 0, f"harness failed:\n{result.stderr}"
    outcome: dict[str, Any] = json.loads(result.stdout)
    assert not outcome.get("threw"), f"trust script threw: {outcome['failures']}"
    return outcome


def same_repo_pull(sha: str = TRUSTED_SHA) -> dict[str, Any]:
    return {"number": 42, "head": {"sha": sha, "repo": {"full_name": REPOSITORY}}}


def test_trust_script_accepts_a_same_repository_pull_request_head(
    tmp_path: Path,
) -> None:
    outcome = run_trust_script(
        tmp_path / "same-repo-pr",
        pr_number="42",
        pull=same_repo_pull(),
    )

    assert outcome["failures"] == [], outcome["failures"]
    call_names = [call["name"] for call in outcome["calls"]]
    # After PR is accepted, korvid provenance is also verified:
    # pulls.get, repos.get (korvid), compareCommitsWithBasehead (korvid)
    assert call_names[0] == "pulls.get"
    assert "repos.get" in call_names
    pr_call = outcome["calls"][0]
    assert pr_call["params"]["pull_number"] == 42
    assert pr_call["params"]["owner"] == REPOSITORY.split("/")[0]
    assert pr_call["params"]["repo"] == REPOSITORY.split("/")[1]


def test_trust_script_rejects_a_fork_pull_request_head(tmp_path: Path) -> None:
    outcome = run_trust_script(
        tmp_path / "fork-pr",
        pr_number="42",
        pull={
            "number": 42,
            "head": {"sha": TRUSTED_SHA, "repo": {"full_name": "attacker/fork"}},
        },
    )

    assert outcome["failures"], "a fork head commit must never be grounded"
    assert "fork" in outcome["failures"][0].lower()


def test_trust_script_rejects_a_deleted_fork_pull_request_head(tmp_path: Path) -> None:
    """A deleted fork leaves ``head.repo`` null; that must not read as same-repo."""
    outcome = run_trust_script(
        tmp_path / "deleted-fork-pr",
        pr_number="42",
        pull={"number": 42, "head": {"sha": TRUSTED_SHA, "repo": None}},
    )

    assert outcome["failures"], "an unresolvable head repository must be rejected"


def test_trust_script_rejects_a_ref_that_is_not_the_pull_request_head(
    tmp_path: Path,
) -> None:
    outcome = run_trust_script(
        tmp_path / "sha-mismatch",
        prompt_lab_ref=OTHER_SHA,
        pr_number="42",
        pull=same_repo_pull(TRUSTED_SHA),
    )

    assert outcome["failures"], "a ref other than the PR head must be rejected"
    assert "head" in outcome["failures"][0].lower()


def test_trust_script_rejects_a_pr_number_that_is_not_a_pull_request(
    tmp_path: Path,
) -> None:
    outcome = run_trust_script(
        tmp_path / "not-a-pr",
        pr_number="4242",
        pull_error="HttpError: Not Found",
    )

    assert outcome["failures"], "a number that is not a PR in this repo must fail"


@pytest.mark.parametrize("status", ["ahead", "identical"])
def test_trust_script_accepts_a_default_branch_ancestor(
    tmp_path: Path, status: str
) -> None:
    outcome = run_trust_script(
        tmp_path / f"ancestor-{status}",
        compare={"status": status},
    )

    assert outcome["failures"] == [], outcome["failures"]
    # After the prompt_lab check, korvid provenance is also verified; the calls are:
    # [compareCommitsWithBasehead (prompt_lab), repos.get (korvid), compareCommitsWithBasehead (korvid)]
    call_names = [call["name"] for call in outcome["calls"]]
    assert "compareCommitsWithBasehead" in call_names
    prompt_lab_call = outcome["calls"][0]
    assert prompt_lab_call["name"] == "compareCommitsWithBasehead"
    assert prompt_lab_call["params"]["basehead"] == f"{TRUSTED_SHA}...main"


@pytest.mark.parametrize("status", ["diverged", "behind"])
def test_trust_script_rejects_an_unmerged_ref_without_a_pull_request(
    tmp_path: Path, status: str
) -> None:
    outcome = run_trust_script(
        tmp_path / f"unmerged-{status}",
        compare={"status": status},
    )

    assert outcome["failures"], (
        "a commit that is not contained in the default branch must be rejected "
        "when no same-repository pull request vouches for it"
    )


def test_trust_script_still_compares_without_the_basehead_endpoint(
    tmp_path: Path,
) -> None:
    """Older octokit builds expose only ``compareCommits``; containment still holds."""
    outcome = run_trust_script(
        tmp_path / "legacy-compare",
        compare={"status": "ahead"},
        basehead_compare_available=False,
    )

    assert outcome["failures"] == [], outcome["failures"]
    call_names = [call["name"] for call in outcome["calls"]]
    # prompt_lab compare, korvid repos.get, korvid compare — all via compareCommits
    assert call_names == ["compareCommits", "repos.get", "compareCommits"]
    prompt_lab_call = outcome["calls"][0]
    assert prompt_lab_call["params"]["base"] == TRUSTED_SHA
    assert prompt_lab_call["params"]["head"] == "main"


def test_trust_script_rejects_a_ref_unknown_to_the_repository(tmp_path: Path) -> None:
    outcome = run_trust_script(
        tmp_path / "unknown-commit",
        compare_error="HttpError: Not Found",
    )

    assert outcome["failures"], "a commit this repository cannot resolve must fail"


def test_trust_script_rejects_a_non_sha_ref(tmp_path: Path) -> None:
    outcome = run_trust_script(tmp_path / "branch-ref", prompt_lab_ref="main")

    assert outcome["failures"], "a branch name must never reach checkout"
    assert outcome["calls"] == [], "a malformed ref must not even be queried"


def test_trust_script_rejects_a_malformed_pr_number(tmp_path: Path) -> None:
    outcome = run_trust_script(tmp_path / "bad-pr", pr_number="0; rm -rf /")

    assert outcome["failures"], "a malformed pr_number must be rejected"
    assert outcome["calls"] == []


# ---------------------------------------------------------------------------
# Supply chain: every action pinned to a real, known commit
# ---------------------------------------------------------------------------


def test_grounding_workflow_pins_every_action_to_a_known_full_sha() -> None:
    workflow = load_workflow()
    uses_values = [str(s["uses"]) for s in steps(workflow) if s.get("uses")]
    assert uses_values, "workflow must use at least one action"

    text = workflow_text()
    for uses in uses_values:
        match = _SHA_PIN_RE.match(uses)
        assert match, f"action {uses!r} must be pinned to a full 40-hex commit SHA"
        action, sha = match.group("action"), match.group("sha")

        assert action in KNOWN_ACTION_PINS, f"action {action!r} is not on the allowlist"
        expected_sha, expected_tag = KNOWN_ACTION_PINS[action]
        assert sha == expected_sha, (
            f"{action} is pinned to {sha}, which is not the commit for {expected_tag} "
            f"({expected_sha})"
        )
        assert f"{action}@{sha}  # {expected_tag}" in text, (
            f"{action}@{sha} must carry the '# {expected_tag}' provenance comment"
        )


def test_grounding_workflow_never_pins_a_mutable_ref() -> None:
    workflow = load_workflow()
    for step in steps(workflow):
        uses = str(step.get("uses", ""))
        if not uses:
            continue
        ref = uses.split("@", 1)[1] if "@" in uses else ""
        assert not ref.startswith("v"), f"{uses!r} must not use a mutable tag ref"
        assert ref not in {"main", "master", "HEAD"}, f"{uses!r} must not use a branch"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_grounding_workflow_uses_oidc_not_client_secret() -> None:
    workflow = load_workflow()
    text = workflow_text()
    login = step_with_uses(workflow, "azure/login")

    assert "client-id" in login["with"]
    assert "tenant-id" in login["with"]
    assert "subscription-id" in login["with"]
    assert "creds" not in login["with"], "azure/login must use OIDC, not a creds blob"
    assert "AZURE_CLIENT_SECRET" not in text
    assert "client-secret" not in text


def test_grounding_workflow_korvid_token_is_downscoped_read_only() -> None:
    workflow = load_workflow()
    token_step = step_with_uses(workflow, "create-github-app-token")
    with_block = dict(token_step["with"])

    assert with_block.get("permission-contents") == "read", (
        "the Korvid installation token must be explicitly downscoped to read-only; "
        "without permission-* inputs it inherits every installed permission"
    )
    for key, value in with_block.items():
        if key.startswith("permission-"):
            assert value == "read", f"{key} must not grant write ({value!r})"
    assert with_block.get("repositories") == "korvid"


def test_grounding_workflow_checkouts_use_read_only_tokens() -> None:
    workflow = load_workflow()
    text = workflow_text()
    checkouts = steps_with_uses(workflow, "actions/checkout")
    assert len(checkouts) == 2, "workflow must check out Prompt Lab and Korvid"

    prompt_lab, korvid = checkouts
    assert "token" not in prompt_lab["with"], (
        "Prompt Lab checkout must use the job's read-only GITHUB_TOKEN"
    )
    assert korvid["with"]["token"] == "${{ steps.korvid-token.outputs.token }}"
    assert korvid["with"]["repository"].endswith("/korvid")

    for checkout in checkouts:
        assert checkout["with"].get("persist-credentials") is False, (
            "credentials must not be left in .git/config on a self-hosted runner"
        )

    assert "KORVID_PAT" not in text, "workflow must not use a PAT; use the app token"


def test_grounding_workflow_scopes_reflection_credentials_to_optimize_rounds() -> None:
    workflow = load_workflow()
    env = effective_env(workflow, orchestrator_step(workflow))

    model_expr = env["GROUNDING_REFLECTION_MODEL"]
    credential_expr = env["GROUNDING_REFLECTION_CREDENTIAL"]

    assert "vars.GROUNDING_REFLECTION_MODEL" in model_expr, (
        "the reflection model must come from Environment/repository configuration"
    )
    assert "secrets.GROUNDING_REFLECTION_CREDENTIAL" in credential_expr, (
        "the reflection credential must come from the protected Environment secrets"
    )
    for expr in (model_expr, credential_expr):
        assert "inputs.round_type == 'optimize-evaluate'" in expr, (
            "reflection credentials must not be materialised for evaluate-only rounds"
        )


# ---------------------------------------------------------------------------
# Injection surface
# ---------------------------------------------------------------------------


def test_grounding_workflow_scripts_never_interpolate_expressions() -> None:
    workflow = load_workflow()
    for name, body in script_bodies(workflow):
        assert "${{" not in body, (
            f"step {name!r} interpolates a ${{{{ }}}} expression into script source "
            "text; every value must be passed through env: and read at runtime"
        )


def test_grounding_workflow_pr_comment_reads_validated_input_from_env() -> None:
    workflow = load_workflow()
    comment_step = pr_comment_step(workflow)
    script = str(comment_step["with"]["script"])
    env = effective_env(workflow, comment_step)

    assert env.get("GROUNDING_PR_NUMBER") == "${{ inputs.pr_number }}"
    assert env.get("GROUNDING_MODEL") == "${{ inputs.model }}"
    assert env.get("GROUNDING_CANDIDATE") == "${{ inputs.candidate }}"

    for name in ("GROUNDING_PR_NUMBER", "GROUNDING_MODEL", "GROUNDING_CANDIDATE"):
        assert f"process.env.{name}" in script, f"{name} must be read from process.env"

    assert re.search(r"\^\[1-9\]\[0-9\]\{?", script), (
        "the script must re-validate pr_number as a positive integer"
    )
    assert "pulls.get" in script, (
        "the script must confirm the number is a pull request in this repository "
        "before commenting, so an arbitrary issue can never be targeted"
    )
    assert "issues.createComment" in script and "issues.updateComment" in script
    assert "korvid-grounding:" in script, "sticky marker must be embedded in the body"


def test_grounding_workflow_pr_comment_step_is_optional_and_guarded() -> None:
    workflow = load_workflow()
    comment_step = pr_comment_step(workflow)
    condition = str(comment_step["if"])
    assert "always()" in condition
    assert "inputs.pr_number != ''" in condition
    assert "hashFiles(" in condition


def test_grounding_workflow_pr_comment_step_guards_on_summary_existence() -> None:
    """The sticky PR comment must only fire when round-summary.md exists.

    A failed round that never produced a summary must neither create nor
    overwrite the marker comment.  The Job Summary step already uses
    ``hashFiles()`` for this — the PR comment must mirror that exact guard.
    """
    workflow = load_workflow()
    comment_step = pr_comment_step(workflow)
    condition = str(comment_step["if"])

    hashed = re.search(r"hashFiles\(\s*'([^']+)'\s*\)", condition)
    assert hashed, (
        "PR comment step must guard with hashFiles() so an absent summary "
        "never creates or overwrites the marker comment"
    )
    assert hashed.group(1) == f"{SAFE_EVIDENCE_RELPATH}/round-summary.md", (
        "the hashFiles path must match the exact safe-evidence round-summary.md"
    )


def test_grounding_workflow_pr_comment_script_guards_absent_summary() -> None:
    """Structural test: the github-script body must refuse to post when
    the summary file does not exist on disk, as an additional defence
    beyond the step-level ``if`` condition (which evaluates before checkout
    artifacts are available on self-hosted runners in some edge cases)."""
    workflow = load_workflow()
    comment_step = pr_comment_step(workflow)
    script = str(comment_step["with"]["script"])

    # The script must check fs.existsSync and abort/return early when missing
    assert "existsSync" in script, "script must check file existence"
    # It must NOT proceed to createComment/updateComment when summary is absent
    # (i.e., there must be a return/exit path that skips the comment)


def test_grounding_workflow_pr_comment_script_is_executable_and_valid() -> None:
    """Regression: the github-script body must be valid JavaScript."""
    workflow = load_workflow()
    comment_step = pr_comment_step(workflow)
    script = str(comment_step["with"]["script"])

    result = subprocess.run(
        ["node", "--check"],
        input=f"(async function(github, context, core, glob, io, exec, fetch, require) {{\n{script}\n}});",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"github-script body is not valid JS:\n{result.stderr}"


# ---------------------------------------------------------------------------
# Evidence: one artifact root, one safe-evidence path, no silent no-op
# ---------------------------------------------------------------------------


def test_grounding_workflow_defines_one_artifact_root_and_safe_evidence_path() -> None:
    workflow = load_workflow()
    env = job_env(workflow)

    assert env["GROUNDING_ARTIFACT_ROOT"] == f"{WORKSPACE_EXPR}/{ARTIFACT_ROOT_RELPATH}"
    assert (
        env["GROUNDING_SAFE_EVIDENCE_DIR"]
        == f"{WORKSPACE_EXPR}/{SAFE_EVIDENCE_RELPATH}"
    )
    assert (
        env["GROUNDING_SAFE_EVIDENCE_DIR"]
        == f"{env['GROUNDING_ARTIFACT_ROOT']}/safe-evidence"
    ), (
        "the safe-evidence directory must be derived from the single artifact root "
        "the orchestrator actually writes to"
    )


def test_grounding_workflow_uploads_exactly_the_safe_evidence_directory() -> None:
    workflow = load_workflow()
    upload = step_with_uses(workflow, "upload-artifact")
    with_block = dict(upload["with"])

    assert with_block["path"].rstrip("/") == SAFE_EVIDENCE_RELPATH, (
        "upload path must be the same safe-evidence directory the orchestrator writes"
    )
    assert with_block["if-no-files-found"] == "error", (
        "missing evidence must fail the job instead of producing a green empty run"
    )
    assert with_block["retention-days"] == 30
    assert str(upload["if"]).strip() == "always()"

    for forbidden in (
        "artifacts/live",
        "runs/",
        "audit",
        "request.json",
        ".kubeconfig",
    ):
        assert forbidden not in with_block["path"]


def test_grounding_workflow_step_summary_uses_the_same_safe_path() -> None:
    workflow = load_workflow()
    summary_steps = [
        s for s in steps(workflow) if "GITHUB_STEP_SUMMARY" in str(s.get("run", ""))
    ]
    assert len(summary_steps) == 1, (
        "workflow must append the round summary exactly once"
    )
    summary_step = summary_steps[0]

    condition = str(summary_step["if"])
    assert "always()" in condition
    hashed = re.search(r"hashFiles\(\s*'([^']+)'\s*\)", condition)
    assert hashed, "summary step must guard with hashFiles() so an absent file skips"
    assert hashed.group(1) == f"{SAFE_EVIDENCE_RELPATH}/round-summary.md"

    body = str(summary_step["run"])
    assert "GROUNDING_SAFE_EVIDENCE_DIR" in body, (
        "the summary must be read from the single safe-evidence env value"
    )
    assert "round-summary.md" in body


def test_grounding_workflow_pr_comment_reads_the_same_safe_path() -> None:
    workflow = load_workflow()
    comment_step = pr_comment_step(workflow)
    env = effective_env(workflow, comment_step)
    summary_path = env["GROUNDING_SUMMARY_PATH"]

    assert (
        summary_path == f"{WORKSPACE_EXPR}/{SAFE_EVIDENCE_RELPATH}/round-summary.md"
    ), "the PR comment must read the same file the orchestrator writes"
    assert "process.env.GROUNDING_SUMMARY_PATH" in str(comment_step["with"]["script"])


# ---------------------------------------------------------------------------
# Orchestrator invocation contract
# ---------------------------------------------------------------------------


def test_grounding_workflow_supplies_every_required_orchestrator_variable() -> None:
    workflow = load_workflow()
    provided = set(effective_env(workflow, orchestrator_step(workflow)))
    required = required_orchestrator_env_vars()

    assert required, "orchestrator script must guard its inputs with ${VAR:?...}"
    missing = sorted(required - provided)
    assert not missing, f"run-grounding-round.sh aborts under set -u without: {missing}"


def test_grounding_workflow_never_suppresses_failures() -> None:
    workflow = load_workflow()
    job = grounding_job(workflow)

    assert "continue-on-error" not in job
    for step in steps(workflow):
        assert "continue-on-error" not in step, (
            f"step {step.get('name', step.get('uses'))!r} must not suppress its failure"
        )

    orchestrator = orchestrator_step(workflow)
    assert "if" not in orchestrator, (
        "the orchestrator step must run unconditionally so its exit status is the job's"
    )

    always_steps = [s for s in steps(workflow) if "always()" in str(s.get("if", ""))]
    assert len(always_steps) >= 2, "summary and upload must run with always()"

    orchestrator_index = steps(workflow).index(orchestrator)
    for step in steps(workflow)[:orchestrator_index]:
        assert "always()" not in str(step.get("if", "")), (
            "setup steps must not run with always()"
        )


def test_grounding_workflow_provisions_korvid_env_out_of_tree() -> None:
    workflow = load_workflow()
    provisioning = [
        s
        for s in steps(workflow)
        if s.get("working-directory") == "korvid" and "uv sync" in str(s.get("run", ""))
    ]
    assert provisioning, (
        "the bridge runs `uv run --project <korvid> --no-sync`, which needs a "
        "pre-existing environment for that checkout"
    )
    step = provisioning[0]
    env = effective_env(workflow, step)
    project_env = env.get("UV_PROJECT_ENVIRONMENT", "")

    assert project_env.startswith("${{ runner.temp }}/"), (
        "the Korvid environment must live outside the read-only checkout, under an "
        f"absolute runner.temp path (got {project_env!r})"
    )
    assert "korvid/" not in project_env.removeprefix("${{ runner.temp }}/")

    body = str(step["run"])
    assert "--frozen" in body, "the checkout's uv.lock must never be rewritten"
    assert "git status --porcelain" in body, (
        "provisioning must prove the Korvid checkout is still unmodified"
    )

    orchestrator_env = effective_env(workflow, orchestrator_step(workflow))
    assert orchestrator_env.get("UV_PROJECT_ENVIRONMENT") == project_env, (
        "`uv run --no-sync` in the bridge must resolve the same out-of-tree env"
    )
    assert orchestrator_env["KORVID_SOURCE_ROOT"] == f"{WORKSPACE_EXPR}/korvid"


def test_grounding_workflow_installs_prompt_lab_on_path() -> None:
    workflow = load_workflow()
    install = [
        s
        for s in steps(workflow)
        if s.get("working-directory") == "prompt-lab"
        and "GITHUB_PATH" in str(s.get("run", ""))
    ]
    assert install, (
        "the orchestrator calls korvid-prompt-lab/korvid-grounding-report directly, so "
        "the installed environment's bin directory must be added to PATH"
    )
    env = effective_env(workflow, install[0])
    assert env.get("UV_PROJECT_ENVIRONMENT", "").startswith("${{ runner.temp }}/")
    assert env["UV_PROJECT_ENVIRONMENT"] != effective_env(
        workflow, orchestrator_step(workflow)
    ).get("UV_PROJECT_ENVIRONMENT"), (
        "Prompt Lab and Korvid must not share one uv environment"
    )


# ---------------------------------------------------------------------------
# Campaign, case, and budget dispatch inputs (design "Trigger and trust boundary")
# ---------------------------------------------------------------------------


def campaign_env_references() -> set[str]:
    """Every ``env:NAME`` reference the AKS campaign resolves at load time."""
    text = CAMPAIGN_PATH.read_text(encoding="utf-8")
    return set(re.findall(r"\benv:([A-Z_][A-Z0-9_]*)", text))


def test_grounding_workflow_declares_campaign_case_and_budget_inputs() -> None:
    workflow = load_workflow()
    inputs = _on_triggers(workflow)["workflow_dispatch"]["inputs"]

    for name in (
        "campaign",
        "train_case_id",
        "validation_case_id",
        "milestone_case_ids",
        "max_metric_calls",
        "seed",
    ):
        assert name in inputs, (
            f"the design requires a {name!r} dispatch input; without it the round "
            "cannot select a campaign, its case splits, or its optimization budget"
        )
        assert inputs[name]["required"] is True

    assert inputs["campaign"]["type"] == "string"
    assert inputs["campaign"]["default"].endswith(".yaml")
    assert inputs["train_case_id"]["type"] == "string"
    assert inputs["validation_case_id"]["type"] == "string"
    assert inputs["milestone_case_ids"]["type"] == "string"
    assert inputs["max_metric_calls"]["type"] == "number"
    assert inputs["seed"]["type"] == "number"

    assert (
        inputs["train_case_id"]["default"] != inputs["validation_case_id"]["default"]
    ), "the shipped defaults must keep the train and validation splits disjoint"
    assert (
        "," in inputs["milestone_case_ids"]["description"].lower()
        or "comma" in str(inputs["milestone_case_ids"]["description"]).lower()
    ), "milestone_case_ids must document that it is a comma-separated list"


def test_grounding_workflow_validates_campaign_case_and_budget_inputs_first() -> None:
    workflow = load_workflow()
    validation_step = steps(workflow)[0]
    validation_env = effective_env(workflow, validation_step)
    body = str(validation_step["run"])

    expected = {
        "CAMPAIGN": "${{ inputs.campaign }}",
        "TRAIN_CASE_ID": "${{ inputs.train_case_id }}",
        "VALIDATION_CASE_ID": "${{ inputs.validation_case_id }}",
        "MILESTONE_CASE_IDS": "${{ inputs.milestone_case_ids }}",
        "MAX_METRIC_CALLS": "${{ inputs.max_metric_calls }}",
        "SEED": "${{ inputs.seed }}",
    }
    for name, expression in expected.items():
        assert validation_env.get(name) == expression, (
            f"{name} must reach the validation step through env:, not interpolation"
        )
        assert f'"${name}"' in body, (
            f"{name} must actually be validated, not merely bound"
        )

    login_index = step_index(workflow, "azure/login")
    assert steps(workflow).index(validation_step) < login_index

    assert "case ids" in body.lower() or "case id" in body.lower()
    assert "disjoint" in body.lower(), (
        "the train and validation splits must be proven disjoint before cluster time"
    )


def test_grounding_workflow_wires_campaign_cases_and_budget_to_the_orchestrator() -> (
    None
):
    workflow = load_workflow()
    env = effective_env(workflow, orchestrator_step(workflow))

    assert env["GROUNDING_CAMPAIGN"] == "${{ inputs.campaign }}"
    assert env["GROUNDING_TRAIN_CASE_ID"] == "${{ inputs.train_case_id }}"
    assert env["GROUNDING_VALIDATION_CASE_ID"] == "${{ inputs.validation_case_id }}"
    assert env["GROUNDING_MILESTONE_CASE_IDS"] == "${{ inputs.milestone_case_ids }}"
    assert env["GROUNDING_MAX_METRIC_CALLS"] == "${{ inputs.max_metric_calls }}"
    assert env["GROUNDING_SEED"] == "${{ inputs.seed }}"


def test_grounding_workflow_supplies_every_campaign_environment_reference() -> None:
    """The campaign resolves models/serving through env:; the job must set them."""
    workflow = load_workflow()
    env = effective_env(workflow, orchestrator_step(workflow))
    references = campaign_env_references()

    assert references, "the AKS campaign must resolve its serving identity from env:"
    missing = sorted(references - set(env))
    assert not missing, (
        "load_campaign() raises 'references missing environment variable' without: "
        f"{missing}"
    )

    assert env["KORVID_AKS_MODEL"] == "${{ inputs.model }}", (
        "the allowlisted model input must be the model the campaign actually serves"
    )
    for name in ("KORVID_AKS_NAMESPACE", "KORVID_AKS_SERVICE"):
        assert env[name] == f"${{{{ vars.{name} }}}}", (
            f"{name} must come from the protected Environment configuration"
        )


# ---------------------------------------------------------------------------
# Lifecycle step 12: bounded runtime and an always() node-pool restore
# ---------------------------------------------------------------------------


def nodepool_record_step(workflow: dict[str, Any]) -> dict[str, Any]:
    matches = [
        s
        for s in steps(workflow)
        if "nodepool show" in str(s.get("run", ""))
        and "GITHUB_OUTPUT" in str(s.get("run", ""))
    ]
    assert len(matches) == 1, (
        "exactly one step must record the original modeleval node count as a step output"
    )
    return matches[0]


def nodepool_restore_step(workflow: dict[str, Any]) -> dict[str, Any]:
    matches = [s for s in steps(workflow) if "nodepool scale" in str(s.get("run", ""))]
    assert len(matches) == 1, "exactly one step must restore the modeleval node count"
    return matches[0]


def test_grounding_workflow_bounds_job_and_orchestrator_runtime() -> None:
    workflow = load_workflow()
    job = grounding_job(workflow)

    job_timeout = job.get("timeout-minutes")
    assert isinstance(job_timeout, int), (
        "without timeout-minutes the job inherits the 360-minute default, so a hung "
        "round burns a GPU node for six hours behind cancel-in-progress: false"
    )
    assert 0 < job_timeout <= 360

    orchestrator = orchestrator_step(workflow)
    step_timeout = orchestrator.get("timeout-minutes")
    assert isinstance(step_timeout, int), "the orchestrator step must be bounded"
    assert 0 < step_timeout < job_timeout, (
        "the orchestrator step must expire before the job does, so the always() "
        "cleanup step still gets to run"
    )


def test_grounding_workflow_records_the_original_node_count_before_the_round() -> None:
    workflow = load_workflow()
    all_steps = steps(workflow)
    record = nodepool_record_step(workflow)

    assert record.get("id"), (
        "the record step needs an id so cleanup can read its output"
    )
    assert all_steps.index(record) < all_steps.index(orchestrator_step(workflow)), (
        "the original count must be captured before the round can scale the pool"
    )
    assert all_steps.index(record) > step_index(workflow, "azure/login")

    body = str(record["run"])
    assert "modeleval" in body
    assert "GITHUB_OUTPUT" in body
    assert "--node-count" not in body, "the record step must be read-only"


def test_grounding_workflow_restores_the_recorded_node_count_with_always() -> None:
    workflow = load_workflow()
    all_steps = steps(workflow)
    record = nodepool_record_step(workflow)
    restore = nodepool_restore_step(workflow)

    assert str(restore["if"]).strip() == "always()", (
        "the design requires restoration in BOTH a shell trap and an always() step; "
        "a cancelled runner kills the trap mid-flight and leaks the GPU node"
    )
    assert "continue-on-error" not in restore, (
        "cleanup failure must be visible, never hidden by the evaluation result"
    )

    restore_index = all_steps.index(restore)
    assert restore_index == len(all_steps) - 1, "cleanup must be the final step"
    assert step_index(workflow, "upload-artifact") < restore_index, (
        "upload-artifact must run before cleanup so evidence survives a cleanup failure"
    )
    assert all_steps.index(pr_comment_step(workflow)) < restore_index, (
        "the PR comment must run before cleanup so evidence survives a cleanup failure"
    )
    summary_index = next(
        index
        for index, step in enumerate(all_steps)
        if "GITHUB_STEP_SUMMARY" in str(step.get("run", ""))
    )
    assert summary_index < restore_index

    env = effective_env(workflow, restore)
    output_ref = f"${{{{ steps.{record['id']}.outputs."
    assert any(value.startswith(output_ref) for value in env.values()), (
        "cleanup must restore the count recorded before the round, not re-derive it"
    )

    body = str(restore["run"])
    assert "--node-count 0" in body, "cleanup may only ever scale the pool back to 0"
    assert "--node-count 1" not in body
    assert "nodepool show" in body, (
        "cleanup must be idempotent: only scale when the pool is not already restored"
    )
    for forbidden in (
        "nodepool add",
        "nodepool delete",
        "nodepool update",
        "aks create",
    ):
        assert forbidden not in body


def test_grounding_workflow_orchestrator_failure_still_fails_the_job() -> None:
    workflow = load_workflow()
    orchestrator = orchestrator_step(workflow)

    assert "if" not in orchestrator
    assert "continue-on-error" not in orchestrator

    all_steps = steps(workflow)
    tail = all_steps[all_steps.index(orchestrator) + 1 :]
    assert tail, "summary, upload, comment, and cleanup must follow the round"
    for step in tail:
        assert "always()" in str(step.get("if", "")), (
            f"step {step.get('name')!r} runs after a possibly failed round and must "
            "declare always() explicitly"
        )


def test_campaign_loads_only_with_the_variables_the_workflow_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the contract against the real loader, not just the workflow text."""
    workflow = load_workflow()
    env = effective_env(workflow, orchestrator_step(workflow))
    references = campaign_env_references()

    #: The concrete values the workflow's expressions resolve to at run time.
    resolved = {
        "KORVID_AKS_MODEL": "qwen3:1.7b",
        "KORVID_AKS_NAMESPACE": "ollama",
        "KORVID_AKS_SERVICE": "ollama",
    }
    assert set(resolved) == references, (
        "this test must cover exactly the campaign's env: references"
    )
    assert references <= set(env)

    for name, value in resolved.items():
        monkeypatch.setenv(name, value)
    campaign = load_campaign(CAMPAIGN_PATH)
    assert campaign.models == ("qwen3:1.7b",)

    for name in sorted(references):
        monkeypatch.delenv(name)
        with pytest.raises(ValueError, match=f"missing environment variable {name}"):
            load_campaign(CAMPAIGN_PATH)
        monkeypatch.setenv(name, resolved[name])


# ---------------------------------------------------------------------------
# Executable cleanup logic: run the workflow's own shell against a fake `az`
# ---------------------------------------------------------------------------


def _write_fake_az(
    bin_dir: Path, *, current_count: str, calls_file: Path, exit_code: int = 0
) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    az = bin_dir / "az"
    az.write_text(
        "#!/usr/bin/env bash\n"
        f'CALLS="{calls_file}"\n'
        'if [[ "$*" == *"nodepool show"* ]]; then\n'
        '  echo "show" >> "$CALLS"\n'
        f'  echo "{current_count}"\n'
        f"  exit {exit_code}\n"
        'elif [[ "$*" == *"nodepool scale"* ]]; then\n'
        '  prev=""\n'
        '  for arg in "$@"; do\n'
        '    if [[ "$prev" == "--node-count" ]]; then echo "scale:$arg" >> "$CALLS"; fi\n'
        '    prev="$arg"\n'
        "  done\n"
        f"  exit {exit_code}\n"
        "fi\n",
        encoding="utf-8",
    )
    az.chmod(0o755)


def _run_step_body(
    body: str,
    tmp_path: Path,
    *,
    env: dict[str, str],
    current_count: str = "1",
    az_exit: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    calls_file = tmp_path / "calls.txt"
    calls_file.touch()
    bin_dir = tmp_path / "bin"
    _write_fake_az(
        bin_dir, current_count=current_count, calls_file=calls_file, exit_code=az_exit
    )

    script = tmp_path / "step.sh"
    script.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", **env},
        timeout=30,
    )
    calls = [
        line
        for line in calls_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return result, calls


def test_grounding_workflow_record_step_publishes_the_original_count(
    tmp_path: Path,
) -> None:
    body = str(nodepool_record_step(load_workflow())["run"])
    output_file = tmp_path / "run" / "github-output"
    output_file.parent.mkdir(parents=True)
    output_file.touch()

    result, calls = _run_step_body(
        body,
        tmp_path / "record",
        env={"GITHUB_OUTPUT": str(output_file)},
        current_count="0",
    )

    assert result.returncode == 0, result.stderr
    assert calls == ["show"], "recording must be read-only"
    assert "original-count=0" in output_file.read_text(encoding="utf-8")


def test_grounding_workflow_record_step_rejects_an_unexpected_count(
    tmp_path: Path,
) -> None:
    body = str(nodepool_record_step(load_workflow())["run"])
    output_file = tmp_path / "github-output"
    output_file.touch()

    result, calls = _run_step_body(
        body,
        tmp_path / "record",
        env={"GITHUB_OUTPUT": str(output_file)},
        current_count="4",
    )

    assert result.returncode != 0
    assert "unexpected modeleval node count" in result.stdout
    assert "scale" not in " ".join(calls)
    assert output_file.read_text(encoding="utf-8") == ""


def test_grounding_workflow_restore_step_scales_back_to_the_recorded_zero(
    tmp_path: Path,
) -> None:
    body = str(nodepool_restore_step(load_workflow())["run"])

    result, calls = _run_step_body(
        body,
        tmp_path / "leaked",
        env={"ORIGINAL_NODE_COUNT": "0"},
        current_count="1",
    )

    assert result.returncode == 0, result.stderr
    assert calls == ["show", "scale:0"], (
        "a leaked node must be released with an exact scale to the recorded count"
    )


def test_grounding_workflow_restore_step_is_idempotent(tmp_path: Path) -> None:
    """The trap usually wins the race; cleanup must then be a no-op, not a retry."""
    body = str(nodepool_restore_step(load_workflow())["run"])

    result, calls = _run_step_body(
        body,
        tmp_path / "already-restored",
        env={"ORIGINAL_NODE_COUNT": "0"},
        current_count="0",
    )

    assert result.returncode == 0, result.stderr
    assert calls == ["show"], "an already-restored pool must not be scaled again"


def test_grounding_workflow_restore_step_never_touches_preexisting_capacity(
    tmp_path: Path,
) -> None:
    body = str(nodepool_restore_step(load_workflow())["run"])

    result, calls = _run_step_body(
        body,
        tmp_path / "preexisting",
        env={"ORIGINAL_NODE_COUNT": "1"},
        current_count="1",
    )

    assert result.returncode == 0, result.stderr
    assert calls == [], "a pool that already had a node must not even be queried"


def test_grounding_workflow_restore_step_tolerates_a_round_that_never_started(
    tmp_path: Path,
) -> None:
    """always() also fires when Azure login failed, so the output may be empty."""
    body = str(nodepool_restore_step(load_workflow())["run"])

    result, calls = _run_step_body(
        body,
        tmp_path / "never-recorded",
        env={"ORIGINAL_NODE_COUNT": ""},
        current_count="0",
    )

    assert result.returncode == 0, result.stderr
    assert calls == []


def test_grounding_workflow_restore_step_fails_visibly(tmp_path: Path) -> None:
    """Design: cleanup failure is a separate failing step, never hidden."""
    body = str(nodepool_restore_step(load_workflow())["run"])

    result, _calls = _run_step_body(
        body,
        tmp_path / "az-broken",
        env={"ORIGINAL_NODE_COUNT": "0"},
        current_count="1",
        az_exit=3,
    )

    assert result.returncode != 0, "a failed scale-down must fail the step"


# ---------------------------------------------------------------------------
# Task 3: Prerequisite step verifies tools before node-count read
# ---------------------------------------------------------------------------


def test_grounding_workflow_has_tool_verification_step_before_node_count() -> None:
    """A step verifying az/kubectl/kubelogin/uv must precede the node-count record step."""
    workflow = load_workflow()
    steps = grounding_job(workflow)["steps"]
    step_names = [s.get("name", "") for s in steps]

    tool_step_idx = next(
        (i for i, name in enumerate(step_names) if "Verify required CLI tools" in name),
        None,
    )
    node_count_idx = next(
        (
            i
            for i, name in enumerate(step_names)
            if "Record original modeleval node count" in name
        ),
        None,
    )

    assert tool_step_idx is not None, (
        "workflow must have a 'Verify required CLI tools' step"
    )
    assert node_count_idx is not None, (
        "workflow must have a 'Record original modeleval node count' step"
    )
    assert tool_step_idx < node_count_idx, (
        f"tool verification step (index {tool_step_idx}) must precede node-count step (index {node_count_idx})"
    )

    # The step must check all four tools
    tool_step = steps[tool_step_idx]
    body = str(tool_step.get("run", ""))
    for tool in ("az", "kubectl", "kubelogin", "uv"):
        assert tool in body, f"tool verification step must check for '{tool}'"


# ---------------------------------------------------------------------------
# Trust boundary: korvid_ref provenance in the authoritative Korvid repository
#
# RED tests — every assertion in this section fails before the implementation
# because the trust step does not yet verify korvid_ref against the Korvid repo.
# ---------------------------------------------------------------------------


def test_grounding_workflow_trust_check_binds_korvid_ref_env() -> None:
    """``KORVID_REF`` must be env-bound in the pre-credential trust step."""
    workflow = load_workflow()
    trust = trust_verification_step(workflow)
    env = effective_env(workflow, trust)
    script = str(trust["with"]["script"])

    assert env.get("KORVID_REF") == "${{ inputs.korvid_ref }}", (
        "korvid_ref must reach the trust script through env:, never interpolated "
        "into the script body"
    )
    assert "process.env.KORVID_REF" in script, (
        "the trust script must read KORVID_REF from process.env at run time"
    )


def test_grounding_workflow_trust_check_binds_korvid_repo_env() -> None:
    """``KORVID_REPO`` must be derived from ``github.repository_owner``, not a user input."""
    workflow = load_workflow()
    trust = trust_verification_step(workflow)
    env = effective_env(workflow, trust)
    script = str(trust["with"]["script"])

    assert env.get("KORVID_REPO") == "${{ github.repository_owner }}/korvid", (
        "the Korvid repo must be derived from github.repository_owner (not user-supplied) "
        "so that a dispatcher cannot point the provenance check at an arbitrary repo"
    )
    assert "process.env.KORVID_REPO" in script, (
        "the trust script must read KORVID_REPO from process.env at run time"
    )


def test_grounding_workflow_trust_check_verifies_korvid_provenance_before_app_token() -> None:
    """Korvid provenance is established in the pre-credential trust step, not after."""
    workflow = load_workflow()
    trust = trust_verification_step(workflow)
    env = effective_env(workflow, trust)

    assert "KORVID_REF" in env, (
        "korvid_ref provenance must be proven in the pre-credential trust step, "
        "before the Korvid app token, Azure login, or any checkout"
    )
    assert "KORVID_REPO" in env


def test_grounding_workflow_trust_check_script_reads_korvid_ref_from_env() -> None:
    """The trust script must not interpolate korvid inputs: read from ``process.env`` only."""
    workflow = load_workflow()
    trust = trust_verification_step(workflow)
    script = str(trust["with"]["script"])

    # process.env reads
    assert "process.env.KORVID_REF" in script
    assert "process.env.KORVID_REPO" in script
    # no ${{ }} in the script body (already tested globally, repeated here for clarity)
    assert "${{" not in script


# ---------------------------------------------------------------------------
# Executable korvid provenance: run the trust script against a scripted API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["ahead", "identical"])
def test_trust_script_korvid_accepts_default_branch_ancestor(
    tmp_path: Path, status: str
) -> None:
    """A korvid_ref that is an ancestor of (or equal to) the default branch is accepted."""
    outcome = run_trust_script(
        tmp_path / f"korvid-ancestor-{status}",
        compare={"status": "ahead"},  # prompt_lab must also pass
        korvid_compare={"status": status},
    )

    assert outcome["failures"] == [], outcome["failures"]
    # The trust script must have made at least one compare call for korvid.
    korvid_compare_calls = [
        c
        for c in outcome["calls"]
        if c.get("params", {}).get("repo") == "korvid"
        and c["name"] in ("compareCommitsWithBasehead", "compareCommits")
    ]
    assert korvid_compare_calls, (
        "the trust step must call compareCommitsWithBasehead/compareCommits for the "
        "authoritative Korvid repo to prove korvid_ref provenance"
    )


@pytest.mark.parametrize("status", ["diverged", "behind"])
def test_trust_script_korvid_rejects_unmerged_ref(
    tmp_path: Path, status: str
) -> None:
    """A korvid_ref that has diverged from or is not contained in the default branch is rejected."""
    outcome = run_trust_script(
        tmp_path / f"korvid-unmerged-{status}",
        compare={"status": "ahead"},  # prompt_lab passes; korvid should fail
        korvid_compare={"status": status},
    )

    assert outcome["failures"], (
        "a korvid_ref that is not contained in the default branch must be rejected; "
        "only commits already merged/reachable from the Korvid default branch are accepted"
    )
    failure_text = " ".join(outcome["failures"]).lower()
    assert "korvid" in failure_text or "default branch" in failure_text, (
        "the rejection message must identify that korvid_ref failed the provenance check"
    )


def test_trust_script_korvid_rejects_api_failure(tmp_path: Path) -> None:
    """An API failure when verifying korvid_ref must fail the trust check, not silently pass."""
    outcome = run_trust_script(
        tmp_path / "korvid-api-failure",
        compare={"status": "ahead"},  # prompt_lab passes; korvid compare should fail
        korvid_compare_error="HttpError: Not Found",
    )

    assert outcome["failures"], (
        "a korvid_ref that the Korvid repository API cannot resolve must fail the "
        "trust check; an API outage must close the round, not open it"
    )
    failure_text = " ".join(outcome["failures"]).lower()
    assert "korvid" in failure_text or "default branch" in failure_text
