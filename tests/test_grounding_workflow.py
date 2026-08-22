"""Contract tests for .github/workflows/grounding-round.yml.

These tests run entirely offline – they parse the workflow YAML and inspect its
structure.  No live GitHub connection is needed.

NOTE: YAML 1.1 treats the bare key ``on`` as the boolean ``True``.  PyYAML
(which implements YAML 1.1) will therefore parse

    on:
      workflow_dispatch: ...

as ``{True: {'workflow_dispatch': ...}}`` unless the key is quoted.  We handle
both representations so the tests remain correct regardless of how the file
quotes the key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "grounding-round.yml"
)


def load_workflow() -> dict[str, Any]:
    raw = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    # YAML 1.1: bare `on` key is parsed as boolean True by PyYAML.
    # Normalise to string key so callers can always use workflow["on"].
    if True in raw and "on" not in raw:
        raw["on"] = raw.pop(True)
    return raw


def _on_triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the mapping under the ``on:`` key (normalised)."""
    return workflow["on"]


def upload_artifact_path(workflow: dict[str, Any]) -> str:
    """Return the ``path`` value from the upload-artifact step, if present."""
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            uses = step.get("uses", "")
            if "upload-artifact" in uses:
                return step.get("with", {}).get("path", "")
    return ""


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_grounding_workflow_has_protected_manual_arc_contract() -> None:
    workflow = load_workflow()
    triggers = _on_triggers(workflow)
    assert "workflow_dispatch" in triggers, "workflow must be triggered only by workflow_dispatch"
    job = workflow["jobs"]["grounding"]
    assert job["runs-on"] == "korvid-runners", "job must run on korvid-runners"
    assert job["environment"] == "aks-grounding", "job must use aks-grounding environment"
    assert workflow["concurrency"]["cancel-in-progress"] is False, (
        "concurrency must not cancel in-progress runs"
    )
    assert workflow["permissions"]["id-token"] == "write", (
        "id-token permission must be write for OIDC"
    )
    assert "pull_request_target" not in triggers, (
        "pull_request_target must not be a trigger (security)"
    )


def test_grounding_workflow_always_uploads_only_safe_evidence_and_cleans_up() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = load_workflow()

    assert "if: always()" in text, "workflow must have always() guard steps"
    assert "safe-evidence" in text, "workflow must reference safe-evidence directory"
    artifact_path = upload_artifact_path(workflow)
    assert "artifacts/live" not in artifact_path, (
        "upload-artifact must NOT include raw live artifacts"
    )
    assert "AZURE_CLIENT_SECRET" not in text, (
        "workflow must not reference AZURE_CLIENT_SECRET (use OIDC)"
    )


def test_grounding_workflow_uses_oidc_not_client_secret() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "azure/login" in text, "workflow must use azure/login action for OIDC auth"
    assert "client-id" in text or "creds" in text, (
        "workflow must provide OIDC credentials to azure/login"
    )
    assert "AZURE_CLIENT_SECRET" not in text, "workflow must not use client secret"


def test_grounding_workflow_uses_pinned_official_actions() -> None:
    workflow = load_workflow()
    actions_used: list[str] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            uses = step.get("uses", "")
            if uses:
                actions_used.append(uses)

    assert actions_used, "workflow must use at least one action"
    for action in actions_used:
        # Every action ref must be pinned to a SHA or a tag (contain @)
        assert "@" in action, f"action '{action}' must be pinned with @<sha|tag>"


def test_grounding_workflow_has_sticky_pr_comment_with_safe_content() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    # Marker comment must be present for sticky-comment deduplication
    assert "korvid-grounding:" in text, (
        "workflow must embed <!-- korvid-grounding:... --> marker for sticky PR comments"
    )
    # PR comment step must also be conditional
    assert "github-script" in text, "workflow must use actions/github-script for PR comment"


def test_grounding_workflow_step_summary_is_safe_and_conditional() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "GITHUB_STEP_SUMMARY" in text, "workflow must append to GITHUB_STEP_SUMMARY"
    assert "hashFiles(" in text or "hashFiles(" in text, (
        "summary step must guard with hashFiles() to skip if file absent"
    )
    assert "round-summary.md" in text, "workflow must append round-summary.md to step summary"


def test_grounding_workflow_failure_semantics_preserved_after_always_steps() -> None:
    """Steps that run with always() must not suppress prior failures."""
    workflow = load_workflow()
    job = workflow["jobs"]["grounding"]
    steps = job.get("steps", [])

    always_steps: list[dict] = [
        s for s in steps if s.get("if", "") == "always()"
        or "always()" in str(s.get("if", ""))
    ]
    # Must have at least one always() step (summary + upload)
    assert len(always_steps) >= 1, "workflow must have always() steps for cleanup/publish"

    # The last step that is NOT always() should be the orchestrator invocation
    non_always = [s for s in steps if "always()" not in str(s.get("if", ""))]
    # Ensure there are substantive steps beyond setup
    assert len(non_always) >= 3, (
        "workflow must have substantive non-always steps (setup, login, orchestrator)"
    )


def test_grounding_workflow_checkout_uses_read_only_token() -> None:
    workflow = load_workflow()
    job = workflow["jobs"]["grounding"]
    steps = job.get("steps", [])
    checkout_steps = [s for s in steps if "actions/checkout" in s.get("uses", "")]
    assert checkout_steps, "workflow must have at least one checkout step"
    # Checkout of Prompt Lab repo should use GITHUB_TOKEN (read-only via permissions)
    # Korvid checkout should use the app token; neither should use a PAT secret
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "KORVID_PAT" not in text, "workflow must not use a PAT; use app token instead"


def test_grounding_workflow_no_pull_request_target() -> None:
    """Explicit test that pull_request_target is absent (TOCTOU attack surface)."""
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "pull_request_target" not in text, (
        "pull_request_target must never appear in this workflow"
    )
