"""Contract tests binding the shipped Korvid pin to its reviewed declaration.

A grounding round is only executable when the Korvid commit it defaults to can
*both* clear the workflow's pre-credential provenance gate and supply the
operation-journey harness the bridge worker imports.  Those two properties were
verified in different places (a workflow default, a README table, a JavaScript
trust script) and drifted apart: the shipped default was rejected by the shipped
gate, so every default dispatch failed before it began.

These tests make the pin a single reviewed fact.  They are deterministic and
offline: the provenance facts come from :mod:`korvid_prompt_lab.korvid_pin`,
which records a dated snapshot of the GitHub API, and the trust gate is executed
against that snapshot rather than against the live network.  Live re-verification
belongs to ``scripts/verify-korvid-pin.sh``, which a maintainer runs on purpose.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

import pytest
from test_grounding_workflow import (
    load_workflow,
    run_trust_script,
    trust_verification_step,
    workflow_inputs,
)

from korvid_prompt_lab import bridge_worker, korvid_pin
from korvid_prompt_lab.korvid_pin import (
    APPROVED_KORVID_PROVENANCE,
    APPROVED_KORVID_SHA,
    KORVID_DEFAULT_BRANCH,
    KORVID_REPOSITORY,
    PROVENANCE_OPEN_PULL_REQUEST,
    REQUIRED_KORVID_IMPORTS,
    REQUIRED_KORVID_SOURCE_PATHS,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = _REPO_ROOT / "README.md"
VERIFY_SCRIPT_PATH = _REPO_ROOT / "scripts" / "verify-korvid-pin.sh"

SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def vouching_pull_request(
    *,
    number: int | None = None,
    head_sha: str | None = None,
    head_repository: str | None = None,
    base_ref: str | None = None,
) -> dict[str, Any]:
    """The recorded Korvid pull request that vouches for the pin, with overrides."""
    provenance = APPROVED_KORVID_PROVENANCE
    return {
        "number": provenance.pull_request if number is None else number,
        "state": "open",
        "base": {"ref": provenance.base_branch if base_ref is None else base_ref},
        "head": {
            "sha": provenance.head_sha if head_sha is None else head_sha,
            "ref": provenance.branch,
            "repo": {
                "full_name": (
                    provenance.head_repository if head_repository is None else head_repository
                )
            },
        },
    }


def run_gate_for_shipped_default(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Replay the recorded provenance snapshot of the shipped default through the gate."""
    provenance = APPROVED_KORVID_PROVENANCE
    kwargs: dict[str, Any] = {
        "compare": {"status": "ahead"},  # prompt_lab_ref passes; korvid is under test
        "korvid_ref": workflow_korvid_default(),
        "korvid_repo": KORVID_REPOSITORY,
        "korvid_default_branch": KORVID_DEFAULT_BRANCH,
        "korvid_compare": {"status": provenance.default_branch_compare_status},
        "korvid_pulls": [vouching_pull_request()],
        "korvid_pull_compare": {"status": provenance.head_compare_status},
    }
    kwargs.update(overrides)
    return run_trust_script(tmp_path, **kwargs)


def workflow_korvid_default() -> str:
    """The ``korvid_ref`` default the workflow actually ships."""
    return str(workflow_inputs(load_workflow())["korvid_ref"]["default"])


# ---------------------------------------------------------------------------
# The pin itself
# ---------------------------------------------------------------------------


def test_approved_pin_is_an_exact_commit_sha() -> None:
    assert SHA_PATTERN.match(APPROVED_KORVID_SHA), (
        "the approved Korvid pin must be an exact 40-hex commit SHA, never a branch or tag"
    )
    assert SHA_PATTERN.match(APPROVED_KORVID_PROVENANCE.head_sha)


def test_pin_records_the_repository_it_was_proven_against() -> None:
    assert KORVID_REPOSITORY == "hellices/korvid"
    assert APPROVED_KORVID_PROVENANCE.head_repository == KORVID_REPOSITORY, (
        "a pull request whose head repository is a fork can never vouch for the pin"
    )
    assert APPROVED_KORVID_PROVENANCE.base_branch == KORVID_DEFAULT_BRANCH


def test_pin_records_why_the_default_branch_route_is_insufficient() -> None:
    """The declaration must keep the failing default-branch status, not hide it.

    The recorded ``diverged`` status is the whole reason the gate needs a second
    acceptance route.  If someone repins to a default-branch commit they must
    update this fact deliberately, which is exactly the review moment that was
    missing when the gate and the default drifted apart.
    """
    provenance = APPROVED_KORVID_PROVENANCE
    if provenance.kind == PROVENANCE_OPEN_PULL_REQUEST:
        assert provenance.default_branch_compare_status not in ("identical", "ahead"), (
            "a pin routed through a pull request must not also be reachable from the "
            "default branch — if it is, pin the default-branch route instead"
        )
        assert provenance.head_compare_status in ("identical", "ahead"), (
            "the vouching pull request head must contain the pinned commit"
        )


def test_required_source_paths_cover_every_required_korvid_module() -> None:
    """Every first-party Korvid module in the import contract has a checkable path."""
    korvid_modules = {
        module for module in REQUIRED_KORVID_IMPORTS if module.split(".")[0] in ("korvid", "tests")
    }
    covered = {
        path.removeprefix("src/").removesuffix(".py").replace("/", ".")
        for path in REQUIRED_KORVID_SOURCE_PATHS
    }
    missing = {module for module in korvid_modules if not _module_is_covered(module, covered)}
    assert not missing, (
        f"REQUIRED_KORVID_SOURCE_PATHS does not cover {sorted(missing)}; the verify "
        "script would silently skip those modules"
    )


def _module_is_covered(module: str, covered: set[str]) -> bool:
    # ``from tests.evals import operation_app`` names the package, and the file that
    # must exist is the submodule it binds.
    if module in covered:
        return True
    return any(candidate.startswith(module + ".") for candidate in covered)


# ---------------------------------------------------------------------------
# Compatibility: the pin must describe the bridge that actually runs
# ---------------------------------------------------------------------------


def bridge_worker_korvid_imports() -> dict[str, tuple[str, ...]]:
    """Derive the Korvid import contract from the bridge worker's own source."""
    source = Path(bridge_worker.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_import_korvid"
    ]
    assert len(functions) == 1, "bridge_worker must resolve Korvid in exactly one place"

    imports: dict[str, list[str]] = {}
    for node in ast.walk(functions[0]):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.setdefault(alias.name, [])
        elif isinstance(node, ast.ImportFrom):
            names = imports.setdefault(node.module or "", [])
            names.extend(alias.name for alias in node.names)
    return {module: tuple(sorted(names)) for module, names in imports.items()}


def test_pin_import_contract_matches_the_bridge_worker() -> None:
    """The declared compatibility contract is derived from, not parallel to, the code."""
    declared = {module: tuple(sorted(names)) for module, names in REQUIRED_KORVID_IMPORTS.items()}
    assert bridge_worker_korvid_imports() == declared, (
        "REQUIRED_KORVID_IMPORTS drifted from bridge_worker._import_korvid; a Korvid "
        "pin proven against a stale contract can still fail to import at run time"
    )


def test_pin_records_the_patchable_attribute_the_bridge_rebinds() -> None:
    source = Path(bridge_worker.__file__).read_text(encoding="utf-8")
    assert f'"{korvid_pin.REQUIRED_PATCHABLE_ATTRIBUTE}"' in source, (
        "the bridge injects prompt overrides by rebinding this attribute on the "
        "operation harness; the pin must name the same attribute"
    )


# ---------------------------------------------------------------------------
# The shipped workflow default must be the reviewed pin
# ---------------------------------------------------------------------------


def test_workflow_korvid_ref_default_is_the_approved_pin() -> None:
    assert workflow_korvid_default() == APPROVED_KORVID_SHA, (
        "the workflow's korvid_ref default drifted from the reviewed pin in "
        "korvid_prompt_lab.korvid_pin"
    )


def test_workflow_korvid_ref_default_is_an_exact_sha() -> None:
    assert SHA_PATTERN.match(workflow_korvid_default())


# ---------------------------------------------------------------------------
# The shipped default must clear the shipped gate  (the round-two regression)
# ---------------------------------------------------------------------------


def test_shipped_korvid_default_passes_the_shipped_trust_gate(tmp_path: Path) -> None:
    """The default dispatch must survive its own pre-credential provenance gate.

    Replays the recorded facts: the pin is ``diverged`` from the Korvid default
    branch, and it is contained in the head of open, same-repository pull request
    #312 targeting that default branch.  A gate that only accepts default-branch
    containment rejects the shipped default, which is the round-two finding.
    """
    outcome = run_gate_for_shipped_default(tmp_path / "shipped-default")

    assert outcome["failures"] == [], (
        "the shipped korvid_ref default must clear the shipped provenance gate; "
        f"it was rejected with {outcome['failures']}"
    )


def test_trust_gate_consults_open_pull_requests_of_the_authoritative_repo(
    tmp_path: Path,
) -> None:
    """The fallback route must ask the authoritative repo for its *open* pull requests."""
    outcome = run_gate_for_shipped_default(tmp_path / "pull-request-route")

    list_calls = [call for call in outcome["calls"] if call["name"] == "pulls.list"]
    assert list_calls, (
        "a ref that the Korvid default branch does not contain must be checked "
        "against the authoritative repository's open pull requests"
    )
    params = list_calls[0]["params"]
    assert params.get("repo") == "korvid"
    assert params.get("state") == "open", (
        "a closed or merged pull request must never vouch for a ref: only open, "
        "under-review branches of the authoritative repository count"
    )
    assert params.get("base") == KORVID_DEFAULT_BRANCH, (
        "only pull requests targeting the default branch may vouch for a ref"
    )


def test_trust_gate_rejects_a_ref_no_open_pull_request_contains(tmp_path: Path) -> None:
    outcome = run_gate_for_shipped_default(
        tmp_path / "no-vouching-pr",
        korvid_pull_compare={"status": "diverged"},
    )

    assert outcome["failures"], (
        "a ref contained neither in the default branch nor in an open pull request "
        "head must be rejected"
    )


def test_trust_gate_rejects_a_fork_pull_request_that_contains_the_ref(tmp_path: Path) -> None:
    """A fork head must never vouch for a Korvid ref, however it compares."""
    outcome = run_gate_for_shipped_default(
        tmp_path / "fork-pr",
        korvid_pulls=[vouching_pull_request(head_repository="attacker/korvid")],
        korvid_pull_compare={"status": "ahead"},
    )

    assert outcome["failures"], (
        "a pull request opened from a fork must never establish Korvid provenance: "
        "anyone can open one, so its head is unreviewed third-party code"
    )


def test_trust_gate_rejects_a_pull_request_targeting_another_base(tmp_path: Path) -> None:
    outcome = run_gate_for_shipped_default(
        tmp_path / "wrong-base-pr",
        korvid_pulls=[vouching_pull_request(base_ref="some-release-branch")],
        korvid_pull_compare={"status": "ahead"},
    )

    assert outcome["failures"], (
        "only pull requests targeting the Korvid default branch may vouch for a ref"
    )


def test_trust_gate_rejects_when_the_pull_request_api_fails(tmp_path: Path) -> None:
    """An API outage must close the round, not open it."""
    outcome = run_gate_for_shipped_default(
        tmp_path / "pr-api-failure",
        korvid_pulls_error="HttpError: 503",
    )

    assert outcome["failures"], (
        "if the authoritative repository's pull requests cannot be read, provenance "
        "is unproven and the round must fail closed"
    )


def test_trust_gate_still_accepts_a_default_branch_ancestor_without_pull_requests(
    tmp_path: Path,
) -> None:
    """The pull-request route is additive: default-branch containment still passes alone."""
    outcome = run_gate_for_shipped_default(
        tmp_path / "default-branch-route",
        korvid_compare={"status": "ahead"},
        korvid_pulls=[],
    )

    assert outcome["failures"] == [], outcome["failures"]
    assert not [call for call in outcome["calls"] if call["name"] == "pulls.list"], (
        "a ref the default branch already contains must be accepted without listing "
        "pull requests at all"
    )


# ---------------------------------------------------------------------------
# Documentation drift
# ---------------------------------------------------------------------------


def readme_text() -> str:
    return README_PATH.read_text(encoding="utf-8")


def test_readme_names_the_approved_korvid_pin() -> None:
    assert APPROVED_KORVID_SHA in readme_text(), (
        "the README must name the exact pinned Korvid SHA so a dispatcher can see "
        "which revision a default round grounds against"
    )


def test_readme_documents_the_pull_request_provenance_route() -> None:
    text = readme_text().lower()
    assert f"#{APPROVED_KORVID_PROVENANCE.pull_request}" in readme_text(), (
        "the README must name the open pull request that vouches for the pin"
    )
    assert "pull request" in text and "fork" in text
    assert "korvid_ref" in text


def test_readme_korvid_ref_row_describes_both_acceptance_routes() -> None:
    rows = [line for line in readme_text().splitlines() if line.startswith("| `korvid_ref`")]
    assert rows, "the README input table must document korvid_ref"
    row = " ".join(rows).lower()
    assert "default branch" in row
    assert "pull request" in row, (
        "the korvid_ref row still claims default-branch containment is the only "
        "accepted provenance, which the shipped default does not satisfy"
    )


def test_readme_does_not_claim_unmerged_refs_are_always_rejected() -> None:
    """Round two's regression was a doc that contradicted the shipped default."""
    text = readme_text()
    assert "an unmerged branch, a fork, or an arbitrary experiment" not in text, (
        "the README must not state that any unmerged commit is rejected: the shipped "
        "default is an unmerged, under-review commit that is accepted on purpose"
    )


# ---------------------------------------------------------------------------
# Maintainer re-verification
# ---------------------------------------------------------------------------


def test_verify_script_exists_and_is_executable() -> None:
    assert VERIFY_SCRIPT_PATH.is_file(), (
        "maintainers need a scripted way to re-prove the pin against the live "
        "GitHub API without a flaky unit test"
    )
    assert os.access(VERIFY_SCRIPT_PATH, os.X_OK), "the verify script must be executable"


def test_verify_script_reads_the_pin_instead_of_hardcoding_it() -> None:
    body = VERIFY_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "korvid_pin" in body, (
        "the verify script must read the reviewed declaration, not a second copy of "
        "the SHA that can drift"
    )
    assert APPROVED_KORVID_SHA not in body, "the SHA must not be duplicated in the script"


def test_verify_script_checks_provenance_and_compatibility() -> None:
    body = VERIFY_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "gh api" in body, "provenance is re-proven through the GitHub API"
    assert "compare" in body, "provenance requires a compare against the authoritative repo"
    assert "pulls" in body, "the pull-request route must be re-provable too"
    assert "contents" in body, (
        "compatibility requires proving the required Korvid source paths exist at the pin"
    )


def test_verify_script_declares_no_new_dependencies() -> None:
    body = VERIFY_SCRIPT_PATH.read_text(encoding="utf-8")
    for forbidden in ("pip install", "uv add", "npm install", "brew install"):
        assert forbidden not in body, f"the verify script must not install anything ({forbidden})"


def test_verify_script_pin_field_does_not_execute_the_declaration(tmp_path: Path) -> None:
    """A bare-Python verifier must parse the declaration without importing it."""
    import re
    import subprocess
    import sys

    body = VERIFY_SCRIPT_PATH.read_text(encoding="utf-8")

    # Extract the Python snippet from pin_field(), which receives the declaration
    # path and requested field as positional arguments.
    m = re.search(
        r"pin_field\(\) \{[^}]*?\"?\$PYTHON\"? -c '(.*?)'\s+\"\$REPO_ROOT[^\"]+\"\s+\"\$1\"\s*\n\}",
        body,
        re.DOTALL,
    )
    assert m is not None, (
        "could not locate the inline Python snippet inside pin_field(); "
        "the test extraction pattern needs updating if the shell function changed"
    )
    snippet = m.group(1)

    pin_path = tmp_path / "korvid_pin.py"
    pin_path.write_text(
        (
            _REPO_ROOT / "src" / "korvid_prompt_lab" / "korvid_pin.py"
        ).read_text(encoding="utf-8")
        + '\nraise RuntimeError("the verifier executed the declaration")\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", snippet, str(pin_path), "sha"],
        capture_output=True,
        text=True,
        env={},
        check=False,
    )
    assert result.returncode == 0, (
        f"pin_field 'sha' failed under bare Python (-S).\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}\n"
        "The inline snippet must parse the declaration without executing it."
    )
    assert result.stdout.strip() == APPROVED_KORVID_SHA, (
        f"pin_field 'sha' returned unexpected output: {result.stdout.strip()!r}"
    )


# ---------------------------------------------------------------------------
# The workflow must document the boundary it now enforces
# ---------------------------------------------------------------------------


def test_workflow_trust_script_never_interpolates_inputs() -> None:
    script = str(trust_verification_step(load_workflow())["with"]["script"])
    assert "${{" not in script, (
        "the pull-request route must read its values from process.env like the rest "
        "of the trust step"
    )


@pytest.mark.parametrize("needle", ["pulls.list", "full_name", "state:"])
def test_workflow_trust_script_implements_the_pull_request_route(needle: str) -> None:
    script = str(trust_verification_step(load_workflow())["with"]["script"])
    assert needle in script, (
        f"the trust script must implement the open same-repository pull request "
        f"provenance route (missing {needle!r})"
    )
