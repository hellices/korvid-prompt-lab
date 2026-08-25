"""The reviewed Korvid revision this repository grounds against.

A grounding round checks out Korvid at ``korvid_ref`` and runs the bridge worker
inside that checkout.  Two independent things must therefore hold for the shipped
default, and they are easy to satisfy one at a time and get wrong together:

*provenance* — the commit must be proven to be authoritative Korvid code before
any credential exists, and

*compatibility* — the commit must actually contain the operation-journey harness
:mod:`korvid_prompt_lab.bridge_worker` imports.

This module is the single deterministic declaration of both, so the workflow
default, the README, and the bridge import contract can be checked against one
reviewed source of truth offline.  Nothing here touches the network: the
provenance facts are a dated snapshot recorded from the GitHub API, and
``scripts/verify-korvid-pin.sh`` re-verifies them live when a maintainer asks.

Why provenance and runtime importability are both recorded
----------------------------------------------------------
The pinned commit now *is* the reviewed squash merge on ``hellices/korvid``
``main``, so the workflow's default-branch provenance route can trust it
durably. That alone is still insufficient: the bridge imports specific Korvid
symbols at run time, and a file path existing says nothing about whether names
such as ``LIFECYCLE_CHECKPOINTS`` still resolve. This declaration therefore
records both the authoritative provenance snapshot *and* the exact import
contract the bridge and maintainer verifier must prove before any Azure/model
credential or AKS scaling occurs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

#: Authoritative Korvid repository.  The workflow derives this from
#: ``github.repository_owner`` so it is never user-controlled; the literal here is
#: only the reviewed expectation that the derivation resolves to.
KORVID_REPOSITORY = "hellices/korvid"

#: Default branch of :data:`KORVID_REPOSITORY`.
KORVID_DEFAULT_BRANCH = "main"

#: The exact Korvid commit a grounding round defaults to.
#:
#: It is the reviewed squash-merge commit that landed the operation harness on
#: Korvid's default branch while still satisfying the bridge's runtime import
#: contract.
APPROVED_KORVID_SHA = "62bd3cbee2e27369bb81abc0957dae341c2aa434"


@dataclass(frozen=True, slots=True)
class KorvidProvenance:
    """A dated snapshot of how :data:`APPROVED_KORVID_SHA` is proven authoritative.

    ``kind`` names the acceptance route the workflow's pre-credential trust gate
    must take for this pin. ``default_branch_compare_status`` is the recorded
    ``compare/<sha>...<default_branch>`` status when the snapshot was taken.
    """

    #: Acceptance route the trust gate applies.
    kind: str
    #: Pull request in :data:`KORVID_REPOSITORY` that vouches for the pin when the
    #: provenance route is :data:`PROVENANCE_OPEN_PULL_REQUEST`.
    pull_request: int | None
    #: Head branch of that pull request.
    branch: str | None
    #: Base branch of that pull request — must be the default branch.
    base_branch: str | None
    #: Head repository of that pull request — must be the authoritative repo, not a fork.
    head_repository: str | None
    #: Head commit of that pull request when the snapshot was taken. It advances
    #: as the pull request does — the tests only replay it, and
    #: ``scripts/verify-korvid-pin.sh`` re-derives the live head.
    head_sha: str | None
    #: ``compare/<APPROVED_KORVID_SHA>...<head_sha>`` status when the snapshot was taken.
    head_compare_status: str | None
    #: ``compare/<APPROVED_KORVID_SHA>...<default_branch>`` status when the snapshot was taken.
    default_branch_compare_status: str
    #: UTC date the facts above were read from the GitHub API.
    verified_on: str

    def __post_init__(self) -> None:
        if self.kind == PROVENANCE_DEFAULT_BRANCH:
            if self.default_branch_compare_status not in {"identical", "ahead"}:
                raise ValueError(
                    "default-branch provenance requires default_branch_compare_status "
                    "to be identical or ahead"
                )
            extras = {
                "pull_request": self.pull_request,
                "branch": self.branch,
                "base_branch": self.base_branch,
                "head_repository": self.head_repository,
                "head_sha": self.head_sha,
                "head_compare_status": self.head_compare_status,
            }
            present = [name for name, value in extras.items() if value is not None]
            if present:
                raise ValueError(
                    "default-branch provenance must not carry pull-request fields: "
                    + ", ".join(present)
                )
            return

        if self.kind == PROVENANCE_OPEN_PULL_REQUEST:
            required = {
                "pull_request": self.pull_request,
                "branch": self.branch,
                "base_branch": self.base_branch,
                "head_repository": self.head_repository,
                "head_sha": self.head_sha,
                "head_compare_status": self.head_compare_status,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "pull-request provenance requires: " + ", ".join(missing)
                )
            return

        raise ValueError(f"unknown Korvid provenance kind: {self.kind}")


#: Acceptance route names the trust gate implements.
PROVENANCE_DEFAULT_BRANCH = "default_branch_containment"
PROVENANCE_OPEN_PULL_REQUEST = "open_pull_request_containment"

#: Recorded provenance of :data:`APPROVED_KORVID_SHA`.
APPROVED_KORVID_PROVENANCE = KorvidProvenance(
    kind=PROVENANCE_DEFAULT_BRANCH,
    pull_request=None,
    branch=None,
    base_branch=None,
    head_repository=None,
    head_sha=None,
    head_compare_status=None,
    default_branch_compare_status="identical",
    verified_on="2026-08-26",
)

#: Every module :func:`korvid_prompt_lab.bridge_worker._import_korvid` imports,
#: mapped to the names it binds.  An empty tuple means a plain ``import <module>``.
#:
#: This is the compatibility half of the pin: a Korvid commit is only usable if it
#: exposes all of these.  A test derives the same mapping from the bridge worker's
#: own source, so the declaration cannot drift away from the code it describes.
REQUIRED_KORVID_IMPORTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "httpx": (),
        "korvid.agent.profiles": ("PromptOverrides",),
        "korvid.evals.operation": (
            "LIFECYCLE_CHECKPOINTS",
            "bundled_operations_dir",
            "load_operation_journeys",
        ),
        "korvid.evals.scripted": ("ScriptedProvider",),
        "korvid.providers.openai_compat": ("OpenAICompatProvider", "ProviderError"),
        "korvid.providers.static_creds": ("StaticHeaderSource",),
        "tests.evals": ("operation_app",),
        "tests.evals.operation_campaign": ("approval_timeout_for",),
        "tests.evals.operation_scripts": ("OPERATION_SCRIPTS",),
        "tests.ui.waits": ("WaitTimeout",),
    }
)

#: Paths inside a Korvid checkout that must exist at :data:`APPROVED_KORVID_SHA`
#: for :data:`REQUIRED_KORVID_IMPORTS` to resolve.  ``httpx`` is a Korvid
#: dependency rather than Korvid source, so it has no path here.
REQUIRED_KORVID_SOURCE_PATHS: tuple[str, ...] = (
    "src/korvid/agent/profiles.py",
    "src/korvid/evals/operation.py",
    "src/korvid/evals/scripted.py",
    "src/korvid/providers/openai_compat.py",
    "src/korvid/providers/static_creds.py",
    "tests/evals/operation_app.py",
    "tests/evals/operation_campaign.py",
    "tests/evals/operation_scripts.py",
    "tests/ui/waits.py",
)

#: Attribute the bridge rebinds on ``tests.evals.operation_app`` to inject prompt
#: overrides.  ``operation_app`` imports it rather than defining it, so the pin
#: records the module the harness must expose it *on*, not where it is written.
REQUIRED_PATCHABLE_ATTRIBUTE = "build_profile"


def approved_pin_summary() -> str:
    """One-line human summary of the pin, used by reports and the verify script."""
    provenance = APPROVED_KORVID_PROVENANCE
    if provenance.kind == PROVENANCE_OPEN_PULL_REQUEST:
        return (
            f"{KORVID_REPOSITORY}@{APPROVED_KORVID_SHA} "
            f"(open PR #{provenance.pull_request} {provenance.branch} -> {provenance.base_branch}; "
            f"compare vs {KORVID_DEFAULT_BRANCH}: {provenance.default_branch_compare_status}; "
            f"PR head compare: {provenance.head_compare_status}; verified {provenance.verified_on})"
        )
    return (
        f"{KORVID_REPOSITORY}@{APPROVED_KORVID_SHA} "
        f"(default branch {KORVID_DEFAULT_BRANCH}; "
        f"compare vs {KORVID_DEFAULT_BRANCH}: {provenance.default_branch_compare_status}; "
        f"verified {provenance.verified_on})"
    )
