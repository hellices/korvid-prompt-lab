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

Why the pin is *not* a default-branch commit
--------------------------------------------
The harness the bridge imports — ``korvid.evals.operation``,
``tests.evals.operation_app``, ``tests.evals.operation_campaign`` and
``tests.evals.operation_scripts`` — has never existed on ``hellices/korvid``
``main``.  It is introduced by the open pull request recorded below.  Repinning
to a ``main`` commit would satisfy a default-branch-only provenance gate and then
fail at run time with "korvid operation harness is not importable", *after* the
Korvid app token, the Azure OIDC session, and the GPU node pool had been spent.
Trading a cheap pre-credential rejection for an expensive post-credential failure
is strictly worse, so the pin stays on the reviewed pull-request commit and the
provenance gate proves that commit is authoritative Korvid code.
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
#: It is the newest commit on the reviewed branch that is *both* fully green in
#: Korvid CI *and* contains every bridge dependency.  The branch has advanced past
#: it, but every later commit either fails CI or leaves the bridge dependencies
#: byte-identical, so moving the pin forward would buy nothing and give up the
#: green signal.
APPROVED_KORVID_SHA = "fc7eece2adb66a5b2a18d378bdfd7503ddbdd2ca"


@dataclass(frozen=True, slots=True)
class KorvidProvenance:
    """A dated snapshot of how :data:`APPROVED_KORVID_SHA` is proven authoritative.

    ``kind`` names the acceptance route the workflow's pre-credential trust gate
    must take for this pin.  ``default_branch_compare_status`` is the recorded
    ``compare/<sha>...<default_branch>`` status — keeping the *failing* status in
    the declaration is deliberate: it is the fact that makes the default-branch
    route insufficient, and a test replays it so the gate can never silently
    regress to a rule the shipped default cannot pass.
    """

    #: Acceptance route: containment in an open same-repository pull request head.
    kind: str
    #: Pull request in :data:`KORVID_REPOSITORY` that vouches for the pin.
    pull_request: int
    #: Head branch of that pull request.
    branch: str
    #: Base branch of that pull request — must be the default branch.
    base_branch: str
    #: Head repository of that pull request — must be the authoritative repo, not a fork.
    head_repository: str
    #: Head commit of that pull request when the snapshot was taken.  It advances
    #: as the pull request does — the tests only replay it, and
    #: ``scripts/verify-korvid-pin.sh`` re-derives the live head.
    head_sha: str
    #: ``compare/<APPROVED_KORVID_SHA>...<head_sha>`` status when the snapshot was taken.
    head_compare_status: str
    #: ``compare/<APPROVED_KORVID_SHA>...<default_branch>`` status when the snapshot was taken.
    default_branch_compare_status: str
    #: UTC date the facts above were read from the GitHub API.
    verified_on: str


#: Acceptance route names the trust gate implements.
PROVENANCE_DEFAULT_BRANCH = "default_branch_containment"
PROVENANCE_OPEN_PULL_REQUEST = "open_pull_request_containment"

#: Recorded provenance of :data:`APPROVED_KORVID_SHA`.
APPROVED_KORVID_PROVENANCE = KorvidProvenance(
    kind=PROVENANCE_OPEN_PULL_REQUEST,
    pull_request=312,
    branch="feat/307-small-operator-foundation",
    base_branch=KORVID_DEFAULT_BRANCH,
    head_repository=KORVID_REPOSITORY,
    head_sha="525378f09e76fc7e869335a6f38133b0d3558407",
    head_compare_status="ahead",
    default_branch_compare_status="diverged",
    verified_on="2026-08-22",
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
    return (
        f"{KORVID_REPOSITORY}@{APPROVED_KORVID_SHA} "
        f"(open PR #{provenance.pull_request} {provenance.branch} -> {provenance.base_branch}; "
        f"compare vs {KORVID_DEFAULT_BRANCH}: {provenance.default_branch_compare_status})"
    )
