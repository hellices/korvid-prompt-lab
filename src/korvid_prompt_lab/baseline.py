"""Materialize Korvid's installed shipped prompt as a Prompt Lab seed candidate.

``korvid-prompt-lab korvid-baseline`` never hard-codes a prompt string in this
repository. Instead it imports ``korvid.agent.profiles.build_profile`` from the
installed ``korvid[agent]`` wheel and reads the selected profile's current
``system_prompt`` at runtime, so the seed candidate used for comparison is
always exactly what the installed distribution ships.

``readonly=True`` and ``resize_supported=False`` are fixed inputs to
``build_profile`` because this command only ever seeds the read-only
optimization scope this repository targets; neither flag (nor
``observability_backends``/``overrides``, both left at their defaults) changes
``system_prompt`` composition, so the materialized prompt is identical to
what any other caller of ``build_profile`` for the same profile name would see.
"""

from __future__ import annotations

import os
import sys
import tempfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from korvid.agent.profiles import build_profile

from .contracts import Candidate

#: Distribution name the shipped prompt and version metadata are read from.
KORVID_DISTRIBUTION_NAME = "korvid"

#: Profiles ``korvid.agent.profiles.build_profile`` accepts.
PROFILE_NAMES = ("full", "small")


def korvid_distribution_version() -> str:
    """Installed Korvid distribution version, failing closed if unavailable."""
    try:
        return _distribution_version(KORVID_DISTRIBUTION_NAME)
    except PackageNotFoundError as exc:
        raise ValueError(
            f"installed Korvid distribution metadata is unavailable for "
            f"{KORVID_DISTRIBUTION_NAME!r}; korvid-baseline refuses to record an "
            "unknown distribution version"
        ) from exc


def build_baseline_candidate(profile: str) -> Candidate:
    """Materialize *profile*'s current shipped system prompt as a Candidate.

    Fails closed (raises ``ValueError``) if the profile name is invalid, the
    installed Korvid distribution's package metadata is unavailable, or the
    resulting system prompt is blank.
    """
    if profile not in PROFILE_NAMES:
        raise ValueError(
            f"profile must be one of {PROFILE_NAMES}, got {profile!r}"
        )

    korvid_version = korvid_distribution_version()

    try:
        agent_profile = build_profile(profile, readonly=True, resize_supported=False)
    except ValueError as exc:
        raise ValueError(
            f"installed Korvid rejected profile {profile!r}: {exc}"
        ) from exc

    system_prompt = agent_profile.system_prompt
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError(
            f"installed Korvid profile {profile!r} produced a blank system prompt"
        )

    return Candidate.from_mapping(
        {
            "schema_version": 1,
            "candidate_id": f"korvid-baseline-{profile}",
            "components": {"system": system_prompt},
            "metadata": {
                "korvid_version": korvid_version,
                "profile": profile,
            },
        }
    )


def render_baseline_yaml(candidate: Candidate) -> str:
    """Exact YAML text a baseline candidate is written as."""
    payload = {
        "schema_version": candidate.schema_version,
        "candidate_id": candidate.candidate_id,
        "components": candidate.components,
        "metadata": candidate.metadata,
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def write_baseline_candidate(candidate: Candidate, output_path: Path | str) -> Path:
    """Write *candidate* to *output_path*, failing closed on collision.

    Refuses to write if ``output_path`` already exists or is a symlink
    (including a dangling one). The write is atomic: the destination either
    ends up with the full exact content, or is not created at all.
    """
    output_path = Path(output_path)
    if output_path.is_symlink():
        raise FileExistsError(
            f"refusing to write baseline candidate over a symlink: {output_path}"
        )
    if output_path.exists():
        raise FileExistsError(
            f"baseline candidate output already exists: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = render_baseline_yaml(candidate)

    fd, temp_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.chmod(0o644)
        os.link(temp_path, output_path)
    finally:
        primary_error = sys.exception()
        try:
            temp_path.unlink(missing_ok=True)
        except OSError as exc:
            if primary_error is not None:
                primary_error.add_note(
                    f"could not remove temporary baseline file {temp_path}: {exc}"
                )
            else:
                raise

    return output_path
