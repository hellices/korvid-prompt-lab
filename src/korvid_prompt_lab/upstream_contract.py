"""Control-plane contract for unchanged Korvid scenario and journey sources."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from .contracts import Candidate, EvalCase

KORVID_VERSION = "0.4.1"
KORVID_REVISION = "33c483e041006eb20259a024ed85a9323e52c8f0"

_REFERENCE = re.compile(r"^(scenarios|journeys)/([a-z0-9][a-z0-9-]*)$")


@dataclass(frozen=True, slots=True)
class SourceCase:
    reference: str
    case_id: str
    kind: str
    prompt: str
    source_path: str
    source_sha256: str
    questions: tuple[str, ...]

    def eval_case(self, model: str) -> EvalCase:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("source case model must be a non-blank string")
        return EvalCase(
            case_id=self.case_id,
            template_id=f"korvid-{self.kind}",
            prompt=self.prompt,
            models=(model,),
        )


def _tracked(root: Path, relative: str) -> bool:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env=env,
    )
    return result.returncode == 0 and result.stdout.strip() == relative


def _source_case(root: Path, reference: str) -> SourceCase:
    match = _REFERENCE.fullmatch(reference)
    if match is None:
        raise ValueError(f"invalid Korvid source reference: {reference!r}")
    directory, name = match.groups()
    kind = "scenario" if directory == "scenarios" else "journey"
    relative = f"src/korvid/evals/{directory}/{name}.yaml"
    if not _tracked(root, relative):
        raise ValueError(f"unknown tracked Korvid source reference: {reference}")
    path = root / relative
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Korvid source reference is unavailable: {reference}") from exc
    data = yaml.safe_load(source)
    if not isinstance(data, dict):
        raise ValueError(f"Korvid source must be a YAML mapping: {reference}")  # noqa: TRY004
    case_id = data.get("id")
    if case_id != name:
        raise ValueError(f"Korvid source filename and declared id differ: {reference}")
    if kind == "scenario":
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Korvid scenario question is missing: {reference}")
        questions: tuple[str, ...] = (question,)
        prompt = question
    else:
        turns = data.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"Korvid journey turns are missing: {reference}")
        questions_list: list[str] = []
        for turn in turns:
            user = turn.get("user") if isinstance(turn, dict) else None
            if not isinstance(user, str) or not user.strip():
                raise ValueError(f"Korvid journey has an invalid user turn: {reference}")
            questions_list.append(user)
        questions = tuple(questions_list)
        prompt = json.dumps(questions_list, ensure_ascii=False)
    return SourceCase(
        reference=reference,
        case_id=name,
        kind=kind,
        prompt=prompt,
        source_path=relative,
        source_sha256=hashlib.sha256(source).hexdigest(),
        questions=questions,
    )


def load_source_cases(root: Path, references: Sequence[str]) -> tuple[SourceCase, ...]:
    root = root.resolve(strict=True)
    if not references:
        raise ValueError("Korvid source references must not be empty")
    cases = tuple(_source_case(root, reference) for reference in references)
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate Korvid source case id")
    return cases


def tier_pack_from_candidate(candidate: Candidate) -> str:
    components = candidate.components
    if set(components) != {"tier_pack"}:
        raise ValueError("upstream candidates support only the tier_pack component")
    text = components["tier_pack"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("tier_pack must be a non-blank string")
    return text


def prompt_candidate(text: str) -> Candidate:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("tier_pack must be a non-blank string")
    return Candidate(
        schema_version=1,
        candidate_id="korvid-upstream-prompt",
        _components=(("tier_pack", text),),
        _metadata=(
            ("scope", "korvid-upstream-tier-pack"),
            ("korvid_version", KORVID_VERSION),
            ("korvid_revision", KORVID_REVISION),
        ),
    )
