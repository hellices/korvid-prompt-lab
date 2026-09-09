from __future__ import annotations

import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]
import litellm  # type: ignore[import-untyped]

from .artifacts import write_json_artifact
from .contracts import Candidate
from .experiment_budget import BudgetExhausted, ExperimentBudget
from .upstream_contract import tier_pack_from_candidate

_LITELLM_PROVIDER_ERRORS: tuple[type[Exception], ...] = tuple(
    litellm.exceptions.LITELLM_EXCEPTION_TYPES
)


class ReflectionProposalSignature(dspy.Signature):
    current_component_text: str = dspy.InputField(desc="Current prompt component text.")
    reflection_records_json: str = dspy.InputField(desc="Compact reflection records encoded as JSON.")
    revised_component_text: str = dspy.OutputField(desc="Improved prompt component text.")


class DSPyInstructionProposer:
    def __init__(
        self,
        reflection_lm: object,
        *,
        budget: ExperimentBudget,
    ) -> None:
        self.reflection_lm = reflection_lm
        self.budget = budget
        self._predictor: dspy.Predict | None = None

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        proposals: dict[str, str] = {}
        for component_name in components_to_update:
            if component_name not in candidate:
                raise ValueError(f"missing candidate component: {component_name}")
            records = list(reflective_dataset.get(component_name, ()))
            reflection_lm = self._reflection_lm_for_call()
            try:
                predictor_kwargs = {
                    "current_component_text": candidate[component_name],
                    "reflection_records_json": json.dumps(records, ensure_ascii=False, sort_keys=True),
                    "lm": reflection_lm,
                }
                with dspy.context(adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
                    prediction = self._get_predictor()(**predictor_kwargs)
            except dspy.AdapterParseError:
                raise ProposalRejected("unparseable_teacher_response") from None
            except _LITELLM_PROVIDER_ERRORS as exc:
                self.budget.check()
                raise ProposalProviderError(_provider_error_label(exc)) from None
            self.budget.check()
            revised = getattr(prediction, "revised_component_text", None)
            if not isinstance(revised, str) or not revised.strip():
                raise ProposalRejected("blank_proposal")
            proposals[component_name] = revised.strip()
        return proposals

    def _get_predictor(self) -> dspy.Predict:
        if self._predictor is None:
            self._predictor = dspy.Predict(ReflectionProposalSignature)
        return self._predictor

    def _reflection_lm_for_call(self) -> object:
        self.budget.check()
        if not isinstance(self.reflection_lm, dspy.LM):
            return self.reflection_lm

        remaining = self.budget.remaining_seconds
        lm_kwargs = getattr(self.reflection_lm, "kwargs", {})
        if "timeout" not in lm_kwargs:
            timeout = remaining
        else:
            configured = lm_kwargs["timeout"]
            if (
            isinstance(configured, bool)
            or not isinstance(configured, (int, float))
            or not math.isfinite(float(configured))
            or configured <= 0
            ):
                raise ValueError("reflection LM timeout must be a finite positive number")
            timeout = min(float(configured), remaining)
        return self.reflection_lm.copy(timeout=timeout, num_retries=0)


class ProposalRejected(ValueError):
    def __init__(self, error_label: str) -> None:
        self.error_label = error_label
        super().__init__(f"proposal rejected: {error_label}")


class DuplicateProposal(RuntimeError):
    pass


class ProposalProviderError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"proposal provider failed: {reason}")


class AuditedProposalSource:
    def __init__(
        self,
        *,
        proposer: Any,
        source: str,
        invocation_dir: Path,
        seed_candidate: Candidate,
        budget: ExperimentBudget,
    ) -> None:
        self.proposer = proposer
        self.source = source
        self.invocation_dir = invocation_dir
        self.seed_candidate = seed_candidate
        self.budget = budget
        self.pending_exception: Exception | None = None
        tier_pack_from_candidate(seed_candidate)
        self._records: list[dict[str, Any]] = []
        self._record_paths: list[str] = []
        self._seen_fingerprints = {seed_candidate.fingerprint}
        self._candidate_fingerprints: list[str] = []
        self._write_audit()

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        try:
            self.budget.consume_proposal()
        except BudgetExhausted as exc:
            self.pending_exception = exc
            raise

        attempt = len(self._records) + 1
        current_candidate = self._candidate(candidate)
        record: dict[str, Any] = {
            "schema_version": 1,
            "attempt": attempt,
            "source": self.source,
            "components_to_update": list(components_to_update),
            "reflective_dataset": _json_copy(reflective_dataset),
            "current_candidate_fingerprint": current_candidate.fingerprint,
        }

        try:
            proposed_value = self.proposer(
                candidate,
                reflective_dataset,
                components_to_update,
            )
        finally:
            failure = sys.exception()
            if isinstance(failure, Exception):
                self._record_source_failure(record, failure)

        record["proposed_components"] = _audit_proposed_components(proposed_value)
        try:
            proposed = self._validate_proposal(
                candidate=current_candidate,
                proposed_value=proposed_value,
                components_to_update=components_to_update,
            )
        except ValueError as exc:
            label = "blank_proposal" if "non-empty" in str(exc) else "malformed_proposal"
            self._reject(record, ProposalRejected(label))

        fingerprint = proposed.fingerprint
        record["proposed_candidate_fingerprint"] = fingerprint
        if fingerprint in self._seen_fingerprints:
            record["status"] = "duplicate"
            self._persist(record)
            raise DuplicateProposal("duplicate proposal")

        self._seen_fingerprints.add(fingerprint)
        self._candidate_fingerprints.append(fingerprint)
        record["status"] = "valid"
        self._persist(record)
        return dict(proposed_value)

    def summary(self) -> dict[str, Any]:
        statuses = Counter(record["status"] for record in self._records)
        error_labels = Counter(
            record["error_label"]
            for record in self._records
            if isinstance(record.get("error_label"), str)
        )
        return {
            "proposal_attempts": len(self._records),
            "distinct_proposals": statuses["valid"],
            "duplicate_proposals": statuses["duplicate"],
            "invalid_proposals": statuses["invalid"],
            "provider_errors": statuses["provider_error"],
            "error_labels": dict(sorted(error_labels.items())),
            "candidate_fingerprints": list(self._candidate_fingerprints),
            "proposal_record_paths": list(self._record_paths),
        }

    def _candidate(self, components: Mapping[str, str]) -> Candidate:
        return Candidate.from_mapping(
            {
                "schema_version": self.seed_candidate.schema_version,
                "candidate_id": self.seed_candidate.candidate_id,
                "components": dict(components),
                "metadata": self.seed_candidate.metadata,
            }
        )

    def _validate_proposal(
        self,
        *,
        candidate: Candidate,
        proposed_value: object,
        components_to_update: Sequence[str],
    ) -> Candidate:
        if not isinstance(proposed_value, Mapping):
            raise ValueError("proposal must be a mapping")  # noqa: TRY004 - converted to ProposalRejected
        if set(proposed_value) != set(components_to_update):
            raise ValueError("proposal components must exactly match requested components")
        if any(not isinstance(name, str) or not isinstance(text, str) for name, text in proposed_value.items()):
            raise ValueError("proposal components must map strings to strings")
        combined = candidate.components
        combined.update(proposed_value)
        proposed = self._candidate(combined)
        tier_pack_from_candidate(proposed)
        return proposed

    def _reject(self, record: dict[str, Any], exc: ProposalRejected) -> None:
        record["status"] = "invalid"
        record["error_label"] = exc.error_label
        self.pending_exception = exc
        self._persist(record)
        raise exc

    def _record_source_failure(self, record: dict[str, Any], failure: Exception) -> None:
        if isinstance(failure, ProposalRejected):
            record["status"] = "invalid"
            record["error_label"] = failure.error_label
        elif isinstance(failure, BudgetExhausted):
            record["status"] = "provider_error"
            record["error_label"] = f"budget_{failure.reason}"
        else:
            record["status"] = "provider_error"
            record["error_label"] = (
                failure.reason
                if isinstance(failure, ProposalProviderError)
                else _provider_error_label(failure)
            )
        self.pending_exception = failure
        self._persist(record)

    def _persist(self, record: dict[str, Any]) -> None:
        path = self.invocation_dir / "proposals" / f"{record['attempt']:04d}.json"
        relative_path = path.relative_to(self.invocation_dir).as_posix()
        write_json_artifact(path, record)
        self._records.append(record)
        self._record_paths.append(relative_path)
        self._write_audit()

    def _write_audit(self) -> None:
        write_json_artifact(
            self.invocation_dir / "proposal-audit.json",
            {
                "schema_version": 1,
                "source": self.source,
                **self.summary(),
            },
        )


def _json_copy(value: object) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _audit_proposed_components(value: object) -> object:
    if not isinstance(value, Mapping):
        return {"value_type": type(value).__name__}
    return {
        str(name): text if isinstance(text, (str, int, float, bool)) or text is None else type(text).__name__
        for name, text in value.items()
    }


def _provider_error_label(exc: Exception) -> str:
    name = type(exc).__name__
    label = "".join(f"_{character.lower()}" if character.isupper() else character for character in name).lstrip("_")
    return f"provider_{label}"
