"""Native GEPA plumbing with scripted responses, not model-quality evidence."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from korvid_prompt_lab.config import load_campaign, load_candidate
from korvid_prompt_lab.contracts import Candidate, EvalCase
from korvid_prompt_lab.native import KorvidNativeRunner, initialize_native
from korvid_prompt_lab.native_contract import (
    find_native_case,
    rules_candidate,
    rules_from_candidate,
)
from korvid_prompt_lab.optimize import optimize_campaign
from korvid_prompt_lab.scoring import BridgeResult

pytestmark = pytest.mark.skipif(
    not os.environ.get("KORVID_NATIVE_SOURCE_ROOT"),
    reason="requires pinned native source environment",
)


class RuleAwareScriptedRunner(KorvidNativeRunner):
    def run(
        self, candidate: Candidate, case: EvalCase, run_dir: Path | str, *,
        repetition: int = 1, seed: int = 0,
    ) -> BridgeResult:
        def script(selected: EvalCase) -> list[list[dict[str, Any]]]:
            final = [{"type": "text_delta", "text": "Done."}, {"type": "done"}]
            if not rules_from_candidate(candidate):
                return [final]
            namespace = find_native_case(selected.case_id).namespace
            return [[{
                "type": "tool_call", "id": "a", "name": "list_resources",
                "arguments": json.dumps({"kind": "pods", "namespace": namespace}),
            }, {"type": "done"}], final]

        return KorvidNativeRunner(self.campaign, script_factory=script).run(
            candidate, case, run_dir, repetition=repetition, seed=seed,
        )


def test_native_gepa_can_select_and_persist_additive_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KORVID_NATIVE_MODEL_URL", "http://127.0.0.1:11434")
    _, path = initialize_native(tmp_path / "setup", model="ollama/qwen3:0.6b")
    campaign = load_campaign(path)
    seen: list[object] = []

    def propose(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        assert components_to_update == ["rules"]
        assert "runtime_policy" in reflective_dataset["rules"][0]["Inputs"]
        seen.append(reflective_dataset)
        return {"rules": json.dumps(["For a requested Pod view, use list_resources in the requested namespace."])}

    result = optimize_campaign(
        runner=RuleAwareScriptedRunner(campaign),
        seed_candidate=rules_candidate([]),
        train_cases=[case for case in campaign.cases if case.case_id == "train-pods"],
        validation_cases=[case for case in campaign.cases if case.case_id == "validation-pods"],
        artifact_root=tmp_path / "optimize", max_metric_calls=6, candidate_proposer=propose,
    )
    assert seen
    persisted = load_candidate(result.best_candidate_path)
    assert rules_from_candidate(persisted)
    summary = json.loads(result.summary_path.read_text())
    assert summary["execution_modes"] == ["scripted"]
    assert summary["best_candidate_differs_from_seed"] is True
    assert not set(summary["train_case_ids"] + summary["validation_case_ids"]) & {
        case.case_id for case in campaign.cases if case.case_id.startswith("holdout-")
    }
