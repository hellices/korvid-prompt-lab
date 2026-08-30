"""Synthetic ``python -m korvid.evals`` fixture for korvid_readonly runner tests.

Mimics only the CLI surface :class:`~korvid_prompt_lab.korvid_readonly.KorvidReadonlyRunner`
depends on: the ``--scenarios/--reps/--profile/--system-prompt-file/--prompt-append-file/--json``
argv shape and the ``KORVID_EVAL_*`` environment contract. Behavior is switched by the
``FAKE_KORVID_EVALS_MODE`` environment variable so tests can drive every runner outcome
without contacting a model or depending on which real bundled scenario id is under test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from korvid.agent.profiles import PromptOverrides, build_profile
from korvid.evals.__main__ import prompt_fingerprint
from korvid.evals.runner import _eval_tools


def _first_scenario_id(scenarios_dir: Path) -> str:
    for path in sorted(scenarios_dir.glob("*.yaml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("id:"):
                return line.split(":", 1)[1].strip()
    return "unknown-scenario"


def _record_invocation(args: argparse.Namespace) -> None:
    """Write everything a test needs to assert about this invocation.

    Always written next to the requested ``--json`` output (a path unique
    to this one invocation's run directory) so concurrent invocations never
    need to coordinate through a shared, mutable environment variable.
    ``FAKE_KORVID_EVALS_RECORD``, when set, additionally copies the same
    record to a caller-chosen path for tests that only run one invocation
    at a time.
    """
    record = {
        "argv": sys.argv[1:],
        "env": {
            key: os.environ.get(key)
            for key in (
                "KORVID_EVAL_BASE_URL",
                "KORVID_EVAL_MODEL",
                "KORVID_EVAL_TIMEOUT_SECONDS",
                "KORVID_EVAL_API_KEY",
            )
        },
        "system_prompt": args.system_prompt_file.read_text(encoding="utf-8").strip(),
        "prompt_append": (
            args.prompt_append_file.read_text(encoding="utf-8").strip()
            if args.prompt_append_file is not None
            else None
        ),
        "scenario_files": sorted(p.name for p in args.scenarios.glob("*.yaml")),
    }
    encoded = json.dumps(record)
    args.json.with_name(args.json.name + ".record.json").write_text(
        encoded, encoding="utf-8"
    )

    record_path = os.environ.get("FAKE_KORVID_EVALS_RECORD")
    if record_path:
        Path(record_path).write_text(encoded, encoding="utf-8")


def _default_run(mode: str) -> dict[str, Any]:
    run: dict[str, Any] = {
        "grade": {
            "diagnosis_success": True,
            "evidence_fetched": True,
            "missing_mentions": [],
            "forbidden_mentions": [],
            "missing_evidence": [],
        },
        "citations": {
            "cited": ["ref-1"],
            "unsupported": [],
            "uncited_evidence": [],
            "coverage": 1.0,
            "precision": 1.0,
        },
        "answer": "diagnosis complete",
        "iterations": 2,
        "tool_calls": 4,
        "resolvable_tool_calls": 4,
        "on_target_tool_calls": 3,
        "malformed_tool_calls": 0,
        "write_attempts": 0,
        "safety_violations": 0,
        "input_tokens": 100,
        "output_tokens": 50,
        "tokens_estimated": False,
        "wall_time_s": 1.5,
        "error": None,
    }

    if mode == "diagnosis-failed":
        run["grade"] = {
            **run["grade"],
            "diagnosis_success": False,
            "missing_mentions": [["oom"]],
        }
    elif mode == "evidence-missing":
        run["grade"] = {
            **run["grade"],
            "evidence_fetched": False,
            "missing_evidence": [
                [{"tool": "get_events", "contains": "OOM", "args": {}}]
            ],
        }
    elif mode == "no-tool-calls":
        run["tool_calls"] = 0
        run["on_target_tool_calls"] = 0
        run["resolvable_tool_calls"] = 0
    elif mode == "partial-efficiency":
        run["tool_calls"] = 4
        run["on_target_tool_calls"] = 1
    elif mode == "hard-failure":
        run["write_attempts"] = 2
        run["safety_violations"] = 1
    elif mode == "write-attempt-only":
        run["write_attempts"] = 1
        run["safety_violations"] = 0
    elif mode == "model-failure":
        run["error"] = "provider returned no tokens"
        run["answer"] = ""
    elif mode == "malformed-tool-calls":
        run["malformed_tool_calls"] = 2
    elif mode == "invalid-citation-coverage":
        run["citations"]["coverage"] = 1.5
    elif mode == "invalid-citation-precision":
        run["citations"]["precision"] = float("nan")
    elif mode == "negative-wall-time":
        run["wall_time_s"] = -1.0
    elif mode == "nonfinite-wall-time":
        run["wall_time_s"] = float("nan")
    return run


#: Substring a candidate's "system" (optionally plus "append") component must
#: contain for `FAKE_KORVID_EVALS_MODE=prompt-driven` to grade it as tuned.
#: Mirrors `tests/fixtures/fake_korvid_bridge.py`'s TUNED_MARKER convention so
#: GEPA integration tests read the same way across both execution modes.
TUNED_MARKER = "korvid-tuned"


def _prompt_driven_run(tuned: bool) -> dict[str, Any]:
    """A large, deterministic quality gap so real (non-mocked) GEPA search
    reliably prefers a tuned candidate over an untuned one, without any
    network call or reflection LM."""
    if tuned:
        return {
            "grade": {
                "diagnosis_success": True,
                "evidence_fetched": True,
                "missing_mentions": [],
                "forbidden_mentions": [],
                "missing_evidence": [],
            },
            "citations": {
                "cited": ["ref-1"],
                "unsupported": [],
                "uncited_evidence": [],
                "coverage": 1.0,
                "precision": 1.0,
            },
            "answer": "diagnosis complete with cited evidence",
            "iterations": 2,
            "tool_calls": 4,
            "resolvable_tool_calls": 4,
            "on_target_tool_calls": 4,
            "malformed_tool_calls": 0,
            "write_attempts": 0,
            "safety_violations": 0,
            "input_tokens": 120,
            "output_tokens": 60,
            "tokens_estimated": False,
            "wall_time_s": 1.2,
            "error": None,
        }
    return {
        "grade": {
            "diagnosis_success": False,
            "evidence_fetched": False,
            "missing_mentions": [["oom"]],
            "forbidden_mentions": [],
            "missing_evidence": [[{"tool": "get_events", "contains": "OOM", "args": {}}]],
        },
        "citations": {
            "cited": [],
            "unsupported": [],
            "uncited_evidence": ["ref-1"],
            "coverage": 0.0,
            "precision": 0.0,
        },
        "answer": "unable to confirm the root cause",
        "iterations": 2,
        "tool_calls": 4,
        "resolvable_tool_calls": 4,
        "on_target_tool_calls": 1,
        "malformed_tool_calls": 1,
        "write_attempts": 0,
        "safety_violations": 0,
        "input_tokens": 120,
        "output_tokens": 60,
        "tokens_estimated": False,
        "wall_time_s": 1.4,
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--reps", type=int, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--system-prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-append-file", type=Path, default=None)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    _record_invocation(args)

    mode = os.environ.get("FAKE_KORVID_EVALS_MODE", "completed")

    if mode == "timeout":
        time.sleep(5.0)
        return 0
    if mode == "nonzero-exit":
        print("boom: provider unreachable", file=sys.stderr)
        return 3
    if mode == "missing-output":
        return 0
    if mode == "malformed-json":
        args.json.write_text("{not json", encoding="utf-8")
        return _exit_code(default=0)

    scenario_id = _first_scenario_id(args.scenarios)
    if mode == "identity-mismatch":
        scenario_id = f"{scenario_id}-wrong"

    if mode == "prompt-driven":
        combined = args.system_prompt_file.read_text(encoding="utf-8")
        if args.prompt_append_file is not None:
            combined += args.prompt_append_file.read_text(encoding="utf-8")
        run = _prompt_driven_run(TUNED_MARKER in combined)
    else:
        run = _default_run(mode)
    scenario_entry: dict[str, Any] = {
        "scenario": scenario_id,
        "root_cause": "oom_killed",
        "successes": 1
        if run["error"] is None and run["grade"]["diagnosis_success"]
        else 0,
        "evidence_hits": 1
        if run["error"] is None and run["grade"]["evidence_fetched"]
        else 0,
        "runs": [run],
    }

    if mode == "extra-scenario":
        scenarios_payload = [
            scenario_entry,
            {**scenario_entry, "scenario": f"{scenario_id}-2"},
        ]
    elif mode == "extra-run":
        scenarios_payload = [
            {**scenario_entry, "runs": [run, _default_run("completed")]}
        ]
    else:
        scenarios_payload = [scenario_entry]

    overrides = PromptOverrides(
        system=args.system_prompt_file.read_text(encoding="utf-8").strip(),
        append=(
            args.prompt_append_file.read_text(encoding="utf-8").strip()
            if args.prompt_append_file is not None
            else None
        ),
    )
    profile = build_profile(
        args.profile,
        readonly=False,
        resize_supported=True,
        overrides=overrides,
    )
    offered_tools = _eval_tools(profile)
    meta: dict[str, Any] = {
        "profile": args.profile,
        "prompts": prompt_fingerprint(profile, tools=offered_tools),
        "tools": {"omitted": [], "count": len(offered_tools)},
        "serving": {"model": os.environ["KORVID_EVAL_MODEL"]},
    }
    if mode == "missing-meta":
        meta.pop("profile")
    elif mode == "wrong-profile":
        meta["profile"] = "full" if args.profile == "small" else "small"
    elif mode == "wrong-prompt-fingerprint":
        meta["prompts"] = {"source": "override", "sha256": "0" * 64}
    elif mode == "wrong-model":
        meta["serving"] = {"model": "different-model"}
    if mode == "malformed-scenario-summary":
        scenario_entry["successes"] = "1"

    payload = {
        "meta": meta,
        "scenarios": scenarios_payload,
    }
    args.json.write_text(json.dumps(payload), encoding="utf-8")
    if mode == "model-failure":
        # Mirrors the real installed Korvid 0.3 CLI: a genuine model failure
        # still writes valid exactly-one-scenario/exactly-one-run JSON with
        # ``run.error`` populated, but the process itself exits nonzero.
        return _exit_code(default=1)
    return _exit_code(default=0)


def _exit_code(*, default: int) -> int:
    """The process exit code to return, overridable by tests independent of

    ``FAKE_KORVID_EVALS_MODE`` so a test can combine an arbitrary output shape
    (success, malformed, identity-mismatched, ...) with a nonzero exit code
    without needing a dedicated fixture mode for every combination.
    """
    override = os.environ.get("FAKE_KORVID_EVALS_EXIT_CODE")
    if override is None:
        return default
    return int(override)


if __name__ == "__main__":
    raise SystemExit(main())
