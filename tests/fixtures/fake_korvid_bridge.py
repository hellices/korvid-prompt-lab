from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def _candidate_fingerprint(candidate: dict[str, object]) -> str:
    payload = {
        "schema_version": candidate["schema_version"],
        "candidate_id": candidate["candidate_id"],
        "components": candidate["components"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _case_tags(case_id: str) -> set[str]:
    if "[" not in case_id or "]" not in case_id:
        return set()
    body = case_id.split("[", 1)[1].rsplit("]", 1)[0]
    return {tag.strip() for tag in body.split(",") if tag.strip()}


TUNED_MARKER = "korvid-tuned"

#: Protocol 2 added the typed ``execution_mode`` every response must declare.
PROTOCOL_VERSION = 2


def _default_grade(candidate: dict[str, object]) -> dict[str, object]:
    """Grade a healthy, fully completed run, rewarding candidates that carry the tuned marker."""
    components = candidate.get("components")
    tuned = isinstance(components, dict) and any(
        isinstance(value, str) and TUNED_MARKER in value for value in components.values()
    )
    if tuned:
        return {"completion": 1.0, "verification": 0.9, "efficiency": 0.8, "hard_failures": []}
    return {"completion": 1.0, "verification": 0.8, "efficiency": 0.7, "hard_failures": []}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    tags = _case_tags(request["case"]["case_id"])
    response_path = Path(args.response)
    fingerprint = _candidate_fingerprint(request["candidate"])
    request_identity = {
        "case_id": request["case"]["case_id"],
        "template_id": request["case"]["template_id"],
        "model": request["case"]["model"],
        "repetition": request["case"]["repetition"],
        "seed": request["case"]["seed"],
    }

    if "timeout" in tags:
        time.sleep(0.5)
        return 0

    if "non-zero-exit" in tags:
        return 7

    if "missing-output" in tags:
        return 0

    if "malformed-json" in tags:
        response_path.write_text("{not json", encoding="utf-8")
        return 0

    if "wrong-shape" in tags:
        response_path.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
        return 0

    if "invalid-utf8" in tags:
        response_path.write_bytes(b"\x80\x81\x82")
        return 0

    if "response-directory" in tags:
        response_path.mkdir(parents=True, exist_ok=True)
        return 0

    protocol_version = PROTOCOL_VERSION + 1 if "protocol-mismatch" in tags else PROTOCOL_VERSION
    candidate_fingerprint = fingerprint + "-wrong" if "fingerprint-mismatch" in tags else fingerprint

    if "systemic-status" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "system_failure",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": None,
            "answer": "",
            "journal": {"checkpoints": []},
            "usage": {},
            "error": "bridge backend unavailable",
        }
    elif "model-failure" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "model_failure",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": None,
            "answer": "",
            "journal": {"checkpoints": ["dispatch"]},
            "usage": {"completion_tokens": 0},
            "error": "model returned no tokens",
        }
    elif "model-failure-with-grade" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "model_failure",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": {
                "completion": 0.1,
                "verification": 0.2,
                "efficiency": 0.3,
                "hard_failures": [],
            },
            "answer": "",
            "journal": {"checkpoints": ["dispatch"]},
            "usage": {"completion_tokens": 0},
            "error": "model returned no tokens",
        }
    elif "model-failure-missing-grade" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "model_failure",
            "candidate_fingerprint": candidate_fingerprint,
            "answer": "",
            "journal": {"checkpoints": ["dispatch"]},
            "usage": {"completion_tokens": 0},
            "error": "model returned no tokens",
        }
    elif "bad-journal" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": {
                "completion": 0.9,
                "verification": 0.8,
                "efficiency": 0.7,
                "hard_failures": [],
            },
            "answer": "verified",
            "journal": "not-a-mapping",
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    elif "bad-usage" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": {
                "completion": 0.9,
                "verification": 0.8,
                "efficiency": 0.7,
                "hard_failures": [],
            },
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": "not-a-mapping",
            "error": None,
        }
    elif "bad-error" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "model_failure",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": None,
            "answer": "",
            "journal": {"checkpoints": []},
            "usage": {},
            "error": {"message": "wrong-shape"},
        }
    elif "missing-grade" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": None,
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    elif "bad-grade" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": ["wrong-shape"],
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    elif "bool-grade" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": {
                "completion": True,
                "verification": 0.8,
                "efficiency": 0.7,
                "hard_failures": [],
            },
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    elif "bad-protocol-type" in tags:
        payload = {
            "protocol_version": True,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": {
                "completion": 0.9,
                "verification": 0.8,
                "efficiency": 0.7,
                "hard_failures": [],
            },
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    elif "bad-status-type" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": True,
            "candidate_fingerprint": candidate_fingerprint,
            "grade": {
                "completion": 0.9,
                "verification": 0.8,
                "efficiency": 0.7,
                "hard_failures": [],
            },
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    elif "bad-fingerprint-type" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": {"value": candidate_fingerprint},
            "grade": {
                "completion": 0.9,
                "verification": 0.8,
                "efficiency": 0.7,
                "hard_failures": [],
            },
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    elif "missing-answer" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": {
                "completion": 0.9,
                "verification": 0.8,
                "efficiency": 0.7,
                "hard_failures": [],
            },
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    elif "missing-journal" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": {
                "completion": 0.9,
                "verification": 0.8,
                "efficiency": 0.7,
                "hard_failures": [],
            },
            "answer": "verified",
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    elif "missing-usage" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": {
                "completion": 0.9,
                "verification": 0.8,
                "efficiency": 0.7,
                "hard_failures": [],
            },
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "error": None,
        }
    elif "extra-response-field" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": {
                "completion": 0.9,
                "verification": 0.8,
                "efficiency": 0.7,
                "hard_failures": [],
            },
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": {"completion_tokens": 12},
            "error": None,
            "unexpected": True,
        }
    elif "extra-grade-field" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": {
                "completion": 0.9,
                "verification": 0.8,
                "efficiency": 0.7,
                "hard_failures": [],
                "unexpected": True,
            },
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    elif "partial-completion" in tags:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": {
                "completion": 0.0,
                "verification": 0.9,
                "efficiency": 0.9,
                "hard_failures": [],
            },
            "answer": "dispatched without finishing the operation",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": {"completion_tokens": 12},
            "error": None,
        }
    elif "flaky-after-2" in tags and int(request["case"]["repetition"]) > 2:
        payload = {
            "protocol_version": protocol_version,
            "status": "model_failure",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": None,
            "answer": "",
            "journal": {"checkpoints": ["dispatch"]},
            "usage": {"completion_tokens": 0},
            "error": "model lost the postcondition after two repetitions",
        }
    else:
        payload = {
            "protocol_version": protocol_version,
            "status": "completed",
            "candidate_fingerprint": candidate_fingerprint,
            "grade": _default_grade(request["candidate"]),
            "answer": "verified",
            "journal": {"checkpoints": ["dispatch", "verify"]},
            "usage": {"completion_tokens": 12},
            "error": None,
        }

    if isinstance(payload, dict):
        journal = payload.get("journal")
        if isinstance(journal, dict):
            journal["model_endpoint"] = request["runtime"].get("model_endpoint")
        if "missing-execution-mode" in tags:
            payload.pop("execution_mode", None)
        elif "bad-execution-mode-type" in tags:
            payload["execution_mode"] = 7
        elif "unknown-execution-mode" in tags:
            payload["execution_mode"] = "simulated"
        elif "scripted-mode" in tags:
            payload["execution_mode"] = "scripted"
        else:
            payload["execution_mode"] = "live"
        if "identity-mismatch" in tags:
            payload["request_identity"] = {
                **request_identity,
                "seed": request_identity["seed"] + 1,
            }
        else:
            payload.setdefault("request_identity", request_identity)

    response_path.write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
