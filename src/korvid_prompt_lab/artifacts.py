from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json_artifact(path: Path | str, payload: Any) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = artifact_path.with_name(f"{artifact_path.name}.tmp")
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp_path.write_text(encoded, encoding="utf-8")
    os.replace(temp_path, artifact_path)
    return artifact_path
