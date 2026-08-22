#!/usr/bin/env python3
"""A real, nested process tree that outlives a naive single-PID kill.

Used to prove that a bridge timeout tears down every descendant. Each level
records its own PID, then spawns the next level and waits for it; the deepest
level sleeps and only then writes a response, so a surviving descendant is
observable both as a live PID and as a late ``response.json``.

Stdlib only, so it runs both as ``KORVID_UV_BIN`` (an executable stand-in for
``uv``) and as a campaign ``serving.command`` launcher.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

#: How many child processes to spawn below this one.
DEPTH_ENV = "FAKE_TREE_DEPTH"
#: File every level appends "<level>:<pid>" to.
PID_FILE_ENV = "FAKE_TREE_PID_FILE"
#: Set by each level for the next; absent means "I am the top level".
LEVEL_ENV = "FAKE_TREE_LEVEL"
#: How long the deepest level sleeps before writing its late response.
SLEEP_ENV = "FAKE_TREE_SLEEP"
#: When set, the deepest level ignores SIGTERM so only SIGKILL can stop it.
IGNORE_SIGTERM_ENV = "FAKE_TREE_IGNORE_SIGTERM"


def _record(pid_file: Path, label: str, pid: int) -> None:
    with pid_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{label}:{pid}\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = sys.argv[1:]
    response_path = Path(args[args.index("--response") + 1])
    pid_file = Path(os.environ[PID_FILE_ENV])
    depth = int(os.environ.get(DEPTH_ENV, "2"))
    level = int(os.environ.get(LEVEL_ENV, "0"))

    if level == 0:
        _record(pid_file, "parent", os.getppid())
    _record(pid_file, f"level-{level}", os.getpid())

    if level >= depth:
        if os.environ.get(IGNORE_SIGTERM_ENV):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(float(os.environ.get(SLEEP_ENV, "30")))
        response_path.write_text('{"late": true}\n', encoding="utf-8")
        return 0

    child = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), *args],
        env={**os.environ, LEVEL_ENV: str(level + 1)},
        shell=False,
    )
    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
