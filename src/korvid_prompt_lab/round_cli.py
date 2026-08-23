from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .rounds import write_safe_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="korvid-grounding-report")
    parser.add_argument("--artifact-root", type=Path, required=True, help="Directory containing evaluation artifacts.")
    parser.add_argument(
        "--optimize-artifact-root",
        type=Path,
        help="Optional directory containing optimization-summary.json and best-candidate.yaml artifacts.",
    )
    parser.add_argument("--safe-output", type=Path, required=True, help="New directory for allowlisted safe evidence.")
    parser.add_argument("--prompt-lab-revision", required=True, help="Prompt Lab git revision for the round.")
    parser.add_argument("--korvid-revision", required=True, help="Korvid git revision for the round.")
    parser.add_argument("--workflow-run-url", required=True, help="GitHub Actions workflow run URL.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_root = write_safe_evidence(
        args.artifact_root,
        args.safe_output,
        optimize_artifact_root=args.optimize_artifact_root,
        prompt_lab_revision=args.prompt_lab_revision,
        korvid_revision=args.korvid_revision,
        workflow_run_url=args.workflow_run_url,
    )
    print(output_root / "round-summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
