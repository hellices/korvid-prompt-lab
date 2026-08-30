from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from korvid.agent.profiles import build_profile as installed_build_profile

from korvid_prompt_lab.baseline import (
    KORVID_DISTRIBUTION_NAME,
    PROFILE_NAMES,
    build_baseline_candidate,
    korvid_distribution_version,
    render_baseline_yaml,
    write_baseline_candidate,
)
from korvid_prompt_lab.cli import main


def _installed_prompt(profile: str) -> str:
    # Overrides are never applied by korvid-baseline, and readonly/resize_supported
    # do not affect system_prompt composition, so any fixed values reproduce the
    # exact shipped prompt used at runtime.
    return installed_build_profile(profile, readonly=True, resize_supported=False).system_prompt


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(args)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_profile_names_are_small_and_full() -> None:
    assert PROFILE_NAMES == ("full", "small")


@pytest.mark.parametrize("profile", ["small", "full"])
def test_build_baseline_candidate_matches_installed_profile_prompt(profile: str) -> None:
    candidate = build_baseline_candidate(profile)

    assert candidate.schema_version == 1
    assert candidate.candidate_id == f"korvid-baseline-{profile}"
    assert candidate.components == {"system": _installed_prompt(profile)}


def test_build_baseline_candidate_records_metadata_without_affecting_fingerprint() -> None:
    candidate = build_baseline_candidate("small")

    assert candidate.metadata == {
        "korvid_version": korvid_distribution_version(),
        "profile": "small",
    }

    bare = build_baseline_candidate("small")
    # Metadata is recorded identically, but fingerprint must only ever depend on
    # schema_version/candidate_id/components per Candidate.fingerprint semantics.
    assert candidate.fingerprint == bare.fingerprint


def test_build_baseline_candidate_fingerprint_is_stable_across_calls() -> None:
    first = build_baseline_candidate("small")
    second = build_baseline_candidate("small")

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != build_baseline_candidate("full").fingerprint


def test_build_baseline_candidate_rejects_invalid_profile() -> None:
    with pytest.raises(ValueError, match="profile"):
        build_baseline_candidate("medium")


def test_build_baseline_candidate_fails_closed_on_blank_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BlankProfile:
        system_prompt = "   "

    monkeypatch.setattr(
        "korvid_prompt_lab.baseline.build_profile",
        lambda name, *, readonly, resize_supported: _BlankProfile(),
    )

    with pytest.raises(ValueError, match="blank"):
        build_baseline_candidate("small")


def test_build_baseline_candidate_fails_closed_when_package_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_name: str) -> str:
        raise PackageNotFoundError(KORVID_DISTRIBUTION_NAME)

    monkeypatch.setattr("korvid_prompt_lab.baseline._distribution_version", _raise)

    with pytest.raises(ValueError, match="metadata"):
        build_baseline_candidate("small")


def test_render_baseline_yaml_is_exact(tmp_path: Path) -> None:
    candidate = build_baseline_candidate("small")

    content = render_baseline_yaml(candidate)

    assert content == yaml.safe_dump(
        {
            "schema_version": 1,
            "candidate_id": "korvid-baseline-small",
            "components": {"system": _installed_prompt("small")},
            "metadata": {
                "korvid_version": korvid_distribution_version(),
                "profile": "small",
            },
        },
        sort_keys=False,
        allow_unicode=True,
    )


def test_write_baseline_candidate_rejects_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "baseline.yaml"
    output.write_text("pre-existing", encoding="utf-8")
    candidate = build_baseline_candidate("small")

    with pytest.raises(FileExistsError):
        write_baseline_candidate(candidate, output)

    assert output.read_text(encoding="utf-8") == "pre-existing"


def test_write_baseline_candidate_rejects_symlink_output(tmp_path: Path) -> None:
    real_target = tmp_path / "elsewhere.yaml"
    real_target.write_text("elsewhere", encoding="utf-8")
    output = tmp_path / "baseline.yaml"
    output.symlink_to(real_target)
    candidate = build_baseline_candidate("small")

    with pytest.raises(FileExistsError):
        write_baseline_candidate(candidate, output)

    assert real_target.read_text(encoding="utf-8") == "elsewhere"


def test_write_baseline_candidate_rejects_dangling_symlink_output(tmp_path: Path) -> None:
    missing_target = tmp_path / "missing.yaml"
    output = tmp_path / "baseline.yaml"
    output.symlink_to(missing_target)
    candidate = build_baseline_candidate("small")

    with pytest.raises(FileExistsError):
        write_baseline_candidate(candidate, output)

    assert not missing_target.exists()


def test_write_baseline_candidate_writes_exact_yaml(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "baseline.yaml"
    candidate = build_baseline_candidate("full")

    written_path = write_baseline_candidate(candidate, output)

    assert written_path == output
    assert output.read_text(encoding="utf-8") == render_baseline_yaml(candidate)


def test_write_baseline_candidate_is_not_visible_before_fsync(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "baseline.yaml"
    candidate = build_baseline_candidate("small")
    real_fsync = os.fsync

    def assert_destination_is_private(fd: int) -> None:
        assert not output.exists()
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", assert_destination_is_private)

    write_baseline_candidate(candidate, output)

    assert output.read_text(encoding="utf-8") == render_baseline_yaml(candidate)
    assert list(tmp_path.iterdir()) == [output]


def test_write_baseline_cleanup_failure_does_not_mask_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "baseline.yaml"
    candidate = build_baseline_candidate("small")

    monkeypatch.setattr(
        os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("fsync failed"))
    )
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda _path, **_kwargs: (_ for _ in ()).throw(OSError("cleanup failed")),
    )

    with pytest.raises(OSError, match="fsync failed") as excinfo:
        write_baseline_candidate(candidate, output)

    assert any("cleanup failed" in note for note in excinfo.value.__notes__)
    assert not output.exists()


def test_cli_korvid_baseline_writes_small_profile_candidate(tmp_path: Path) -> None:
    output = tmp_path / "korvid-baseline-small.yaml"

    exit_code, stdout, stderr = _run_cli(
        ["korvid-baseline", "--profile", "small", "--output", str(output)]
    )

    assert exit_code == 0, stderr
    assert "korvid-baseline-small" in stdout

    candidate = build_baseline_candidate("small")
    assert output.read_text(encoding="utf-8") == render_baseline_yaml(candidate)


def test_cli_korvid_baseline_writes_full_profile_candidate(tmp_path: Path) -> None:
    output = tmp_path / "korvid-baseline-full.yaml"

    exit_code, _stdout, stderr = _run_cli(
        ["korvid-baseline", "--profile", "full", "--output", str(output)]
    )

    assert exit_code == 0, stderr
    candidate = build_baseline_candidate("full")
    assert output.read_text(encoding="utf-8") == render_baseline_yaml(candidate)


def test_cli_korvid_baseline_rejects_invalid_profile(tmp_path: Path) -> None:
    output = tmp_path / "baseline.yaml"

    with pytest.raises(SystemExit) as excinfo:
        main(["korvid-baseline", "--profile", "bogus", "--output", str(output)])

    assert excinfo.value.code != 0
    assert not output.exists()


def test_cli_korvid_baseline_rejects_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "baseline.yaml"
    output.write_text("pre-existing", encoding="utf-8")

    exit_code, _stdout, stderr = _run_cli(
        ["korvid-baseline", "--profile", "small", "--output", str(output)]
    )

    assert exit_code != 0
    assert "korvid-baseline failed" in stderr
    assert output.read_text(encoding="utf-8") == "pre-existing"


def test_cli_korvid_baseline_rejects_symlink_output(tmp_path: Path) -> None:
    real_target = tmp_path / "elsewhere.yaml"
    real_target.write_text("elsewhere", encoding="utf-8")
    output = tmp_path / "baseline.yaml"
    output.symlink_to(real_target)

    exit_code, _stdout, _stderr = _run_cli(
        ["korvid-baseline", "--profile", "small", "--output", str(output)]
    )

    assert exit_code != 0
    assert real_target.read_text(encoding="utf-8") == "elsewhere"
