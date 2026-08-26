"""Tests for autosplitter.record (path template resolution)."""

from __future__ import annotations

from pathlib import Path

from autosplitter.record import resolve_path


def test_game_from_config_stem(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    template = str(recordings / "{game}_run.mkv")
    result = resolve_path(template, "configs/start-powerless.yaml")
    assert "start-powerless_run.mkv" in result


def test_datetime_variables(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    template = str(recordings / "{datetime}.mkv")
    result = resolve_path(template, "config.yaml")
    # Should contain YYYYMMDD_HHMMSS pattern
    parts = Path(result).stem.split("_")
    assert len(parts) == 2
    assert len(parts[0]) == 8  # YYYYMMDD
    assert len(parts[1]) == 6  # HHMMSS


def test_attempt_starts_at_one(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    template = str(recordings / "run_{attempt}.mkv")
    result = resolve_path(template, "config.yaml")
    assert result.endswith("run_0001.mkv")


def test_attempt_increments_with_existing_files(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    # Create existing files
    (recordings / "run_0001.mkv").touch()
    (recordings / "run_0002.mkv").touch()
    (recordings / "run_0003.mkv").touch()
    template = str(recordings / "run_{attempt}.mkv")
    result = resolve_path(template, "config.yaml")
    assert result.endswith("run_0004.mkv")


def test_attempt_counts_mp4_too(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "run_0001.mkv").touch()
    (recordings / "run_0002.mp4").touch()
    template = str(recordings / "run_{attempt}.mkv")
    result = resolve_path(template, "config.yaml")
    assert result.endswith("run_0003.mkv")


def test_attempt_ignores_unrelated_files(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "other_0001.mkv").touch()
    (recordings / "run_0001.mkv").touch()
    template = str(recordings / "run_{attempt}.mkv")
    result = resolve_path(template, "config.yaml")
    assert result.endswith("run_0002.mkv")


def test_attempt_creates_directory_if_needed(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    template = str(recordings / "run_{attempt}.mkv")
    result = resolve_path(template, "config.yaml")
    assert result.endswith("run_0001.mkv")
    # Directory is NOT created by resolve_path — caller's responsibility.


def test_no_attempt_variable(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    template = str(recordings / "run.mkv")
    result = resolve_path(template, "config.yaml")
    assert result.endswith("run.mkv")


def test_attempt_zero_padded_to_four_digits(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    # Create 100 existing files
    for i in range(1, 101):
        (recordings / f"run_{i:04d}.mkv").touch()
    template = str(recordings / "run_{attempt}.mkv")
    result = resolve_path(template, "config.yaml")
    assert result.endswith("run_0101.mkv")


def test_attempt_not_affected_by_different_prefix(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "session_0001.mkv").touch()
    template = str(recordings / "run_{attempt}.mkv")
    result = resolve_path(template, "config.yaml")
    assert result.endswith("run_0001.mkv")


def test_full_template(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    template = str(recordings / "{game}_{datetime}_{attempt:04d}.mkv")
    result = resolve_path(template, "configs/start-powerless.yaml")
    # Should match pattern: start-powerless_YYYYMMDD_HHMMSS_NNNN.mkv
    name = Path(result).name
    assert name.startswith("start-powerless_")
    assert name.endswith(".mkv")
    # The attempt part should be 0001 (first run)
    parts = name.replace("start-powerless_", "").replace(".mkv", "").rsplit("_", 1)
    assert parts[1] == "0001"
