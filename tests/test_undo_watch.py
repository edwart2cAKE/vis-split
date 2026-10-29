"""Tests for autosplitter.livesplit.sync.UndoWatch."""

from __future__ import annotations

from autosplitter.livesplit.sync import CONFIRM_POLLS, HOLDOFF_SECONDS, UndoWatch


def obs(
    watch: UndoWatch,
    *,
    phase: str | None = "Running",
    ls: int | None = 0,
    expected: int = 1,
    now: float = 100.0,
    last: float | None = 0.0,
    suppress: bool = False,
    ok: bool = True,
) -> int | None:
    return watch.observe(phase, ls, expected, now, last, suppress, ok)


def armed() -> UndoWatch:
    watch = UndoWatch()
    assert obs(watch, ls=0, expected=0) is None  # in-step Running poll arms
    return watch


def test_constants() -> None:
    assert CONFIRM_POLLS == 2
    assert HOLDOFF_SECONDS == 3.0


def test_one_poll_insufficient_two_sufficient() -> None:
    watch = armed()
    assert obs(watch) is None
    assert obs(watch) == 0


def test_count_restarts_after_confirmation() -> None:
    watch = armed()
    obs(watch)
    assert obs(watch) == 0
    assert obs(watch) is None


def test_ls_index_change_resets_count() -> None:
    watch = armed()
    assert obs(watch, ls=0, expected=2) is None
    assert obs(watch, ls=1, expected=2) is None
    assert obs(watch, ls=1, expected=2) == 1


def test_expected_change_resets_count() -> None:
    watch = armed()
    assert obs(watch, ls=0, expected=1) is None
    assert obs(watch, ls=0, expected=2) is None
    assert obs(watch, ls=0, expected=2) == 0


def test_non_candidate_poll_resets_count() -> None:
    watch = armed()
    obs(watch)
    assert obs(watch, ls=1) is None  # agrees: not a candidate
    assert obs(watch) is None
    assert obs(watch) == 0


def test_holdoff() -> None:
    watch = armed()
    assert obs(watch, now=101.0, last=100.0) is None
    assert obs(watch, now=102.9, last=100.0) is None
    assert obs(watch, now=103.0, last=100.0) is None  # first candidate poll
    assert obs(watch, now=105.0, last=100.0) == 0


def test_no_last_split_time_means_no_holdoff() -> None:
    watch = armed()
    obs(watch, last=None)
    assert obs(watch, last=None) == 0


def test_suppress_blocks() -> None:
    watch = armed()
    for _ in range(5):
        assert obs(watch, suppress=True) is None


def test_phase_not_running_blocks() -> None:
    watch = armed()
    for phase in ("Paused", "Ended", "NotRunning", None):
        for _ in range(3):
            assert obs(watch, phase=phase) is None


def test_ls_ahead_or_equal_ignored() -> None:
    watch = armed()
    for _ in range(3):
        assert obs(watch, ls=2, expected=1) is None
        assert obs(watch, ls=1, expected=1) is None


def test_unknown_or_negative_index_ignored() -> None:
    watch = armed()
    for _ in range(3):
        assert obs(watch, ls=None) is None
        assert obs(watch, ls=-1) is None


def test_rollback_not_ok_blocks() -> None:
    watch = armed()
    for _ in range(3):
        assert obs(watch, ok=False) is None


def test_unarmed_watch_never_confirms() -> None:
    watch = UndoWatch()
    for _ in range(4):
        assert obs(watch) is None


def test_arming_poll_ignores_holdoff_and_suppress() -> None:
    watch = UndoWatch()
    assert obs(watch, ls=1, expected=1, now=1.0, last=0.9, suppress=True) is None
    assert obs(watch, ls=0, expected=1) is None
    assert obs(watch, ls=0, expected=1) == 0


def test_non_running_phase_disarms() -> None:
    watch = armed()
    assert obs(watch, phase="NotRunning", ls=-1) is None
    assert obs(watch) is None
    assert obs(watch) is None  # still unarmed


def test_rearm_after_non_running() -> None:
    watch = armed()
    obs(watch, phase="Paused")
    assert obs(watch, ls=1, expected=1) is None  # re-arms
    assert obs(watch) is None
    assert obs(watch) == 0


def test_disarm_clears_arming_and_count() -> None:
    watch = armed()
    obs(watch)
    watch.disarm()
    assert obs(watch) is None
    assert obs(watch) is None
