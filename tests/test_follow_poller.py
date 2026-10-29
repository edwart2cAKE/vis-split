"""Tests for main._FollowPoller: consecutive polls, split-count handling."""

from __future__ import annotations

from typing import cast, final

import pytest

from autosplitter import main as main_mod
from autosplitter.config import (
    DEFAULT_ROI,
    Capture,
    Color,
    Config,
    Detection,
    LiveSplit,
    Split,
)
from autosplitter.detection import TransitionDetector
from autosplitter.livesplit import LiveSplitClient

BLACK = Color(name="black", rgb=(0, 0, 0), tolerance=15.0)


def machine(n: int) -> TransitionDetector:
    cfg = Config(
        capture=Capture(fps=60.0, stream_url="udp://127.0.0.1:9733", width=1920, height=1080),
        roi=DEFAULT_ROI,
        detection=Detection(required_frames=1, reset_threshold=0.80, restart_quiet_seconds=3.0),
        colors={"black": BLACK},
        splits=tuple(
            Split(id=i + 1, name=chr(ord("A") + i), color="black", threshold=0.95)
            for i in range(n)
        ),
        livesplit=LiveSplit(host="127.0.0.1", port=16834),
    )
    return TransitionDetector(cfg)


@final
class Fake:
    def __init__(
        self,
        phase: str | None = "Running",
        index: int | None = 2,
        count: int | None = 5,
    ) -> None:
        self.phase: str | None = phase
        self.index: int | None = index
        self.count: int | None = count

    def get_current_timer_phase(self) -> str | None:
        return self.phase

    def get_split_index(self) -> int | None:
        return self.index

    def get_current_split_name(self) -> str | None:
        return "X"

    def get_split_count(self) -> int | None:
        return self.count


def do_poll(p: main_mod._FollowPoller, c: Fake, m: TransitionDetector) -> main_mod._FollowPollResult:  # pyright: ignore[reportPrivateUsage]
    return p.poll(cast(LiveSplitClient, cast(object, c)), m, "X", None, False)


def new(n: int = 5) -> main_mod._FollowPoller:  # pyright: ignore[reportPrivateUsage]
    return main_mod._FollowPoller(n)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("failure", ["phase", "index"])
def test_failed_poll_breaks_consecutive_run(failure: str) -> None:
    p, m, c = new(), machine(5), Fake()
    assert do_poll(p, c, m).message is None  # candidate 1
    if failure == "phase":
        c.phase = None
    else:
        c.index = None
    assert not do_poll(p, c, m).ok
    c.phase, c.index = "Running", 2
    assert do_poll(p, c, m).message is None  # restarted: count 1 again
    assert m.current_split_index == 0
    assert do_poll(p, c, m).message is not None  # two consecutive
    assert m.current_split_index == 2


def test_two_consecutive_polls_still_skip() -> None:
    p, m, c = new(), machine(5), Fake()
    do_poll(p, c, m)
    assert do_poll(p, c, m).message is not None


def test_count_unavailable_notice_once_and_skip_kept(
    capsys: pytest.CaptureFixture[str],
) -> None:
    p, m, c = new(), machine(5), Fake(count=None)
    do_poll(p, c, m)
    out = capsys.readouterr().out
    assert out.count("split count unavailable") == 1
    r = do_poll(p, c, m)
    assert "split count unavailable" not in capsys.readouterr().out
    assert r.message is not None  # skip still works


def test_count_unavailable_notice_again_after_reconnect(
    capsys: pytest.CaptureFixture[str],
) -> None:
    p, m, c = new(), machine(5), Fake(count=None)
    do_poll(p, c, m)
    p.connection_lost()
    do_poll(p, c, m)
    assert capsys.readouterr().out.count("split count unavailable") == 2


def test_count_mismatch_disables_skip_and_reconnect_reenables(
    capsys: pytest.CaptureFixture[str],
) -> None:
    p, m, c = new(), machine(5), Fake(count=7)
    for _ in range(3):
        assert do_poll(p, c, m).message is None
    assert "skip-forward following disabled" in capsys.readouterr().out
    assert m.current_split_index == 0
    p.connection_lost()
    c.count = 5
    do_poll(p, c, m)
    assert do_poll(p, c, m).message is not None
