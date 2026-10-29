"""Poller-level tests for follow_timer lifecycle actions (increment 2)."""

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
    StartTransition,
)
from autosplitter.detection import DetectionResult, State, TransitionDetector
from autosplitter.livesplit import Action, Decision, LiveSplitClient

BLACK = Color(name="black", rgb=(0, 0, 0), tolerance=15.0)
WHITE = Color(name="white", rgb=(255, 255, 255), tolerance=15.0)


def machine(n: int = 4, start: bool = False) -> TransitionDetector:
    cfg = Config(
        capture=Capture(fps=60.0, stream_url="udp://127.0.0.1:9733", width=1920, height=1080),
        roi=DEFAULT_ROI,
        detection=Detection(required_frames=1, reset_threshold=0.80, restart_quiet_seconds=3.0),
        colors={"black": BLACK, "white": WHITE},
        splits=tuple(
            Split(id=i + 1, name=chr(ord("A") + i), color="black", threshold=0.95)
            for i in range(n)
        ),
        livesplit=LiveSplit(host="127.0.0.1", port=16834),
        start_transition=StartTransition(name="Start", color="white", threshold=0.95)
        if start
        else None,
    )
    return TransitionDetector(cfg)


def feed(m: TransitionDetector, *ratios: float) -> int:
    n = 0
    for r in ratios:
        if m.feed_result(DetectionResult(int(r * 100), 100, r)) is not None:
            n += 1
    return n


@final
class Fake:
    def __init__(self, phase: str | None = "Running", index: int | None = 0, count: int | None = 4) -> None:
        self.phase: str | None = phase
        self.index: int | None = index
        self.count: int | None = count
        self.calls: list[str] = []

    def get_current_timer_phase(self) -> str | None:
        self.calls.append("phase")
        return self.phase

    def get_split_index(self) -> int | None:
        self.calls.append("index")
        return self.index

    def get_current_split_name(self) -> str | None:
        self.calls.append("name")
        return "X"

    def get_split_count(self) -> int | None:
        self.calls.append("count")
        return self.count


def do_poll(
    p: main_mod._FollowPoller,  # pyright: ignore[reportPrivateUsage]
    c: Fake,
    m: TransitionDetector,
    last: float | None = None,
) -> main_mod._FollowPollResult:  # pyright: ignore[reportPrivateUsage]
    return p.poll(cast(LiveSplitClient, cast(object, c)), m, "X", last, False)


def new(n: int = 4, loop: bool = False) -> main_mod._FollowPoller:  # pyright: ignore[reportPrivateUsage]
    return main_mod._FollowPoller(n, loop)  # pyright: ignore[reportPrivateUsage]


def test_reset_rewinds_machine_and_clears_last_split() -> None:
    p, m, c = new(), machine(), Fake(index=2)
    m.advance_to(2)
    feed(m, 0.0)
    do_poll(p, c, m)  # activity seen
    c.phase = "NotRunning"
    calls_before = len(c.calls)
    r = do_poll(p, c, m, last=5.0)
    assert c.calls[calls_before:] == ["phase"]  # NotRunning: no index/name
    assert r.message is not None and "RESET" in r.message
    assert r.clear_last_split and not r.ok and r.suppress is False
    assert m.state is State.WAITING_FOR_RESET and m.current_split_index == 0
    assert feed(m, 1.0, 1.0, 1.0) == 0  # lit split-0 color cannot fire


def test_reset_with_gate_returns_to_waiting_for_start() -> None:
    p, m, c = new(), machine(start=True), Fake(index=0)
    m.advance_to(0)
    feed(m, 0.0)
    do_poll(p, c, m)
    c.phase = "NotRunning"
    assert do_poll(p, c, m).message is not None
    assert m.state is State.WAITING_FOR_START


def test_never_started_notrunning_is_ignored() -> None:
    p, m, c = new(), machine(), Fake(phase="NotRunning")
    m.advance_to(2)
    for _ in range(3):
        assert do_poll(p, c, m).message is None
    assert m.current_split_index == 2


def test_reset_only_once() -> None:
    p, m, c = new(), machine(), Fake(index=0)
    do_poll(p, c, m)
    c.phase = "NotRunning"
    assert do_poll(p, c, m).message is not None
    assert do_poll(p, c, m).message is None


def test_reconnect_makes_idle_timer_not_a_reset() -> None:
    p, m, c = new(), machine(), Fake(index=0)
    do_poll(p, c, m)
    p.connection_lost()
    c.phase = "NotRunning"
    assert do_poll(p, c, m).message is None


def test_manual_start_leaves_gate() -> None:
    p, m, c = new(), machine(start=True), Fake(index=0)
    r = do_poll(p, c, m)
    assert r.message is not None and "timer started" in r.message
    assert not r.clear_last_split
    assert m.state is State.WAITING_FOR_RESET and m.current_split_index == 0


def test_manual_start_refused_by_machine_leaves_state() -> None:
    m = machine(start=True)
    assert main_mod._apply_decision(m, Decision(Action.MANUAL_START, 9)) is None  # pyright: ignore[reportPrivateUsage]
    assert m.waiting_for_start


def test_run_ended_two_polls_goes_done() -> None:
    p, m, c = new(), machine(), Fake(phase="Running", index=0)
    do_poll(p, c, m)  # prime: timer seen Running
    c.phase, c.index = "Ended", 4
    assert do_poll(p, c, m).message is None
    r = do_poll(p, c, m)
    assert r.message is not None and "run ended" in r.message
    assert m.finished


def test_stale_ended_never_ends_run() -> None:
    p, m, c = new(), machine(), Fake(phase="Ended", index=4)
    for _ in range(5):
        assert do_poll(p, c, m).message is None
    assert not m.finished


def test_reconnect_requires_running_again_before_run_end() -> None:
    p, m, c = new(), machine(), Fake(phase="Running", index=0)
    do_poll(p, c, m)
    p.connection_lost()
    c.phase, c.index = "Ended", 4
    for _ in range(4):
        assert do_poll(p, c, m).message is None
    assert not m.finished


def test_run_ended_refused_in_transition() -> None:
    m = machine()
    feed(m, 1.0)
    assert m.state is State.TRANSITION
    assert main_mod._apply_decision(m, Decision(Action.RUN_ENDED, 4)) is None  # pyright: ignore[reportPrivateUsage]
    assert m.state is State.TRANSITION


def test_count_mismatch_disables_run_end_and_manual_start() -> None:
    p, m, c = new(), machine(start=True), Fake(index=0, count=9)
    for _ in range(3):
        assert do_poll(p, c, m).message is None
    assert m.waiting_for_start
    p2, m2, c2 = new(), machine(), Fake(phase="Ended", index=4, count=9)
    for _ in range(3):
        assert do_poll(p2, c2, m2).message is None
    assert not m2.finished


def test_reset_works_despite_count_mismatch() -> None:
    p, m, c = new(), machine(), Fake(index=0, count=9)
    do_poll(p, c, m)
    c.phase = "NotRunning"
    assert do_poll(p, c, m).message is not None


def test_reset_while_done_needs_loop() -> None:
    for loop, rewound in ((False, False), (True, True)):
        p, m, c = new(loop=loop), machine(), Fake(phase="Ended", index=4)
        m.advance_to(4)
        do_poll(p, c, m)
        c.phase = "NotRunning"
        r = do_poll(p, c, m)
        assert (r.message is not None) is rewound
        assert m.finished is (not rewound)


def test_loop_undo_from_ended_via_skip_after_rearm() -> None:
    """Final split undone: DONE already rearmed to NORMAL idx 0 by --loop."""
    p, m, c = new(loop=True), machine(), Fake(phase="Ended", index=4)
    m.advance_to(4)
    do_poll(p, c, m)
    m.reset()  # color-quiet rearm
    p.disarm()
    c.phase, c.index = "Running", 3
    assert do_poll(p, c, m).message is None
    r = do_poll(p, c, m)
    assert r.message is not None and "SKIP/manual" in r.message
    assert m.current_split_index == 3


def test_loop_undo_from_ended_via_manual_start_with_gate() -> None:
    p, m, c = new(loop=True), machine(start=True), Fake(phase="Running", index=3)
    r = do_poll(p, c, m)
    assert r.message is not None and "timer started" in r.message
    assert m.current_split_index == 3


def test_rollback_from_done_without_loop_still_possible() -> None:
    m = machine()
    m.advance_to(4)
    assert m.rollback_to(3)
    assert not m.finished


@pytest.mark.parametrize("n", [1, 2])
def test_followed_action_message_flags(n: int) -> None:
    p, m, c = new(), machine(), Fake(index=n + 1)
    do_poll(p, c, m)
    r = do_poll(p, c, m)
    assert r.message is not None and not r.clear_last_split


def newp(n: int = 4, loop: bool = False) -> main_mod._FollowPoller:  # pyright: ignore[reportPrivateUsage]
    return main_mod._FollowPoller(n, loop, True)  # pyright: ignore[reportPrivateUsage]


class TestPausePoller:
    def test_pause_resume_flags_logs_and_no_hold_off(self) -> None:
        p, m, c = newp(), machine(), Fake(index=0)
        do_poll(p, c, m)
        assert not p.paused
        c.phase = "Paused"
        r = do_poll(p, c, m)
        assert r.message == "LiveSplit: timer PAUSED — detector frozen"
        assert p.paused and not r.touch_last_split and not r.clear_last_split
        c.phase = "Running"
        r = do_poll(p, c, m)
        assert r.message == "LiveSplit: timer RESUMED"
        assert not p.paused and not r.touch_last_split

    def test_suppress_not_cleared_by_pause(self) -> None:
        p, m, c = newp(), machine(), Fake(index=0)
        m.advance_to(2)  # expected 2 != LiveSplit index 0: suppress stays
        do_poll(p, c, m)
        c.phase = "Paused"
        r = p.poll(cast(LiveSplitClient, cast(object, c)), m, "X", None, True)
        assert r.suppress is True

    def test_pause_disabled_by_default(self) -> None:
        p, m, c = new(), machine(), Fake(index=0)
        do_poll(p, c, m)
        c.phase = "Paused"
        assert do_poll(p, c, m).message is None
        assert not p.paused

    def test_streak_cleared_so_lit_color_needs_full_debounce(self) -> None:
        cfg_m = machine()  # required_frames=1 in this helper; use streak API directly
        p, c = newp(), Fake(index=0)
        m = cfg_m
        do_poll(p, c, m)
        c.phase = "Paused"
        do_poll(p, c, m)
        # frozen: caller does not feed. On resume the machine is untouched.
        c.phase = "Running"
        do_poll(p, c, m)
        assert m.state is State.NORMAL and m.current_split_index == 0
        assert feed(m, 1.0) == 1  # exactly one split afterwards
        assert feed(m, 1.0, 1.0) == 0

    @pytest.mark.parametrize("fire", [1, 2])
    def test_pause_in_transition_or_waiting_stays_put(self, fire: int) -> None:
        p, m, c = newp(), machine(), Fake(index=0)
        do_poll(p, c, m)
        feed(m, *([1.0] * fire))
        before = m.state
        c.phase = "Paused"
        c.index = 1
        do_poll(p, c, m)
        assert p.paused
        assert m.state is before

    def test_reconnect_and_reset_clear_paused(self) -> None:
        p, m, c = newp(), machine(), Fake(index=0)
        do_poll(p, c, m)
        c.phase = "Paused"
        do_poll(p, c, m)
        p.connection_lost()
        assert not p.paused
        c.phase = "Paused"
        do_poll(p, c, m)
        assert p.paused
        p.disarm()
        assert not p.paused
        c.phase = "Running"
        do_poll(p, c, m)
        c.phase = "Paused"
        do_poll(p, c, m)
        c.phase = "NotRunning"
        do_poll(p, c, m)
        assert not p.paused


class TestThawValve:
    def _frozen(self) -> tuple[main_mod._FollowPoller, TransitionDetector, Fake]:  # pyright: ignore[reportPrivateUsage]
        p, m, c = newp(), machine(), Fake(index=0)
        do_poll(p, c, m)
        c.phase = "Paused"
        do_poll(p, c, m)
        assert p.paused
        return p, m, c

    def test_three_failures_thaw(self, capsys: pytest.CaptureFixture[str]) -> None:
        p, m, c = self._frozen()
        c.phase = None
        assert do_poll(p, c, m).message is None
        assert do_poll(p, c, m).message is None
        r = do_poll(p, c, m)
        assert r.message == "LiveSplit: no answers while paused — detector unfrozen"
        assert not p.paused and not r.touch_last_split
        _ = capsys

    def test_index_failures_thaw_too(self) -> None:
        p, m, c = self._frozen()
        c.index = None
        do_poll(p, c, m)
        do_poll(p, c, m)
        assert p.paused
        do_poll(p, c, m)
        assert not p.paused

    def test_two_failures_do_not_thaw(self) -> None:
        p, m, c = self._frozen()
        c.phase = None
        do_poll(p, c, m)
        do_poll(p, c, m)
        assert p.paused

    def test_success_resets_count(self) -> None:
        p, m, c = self._frozen()
        c.phase = None
        do_poll(p, c, m)
        do_poll(p, c, m)
        c.phase = "Paused"
        do_poll(p, c, m)  # answered
        c.phase = None
        do_poll(p, c, m)
        do_poll(p, c, m)
        assert p.paused

    def test_refreezes_after_thaw(self) -> None:
        p, m, c = self._frozen()
        c.phase = None
        for _ in range(3):
            do_poll(p, c, m)
        assert not p.paused
        c.phase = "Paused"
        r = do_poll(p, c, m)
        assert r.message == "LiveSplit: timer PAUSED — detector frozen" and p.paused

    def test_connection_lost_resets_count(self) -> None:
        p, m, c = self._frozen()
        c.phase = None
        do_poll(p, c, m)
        do_poll(p, c, m)
        p.connection_lost()
        c.phase = "Paused"
        do_poll(p, c, m)
        c.phase = None
        do_poll(p, c, m)
        do_poll(p, c, m)
        assert p.paused

    def test_not_thawed_when_not_paused(self) -> None:
        p, m, c = newp(), machine(), Fake(index=0)
        do_poll(p, c, m)
        c.phase = None
        for _ in range(5):
            assert do_poll(p, c, m).message is None
