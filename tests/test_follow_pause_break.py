"""Adversarial tests for follow_pause (increment 3)."""

from __future__ import annotations

from typing import cast, final

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
from autosplitter.livesplit import Action, LiveSplitClient, TimerFollower

BLACK = Color(name="black", rgb=(0, 0, 0), tolerance=15.0)
WHITE = Color(name="white", rgb=(255, 255, 255), tolerance=15.0)


def machine(n: int = 3, frames: int = 3, start: bool = False) -> TransitionDetector:
    cfg = Config(
        capture=Capture(fps=60.0, stream_url="udp://127.0.0.1:9733", width=1920, height=1080),
        roi=DEFAULT_ROI,
        detection=Detection(required_frames=frames, reset_threshold=0.80, restart_quiet_seconds=3.0),
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


@final
class Fake:
    def __init__(self, phase: str | None = "Running", index: int | None = 0) -> None:
        self.phase: str | None = phase
        self.index: int | None = index

    def get_current_timer_phase(self) -> str | None:
        return self.phase

    def get_split_index(self) -> int | None:
        return self.index

    def get_current_split_name(self) -> str | None:
        return "X"

    def get_split_count(self) -> int | None:
        return 3


def poller(loop: bool = False) -> main_mod._FollowPoller:  # pyright: ignore[reportPrivateUsage]
    return main_mod._FollowPoller(3, loop, True)  # pyright: ignore[reportPrivateUsage]


def poll(
    p: main_mod._FollowPoller,  # pyright: ignore[reportPrivateUsage]
    c: Fake,
    m: TransitionDetector,
) -> main_mod._FollowPollResult:  # pyright: ignore[reportPrivateUsage]
    return p.poll(cast(LiveSplitClient, cast(object, c)), m, "X", None, False)


def frames(
    p: main_mod._FollowPoller,  # pyright: ignore[reportPrivateUsage]
    m: TransitionDetector,
    *ratios: float,
) -> int:
    """Mimic _run_live's frame loop: no feed while p.paused."""
    n = 0
    for r in ratios:
        if p.paused:
            continue
        if m.feed_result(DetectionResult(int(r * 100), 100, r)) is not None:
            n += 1
    return n


# ---- end-to-end harness (poller + real machine + frame loop) ----------------


def test_no_split_while_paused_and_full_debounce_after_resume() -> None:
    p, m, c = poller(), machine(frames=3), Fake(index=0)
    poll(p, c, m)
    assert frames(p, m, 1.0, 1.0) == 0  # partial streak 2/3
    c.phase = "Paused"
    poll(p, c, m)
    assert frames(p, m, *([1.0] * 20)) == 0  # frozen: lit colour ignored
    c.phase = "Running"
    poll(p, c, m)
    assert frames(p, m, 1.0, 1.0) == 0  # restarts from zero
    assert frames(p, m, 1.0) == 1
    assert frames(p, m, *([1.0] * 10)) == 0  # no duplicate


def test_pause_while_transition_then_resume_no_duplicate() -> None:
    p, m, c = poller(), machine(frames=2), Fake(index=0)
    poll(p, c, m)
    assert frames(p, m, 1.0, 1.0) == 1
    assert m.state is State.TRANSITION
    c.phase = "Paused"
    c.index = 1
    poll(p, c, m)
    c.phase = "Running"
    poll(p, c, m)
    assert frames(p, m, 1.0, 1.0, 1.0) == 0
    assert m.state is State.WAITING_FOR_RESET


def test_pause_during_waiting_for_reset_then_clear_and_next_split() -> None:
    p, m, c = poller(), machine(frames=2), Fake(index=0)
    poll(p, c, m)
    frames(p, m, 1.0, 1.0, 0.9)
    assert m.state is State.WAITING_FOR_RESET
    c.phase, c.index = "Paused", 1
    poll(p, c, m)
    frames(p, m, 0.0, 0.0)  # clearing frames ignored while frozen
    assert m.state is State.WAITING_FOR_RESET
    c.phase = "Running"
    poll(p, c, m)
    frames(p, m, 0.0)
    assert m.state is State.NORMAL and m.current_split_index == 1
    assert frames(p, m, 1.0, 1.0) == 1


def test_flapping_pause_resume_never_splits_with_partial_streaks() -> None:
    p, m, c = poller(), machine(frames=3), Fake(index=0)
    poll(p, c, m)
    total = 0
    for _ in range(10):
        total += frames(p, m, 1.0, 1.0)  # 2 of 3 each time
        c.phase = "Paused"
        poll(p, c, m)
        c.phase = "Running"
        poll(p, c, m)
    assert total == 0


# ---- lost PAUSE/RESUME when the index query fails ---------------------------


def test_pause_poll_with_failed_index_still_freezes_and_clears_streak() -> None:
    """follower flips to paused on the first Paused poll; if get_split_index()
    fails that poll, poll() returns early via _clear_pending, discarding the
    PAUSE decision: no clear_streak, mirror flag not updated."""
    p, m, c = poller(), machine(frames=3), Fake(index=0)
    poll(p, c, m)
    frames(p, m, 1.0, 1.0)
    c.phase, c.index = "Paused", None
    poll(p, c, m)
    assert p.paused, "frozen flag lost when index query fails on the PAUSE poll"
    c.index = 0
    poll(p, c, m)
    c.phase = "Running"
    poll(p, c, m)
    assert frames(p, m, 1.0) == 0, "stale partial streak survived the pause"


def test_resume_poll_with_failed_index_still_unfreezes() -> None:
    p, m, c = poller(), machine(frames=3), Fake(index=0)
    poll(p, c, m)
    c.phase = "Paused"
    poll(p, c, m)
    assert p.paused
    c.phase, c.index = "Running", None
    poll(p, c, m)
    assert not p.paused, "detector stays frozen after RESUME swallowed by failed index"


def test_resume_with_failed_index_then_ok_restarts_debounce() -> None:
    p, m, c = poller(), machine(frames=3), Fake(index=0)
    poll(p, c, m)
    c.phase = "Paused"
    poll(p, c, m)
    c.phase, c.index = "Running", None
    poll(p, c, m)
    c.index = 0
    poll(p, c, m)
    assert frames(p, m, 1.0, 1.0) == 0


# ---- stuck-frozen escape hatches -------------------------------------------


def test_connection_lost_unfreezes() -> None:
    p, m, c = poller(), machine(), Fake(index=0)
    poll(p, c, m)
    c.phase = "Paused"
    poll(p, c, m)
    p.connection_lost()
    assert not p.paused
    c.phase = None
    poll(p, c, m)
    assert not p.paused


def test_reconnect_while_still_paused_gives_fresh_pause() -> None:
    p, m, c = poller(), machine(), Fake(index=0)
    poll(p, c, m)
    c.phase = "Paused"
    poll(p, c, m)
    p.connection_lost()
    r = poll(p, c, m)
    assert p.paused
    assert r.message == "LiveSplit: timer PAUSED — detector frozen"


def test_loop_rearm_disarm_while_paused_then_still_paused_refreezes() -> None:
    p, m, c = poller(loop=True), machine(), Fake(index=0)
    poll(p, c, m)
    c.phase = "Paused"
    poll(p, c, m)
    p.disarm()  # what the --loop rearm does
    assert not p.paused
    r = poll(p, c, m)  # timer is still Paused: freezes again, once
    assert p.paused and r.message is not None
    assert poll(p, c, m).message is None


def test_notrunning_while_paused_resets_and_unfreezes() -> None:
    p, m, c = poller(), machine(), Fake(index=0)
    poll(p, c, m)
    c.phase = "Paused"
    poll(p, c, m)
    c.phase = "NotRunning"
    r = poll(p, c, m)
    assert r.message is not None and "RESET" in r.message
    assert not p.paused


def test_none_phase_keeps_frozen_state_unchanged() -> None:
    p, m, c = poller(), machine(), Fake(index=0)
    poll(p, c, m)
    c.phase = "Paused"
    poll(p, c, m)
    c.phase = None
    # Below the safety-valve threshold (3 consecutive failures) the state
    # is kept; the valve itself is covered in test_follow_lifecycle.py.
    for _ in range(2):
        poll(p, c, m)
    assert p.paused


def test_paused_then_ended_unfreezes() -> None:
    """Ended is a stopped timer, not a pause: frozen detector should not linger."""
    f = TimerFollower(pause=True)
    for ph, i in (("Paused", 0), ("Ended", 3)):
        _ = f.observe(phase=ph, ls_index=i, expected=0, n_splits=3, now=0.0,
                      last_split_time=None, suppress=False, can_back=True,
                      can_fwd=True, waiting_for_start=False)
    assert not f.paused


# ---- gate / skip / undo interplay ------------------------------------------


def test_paused_at_gate_then_resume_then_manual_start() -> None:
    p, m, c = poller(), machine(start=True), Fake(index=0)
    c.phase = "Paused"
    assert poll(p, c, m).message is not None
    c.phase = "Running"
    assert poll(p, c, m).message == "LiveSplit: timer RESUMED"
    r = poll(p, c, m)
    assert r.message is not None and "left the start gate" in r.message
    assert not p.paused


def test_skip_forward_confirmed_while_paused_characterization() -> None:
    p, m, c = poller(), machine(), Fake(index=0)
    poll(p, c, m)
    c.phase, c.index = "Paused", 2
    for _ in range(4):
        poll(p, c, m)
    assert m.current_split_index == 2  # skip applies even while frozen


def test_paused_forever_polls_emit_pause_once() -> None:
    p, m, c = poller(), machine(), Fake(index=0)
    poll(p, c, m)
    c.phase = "Paused"
    msgs = [poll(p, c, m).message for _ in range(10)]
    assert msgs[0] is not None and all(x is None for x in msgs[1:])


def test_disabled_pause_never_emits_pause_actions() -> None:
    f = TimerFollower(pause=False)
    for ph in ("Running", "Paused", "Paused", "Running", "Paused"):
        d = f.observe(phase=ph, ls_index=0, expected=0, n_splits=3, now=0.0,
                      last_split_time=None, suppress=False, can_back=True,
                      can_fwd=True, waiting_for_start=False)
        assert d.action not in (Action.PAUSE, Action.RESUME)
        assert not f.paused
