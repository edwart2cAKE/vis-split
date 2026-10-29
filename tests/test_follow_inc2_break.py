"""Adversarial tests for follow_timer increment 2 (reset / manual start / run end)."""

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
from autosplitter.detection import (
    DetectionResult,
    StartEvent,
    State,
    TransitionDetector,
)
from autosplitter.livesplit import Action, LiveSplitClient, TimerFollower

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


def feed(m: TransitionDetector, *ratios: float) -> list[object]:
    out: list[object] = []
    for r in ratios:
        ev = m.feed_result(DetectionResult(int(r * 100), 100, r))
        if ev is not None:
            out.append(ev)
    return out


@final
class Fake:
    def __init__(self, phase: str | None = "Running", index: int | None = 0, count: int | None = 4) -> None:
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


def poll(
    p: main_mod._FollowPoller,  # pyright: ignore[reportPrivateUsage]
    c: Fake,
    m: TransitionDetector,
    last: float | None = None,
) -> main_mod._FollowPollResult:  # pyright: ignore[reportPrivateUsage]
    return p.poll(cast(LiveSplitClient, cast(object, c)), m, "X", last, False)


def new(n: int = 4, loop: bool = False) -> main_mod._FollowPoller:  # pyright: ignore[reportPrivateUsage]
    return main_mod._FollowPoller(n, loop)  # pyright: ignore[reportPrivateUsage]


def fobs(f: TimerFollower, **kw: object) -> Action:
    base: dict[str, object] = {
        "phase": "Running", "ls_index": 0, "expected": 0, "n_splits": 4, "now": 1.0,
        "last_split_time": None, "suppress": False, "can_back": True, "can_fwd": True,
        "waiting_for_start": False,
    }
    base.update(kw)
    return f.observe(**base).action  # pyright: ignore[reportArgumentType]


# ---- strengthened versions of the four edited increment-1 tests ----------


@pytest.mark.parametrize("phase", ["Running", "Paused"])
def test_gate_never_skips_only_manual_start(phase: str) -> None:
    f = TimerFollower()
    seen = {fobs(f, phase=phase, ls_index=2, waiting_for_start=True) for _ in range(5)}
    assert Action.SKIP_FORWARD not in seen
    assert seen == {Action.MANUAL_START}  # single poll, repeated each time


def test_gate_with_cannot_fwd_does_nothing() -> None:
    f = TimerFollower()
    for _ in range(5):
        assert fobs(f, ls_index=2, waiting_for_start=True, can_fwd=False) is Action.NONE


def test_gate_negative_index_and_ended_are_not_manual_start() -> None:
    f = TimerFollower()
    assert fobs(f, ls_index=-1, waiting_for_start=True) is Action.NONE
    assert fobs(f, phase="Ended", ls_index=2, waiting_for_start=True) is Action.NONE


def test_poller_gate_stays_closed_after_manual_start_and_no_skip_follows() -> None:
    p, m, c = new(3), machine(3, start=True), Fake(index=2, count=3)
    assert poll(p, c, m).message is not None
    for _ in range(5):
        assert poll(p, c, m).message is None
    assert m.current_split_index == 2 and m.state is State.WAITING_FOR_RESET


def test_apply_decision_manual_start_run_ended_refusals() -> None:
    apply = main_mod._apply_decision  # pyright: ignore[reportPrivateUsage]
    from autosplitter.livesplit import Decision

    m = machine(3)
    assert apply(m, Decision(Action.MANUAL_START)) is None  # no index
    m2 = machine(3)
    assert apply(m2, Decision(Action.RUN_ENDED, 99)) is None
    # Increment 3: PAUSE/RESUME are applied (clear the streak, log only).
    assert apply(m2, Decision(Action.RESUME, 1)) == "LiveSplit: timer RESUMED"
    assert apply(m2, Decision(Action.PAUSE, 1)) == "LiveSplit: timer PAUSED — detector frozen"


# ---- RESET -----------------------------------------------------------------


def test_reset_during_transition_and_waiting_for_reset() -> None:
    for pre in ("transition", "wfr"):
        p, m, c = new(), machine(), Fake(index=0)
        poll(p, c, m)
        if pre == "transition":
            feed(m, 1.0)
            assert m.state is State.TRANSITION
        else:
            feed(m, 1.0, 0.9)
            assert m.state is State.WAITING_FOR_RESET
        c.phase = "NotRunning"
        r = poll(p, c, m)
        assert r.message is not None
        assert m.current_split_index == 0 and m.state is State.WAITING_FOR_RESET
        assert feed(m, 1.0, 1.0) == []
        feed(m, 0.0)  # quiet -> NORMAL
        assert m.state is State.NORMAL
        assert len(feed(m, 1.0)) == 1


def test_reset_with_gate_lit_color_refires_start() -> None:
    """Characterization: with a gate, reset(settle) == reset(); a still-lit gate
    colour fires StartEvent immediately after a manual LiveSplit reset (and
    would send `start` with auto_start_timer)."""
    p, m, c = new(), machine(start=True), Fake(index=0)
    m.advance_to(0)
    poll(p, c, m)
    c.phase = "NotRunning"
    poll(p, c, m)
    assert m.state is State.WAITING_FOR_START
    evs = feed(m, 1.0)
    assert len(evs) == 1 and isinstance(evs[0], StartEvent)


def test_flapping_notrunning_running_resets_each_time_and_never_crashes() -> None:
    p, m, c = new(), machine(), Fake(index=0)
    resets = 0
    for i in range(10):
        c.phase = "Running" if i % 2 == 0 else "NotRunning"
        r = poll(p, c, m)
        if r.message and "RESET" in r.message:
            resets += 1
    assert resets == 5  # NotRunning at i=1,3,5,7,9, each after Running
    assert m.current_split_index == 0


def test_reset_after_midrun_then_restart_between_polls_rolls_back() -> None:
    """User resets+restarts inside one poll interval: only Running L=0 seen."""
    p, m, c = new(), machine(), Fake(index=2)
    m.advance_to(2)
    feed(m, 0.0)
    poll(p, c, m, last=None)  # arm
    c.index = 0
    poll(p, c, m, last=None)
    r = poll(p, c, m, last=None)
    assert r.message is not None and "UNSPLIT" in r.message
    assert m.current_split_index == 0


def test_stale_running_after_reset_does_not_manual_start_without_gate() -> None:
    p, m, c = new(), machine(), Fake(index=1)
    poll(p, c, m)
    c.phase = "NotRunning"
    poll(p, c, m)
    c.phase = "Running"
    c.index = 0
    for _ in range(4):
        assert poll(p, c, m).message is None
    assert m.current_split_index == 0


def test_reset_rewound_done_then_followed_rerun() -> None:
    p, m, c = new(loop=True), machine(), Fake(phase="Ended", index=4)
    m.advance_to(4)
    poll(p, c, m)
    c.phase = "NotRunning"
    assert poll(p, c, m).message is not None
    c.phase, c.index = "Running", 0
    for _ in range(3):
        assert poll(p, c, m).message is None
    assert m.state is State.WAITING_FOR_RESET


# ---- MANUAL_START ---------------------------------------------------------


def test_manual_start_n1() -> None:
    p, m, c = new(1), machine(1, start=True), Fake(index=0, count=1)
    assert poll(p, c, m).message is not None
    feed(m, 0.0)  # settle
    assert m.state is State.NORMAL
    evs = feed(m, 1.0)
    assert len(evs) == 1
    feed(m, 1.0)
    assert m.finished


def test_manual_start_with_gate_lit_does_not_double_fire() -> None:
    p, m, c = new(), machine(start=True), Fake(index=0)
    poll(p, c, m)
    assert feed(m, 1.0, 1.0, 1.0) == []  # lit gate color: no StartEvent


def test_manual_start_L_gt0_then_split_color_does_not_fire_until_quiet() -> None:
    p, m, c = new(), machine(start=True), Fake(index=2)
    poll(p, c, m)
    assert m.current_split_index == 2
    assert feed(m, 1.0, 1.0) == []
    feed(m, 0.0)
    assert len(feed(m, 1.0)) == 1


def test_manual_start_after_start_event_in_flight_no_effect() -> None:
    p, m, c = new(), machine(start=True), Fake(index=0)
    evs = feed(m, 1.0)
    assert isinstance(evs[0], StartEvent) and m.state is State.TRANSITION
    for _ in range(3):
        assert poll(p, c, m).message is None


# ---- RUN_ENDED ------------------------------------------------------------


def test_run_ended_last_split_in_flight() -> None:
    p, m, c = new(), machine(), Fake(phase="Ended", index=4)
    m.advance_to(3)
    feed(m, 0.0)
    feed(m, 1.0)  # last split fires
    for _ in range(4):
        _ = poll(p, c, m)  # must not crash
    feed(m, 0.0)
    assert m.finished


def test_run_ended_from_waiting_for_reset() -> None:
    p, m, c = new(), machine(), Fake(phase="Ended", index=4)
    feed(m, 1.0, 0.9)  # split 0 fired, WAITING_FOR_RESET, E=1
    c.phase, c.index = "Running", 1  # prime: timer seen Running (in step)
    poll(p, c, m)
    c.phase, c.index = "Ended", 4
    poll(p, c, m)
    assert poll(p, c, m).message is not None
    assert m.finished


def test_ended_unsplit_ended_again() -> None:
    p, m, c = new(), machine(), Fake(phase="Ended", index=4)
    m.advance_to(4)
    poll(p, c, m)  # arm at DONE
    c.phase, c.index = "Running", 3
    poll(p, c, m)
    r = poll(p, c, m)
    assert r.message is not None and "UNSPLIT" in r.message
    assert not m.finished
    c.phase, c.index = "Ended", 4
    poll(p, c, m)
    r = poll(p, c, m)
    assert r.message is not None and m.finished


def test_run_ended_not_repeated_after_rearm_with_loop() -> None:
    """--loop colour-quiet rearm while LiveSplit is still Ended must not drag
    the machine straight back to DONE (ping-pong every restart_quiet_seconds)."""
    p, m, c = new(loop=True), machine(), Fake(phase="Ended", index=4)
    m.advance_to(4)
    poll(p, c, m)
    m.reset()  # quiet rearm in _run_live
    p.disarm()
    for _ in range(4):
        _ = poll(p, c, m)
    assert not m.finished


def test_run_ended_not_triggered_on_stale_ended_at_startup() -> None:
    """Detector started while LiveSplit still shows a finished run: the
    machine should not declare the (new) run DONE."""
    p, m, c = new(), machine(), Fake(phase="Ended", index=4)
    for _ in range(4):
        _ = poll(p, c, m)
    assert not m.finished


def test_run_ended_flapping_identity_requires_consecutive() -> None:
    p, m, c = new(), machine(), Fake(phase="Running", index=0)
    poll(p, c, m)  # prime: timer seen Running
    c.phase, c.index = "Ended", 4
    poll(p, c, m)
    c.phase = None
    poll(p, c, m)
    c.phase = "Ended"
    assert poll(p, c, m).message is None
    assert poll(p, c, m).message is not None


def test_weird_timestamps_do_not_crash() -> None:
    f = TimerFollower()
    for now, last in ((0.0, 1e12), (-5.0, -1e9), (float("inf"), 0.0), (1.0, float("nan"))):
        _ = fobs(f, now=now, last_split_time=last, ls_index=1, expected=3)


def test_count_mismatch_plus_reset_then_ended_not_done() -> None:
    p, m, c = new(), machine(), Fake(index=2, count=9)
    m.advance_to(2)
    poll(p, c, m)
    c.phase = "NotRunning"
    assert poll(p, c, m).message is not None
    c.phase, c.index = "Ended", 4
    for _ in range(4):
        assert poll(p, c, m).message is None


def test_connection_lost_clears_run_end_candidate() -> None:
    p, m, c = new(), machine(), Fake(phase="Ended", index=4)
    poll(p, c, m)
    p.connection_lost()
    # candidate must restart (count re-checked too)
    assert poll(p, c, m).message is None
