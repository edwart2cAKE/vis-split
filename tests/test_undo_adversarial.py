"""Adversarial tests for the Unsplit-resync feature (rollback_to, UndoWatch,
get_split_index). Tests that fail here expose real defects; do not xfail."""

from __future__ import annotations

import socket
import threading
import time
from typing import cast, final

import pytest

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
    SplitEvent,
    StartEvent,
    State,
    TransitionDetector,
)
from autosplitter.livesplit import LiveSplitClient
from autosplitter.livesplit.sync import HOLDOFF_SECONDS, UndoWatch

BLACK = Color(name="black", rgb=(0, 0, 0), tolerance=15.0)
WHITE = Color(name="white", rgb=(255, 255, 255), tolerance=15.0)


def make_config(
    splits: tuple[Split, ...],
    required_frames: int = 3,
    start_transition: StartTransition | None = None,
) -> Config:
    return Config(
        capture=Capture(fps=60.0, stream_url="udp://127.0.0.1:9733", width=1920, height=1080),
        roi=DEFAULT_ROI,
        detection=Detection(
            required_frames=required_frames, reset_threshold=0.80, restart_quiet_seconds=3.0
        ),
        colors={"black": BLACK, "white": WHITE},
        splits=splits,
        livesplit=LiveSplit(host="127.0.0.1", port=16834),
        start_transition=start_transition,
    )


def res(ratio: float) -> DetectionResult:
    return DetectionResult(matched_pixels=int(ratio * 100), total_pixels=100, ratio=ratio)


def events_of(m: TransitionDetector, ratios: list[float]) -> list[SplitEvent | StartEvent]:
    out: list[SplitEvent | StartEvent] = []
    for r in ratios:
        e = m.feed_result(res(r))
        if e is not None:
            out.append(e)
    return out


def splits(n: int) -> tuple[Split, ...]:
    return tuple(
        Split(id=i + 1, name=chr(ord("A") + i), color="black", threshold=0.95)
        for i in range(n)
    )


def machine(n: int, *, frames: int = 1, start: bool = False) -> TransitionDetector:
    gate = StartTransition(name="Start", color="white", threshold=0.95) if start else None
    return TransitionDetector(
        make_config(splits(n), required_frames=frames, start_transition=gate)
    )


def ids(events: list[SplitEvent | StartEvent]) -> list[int]:
    return [e.split_id for e in events if isinstance(e, SplitEvent)]


# ---------------------------------------------------------------- state machine


def test_second_rollback_to_same_index_is_rejected() -> None:
    m = machine(3)
    events_of(m, [1.0, 1.0])
    assert m.rollback_to(0)
    assert m.expected_ls_index == 0
    assert m.rollback_to(0) is False


def test_rollback_expected_index_consistent_after_rollback() -> None:
    m = machine(4)
    events_of(m, [1.0, 1.0, 0.0, 1.0, 1.0])  # A, B fired; waiting; LS=2
    assert m.expected_ls_index == 2
    assert m.rollback_to(1)
    assert m.expected_ls_index == 1
    assert m.rollback_to(0)  # second undo, lower
    assert m.expected_ls_index == 0
    assert m.current_split_index == 0
    assert m.state is State.WAITING_FOR_RESET


def test_after_rollback_next_split_emitted_is_the_rolled_back_one() -> None:
    m = machine(3)
    events_of(m, [1.0, 1.0, 0.0, 1.0, 1.0])  # A,B fired
    assert m.rollback_to(1)
    out = events_of(m, [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    assert ids(out) == [2, 3]


def test_dead_zone_ratios_after_rollback() -> None:
    m = machine(3)
    events_of(m, [1.0, 1.0])
    assert m.rollback_to(0)
    for r in (0.99, 0.95, 0.85, 0.80, 0.8000001):
        assert m.feed_result(res(r)) is None
        assert m.state is State.WAITING_FOR_RESET
    assert m.feed_result(res(0.79999)) is None
    assert m.state is State.NORMAL and m.current_split_index == 0


def test_n1_split_rollback_from_done_full_cycle() -> None:
    m = machine(1)
    assert len(events_of(m, [1.0, 1.0, 1.0])) == 1
    assert m.finished
    assert m.expected_ls_index == 1
    assert m.rollback_to(0)
    assert m.expected_ls_index == 0
    assert events_of(m, [1.0] * 30) == []  # still lit, no double split
    assert events_of(m, [0.0]) == []
    assert m.state is State.NORMAL
    assert len(events_of(m, [1.0, 1.0, 1.0])) == 1
    assert m.finished


def test_rollback_from_transition_of_last_split_then_retry_after_done() -> None:
    m = machine(2)
    events_of(m, [1.0, 1.0, 0.0, 1.0])  # last split just fired -> TRANSITION
    assert m.state is State.TRANSITION
    assert m.rollback_to(1) is False
    m.feed_result(res(1.0))
    assert m.finished
    assert m.rollback_to(1)


def test_reset_after_rollback_clears_start_flag() -> None:
    """--loop reset() following a rollback must not leave stale settle flag."""
    m = machine(2)
    events_of(m, [1.0, 1.0])
    assert m.rollback_to(0)
    m.reset()
    assert m.state is State.NORMAL and m.expected_ls_index == 0
    m.feed_result(res(1.0))  # fires A
    assert m.state is State.TRANSITION
    assert m.expected_ls_index == 1  # would be 0 if flag leaked
    m.feed_result(res(1.0))
    m.feed_result(res(0.0))
    assert m.current_split_index == 1  # advanced normally


def test_reset_after_rollback_from_done_with_gate() -> None:
    m = machine(2, start=True)
    events_of(m, [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    assert m.finished
    assert m.rollback_to(1)
    m.reset()
    assert m.state is State.WAITING_FOR_START
    assert m.expected_ls_index == 0
    assert m.rollback_to(0) is False


def test_gate_settling_rejects_rollback() -> None:
    m = machine(3, start=True)
    events_of(m, [1.0, 1.0])
    assert m.state is State.WAITING_FOR_RESET
    assert m.rollback_to(0) is False


def test_gate_rollback_of_first_split_does_not_return_to_gate() -> None:
    m = machine(3, start=True)
    events_of(m, [1.0, 1.0, 0.0, 1.0, 1.0])  # gate, settle, A fired
    assert m.rollback_to(0)
    assert m.state is State.WAITING_FOR_RESET
    m.feed_result(res(0.0))
    assert m.state is State.NORMAL
    assert ids(events_of(m, [1.0])) == [1]


def test_rollback_from_done_multi_level_replays_all_splits() -> None:
    m = machine(3)
    events_of(m, [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    assert m.finished
    assert m.rollback_to(0)
    out = events_of(m, [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    assert ids(out) == [1, 2, 3]
    assert m.finished and m.expected_ls_index == 3


def test_rollback_to_last_index_from_done_then_refire_and_done_again() -> None:
    m = machine(2)
    events_of(m, [1.0, 1.0, 0.0, 1.0, 1.0])
    assert m.finished
    assert m.rollback_to(1)
    assert m.current_split is not None and m.current_split.name == "B"
    assert events_of(m, [0.0, 1.0]) != []
    m.feed_result(res(1.0))
    assert m.finished


def test_rollback_with_debounce_3_lit_color_no_refire() -> None:
    m = machine(3, frames=3)
    events_of(m, [1.0] * 4)
    assert m.rollback_to(0)
    assert events_of(m, [1.0] * 50) == []


def test_rollback_bool_and_wrong_types_do_not_crash() -> None:
    m = machine(3)
    events_of(m, [1.0, 1.0, 0.0])
    assert m.rollback_to(-1) is False
    assert m.rollback_to(10**20) is False


# --------------------------------------------------------------------- UndoWatch


def watch_obs(w: UndoWatch, ls: int | None, expected: int = 2, now: float = 100.0,
              last: float | None = None) -> int | None:
    return w.observe("Running", ls, expected, now, last, False, True)


def test_flapping_values_never_confirm() -> None:
    w = UndoWatch()
    for i in range(20):
        assert watch_obs(w, i % 2) is None


def test_flapping_with_none_between_never_confirms() -> None:
    w = UndoWatch()
    for _ in range(10):
        assert watch_obs(w, 0) is None
        assert watch_obs(w, None) is None


def test_time_going_backwards_blocks() -> None:
    w = UndoWatch()
    for _ in range(4):
        assert watch_obs(w, 0, now=50.0, last=100.0) is None


def test_nan_time_blocks() -> None:
    w = UndoWatch()
    for _ in range(4):
        assert watch_obs(w, 0, now=float("nan"), last=1.0) is None


def armed_watch() -> UndoWatch:
    """UndoWatch already armed by an in-step Running poll."""
    w = UndoWatch()
    assert w.observe("Running", 0, 0, 0.0, None, False, True) is None
    return w


def test_holdoff_exact_boundary_inclusive() -> None:
    w = armed_watch()
    assert watch_obs(w, 0, now=100.0 + HOLDOFF_SECONDS, last=100.0) is None
    assert watch_obs(w, 0, now=100.0 + HOLDOFF_SECONDS, last=100.0) == 0


def test_holdoff_interrupts_confirmation() -> None:
    """A split sent between two agreeing polls must void the first poll."""
    w = armed_watch()
    assert watch_obs(w, 0, now=100.0, last=0.0) is None
    assert watch_obs(w, 0, now=102.0, last=101.0) is None  # new split, hold-off
    assert watch_obs(w, 0, now=106.0, last=101.0) is None  # count restarted
    assert watch_obs(w, 0, now=108.0, last=101.0) == 0


def test_huge_index_is_ignored() -> None:
    w = UndoWatch()
    for _ in range(3):
        assert watch_obs(w, int("1" + "0" * 30), expected=2) is None


def test_confirm_returns_ls_index_not_expected() -> None:
    w = armed_watch()
    watch_obs(w, 1, expected=5)
    assert watch_obs(w, 1, expected=5) == 1


def test_expected_zero_never_candidate() -> None:
    w = UndoWatch()
    for _ in range(4):
        assert watch_obs(w, 0, expected=0) is None


def test_polling_simulation_false_split_unsplit_then_correct_split() -> None:
    """Mimic _run_live poll block against a real machine end to end."""
    m = machine(3)
    w = UndoWatch()
    t = 0.0
    events_of(m, [1.0, 1.0])  # false split A at t=0, LS index becomes 1
    last = 0.0
    # First Running poll shows LS in step with the detector: arms the watch.
    assert w.observe("Running", 1, m.expected_ls_index, 1.0, last, False, True) is None
    ls = 1
    fired: list[int] = []
    # user presses Unsplit at t=1; polls every 2 s
    for t in (2.0, 4.0, 6.0, 8.0):
        if t >= 1.0:
            ls = 0
        target = w.observe("Running", ls, m.expected_ls_index, t, last, False,
                           m.state in (State.NORMAL, State.WAITING_FOR_RESET, State.DONE))
        if target is not None:
            assert m.rollback_to(target)
    assert m.current_split_index == 0 and m.expected_ls_index == 0
    m.feed_result(res(0.0))
    fired = ids(events_of(m, [1.0, 1.0]))
    assert fired == [1]


def test_timer_started_late_after_detector_progress_is_not_treated_as_undo() -> None:
    """Detector fired A,B while LiveSplit timer was NotRunning (ignored there).
    User then starts timer: LS index 0, Running. This is NOT an unsplit, but
    the watch confirms a rollback to 0 and discards detector progress."""
    m = machine(3)
    events_of(m, [1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    assert m.expected_ls_index == 2
    w = UndoWatch()
    results = [
        w.observe("Running", 0, m.expected_ls_index, 100.0 + 2 * i, 0.0, False, True)
        for i in range(2)
    ]
    assert results == [None, None]


# ------------------------------------------------------------------- client


@final
class ScriptedServer:
    """Replies per received command line using a script of (delay, reply)."""

    def __init__(self, script: list[tuple[float, bytes]]) -> None:
        self.script: list[tuple[float, bytes]] = script
        self.sock: socket.socket = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port: int = cast("tuple[str, int]", self.sock.getsockname())[1]
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        conn = cast("tuple[socket.socket, object]", self.sock.accept())[0]
        buf = b""
        i = 0
        while i < len(self.script):
            try:
                data = conn.recv(4096)
            except OSError:
                return
            if not data:
                return
            buf += data
            while b"\n" in buf and i < len(self.script):
                _, buf = buf.split(b"\n", 1)
                delay, reply = self.script[i]
                i += 1
                threading.Timer(delay, conn.sendall, args=(reply,)).start()

    def close(self) -> None:
        self.sock.close()


def client_for(port: int) -> LiveSplitClient:
    return LiveSplitClient("127.0.0.1", port, timeout_seconds=0.1, reconnect_seconds=60.0)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (b" 3 \n", 3),
        (b"+2\n", 2),
        (b"3.0\n", None),
        (b"3 4\n", None),
        (b"0x3\n", None),
        (b"3\r\r\n", 3),
        (b"3\nextra\n", 3),
        (b"99999999999999999999\n", 99999999999999999999),
        (b"-0\n", 0),
        (b"1_0\n", None),  # Python int() accepts underscores: reports 10
        ("٣\n".encode(), None),  # Arabic-indic digit: int() accepts it
        (b"nan\n", None),
        (b"\xff\xfe\n", None),
    ],
)
def test_weird_replies(reply: bytes, expected: int | None) -> None:
    srv = ScriptedServer([(0.0, reply)])
    c = client_for(srv.port)
    assert c.get_split_index() == expected
    c.close()
    srv.close()


def test_stale_reply_after_timeout_is_not_returned_for_next_query() -> None:
    """A query that timed out leaves its reply in the socket; the next query
    then reads that stale line (off-by-one forever). For getsplitindex this
    yields a stale index, which could confirm a bogus undo."""
    srv = ScriptedServer([(0.4, b"0\n"), (0.0, b"2\n")])
    c = client_for(srv.port)
    assert c.get_split_index() is None  # times out at 0.1 s
    time.sleep(0.5)  # late reply "0" now sitting in the socket
    assert c.get_split_index() == 2  # gets stale 0
    c.close()
    srv.close()
