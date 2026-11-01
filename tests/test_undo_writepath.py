"""Increment 4 write path: break tests for the unsplit / re-split executor.

Three layers:

* in-process ``SLS`` (a stateful fake LiveSplit with a user-action channel
  that fires BETWEEN our queries) driven through the real ``_undo_service``,
  the real ``TransitionDetector`` and the real ``UndoExecutor``;
* a socket-level scripted server (real ``LiveSplitClient``) for transport
  failures that the in-process fake cannot express (EOF after write, slow or
  late replies, asynchronous apply);
* a randomised twin asserting invariants after every step.

Tests that fail expose real defects; do not xfail.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import random
import socket
import threading
import time
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast, final

import numpy as np
import pytest
from test_undo_machine import (  # pyright: ignore[reportImplicitRelativeImport]
    FRAME,
    OCR_U,
    Rig,
    res,
)

from autosplitter import main as mainmod
from autosplitter.config import ConfigError, load
from autosplitter.detection import State
from autosplitter.detection.state_machine import SplitEvent
from autosplitter.livesplit import LiveSplitClient, OutcomeKind, UndoExecutor

OCR_MODE_UNDO = dataclasses.replace(OCR_U, mode="undo")
ALLOWED = frozenset({"phase", "idx", "split", "unsplit", "name", "count"})


# --------------------------------------------------------------------------
# stateful fake LiveSplit (in-process)
# --------------------------------------------------------------------------


@final
class SLS:
    """LiveSplit with real split/unsplit rules and a user-action channel.

    ``hooks[n]`` run just BEFORE the n-th call (1-based) after ``arm()``.
    ``faults[kind]`` is a queue of "ok" | "fail" | "apply_fail" (sends only:
    applied but reported False) consumed per call of that kind.
    """

    def __init__(self, idx: int = 1, count: int = 4, phase: str = "Running") -> None:
        self.idx: int = idx
        self.count: int = count
        self.phase: str = phase
        self.log: list[str] = []
        self.user: list[str] = []
        self.hooks: dict[int, list[Callable[[SLS], None]]] = {}
        self.faults: dict[str, list[str]] = {}
        self.ncall: int = 0

    def arm(self, hooks: dict[int, list[Callable[[SLS], None]]] | None = None) -> None:
        self.ncall = 0
        self.hooks = hooks or {}

    # -- LiveSplit rules
    def _do_split(self) -> None:
        if self.phase == "Running":
            self.idx += 1
            if self.idx >= self.count:
                self.phase = "Ended"

    def _do_unsplit(self) -> None:
        if self.phase in ("Running", "Ended") and self.idx > 0:
            self.idx -= 1
            self.phase = "Running"

    # -- user actions
    def u_split(self) -> None:
        self.user.append("split")
        self._do_split()

    def u_unsplit(self) -> None:
        self.user.append("unsplit")
        self._do_unsplit()

    def u_reset(self) -> None:
        self.user.append("reset")
        self.phase = "NotRunning"
        self.idx = -1

    def u_pause(self) -> None:
        self.user.append("pause")
        if self.phase == "Running":
            self.phase = "Paused"

    # -- client surface
    def _enter(self, kind: str) -> str:
        self.ncall += 1
        self.log.append(kind)
        for hook in self.hooks.get(self.ncall, []):
            hook(self)
        queue = self.faults.get(kind)
        return queue.pop(0) if queue else "ok"

    def get_current_timer_phase(self) -> str | None:
        return None if self._enter("phase") == "fail" else self.phase

    def get_split_index(self) -> int | None:
        return None if self._enter("idx") == "fail" else self.idx

    def get_current_split_name(self) -> str | None:
        return None if self._enter("name") == "fail" else f"S{self.idx + 1}"

    def get_split_count(self) -> int | None:
        return None if self._enter("count") == "fail" else self.count

    def split(self) -> bool:
        fault = self._enter("split")
        if fault != "fail":
            self._do_split()
        return fault == "ok"

    def unsplit(self) -> bool:
        fault = self._enter("unsplit")
        if fault != "fail":
            self._do_unsplit()
        return fault == "ok"

    @property
    def sent(self) -> list[str]:
        return [c for c in self.log if c in ("split", "unsplit")]


@final
class Sim:
    """machine + executor + SLS driven like ``_run_live`` does per frame."""

    def __init__(self, n: int = 4, start: bool = False, mode_undo: bool = True) -> None:
        self.r: Rig = Rig(n, OCR_MODE_UNDO if mode_undo else OCR_U, start)
        self.ls: SLS = SLS(idx=0, count=n)
        self.ex: UndoExecutor = UndoExecutor()
        self.n: int = n
        self.out: list[str] = []
        self.service_moves: int = 0
        self.fires: int = 0
        self.outcomes: list[OutcomeKind] = []

    @property
    def m(self) -> object:
        return self.r.m

    def tick(self, ratio: float, text: str | None, dt: float = 0.25) -> SplitEvent | None:
        """One frame in main's order: detect, OCR/judge, undo service, send."""
        ev = self.r.m.feed_result(res(ratio))
        if text is not None:
            self.r.read(text, dt=dt)
        else:
            self.r.clock.t += dt
            self.r.frame += 1
            self.r.m.check_ocr_resync(FRAME, self.r.frame)
        # Capture lines through a fake client wrapper? _undo_service prints.
        if mainmod._undo_service(self.ls, self.r.m, self.r.m._config, True, self.ex):  # pyright: ignore[reportPrivateUsage, reportArgumentType]
            self.service_moves += 1
        if isinstance(ev, SplitEvent):
            self.fires += 1
            _ = self.ls.split()
        return cast("SplitEvent | None", ev)

    def agree(self) -> None:
        for _ in range(3):
            _ = self.tick(0.0, str(self.r.m.current_split_index + 1), dt=0.1)

    def fire_clear(self, hud: str) -> None:
        """Colour transition then clear, HUD showing ``hud`` throughout."""
        _ = self.tick(1.0, hud)
        _ = self.tick(1.0, hud)
        _ = self.tick(1.0, hud)
        _ = self.tick(0.0, hud)


def committed_death(n: int = 4, start: bool = False, n_prev: int = 0) -> Sim:
    """Sim sitting at detector index n_prev+1 with an UNDO verdict just
    about to be confirmed: everything up to (not including) the confirming read."""
    s = Sim(n, start)
    if start:
        s.fire_clear("")  # start gate (HUD unread)
    for k in range(n_prev + 1):
        s.agree()
        s.fire_clear(str(k + 1))
        if k < n_prev:
            for _ in range(2):
                _ = s.tick(0.0, str(k + 2))
    s.ls.log.clear()
    s.ls.arm()
    return s


def confirm_until_acted(s: Sim, hud: str, max_ticks: int = 60) -> None:
    for _ in range(max_ticks):
        _ = s.tick(0.0, hud)
        if s.ls.log:
            return


def div(s: Sim) -> int | None:
    """Detector's intended LiveSplit index minus LiveSplit's (Running only)."""
    if s.ls.phase != "Running":
        return None
    return s.r.m.expected_ls_index - s.ls.idx


# --------------------------------------------------------------------------
# 1. user action races inside the attempt (matrix)
# --------------------------------------------------------------------------

def _twice(ls: SLS, act: Callable[[SLS], None]) -> None:
    act(ls)
    act(ls)


USER_ACTIONS: dict[str, Callable[[SLS], None]] = {
    "split": SLS.u_split,
    "unsplit": SLS.u_unsplit,
    "reset": SLS.u_reset,
    "pause": SLS.u_pause,
    "double_unsplit": lambda ls: _twice(ls, SLS.u_unsplit),
    "skip2": lambda ls: _twice(ls, SLS.u_split),
}


@pytest.mark.parametrize("n_prev", [0, 1])
@pytest.mark.parametrize("action", sorted(USER_ACTIONS))
@pytest.mark.parametrize("before_call", [1, 2, 3, 4])
def test_user_action_inside_attempt_never_makes_things_worse(
    action: str, before_call: int, n_prev: int
) -> None:
    s = committed_death(n_prev=n_prev)
    s.ls.idx = n_prev + 1
    start_idx = s.ls.idx
    s.ls.arm({before_call: [USER_ACTIONS[action]]})
    # LiveSplit after the user's action alone (our commands ignored)
    twin = SLS(idx=start_idx, count=4)
    USER_ACTIONS[action](twin)
    base = (n_prev + 1) - twin.idx if twin.phase == "Running" else None
    confirm_until_acted(s, str(n_prev + 1))
    # the same frame may have run more calls than the matrix hooks
    assert set(s.ls.log) <= ALLOWED
    assert s.ls.sent.count("unsplit") <= 1
    assert s.ls.sent.count("split") == 0
    after = div(s)
    if after is not None and base is not None:
        assert abs(after) <= abs(base), (s.ls.log, s.ls.user, s.ls.idx, s.r.m.expected_ls_index)
    # settle: more frames never send anything else for this verdict
    before = list(s.ls.sent)
    for _ in range(30):
        _ = s.tick(0.0, str(n_prev + 1))
    assert s.ls.sent == before


def test_user_unsplit_between_precondition_and_send_rolls_detector_back_to_ls() -> None:
    s = committed_death(n_prev=1)
    s.ls.idx = 2
    s.ls.arm({3: [SLS.u_unsplit]})  # before our `unsplit`
    confirm_until_acted(s, "2")
    assert s.ls.idx == 0
    assert s.r.m.current_split_index == 0 and s.r.m.expected_ls_index == 0


def test_user_split_between_precondition_and_send_leaves_detector_in_step() -> None:
    s = committed_death()
    s.ls.arm({3: [SLS.u_split]})
    confirm_until_acted(s, "1")
    assert s.ls.idx == s.r.m.expected_ls_index  # both at 2: undo ignored, no desync
    assert s.ls.sent == ["unsplit"]


def test_idx_zero_is_a_noop_for_livesplit_and_consistent_for_detector() -> None:
    s = committed_death()
    s.ls.idx = 0  # LiveSplit never advanced (earlier split lost)
    s.ls.arm()
    confirm_until_acted(s, "1")
    assert s.ls.sent == []  # nothing sent
    assert s.r.m.current_split_index == 0 and s.ls.idx == 0


def test_timer_ended_or_notrunning_or_paused_sends_nothing() -> None:
    for phase in ("Ended", "NotRunning", "Paused"):
        s = committed_death()
        s.ls.phase = phase
        s.ls.arm()
        confirm_until_acted(s, "1")
        assert s.ls.sent == [], phase
        assert s.r.m.current_split_index == 1, phase  # untouched, not rolled back alone


def test_reset_between_queries_does_not_roll_back_or_resend() -> None:
    s = committed_death()
    s.ls.arm({2: [SLS.u_reset]})  # after phase query, before idx query
    confirm_until_acted(s, "1")
    assert s.ls.sent == []
    assert s.r.m.current_split_index == 1


# --------------------------------------------------------------------------
# 2. single verdict, single unsplit, consumption
# --------------------------------------------------------------------------


def test_one_verdict_never_unsplits_twice_even_when_ignored() -> None:
    s = committed_death()
    s.ls.faults["unsplit"] = ["ok"]
    # the server ignores it: make unsplit a no-op by pausing hook before send
    s.ls.arm({3: [lambda ls: setattr(ls, "phase", "Paused")]})
    confirm_until_acted(s, "1")
    for _ in range(200):
        _ = s.tick(0.0, "1")
    assert s.ls.sent.count("unsplit") == 1


def test_unverified_unsplit_not_repeated_and_not_rolled_back() -> None:
    s = committed_death()
    s.ls.faults["idx"] = ["ok", "fail"]
    s.ls.arm()
    confirm_until_acted(s, "1")
    for _ in range(100):
        _ = s.tick(0.0, "1")
    assert s.ls.sent == ["unsplit"]  # the desync itself is asserted by the transport-hole tests


# --------------------------------------------------------------------------
# 3. undelivered-but-applied (EOF after write) and friends, in-process
# --------------------------------------------------------------------------


def test_apply_fail_unsplit_is_reconciled_without_follow_timer() -> None:
    """unsplit applied by LiveSplit but reported undelivered (what LiveSplitClient._send
    returns on EOF-after-write): the detector must not end up ahead of LiveSplit."""
    s = committed_death()
    s.ls.faults["unsplit"] = ["apply_fail"]
    s.ls.arm()
    confirm_until_acted(s, "1")
    for _ in range(40):
        _ = s.tick(0.0, "1")
    assert s.ls.idx == 0
    assert s.r.m.expected_ls_index == s.ls.idx, "detector ahead of LiveSplit, nothing repairs it"


def test_apply_fail_unsplit_is_repaired_by_follow_timer() -> None:
    s = committed_death()
    poller = mainmod._FollowPoller(4)  # pyright: ignore[reportPrivateUsage]
    _ = poller.poll(s.ls, s.r.m, None, None, False)  # pyright: ignore[reportArgumentType]  # arms: in step
    s.ls.log.clear()
    s.ls.faults["unsplit"] = ["apply_fail"]
    s.ls.arm()
    confirm_until_acted(s, "1")
    for _ in range(4):
        _ = poller.poll(s.ls, s.r.m, None, None, False)  # pyright: ignore[reportArgumentType]
    assert s.r.m.expected_ls_index == s.ls.idx == 0


def test_apply_fail_recovery_resplit_leaves_detector_behind() -> None:
    s = undone_sim()
    s.ls.faults["split"] = ["apply_fail"]
    s.ls.arm()
    for _ in range(6):  # recovery needs the S+1 streak to span undo.confirm_seconds
        _ = s.tick(0.0, "2")
    assert s.ls.idx == 1  # LiveSplit applied the re-split
    assert s.r.m.expected_ls_index == s.ls.idx, "detector behind LiveSplit, nothing repairs it"


def undone_sim(n: int = 4, start: bool = False) -> Sim:
    s = committed_death(n, start)
    confirm_until_acted(s, "1")
    assert s.ls.sent == ["unsplit"] and s.ls.idx == 0
    _ = s.tick(0.0, "1")  # the undo's settle clear
    assert s.r.m.state is State.NORMAL and s.r.m.current_split_index == 0
    s.ls.log.clear()
    s.ls.arm()
    return s


# --------------------------------------------------------------------------
# 4. recovery strictness
# --------------------------------------------------------------------------


def test_recovery_splits_exactly_when_hud_advanced_and_in_step() -> None:
    s = undone_sim()
    for _ in range(6):
        _ = s.tick(0.0, "2")
    assert s.ls.sent == ["split"] and s.ls.idx == 1
    assert s.r.m.expected_ls_index == 1 and s.r.m.current_split_index == 1
    for _ in range(100):
        _ = s.tick(0.0, "2")
    assert s.ls.sent == ["split"]


@pytest.mark.parametrize("seq", [["1", "1", "1"], ["2", "1", "2", "1", "2", "1"], ["3", "3"], ["", ""], ["9", "9"]])
def test_recovery_never_splits_without_a_stable_s_plus_1_hud(seq: list[str]) -> None:
    s = undone_sim()
    for t in seq * 5:
        _ = s.tick(0.0, t)
    assert s.ls.sent == []


def test_recovery_low_confidence_never_splits() -> None:
    s = undone_sim()
    for _ in range(20):
        s.r.read("2", conf=60.0)
        _ = mainmod._undo_service(s.ls, s.r.m, s.r.m._config, True, s.ex)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert s.ls.sent == []


def test_recovery_after_user_manual_split_does_not_double_split() -> None:
    s = undone_sim()
    s.ls.u_split()  # user manually re-splits
    for _ in range(8):
        _ = s.tick(0.0, "2")
    assert s.ls.sent == [] and s.ls.idx == 1


def test_recovery_after_user_unsplit_below_s_does_not_split() -> None:
    s = committed_death(n_prev=1)
    s.ls.idx = 2
    confirm_until_acted(s, "2")
    assert s.ls.idx == 1
    _ = s.tick(0.0, "2")  # settle
    s.ls.u_unsplit()  # user keeps unsplitting: LiveSplit now below S
    s.ls.log.clear()
    for _ in range(8):
        _ = s.tick(0.0, "3")
    assert s.ls.sent == []


def test_recovery_racing_a_user_split_must_not_overshoot_livesplit() -> None:
    """User splits between our idx query and our re-split (no compare-and-swap in
    the protocol): our `split` then pushes LiveSplit one segment past the HUD and
    the log wrongly says 're-split had no effect'. Asserted in-step, so it fails."""
    s = undone_sim()
    s.ls.arm({3: [SLS.u_split]})  # before the re-split
    for _ in range(6):
        _ = s.tick(0.0, "2")
    assert s.ls.idx <= 1, f"LiveSplit overshot to {s.ls.idx}: our re-split skipped a segment"


def test_new_colour_fire_during_recovery_window_cancels_it() -> None:
    s = undone_sim()
    _ = s.tick(1.0, "1")
    _ = s.tick(1.0, "1")  # colour fires split 0 again
    assert s.ls.sent == ["split"]
    s.ls.log.clear()
    for _ in range(8):
        _ = s.tick(0.0, "2")
    assert s.ls.sent == []


@pytest.mark.parametrize("k", range(0, 24, 3))
def test_single_lit_frame_at_verdict_time_does_not_double_fire(k: int) -> None:
    """One lit frame (below required_frames) landing on/near the verdict frame."""
    s = committed_death()
    fires0 = s.fires
    for _ in range(k):
        _ = s.tick(0.0, "1")
    _ = s.tick(1.0, "1")
    for _ in range(60):
        _ = s.tick(0.0, "1")
    assert s.fires == fires0
    assert s.ls.sent == ["unsplit"]
    assert s.r.m.expected_ls_index == s.ls.idx == 0


def test_respawn_flash_exactly_at_undo_time() -> None:
    s = committed_death()
    s.ls.arm()
    # flash on the very frame the verdict confirms
    for _ in range(60):
        _ = s.tick(0.0, "1")
        if s.ls.log:
            break
    sent = list(s.ls.sent)
    assert sent == ["unsplit"]
    # flash: lit twice then clear -> legit new fire of split 0 (only the clear after)
    ev = [s.tick(r, "1") for r in (1.0, 1.0, 1.0, 0.0, 0.0)]
    # the lit colour is held by the settle state: first clear then normal
    assert s.ls.idx in (0, 1)
    assert [e for e in ev if e is not None] == []
    assert s.r.m.state is State.NORMAL and s.r.m.current_split_index == 0


def test_second_real_transition_after_undo_works_and_recovery_stays_off() -> None:
    s = undone_sim()
    ev = [s.tick(r, "1") for r in (1.0, 1.0, 1.0, 0.0)]
    assert sum(isinstance(e, SplitEvent) for e in ev) == 1
    assert s.ls.idx == 1 and s.r.m.current_split_index == 1
    for _ in range(20):
        _ = s.tick(0.0, "2")
    assert s.ls.sent == ["split"]  # only the colour fire, no recovery split


# --------------------------------------------------------------------------
# 5. gate, N=1/2, loop
# --------------------------------------------------------------------------


def test_start_gate_death_on_first_split() -> None:
    s = committed_death(start=True)
    s.ls.idx = 1
    confirm_until_acted(s, "1")
    assert s.ls.sent == ["unsplit"] and s.ls.idx == 0
    assert s.r.m.current_split_index == 0


def test_n2_death_on_first_split_and_recovery() -> None:
    s = undone_sim(n=2)
    for _ in range(6):
        _ = s.tick(0.0, "2")
    assert s.ls.sent == ["split"] and s.ls.idx == 1 and s.r.m.current_split_index == 1


def test_n1_never_attempts_anything() -> None:
    s = Sim(n=1)
    s.agree()
    s.fire_clear("1")
    for _ in range(60):
        _ = s.tick(0.0, "1")
    assert s.ls.sent == ["split"]  # the colour split only
    assert s.r.m.state is State.DONE


def test_loop_reset_between_undo_and_recovery_cancels_recovery() -> None:
    s = undone_sim()
    s.r.m.reset()  # --loop rearm / follow RESET
    for _ in range(8):
        _ = s.tick(0.0, "2")
    assert s.ls.sent == []


def test_follow_reset_settle_cancels_recovery() -> None:
    s = undone_sim()
    s.r.m.reset(settle=True)
    for _ in range(8):
        _ = s.tick(0.0, "2")
    assert s.ls.sent == []


def test_hot_reload_with_fewer_splits_does_not_crash_service() -> None:
    s = committed_death()
    new_cfg = dataclasses.replace(s.r.m._config, splits=s.r.m._config.splits[:2])  # pyright: ignore[reportPrivateUsage]
    s.r.m.reload_config(new_cfg)
    confirm_until_acted(s, "1")  # must not raise


# --------------------------------------------------------------------------
# 6. back-to-back verdicts
# --------------------------------------------------------------------------


def test_duplicate_queued_undo_verdicts_unsplit_once() -> None:
    s = committed_death()
    s.ls.arm()
    # drive until the verdict is queued but do NOT service: call machine directly
    target: int | None = None
    for _ in range(60):
        s.r.read("1")
        if s.r.m.undo_target is not None:
            target = s.r.m.undo_target
            break
    assert target == 0
    ev = s.r.m.take_undo_event()
    assert ev is not None
    s.r.m._undo_events.extend([ev, ev, ev])  # pyright: ignore[reportPrivateUsage]
    s.r.m._undo_target = 0  # pyright: ignore[reportPrivateUsage]
    _ = mainmod._undo_service(s.ls, s.r.m, s.r.m._config, True, s.ex)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert s.ls.sent == ["unsplit"]


def test_executor_called_twice_for_same_target_second_is_skipped() -> None:
    s = committed_death()
    for _ in range(60):
        s.r.read("1")
        if s.r.m.undo_target is not None:
            break
    first = s.ex.attempt(s.ls, s.r.m, 0, 1.0)
    second = s.ex.attempt(s.ls, s.r.m, 0, 2.0)
    assert first.kind is OutcomeKind.APPLIED and second.kind is OutcomeKind.SKIPPED
    assert s.ls.sent == ["unsplit"]


# --------------------------------------------------------------------------
# 7. socket level: real LiveSplitClient vs a scripted server
# --------------------------------------------------------------------------


@final
class Script:
    """Threaded LiveSplit server with injectable transport misbehaviour."""

    def __init__(self, idx: int = 1, count: int = 4) -> None:
        self.idx: int = idx
        self.count: int = count
        self.phase: str = "Running"
        self.lines: list[str] = []
        self.close_after: dict[str, int] = {}   # cmd -> nth occurrence: close right after handling it
        self.delay: dict[str, float] = {}       # reply delay per command
        self.delay_nth: dict[tuple[str, int], float] = {}  # (cmd, nth occurrence) -> delay
        self.count_of: dict[str, int] = {}
        self.async_apply: dict[str, float] = {}  # apply a split/unsplit this many seconds late
        self.stop: bool = False
        self.sock: socket.socket = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port: int = cast("tuple[str, int]", self.sock.getsockname())[1]
        self.lock: threading.Lock = threading.Lock()
        self.conns: list[socket.socket] = []
        self.t: threading.Thread = threading.Thread(target=self._accept, daemon=True)
        self.t.start()

    def _accept(self) -> None:
        self.sock.settimeout(0.05)
        while not self.stop:
            try:
                c = self.sock.accept()[0]
            except OSError:
                continue
            self.conns.append(c)
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _apply(self, line: str) -> None:
        with self.lock:
            if line == "split" and self.phase == "Running":
                self.idx += 1
            elif line == "unsplit" and self.phase == "Running" and self.idx > 0:
                self.idx -= 1

    def _serve(self, c: socket.socket) -> None:
        try:
            self._serve_conn(c)
        except OSError:
            pass  # Script.close() closed the connection under us (EBADF etc.)

    def _serve_conn(self, c: socket.socket) -> None:
        buf: bytes = b""
        c.settimeout(0.05)
        while not self.stop:
            try:
                d = c.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                return
            if not d:
                return
            buf += d
            while b"\n" in buf:
                raw, buf = cast("tuple[bytes, bytes]", tuple(buf.split(b"\n", 1)))
                line = raw.decode()
                self.lines.append(line)
                nth = self.count_of.get(line, 0) + 1
                self.count_of[line] = nth
                reply: str | None = None
                if line in self.async_apply:
                    threading.Timer(self.async_apply[line], self._apply, args=(line,)).start()
                else:
                    self._apply(line)
                if line == "getcurrenttimerphase":
                    reply = self.phase
                elif line == "getsplitindex":
                    reply = str(self.idx)
                elif line == "getsplitcount":
                    reply = str(self.count)
                elif line == "getcurrentsplitname":
                    reply = f"S{self.idx + 1}"
                if line in self.delay:
                    time.sleep(self.delay[line])
                if (line, nth) in self.delay_nth:
                    time.sleep(self.delay_nth[(line, nth)])
                if self.close_after.get(line) == nth:
                    c.close()
                    return
                if reply is not None:
                    try:
                        c.sendall(reply.encode() + b"\n")
                    except OSError:
                        return

    def sent(self) -> list[str]:
        return [x for x in list(self.lines) if x in ("split", "unsplit", "start")]

    def close(self) -> None:
        self.stop = True
        for c in self.conns:
            try:
                c.close()
            except OSError:
                pass
        self.t.join(1.0)
        self.sock.close()


@pytest.fixture
def script() -> Iterator[Script]:
    s = Script()
    yield s
    s.close()


def client_for(srv: Script, timeout: float = 0.3, reconnect: float = 0.05) -> LiveSplitClient:
    return LiveSplitClient("127.0.0.1", srv.port, timeout_seconds=timeout, reconnect_seconds=reconnect)


def socket_death() -> Rig:
    r = Rig(4, OCR_MODE_UNDO)
    r.agree()
    r.fire_and_clear()
    for _ in range(40):
        r.read("1")
    assert r.m.undo_target == 0 and r.m.current_split_index == 1
    return r


def run_exec(r: Rig, c: LiveSplitClient) -> OutcomeKind:
    return UndoExecutor().attempt(c, r.m, 0, time.monotonic()).kind


def test_socket_happy_path_exact_wire(script: Script) -> None:
    r = socket_death()
    c = client_for(script)
    assert run_exec(r, c) is OutcomeKind.APPLIED
    assert script.lines == ["getcurrenttimerphase", "getsplitindex", "unsplit", "getsplitindex"]
    c.close()


HOLES = ["eof_after_unsplit", "no_reply_to_verify", "async_apply"]


def _arm_hole(script: Script, hole: str) -> None:
    if hole == "eof_after_unsplit":
        script.close_after["unsplit"] = 1
    elif hole == "no_reply_to_verify":
        script.close_after["getsplitindex"] = 2  # the verify query dies unanswered
    elif hole == "async_apply":
        script.async_apply["unsplit"] = 0.15


@pytest.mark.parametrize("hole", HOLES)
def test_transport_hole_must_not_leave_detector_ahead_without_follow_timer(
    script: Script, hole: str
) -> None:
    """LiveSplit applied the unsplit but our write path could not tell. With only
    ``ocr.mode: undo`` configured (no follow_timer / resync), the detector then
    stays one split ahead of LiveSplit forever."""
    r = socket_death()
    c = client_for(script, timeout=0.2)
    _arm_hole(script, hole)
    _ = run_exec(r, c)
    time.sleep(0.4)  # let any asynchronous apply land
    c.close()
    assert script.idx == 0, "scenario broken: unsplit should have been applied"
    # advance frames so the machine settles; nothing in this config can repair it
    for _ in range(10):
        _ = r.m.feed_result(res(0.0))
    assert r.m.expected_ls_index == script.idx, (
        f"hole={hole}: detector at {r.m.expected_ls_index}, LiveSplit at {script.idx}"
    )


def test_server_dies_after_precondition_never_rolls_back_alone(script: Script) -> None:
    r = socket_death()
    c = client_for(script, timeout=0.2)
    script.close_after["getsplitindex"] = 1  # dies on the first idx query
    kind = run_exec(r, c)
    assert kind is OutcomeKind.SKIPPED_UNAVAILABLE
    assert script.sent() == []
    assert r.m.current_split_index == 1
    c.close()


def test_hung_server_attempt_is_bounded_and_sends_nothing(script: Script) -> None:
    r = socket_death()
    c = client_for(script, timeout=0.2)
    script.delay["getcurrenttimerphase"] = 1.0
    t0 = time.monotonic()
    kind = run_exec(r, c)
    dt = time.monotonic() - t0
    assert kind is OutcomeKind.SKIPPED_UNAVAILABLE and dt < 0.2 * 2 + 0.3
    assert script.sent() == []
    c.close()


def test_slow_verify_reply_is_unverified_and_later_queries_are_not_misaligned(script: Script) -> None:
    r = socket_death()
    c = client_for(script, timeout=0.15)
    script.delay_nth[("getsplitindex", 2)] = 0.4
    t0 = time.monotonic()
    kind = run_exec(r, c)
    dt = time.monotonic() - t0
    # Increment-4 reconcile: the unanswered verify is re-read (same applied
    # unsplit, confirmed late) instead of being left UNVERIFIED.
    assert kind is OutcomeKind.APPLIED and dt < 0.15 + 0.3 + 0.25
    time.sleep(0.5)
    # the late reply must be discarded, never mistaken for this answer
    assert c.get_current_timer_phase() == "Running"
    assert c.get_split_index() == 0
    c.close()


def test_worst_case_attempt_blocks_at_most_two_timeouts(script: Script) -> None:
    r = socket_death()
    timeout = 0.2
    c = client_for(script, timeout=timeout)
    # phase answers, idx hangs
    script.delay["getsplitindex"] = 2.0
    t0 = time.monotonic()
    _ = run_exec(r, c)
    assert time.monotonic() - t0 < 2 * timeout + 0.3
    c.close()


def test_unsplit_wire_bytes_and_no_extra_commands(script: Script) -> None:
    r = socket_death()
    c = client_for(script)
    _ = run_exec(r, c)
    for _ in range(5):
        _ = mainmod._undo_service(c, r.m, r.m._config, True, UndoExecutor())  # pyright: ignore[reportPrivateUsage]
    assert set(script.lines) <= {"getcurrenttimerphase", "getsplitindex", "unsplit"}
    assert script.lines.count("unsplit") == 1
    c.close()


def test_connection_refused_then_server_appears_attempt_is_not_retried() -> None:
    dead = socket.socket()
    dead.bind(("127.0.0.1", 0))
    port = cast("tuple[str, int]", dead.getsockname())[1]
    dead.close()
    r = socket_death()
    c = LiveSplitClient("127.0.0.1", port, timeout_seconds=0.1, reconnect_seconds=0.01)
    assert run_exec(r, c) is OutcomeKind.SKIPPED_UNAVAILABLE
    assert r.m.undo_target is None  # consumed
    assert r.m.current_split_index == 1


# --------------------------------------------------------------------------
# 8. randomised twin
# --------------------------------------------------------------------------

SEEDS = list(range(40))


def _user(ls: SLS, rng: random.Random) -> None:
    rng.choice([SLS.u_split, SLS.u_unsplit, SLS.u_reset, SLS.u_pause, SLS.u_unsplit])(ls)


def _user_hook(seed: int) -> Callable[[SLS], None]:
    def hook(ls: SLS) -> None:
        _user(ls, random.Random(seed))

    return hook


@pytest.mark.parametrize("seed", SEEDS)
def test_fuzz_invariants(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.choice([2, 3, 4, 5])
    s = Sim(n, start=rng.random() < 0.2)
    if s.r.m.waiting_for_start:
        s.fire_clear("")
    s.ls.idx = 0
    s.ls.phase = "Running"
    sent_before = 0
    unsplits_per_verdict = 0
    for _step in range(rng.randint(4, 14)):
        if s.r.m.finished:
            break
        d = s.r.m.current_split_index
        kind = rng.choice(["genuine", "death", "lag", "flash_death", "user", "hiccup"])
        hud_old = str(d)
        hud_new = str(d + 1)
        s.ls.arm()
        if rng.random() < 0.4:
            call = rng.randint(1, 5)
            s.ls.hooks = {call: [_user_hook(rng.randrange(10**9))]}
        if kind == "hiccup":
            k = rng.choice(["phase", "idx", "unsplit", "split"])
            s.ls.faults[k] = [rng.choice(["fail", "apply_fail"] if k in ("unsplit", "split") else ["fail"])]
        if kind == "user":
            _user(s.ls, rng)
        s.agree()
        s.fire_clear(hud_old)
        sent_before = len(s.ls.sent)
        ticks = rng.randint(10, 60)
        for t in range(ticks):
            if kind == "genuine":
                text = hud_new
            elif kind == "lag":
                text = hud_old if t < ticks // 2 else hud_new
            else:
                text = hud_old
            ratio = 1.0 if (kind == "flash_death" and t in (3, 4, 5)) else 0.0
            _ = s.tick(ratio, text)
            assert set(s.ls.log) <= ALLOWED
        sent = s.ls.sent[sent_before:]
        assert set(sent) <= {"split", "unsplit"}
        unsplits_per_verdict = sent.count("unsplit")
        assert unsplits_per_verdict <= 2  # one per verdict window, a flash may add a window
        assert s.r.m.epoch >= 0
        # machine sanity
        assert 0 <= s.r.m.current_split_index <= n
        s.ls.phase = "Running"
        s.ls.idx = max(s.ls.idx, 0)
        s.ls.log.clear()
        s.ls.user.clear()
        # resync the twin for the next scenario (the harness is the 'user' here)
        s.ls.idx = s.r.m.expected_ls_index if s.ls.faults == {} else s.ls.idx
        s.ls.faults = {}
        s.ls.arm()
        if s.ls.idx != s.r.m.expected_ls_index:
            s.ls.idx = s.r.m.expected_ls_index


@pytest.mark.parametrize("seed", SEEDS)
def test_fuzz_no_hiccup_no_user_stays_in_sync(seed: int) -> None:
    """Without faults or users the detector and LiveSplit must agree after
    every scenario (death / genuine / lag / flash)."""
    rng = random.Random(1000 + seed)
    n = rng.choice([3, 4, 5])
    s = Sim(n)
    s.ls.idx = 0
    for _ in range(rng.randint(2, 6)):
        if s.r.m.finished or s.r.m.current_split_index >= n - 1:
            break
        d = s.r.m.current_split_index
        kind = rng.choice(["genuine", "death", "lag"])
        s.agree()
        s.fire_clear(str(d))
        total = rng.randint(20, 60)
        for t in range(total):
            text = {"genuine": str(d + 1), "death": str(d), "lag": str(d) if t < total // 2 else str(d + 1)}[kind]
            _ = s.tick(0.0, text)
        assert s.ls.log == [] or set(s.ls.log) <= {"phase", "idx", "unsplit", "split"}
        # settle clear
        _ = s.tick(0.0, str(s.r.m.current_split_index + 1))
        assert s.r.m.expected_ls_index == s.ls.idx, (kind, d, s.ls.sent, s.r.m.state)
        s.ls.log.clear()


# --------------------------------------------------------------------------
# 9. hot loop cost with nothing pending
# --------------------------------------------------------------------------


def test_service_idle_cost_and_allocations() -> None:
    s = Sim()
    s.agree()
    c = s.ls
    m = s.r.m
    cfg = m._config  # pyright: ignore[reportPrivateUsage]
    ex = s.ex
    for _ in range(100):
        _ = mainmod._undo_service(c, m, cfg, True, ex)  # pyright: ignore[reportArgumentType, reportPrivateUsage]
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        _ = mainmod._undo_service(c, m, cfg, True, ex)  # pyright: ignore[reportArgumentType, reportPrivateUsage]
    per = (time.perf_counter() - t0) / n
    assert per < 20e-6, f"{per * 1e6:.1f} us per idle service call"
    assert c.log == []
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    for _ in range(5000):
        _ = mainmod._undo_service(c, m, cfg, True, ex)  # pyright: ignore[reportArgumentType, reportPrivateUsage]
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    growth = sum(st.size_diff for st in snap1.compare_to(snap0, "filename") if "tracemalloc" not in str(st.traceback))
    assert growth < 4096, growth


def test_check_ocr_resync_idle_cost_unchanged_by_recovery_fields() -> None:
    r = Rig(4, OCR_MODE_UNDO)
    r.agree()
    n = 20000
    t0 = time.perf_counter()
    for i in range(n):
        r.m.check_ocr_resync(FRAME, r.frame + i)
    per = (time.perf_counter() - t0) / n
    assert per < 30e-6, f"{per * 1e6:.1f} us"


# --------------------------------------------------------------------------
# 10. config and --check
# --------------------------------------------------------------------------

BASE = (
    "capture: {fps: 60, stream_url: 'udp://127.0.0.1:9733', width: 64, height: 36}\n"
    + "roi: {x: 0, y: 0, width: 1, height: 1}\n"
    + "detection: {required_frames: 2, reset_threshold: 0.5}\n"
    + "colors: {black: {rgb: [0, 0, 0], tolerance: 25}}\n"
    + "splits:\n  - {id: 1, name: One, color: black, threshold: 0.9}\n"
    + "  - {id: 2, name: Two, color: black, threshold: 0.9}\n"
    + "livesplit: {host: '127.0.0.1', port: 16834}\n"
)
ROI = "ocr:\n  roi: {x: 0, y: 0, width: 0.5, height: 0.5}\n"
PAT = "  patterns: {'(\\d+)': null}\n"


def _cfg(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "c.yaml"
    base = BASE
    if "mode: undo" in body:  # now requires follow_timer
        base = BASE.replace("port: 16834}", "port: 16834, follow_timer: true}")
    p.write_text(base + body, encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "val",
    ["0", "-1", "'x'", "null", "[]", ".nan", ".inf", "-.inf", "true", "false", "1" + "0" * 400],
)
def test_recover_seconds_bad_values_are_config_errors(tmp_path: Path, val: str) -> None:
    p = _cfg(tmp_path, ROI + PAT + "  mode: undo\n  undo: {recover_seconds: " + val + "}\n")
    try:
        c = load(p)
    except ConfigError:
        return
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"non-ConfigError escaped: {type(exc).__name__}: {exc}")
    assert c.ocr is not None and c.ocr.undo is not None
    assert math.isfinite(c.ocr.undo.recover_seconds) and c.ocr.undo.recover_seconds > 0


@pytest.mark.parametrize("val", ["1", "0", "'yes'", "null", "[]", "2.5"])
def test_recover_late_hud_non_bool_is_config_error(tmp_path: Path, val: str) -> None:
    p = _cfg(tmp_path, ROI + PAT + "  mode: undo\n  undo: {recover_late_hud: " + val + "}\n")
    with pytest.raises(ConfigError):
        _ = load(p)


def test_mode_undo_without_patterns_is_config_error(tmp_path: Path) -> None:
    p = _cfg(tmp_path, ROI + "  mode: undo\n")
    with pytest.raises(ConfigError, match="patterns"):
        _ = load(p)


def test_mode_undo_with_enabled_false_warns_and_never_acts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    srv = Script()
    try:
        p = tmp_path / "live.yaml"
        text = _cfg(
            tmp_path,
            ROI + PAT + "  enabled: false\n  mode: undo\n",
        ).read_text(encoding="utf-8").replace("port: 16834", f"port: {srv.port}")
        p.write_text(text, encoding="utf-8")

        class Src:
            exhausted: bool = False

            def __init__(self, *_a: object, **_k: object) -> None:
                self.n: int = 0

            def read(self) -> np.ndarray:
                self.n += 1
                return np.full((36, 64, 3), 0 if self.n < 6 else 255, np.uint8)

            def close(self) -> None:
                pass

        monkeypatch.setattr(mainmod, "MpegTsFrameSource", Src)
        rc = mainmod.main(["--config", str(p), "--live", "x.ts", "--max-frames", "40", "--livesplit"])
        out = capsys.readouterr().out
    finally:
        srv.close()
    assert rc == 0
    assert "WARN: ocr.mode undo but ocr.enabled is false; no undo will happen" in out
    assert "unsplit" not in srv.lines and srv.sent().count("split") >= 1


def test_check_does_not_claim_active_unsplit_when_ocr_is_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--check summary says 'ACTIVE (will send unsplit)' although ocr.enabled is
    false and `_run_live` will never undo (it prints a WARN at runtime instead)."""
    p = _cfg(tmp_path, ROI + PAT + "  enabled: false\n  mode: undo\n")
    assert mainmod.main(["--config", str(p), "--check"]) == 0
    out = capsys.readouterr().out
    assert "ocr backend: disabled" in out
    assert "ACTIVE" not in out


def test_stale_numeric_reply_from_a_timed_out_poll_never_triggers_an_unsplit(script: Script) -> None:
    """A previous idx query timed out; its late numeric reply must not be read as
    the executor's phase/idx answers and cause a send based on stale data."""
    r = socket_death()
    c = client_for(script, timeout=0.2)
    script.delay_nth[("getsplitindex", 1)] = 0.35
    assert c.get_split_index() is None  # the poller's timed-out query
    kind = run_exec(r, c)
    time.sleep(0.5)
    assert script.sent() in ([], ["unsplit"])
    if script.sent() == ["unsplit"]:
        assert kind in (OutcomeKind.APPLIED, OutcomeKind.UNVERIFIED), kind
    assert r.m.expected_ls_index in (script.idx, 1)
    c.close()


# --------------------------------------------------------------------------
# 11. which configuration repairs which hole (twins of the failing tests)
# --------------------------------------------------------------------------


def test_apply_fail_resplit_is_repaired_by_follow_timer() -> None:
    s = undone_sim()
    poller = mainmod._FollowPoller(4)  # pyright: ignore[reportPrivateUsage]
    _ = poller.poll(s.ls, s.r.m, None, None, False)  # pyright: ignore[reportArgumentType]  # arms
    s.ls.log.clear()
    s.ls.faults["split"] = ["apply_fail"]
    s.ls.arm()
    for _ in range(6):
        _ = s.tick(0.0, "2")
    # the hole is closed by the executor's own reconcile (no follow_timer needed)
    assert s.ls.idx == 1 and s.r.m.expected_ls_index == 1
    for _ in range(4):
        _ = poller.poll(s.ls, s.r.m, None, None, False)  # pyright: ignore[reportArgumentType]
    assert s.r.m.expected_ls_index == s.ls.idx == 1


def test_apply_fail_unsplit_is_repaired_by_resync_on_unsplit_watch() -> None:
    """The UNVERIFIED / NOT DELIVERED log lines promise resync_on_unsplit repairs it."""
    from autosplitter.livesplit import UndoWatch

    s = committed_death()
    watch = UndoWatch()
    assert watch.observe("Running", s.ls.idx, s.r.m.expected_ls_index, 1.0, None, False) is None  # arm
    s.ls.log.clear()
    s.ls.faults["unsplit"] = ["apply_fail"]
    s.ls.arm()
    confirm_until_acted(s, "1")
    # the executor reconciles the "undelivered" (but applied) unsplit itself
    assert s.ls.idx == 0 and s.r.m.expected_ls_index == 0
    for i in range(3):  # ...so the watch has nothing left to repair (no double rollback)
        assert watch.observe("Running", s.ls.idx, s.r.m.expected_ls_index, 10.0 + i, None, False) is None


def test_async_applied_resplit_must_not_leave_detector_behind(script: Script) -> None:
    """Recovery `split` applied by LiveSplit a moment after the verify query."""
    script.idx = 1
    r = socket_death()
    c = client_for(script, timeout=0.3)
    assert run_exec(r, c) is OutcomeKind.APPLIED
    r.m.feed_result(res(0.0))  # the undo's settle clear
    for _ in range(5):  # S+1 streak must span undo.confirm_seconds (1.0 s)
        r.read("2")
    assert r.m.recovery_target == 1
    script.async_apply["split"] = 0.15
    out = UndoExecutor().recover(c, r.m, time.monotonic())
    assert out is not None
    time.sleep(0.4)
    c.close()
    assert script.idx == 1
    assert r.m.expected_ls_index == script.idx, f"detector {r.m.expected_ls_index}, LiveSplit {script.idx}"


# --------------------------------------------------------------------------
# 12. frame-loop level: stall bound and follow_timer flip-flop (real _run_live)
# --------------------------------------------------------------------------

from test_undo_live import (  # pyright: ignore[reportImplicitRelativeImport]
    Hud,
    LsServer,
    Worker,
    write_cfg,
)


def slow_server(idx_delay: float) -> LsServer:
    """LsServer whose getsplitindex answers late (serial, like a busy UI thread)."""
    srv = LsServer()
    real = srv._handle  # pyright: ignore[reportPrivateUsage]

    def handle(line: str) -> str | None:
        if line == "getsplitindex":
            time.sleep(idx_delay)
        return real(line)

    setattr(srv, "_handle", handle)  # noqa: B010
    return srv


def _live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    server: LsServer,
    hud_mode: str,
    frames: int,
    timeout: float = 0.5,
    follow: bool = False,
) -> tuple[str, list[float]]:
    from autosplitter.detection import state_machine as sm

    stamps: list[float] = []
    hud = Hud(server, hud_mode)

    class Src:
        exhausted: bool = False

        def __init__(self, *_a: object, **_k: object) -> None:
            self.n: int = 0

        def read(self) -> np.ndarray:
            time.sleep(0.004)
            stamps.append(time.monotonic())
            self.n += 1
            return np.full((36, 64, 3), 0 if self.n <= 4 else 255, np.uint8)

        def close(self) -> None:
            pass

    def fake_build(_c: object) -> tuple[Worker, str]:
        return Worker(hud), "fake OK"

    monkeypatch.setattr(sm, "build_ocr_worker", fake_build)
    monkeypatch.setattr(mainmod, "MpegTsFrameSource", Src)
    p = write_cfg(tmp_path, server.port, "undo")
    text = p.read_text(encoding="utf-8").replace("timeout_seconds: 0.5", f"timeout_seconds: {timeout}")
    if follow:
        text = text.replace("poll_seconds: 30", "poll_seconds: 0.3")
    p.write_text(text, encoding="utf-8")
    rc = mainmod.main(["--config", str(p), "--live", "x.ts", "--max-frames", str(frames), "--livesplit"])
    assert rc == 0
    return capsys.readouterr().out, stamps


def test_slow_server_stalls_the_frame_loop_for_at_most_two_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    srv = slow_server(1.0)
    timeout = 0.25
    try:
        out, stamps = _live(tmp_path, monkeypatch, capsys, srv, "death", 400, timeout)
    finally:
        srv.stop = True
        time.sleep(1.2)  # let the serial handler finish its sleep
        srv.close()
    gaps = [b - a for a, b in itertools.pairwise(stamps)]
    assert max(gaps) < 2 * timeout + 0.3, f"max inter-frame gap {max(gaps):.2f}s"
    assert len(stamps) == 400  # frames keep flowing after the stall
    assert srv.moves() == ["split"]  # nothing but the colour split ever sent
    assert "undo skipped (server unavailable)" in out
    assert out.count("undo skipped") == 1


@pytest.mark.parametrize(
    ("hud_mode", "expected"),
    [("death", ["split", "unsplit"]), ("late", ["split", "unsplit", "split"]), ("genuine", ["split"])],
)
def test_follow_timer_with_undo_mode_never_flip_flops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    hud_mode: str, expected: list[str],
) -> None:
    srv = LsServer()
    try:
        out, _ = _live(tmp_path, monkeypatch, capsys, srv, hud_mode, 220, follow=True)
    finally:
        srv.close()
    assert srv.moves() == expected, out
    assert "UNSPLIT detected" not in out and "SKIP/manual split detected" not in out, out
