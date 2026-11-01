"""Increment 4: apply_undo, the unsplit executor and the late-HUD recovery."""

from __future__ import annotations

import dataclasses

import pytest
from test_undo_machine import (  # pyright: ignore[reportImplicitRelativeImport]
    OCR_U,
    UNDO,
    Rig,
    res,
    snapshot,
)

from autosplitter.detection import State
from autosplitter.detection.state_machine import SplitEvent
from autosplitter.detection.undo_judge import VerdictKind
from autosplitter.livesplit import (
    Action,
    OutcomeKind,
    TimerFollower,
    UndoExecutor,
    UndoOutcome,
    UndoWatch,
)

OCR_MODE_UNDO = dataclasses.replace(OCR_U, mode="undo")


class FakeClient:
    """Duck-typed LiveSplit client with LiveSplit's real unsplit/split rules."""

    def __init__(self, idx: int | None = 1, phase: str | None = "Running") -> None:
        self.idx: int | None = idx
        self.phase: str | None = phase
        self.calls: list[str] = []
        self.unsplit_ok: bool = True
        self.split_ok: bool = True
        self.apply: bool = True  # server honours the command
        self.extra_back: int = 0  # simultaneous user unsplit(s)
        self.verify_none_after_send: bool = False
        self._sent: bool = False
        self.last_send_wrote: bool = True  # LiveSplitClient attribute (see client.py)

    def get_current_timer_phase(self) -> str | None:
        self.calls.append("phase")
        return self.phase

    def get_split_index(self) -> int | None:
        self.calls.append("idx")
        if self.verify_none_after_send and self._sent:
            return None
        return self.idx

    def unsplit(self) -> bool:
        self.calls.append("unsplit")
        if not self.unsplit_ok:
            self.last_send_wrote = False
            return False
        self._sent = True
        if self.apply and self.idx is not None and self.idx > 0 and self.phase == "Running":
            self.idx = max(0, self.idx - 1 - self.extra_back)
        return True

    def split(self) -> bool:
        self.calls.append("split")
        if not self.split_ok:
            self.last_send_wrote = False
            return False
        self._sent = True
        if self.apply and self.idx is not None and self.phase == "Running":
            self.idx += 1
        return True


def confirmed_death(n_prev: int = 0) -> Rig:
    """Rig whose detector sits at index n_prev+1 with a confirmed UNDO verdict."""
    r = Rig()
    for k in range(n_prev + 1):
        r.agree()
        r.fire_and_clear()
        if k < n_prev:
            r.read(str(k + 2))  # genuine: HUD advanced
            r.events()
    for _ in range(40):
        r.read(str(n_prev + 1))
    assert [v.kind for v in r.events()] == [VerdictKind.UNDO]
    assert r.m.undo_target == n_prev
    assert r.m.state is State.NORMAL and r.m.current_split_index == n_prev + 1
    return r


def run_attempt(r: Rig, c: FakeClient) -> UndoOutcome:
    target = r.m.undo_target
    assert target is not None
    return UndoExecutor().attempt(c, r.m, target, 123.0)


# ------------------------------------------------------------------ machine


def test_apply_undo_from_normal_ends_in_settle_with_bumped_epoch() -> None:
    r = confirmed_death()
    epoch = r.m.epoch
    assert r.m.apply_undo(0)
    assert r.m.state is State.WAITING_FOR_RESET and r.m.current_split_index == 0
    assert r.m.epoch > epoch and r.m.expected_ls_index == 0
    assert r.m.undo_target is None and not r.m.undo_pending
    assert r.m.take_undo_event() is None


def test_apply_undo_still_lit_colour_does_not_refire_until_reset_threshold() -> None:
    r = confirmed_death()
    assert r.m.apply_undo(0)
    for _ in range(20):
        assert r.m.feed_result(res(1.0)) is None
    assert r.m.feed_result(res(0.5)) is None  # == reset_threshold: dead zone
    assert r.m.state is State.WAITING_FOR_RESET
    assert r.m.feed_result(res(0.49)) is None
    assert r.m.state is State.NORMAL and r.m.current_split_index == 0
    assert r.m.take_undo_event() is None and not r.m.undo_pending  # no window reopened
    r.m.feed_result(res(1.0))
    ev = r.m.feed_result(res(1.0))
    assert isinstance(ev, SplitEvent) and ev.split_id == 1  # split 0 fires normally again


def test_apply_undo_from_waiting_for_reset_cancels_pending_advance() -> None:
    r = Rig()
    r.agree()
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    assert r.m.state is State.WAITING_FOR_RESET and r.m.expected_ls_index == 1
    assert r.m.apply_undo(0)
    r.m.feed_result(res(0.0))
    assert r.m.state is State.NORMAL and r.m.current_split_index == 0


@pytest.mark.parametrize("idx", [-1, 1, 2, 9])
def test_apply_undo_refuses_bad_index(idx: int) -> None:
    r = confirmed_death()  # expected_ls_index == 1
    before = snapshot(r.m)
    assert not r.m.apply_undo(idx)
    assert snapshot(r.m) == before


def test_apply_undo_refused_in_transition_and_waiting_for_start() -> None:
    r = Rig()
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    assert r.m.state is State.TRANSITION
    before = snapshot(r.m)
    assert not r.m.apply_undo(0)
    assert snapshot(r.m) == before
    s = Rig(start=True)
    assert s.m.state is State.WAITING_FOR_START and not s.m.apply_undo(0)


def test_rollback_to_from_normal_unchanged() -> None:
    r = confirmed_death()
    assert r.m.rollback_to(0)
    assert r.m.state is State.NORMAL  # user-unsplit path keeps its old semantics


# ----------------------------------------------------------------- executor


def test_success_sends_unsplit_verifies_and_rolls_back() -> None:
    r = confirmed_death()
    c = FakeClient(idx=1)
    out = run_attempt(r, c)
    assert out.kind is OutcomeKind.APPLIED and out.changed and out.time == 123.0
    assert out.lines == ("LiveSplit: UNSPLIT sent -> index 0 (verified)",)
    assert c.calls == ["phase", "idx", "unsplit", "idx"]
    assert r.m.current_split_index == 0 and r.m.state is State.WAITING_FOR_RESET
    assert r.m.undo_target is None


def test_already_at_target_rolls_back_without_sending() -> None:
    r = confirmed_death()
    c = FakeClient(idx=0)
    out = run_attempt(r, c)
    assert out.kind is OutcomeKind.ALREADY_THERE and out.changed
    assert "unsplit" not in c.calls and r.m.current_split_index == 0


def test_simultaneous_user_unsplit_rolls_back_to_livesplit() -> None:
    r = confirmed_death(n_prev=2)  # detector at 3, S = 2
    c = FakeClient(idx=3)
    c.extra_back = 1  # user also pressed unsplit: LiveSplit ends at 1
    out = run_attempt(r, c)
    assert out.kind is OutcomeKind.USER_UNSPLIT and out.changed
    assert any(line.startswith("WARN") for line in out.lines)
    assert r.m.current_split_index == 1 and r.m.expected_ls_index == 1


def test_ignored_unsplit_leaves_detector_untouched() -> None:
    r = confirmed_death()
    c = FakeClient(idx=1)
    c.apply = False
    before = snapshot(r.m)
    out = run_attempt(r, c)
    assert out.kind is OutcomeKind.IGNORED and not out.changed
    assert "no effect" in out.lines[0]
    assert snapshot(r.m) == before and r.m.undo_target is None


def test_unverified_does_not_roll_back() -> None:
    r = confirmed_death()
    c = FakeClient(idx=1)
    c.verify_none_after_send = True
    before = snapshot(r.m)
    out = run_attempt(r, c)
    assert out.kind is OutcomeKind.UNVERIFIED and not out.changed
    assert "detector left at 1" in out.lines[0]
    assert snapshot(r.m) == before


@pytest.mark.parametrize("which", ["phase", "idx"])
def test_server_unavailable_skips_and_never_rolls_back(which: str) -> None:
    r = confirmed_death()
    c = FakeClient(idx=None if which == "idx" else 1, phase=None if which == "phase" else "Running")
    before = snapshot(r.m)
    out = run_attempt(r, c)
    assert out.kind is OutcomeKind.SKIPPED_UNAVAILABLE and not out.changed
    assert out.lines == ("LiveSplit: undo skipped (server unavailable)",)
    assert "unsplit" not in c.calls
    assert snapshot(r.m) == before and r.m.undo_target is None


def test_undelivered_unsplit_no_rollback() -> None:
    r = confirmed_death()
    c = FakeClient(idx=1)
    c.unsplit_ok = False
    before = snapshot(r.m)
    out = run_attempt(r, c)
    assert out.kind is OutcomeKind.NOT_DELIVERED and not out.changed
    assert out.lines == ("LiveSplit: unsplit NOT DELIVERED (server unreachable)",)
    assert snapshot(r.m) == before


@pytest.mark.parametrize("phase", ["Paused", "NotRunning", "Ended", "Weird"])
def test_timer_not_running_skips(phase: str) -> None:
    r = confirmed_death()
    c = FakeClient(idx=1, phase=phase)
    before = snapshot(r.m)
    out = run_attempt(r, c)
    assert out.kind is OutcomeKind.SKIPPED and f"timer is {phase}" in out.lines[0]
    assert "unsplit" not in c.calls and snapshot(r.m) == before


@pytest.mark.parametrize("ls", [2, 5, -1])
def test_unexpected_livesplit_index_skips(ls: int) -> None:
    r = confirmed_death()  # D = 1; D+1 is a manual split: leave it to follow_timer
    c = FakeClient(idx=ls)
    before = snapshot(r.m)
    out = run_attempt(r, c)
    assert out.kind is OutcomeKind.SKIPPED and out.lines[0].startswith("WARN")
    assert "unsplit" not in c.calls and snapshot(r.m) == before


def test_one_attempt_per_verdict_no_requeries() -> None:
    r = confirmed_death()
    c = FakeClient(idx=None)
    run_attempt(r, c)
    n = len(c.calls)
    out = UndoExecutor().attempt(c, r.m, 0, 124.0)  # same verdict again, frames later
    assert out.kind is OutcomeKind.SKIPPED and len(c.calls) == n
    for _ in range(20):
        r.read("1")
    assert r.m.undo_target is None


def test_state_changed_since_verdict_makes_no_queries() -> None:
    r = confirmed_death()
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))  # new colour fire voids the verdict
    c = FakeClient(idx=1)
    out = UndoExecutor().attempt(c, r.m, 0, 1.0)
    assert out.kind is OutcomeKind.SKIPPED and c.calls == []


def test_no_flip_flop_with_follower_or_undo_watch_after_apply() -> None:
    r = confirmed_death()
    c = FakeClient(idx=1)
    assert run_attempt(r, c).kind is OutcomeKind.APPLIED
    follower = TimerFollower()
    watch = UndoWatch()
    for i in range(10):
        decision = follower.observe(
            phase="Running", ls_index=c.idx, expected=r.m.expected_ls_index, n_splits=4,
            now=1000.0 + 10 * i, last_split_time=None, suppress=False,
            can_back=True, can_fwd=True, waiting_for_start=False,
        )
        assert decision.action is Action.NONE
        assert watch.observe("Running", c.idx, r.m.expected_ls_index, 1000.0 + 10 * i, None, False) is None
    assert r.m.current_split_index == 0


def test_no_flip_flop_after_ignored_unsplit() -> None:
    r = confirmed_death()
    c = FakeClient(idx=1)
    c.apply = False
    run_attempt(r, c)
    follower = TimerFollower()
    for i in range(10):
        decision = follower.observe(
            phase="Running", ls_index=c.idx, expected=r.m.expected_ls_index, n_splits=4,
            now=1000.0 + 10 * i, last_split_time=None, suppress=False,
            can_back=True, can_fwd=True, waiting_for_start=False,
        )
        assert decision.action is Action.NONE


# ----------------------------------------------------------------- recovery


def late_hud(r: Rig, n: int = 5) -> None:
    """HUD shows S+1 for ``n`` reads (default spans 1.0 s = undo.confirm_seconds)."""
    for _ in range(n):
        r.read("2")


def undone_rig(ocr: object = OCR_MODE_UNDO) -> tuple[Rig, FakeClient]:
    from autosplitter.config import OcrConfig

    assert isinstance(ocr, OcrConfig)
    r = Rig(ocr=ocr)
    r.agree()
    r.fire_and_clear()
    for _ in range(40):
        r.read("1")
    assert r.m.undo_target == 0
    c = FakeClient(idx=1)
    assert run_attempt(r, c).kind is OutcomeKind.APPLIED
    c.calls.clear()
    r.m.feed_result(res(0.0))  # the undo's own settle clear
    assert r.m.state is State.NORMAL and r.m.current_split_index == 0
    return r, c


def test_late_hud_recovery_resplits_once() -> None:
    r, c = undone_rig()
    assert r.m.recovery_target is None
    late_hud(r)
    assert r.m.recovery_target == 1
    ex = UndoExecutor()
    out = ex.recover(c, r.m, 500.0)
    assert out is not None and out.kind is OutcomeKind.RECOVERED and out.changed
    assert out.lines[0].startswith("OCR: undo regret — HUD advanced late (+")
    assert out.lines[0].endswith("s after undo); re-splitting")
    assert out.lines[1] == "LiveSplit: SPLIT re-sent -> index 1 (verified)"
    # phase, idx, idx (re-read right before sending), split, idx (verify)
    assert c.calls == ["phase", "idx", "idx", "split", "idx"] and c.idx == 1
    assert r.m.current_split_index == 1 and r.m.state is State.WAITING_FOR_RESET
    assert ex.recover(c, r.m, 501.0) is None  # at most once
    late_hud(r)
    assert r.m.recovery_target is None


def test_recovery_needs_every_read_above_the_undo_floor() -> None:
    # filter floor 50, undo floor 80: consensus must not be completed by a
    # streak whose earlier reads were only filter-grade.
    r, _ = undone_rig()
    r.read("2", conf=70.0)
    r.read("2", conf=90.0)
    assert r.m.recovery_target is None
    for _ in range(3):
        r.read("2", conf=90.0)  # streak of 4 at >= 80 spans only 0.75 s
        assert r.m.recovery_target is None
    r.read("2", conf=90.0)  # 5th: spans 1.0 s == confirm_seconds
    assert r.m.recovery_target == 1


def test_recovery_streak_must_span_confirm_seconds() -> None:
    r, _ = undone_rig()
    for _ in range(4):
        r.read("2")  # 4 reads, 0.75 s < confirm_seconds 1.0
    assert r.m.recovery_target is None
    r.read("2")
    assert r.m.recovery_target == 1


def test_recovery_streak_breaks_on_other_reading() -> None:
    r, _ = undone_rig()
    r.read("2", conf=90.0)
    r.read("1", conf=90.0)
    for _ in range(4):
        r.read("2", conf=90.0)
    assert r.m.recovery_target is None
    r.read("2", conf=90.0)
    assert r.m.recovery_target == 1


def test_new_colour_fire_cancels_recovery() -> None:
    r, c = undone_rig()
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))  # fires split 0 again
    late_hud(r)
    assert r.m.recovery_target is None
    assert UndoExecutor().recover(c, r.m, 1.0) is None and c.calls == []


def test_recovery_expires() -> None:
    r, _ = undone_rig()
    r.clock.t += 10.5
    late_hud(r)
    assert r.m.recovery_target is None


def test_recovery_disabled_by_config_and_in_verify_only() -> None:
    off = dataclasses.replace(
        OCR_MODE_UNDO, undo=dataclasses.replace(UNDO, recover_late_hud=False)
    )
    r, _ = undone_rig(off)
    late_hud(r)
    assert r.m.recovery_target is None
    v = Rig()  # verify_only (dry-run config): apply_undo never arms
    v.agree()
    v.fire_and_clear()
    assert v.m.apply_undo(0)
    v.m.feed_result(res(0.0))
    v.read("2")
    v.read("2")
    assert v.m.recovery_target is None


def test_no_recovery_after_already_there_rollback() -> None:
    r2 = Rig(ocr=OCR_MODE_UNDO)
    r2.agree()
    r2.fire_and_clear()
    for _ in range(40):
        r2.read("1")
    assert run_attempt(r2, FakeClient(idx=0)).kind is OutcomeKind.ALREADY_THERE
    r2.m.feed_result(res(0.0))
    r2.read("2")
    r2.read("2")
    assert r2.m.recovery_target is None


@pytest.mark.parametrize(
    "setup",
    ["not_delivered", "no_effect", "unverified", "paused", "wrong_index", "down"],
)
def test_recovery_failures_do_nothing_and_are_consumed(setup: str) -> None:
    r, c = undone_rig()
    late_hud(r)
    assert r.m.recovery_target == 1
    if setup == "not_delivered":
        c.split_ok = False
    elif setup == "no_effect":
        c.apply = False
    elif setup == "unverified":
        c.verify_none_after_send = True
    elif setup == "paused":
        c.phase = "Paused"
    elif setup == "wrong_index":
        c.idx = 3
    else:
        c.phase = None
    before = snapshot(r.m)
    out = UndoExecutor().recover(c, r.m, 9.0)
    assert out is not None and out.kind is OutcomeKind.RECOVERY_SKIPPED and not out.changed
    assert any("WARN" in line or "NOT DELIVERED" in line for line in out.lines)
    assert snapshot(r.m) == before
    assert UndoExecutor().recover(c, r.m, 10.0) is None
    late_hud(r)
    assert r.m.recovery_target is None


def test_recovery_state_changed_before_send() -> None:
    r, c = undone_rig()
    late_hud(r)
    assert r.m.rollback_to(0) is False  # (expected == 0: nothing to roll back)
    r.m.reset()
    assert r.m.recovery_target is None
    assert UndoExecutor().recover(c, r.m, 1.0) is None


def test_hot_reload_cannot_flip_recovery() -> None:
    r, c = undone_rig()
    from test_undo_machine import BASE_OCR, cfg  # pyright: ignore[reportImplicitRelativeImport]

    r.m.reload_config(cfg(4, BASE_OCR))
    late_hud(r)
    assert r.m.recovery_target == 1
    assert c.calls == []
