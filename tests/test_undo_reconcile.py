"""Increment 4 send-back: ambiguity is reconciled, not guessed.

Covers the deferred/synchronous reconcile of an `unsplit` / recovery `split`
whose effect the write path could not confirm, the pre-send re-read, recovery
span, the NOTE and the --check text.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_undo_exec import (  # pyright: ignore[reportImplicitRelativeImport]
    OCR_MODE_UNDO,
    FakeClient,
    confirmed_death,
    undone_rig,
)
from test_undo_machine import Rig, res  # pyright: ignore[reportImplicitRelativeImport]
from test_undo_writepath import (  # pyright: ignore[reportImplicitRelativeImport]
    SLS,
    Script,
    client_for,
    socket_death,
)

from autosplitter import main as mainmod
from autosplitter.detection import State
from autosplitter.livesplit import (
    Action,
    OutcomeKind,
    TimerFollower,
    UndoExecutor,
)

_ = OCR_MODE_UNDO


class Seq:
    """Client whose getsplitindex replies come from a script (last repeats)."""

    def __init__(self, replies: list[int | None], unsplit_ok: bool = True, wrote: bool = True) -> None:
        self.replies: list[int | None] = replies
        self.n_idx: int = 0
        self.calls: list[str] = []
        self.unsplit_ok: bool = unsplit_ok
        self.last_send_wrote: bool = wrote

    def get_current_timer_phase(self) -> str | None:
        self.calls.append("phase")
        return "Running"

    def get_split_index(self) -> int | None:
        self.calls.append("idx")
        i = min(self.n_idx, len(self.replies) - 1)
        self.n_idx += 1
        return self.replies[i]

    def unsplit(self) -> bool:
        self.calls.append("unsplit")
        return self.unsplit_ok

    def split(self) -> bool:
        self.calls.append("split")
        return self.unsplit_ok


class Clock:
    def __init__(self) -> None:
        self.t: float = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, secs: float) -> None:
        self.slept.append(secs)
        self.t += secs


def executor() -> tuple[UndoExecutor, Clock]:
    c = Clock()
    return UndoExecutor(sleep=c.sleep, clock=c), c


# --------------------------------------------------------------- undo paths
# Seq replies: [precondition idx, verify idx, recheck 1, recheck 2, ...]


def test_eof_after_write_applied_is_confirmed_late() -> None:
    r = confirmed_death()
    ex, _ = executor()
    c = Seq([1, 0], unsplit_ok=False, wrote=True)  # "undelivered" but applied
    out = ex.attempt(c, r.m, 0, 1.0)
    assert out.kind is OutcomeKind.APPLIED and out.changed
    assert out.lines == ("LiveSplit: UNSPLIT applied (confirmed late)",)
    assert r.m.current_split_index == 0 and c.calls.count("unsplit") == 1


def test_eof_after_write_not_applied_drops_it() -> None:
    r = confirmed_death()
    ex, clk = executor()
    c = Seq([1, 1, 1, 1], unsplit_ok=False, wrote=True)
    out = ex.attempt(c, r.m, 0, 1.0)
    assert out.kind is OutcomeKind.IGNORED and not out.changed
    assert "no effect" in out.lines[0] and r.m.current_split_index == 1
    assert clk.slept == [0.075, 0.2] and not ex.pending
    assert c.calls.count("unsplit") == 1


def test_no_reply_to_verify_then_late_answer() -> None:
    r = confirmed_death()
    ex, _ = executor()
    c = Seq([1, None, 0])
    out = ex.attempt(c, r.m, 0, 1.0)
    assert out.kind is OutcomeKind.APPLIED and "confirmed late" in out.lines[0]
    assert r.m.current_split_index == 0


def test_async_apply_seen_on_second_recheck() -> None:
    r = confirmed_death()
    ex, clk = executor()
    c = Seq([1, 1, 1, 0])  # verify D, recheck D, recheck D-1
    out = ex.attempt(c, r.m, 0, 1.0)
    assert out.kind is OutcomeKind.APPLIED and clk.slept == [0.075, 0.2]
    assert r.m.current_split_index == 0 and c.calls.count("unsplit") == 1


def test_simultaneous_user_unsplit_found_during_reconcile() -> None:
    r = confirmed_death(n_prev=2)
    ex, _ = executor()
    c = Seq([3, None, 1])
    out = ex.attempt(c, r.m, 2, 1.0)
    assert out.kind is OutcomeKind.USER_UNSPLIT and r.m.current_split_index == 1


def test_deadline_reached_warns_loudly_and_never_rolls_back() -> None:
    r = confirmed_death()
    ex, clk = executor()
    c = Seq([1, None])  # never answers after the precondition
    first = ex.attempt(c, r.m, 0, 1.0)
    assert first.kind is OutcomeKind.UNVERIFIED and ex.pending
    assert "detector left at 1" in first.lines[0]
    queries = c.calls.count("idx")
    assert ex.pending_step(c, r.m) is None  # rate limited: next_at not reached
    assert c.calls.count("idx") == queries
    final = None
    steps = 0
    while final is None and steps < 40:
        clk.t += 0.5
        steps += 1
        before = c.calls.count("idx")
        final = ex.pending_step(c, r.m)
        assert c.calls.count("idx") - before <= 1  # at most one query per step (backs off)
    assert final is not None and final.kind is OutcomeKind.UNCONFIRMED
    assert "could not confirm unsplit" in final.lines[0]
    assert "follow_timer (required for ocr.mode undo) repairs it" in final.lines[0]
    assert not ex.pending and r.m.current_split_index == 1
    assert c.calls.count("unsplit") == 1  # never a second unsplit


def test_deferred_reconcile_confirms_when_server_returns() -> None:
    r = confirmed_death()
    ex, clk = executor()
    c = Seq([1, None, None, None, 0])
    out = ex.attempt(c, r.m, 0, 1.0)
    assert out.kind is OutcomeKind.UNVERIFIED
    clk.t += 0.6
    done = ex.pending_step(c, r.m)
    assert done is not None and done.kind is OutcomeKind.APPLIED
    assert done.lines == ("LiveSplit: UNSPLIT applied (confirmed late)",)
    assert r.m.current_split_index == 0 and not ex.pending


def test_pending_blocks_new_attempts_and_recoveries_and_sends_nothing() -> None:
    r = confirmed_death()
    ex, _ = executor()
    c = Seq([1, None])
    _ = ex.attempt(c, r.m, 0, 1.0)
    assert ex.pending
    calls = list(c.calls)
    again = ex.attempt(c, r.m, 0, 2.0)
    assert again.kind is OutcomeKind.SKIPPED and "still being confirmed" in again.lines[0]
    assert ex.recover(c, r.m, 3.0) is None
    assert c.calls == calls and c.calls.count("unsplit") == 1


def test_definitely_unsent_is_still_not_delivered_without_reconcile() -> None:
    r = confirmed_death()
    ex, clk = executor()
    c = Seq([1], unsplit_ok=False, wrote=False)
    out = ex.attempt(c, r.m, 0, 1.0)
    assert out.kind is OutcomeKind.NOT_DELIVERED and clk.slept == [] and not ex.pending


def test_pending_does_not_fight_follow_timer_rollback() -> None:
    r = confirmed_death()
    ex, clk = executor()
    c = Seq([1, None, None, None, 0])
    _ = ex.attempt(c, r.m, 0, 1.0)
    assert ex.pending
    follower = TimerFollower()
    arm = follower.observe(  # armed earlier by an in-step poll (LiveSplit == detector)
        phase="Running", ls_index=1, expected=1, n_splits=4, now=900.0, last_split_time=None,
        suppress=False, can_back=True, can_fwd=True, waiting_for_start=False,
    )
    assert arm.action is Action.NONE
    for i in range(3):  # LiveSplit is at D-1: the normal user-unsplit path rolls back
        d = follower.observe(
            phase="Running", ls_index=0, expected=r.m.expected_ls_index, n_splits=4,
            now=1000.0 + i, last_split_time=None, suppress=False,
            can_back=True, can_fwd=True, waiting_for_start=False,
        )
        if d.action is Action.ROLLBACK and d.index is not None:
            assert r.m.rollback_to(d.index)
            break
    assert r.m.current_split_index == 0
    clk.t += 0.6
    done = ex.pending_step(c, r.m)  # must not roll back a second time
    assert done is not None and done.kind is OutcomeKind.SKIPPED and not done.changed
    assert r.m.current_split_index == 0 and not ex.pending


def test_pending_step_abandons_when_detector_moved_on() -> None:
    r = confirmed_death()
    ex, clk = executor()
    c = Seq([1, None, None, None, 1])
    _ = ex.attempt(c, r.m, 0, 1.0)
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))  # colour fires again
    assert r.m.state is State.TRANSITION
    clk.t += 0.6
    done = ex.pending_step(c, r.m)
    assert done is not None and done.kind is OutcomeKind.SKIPPED and not done.changed


# ------------------------------------------------------------ recovery paths
# Seq replies: [precondition, pre-send re-read, verify, rechecks...]


def ready_recovery() -> Rig:
    r, _ = undone_rig()
    for _ in range(5):
        r.read("2")
    assert r.m.recovery_target == 1
    return r


def test_recovery_late_confirm() -> None:
    r = ready_recovery()
    ex, _ = executor()
    c = Seq([0, 0, 0, 1], unsplit_ok=False, wrote=True)
    out = ex.recover(c, r.m, 5.0)
    assert out is not None and out.kind is OutcomeKind.RECOVERED
    assert out.lines[1] == "LiveSplit: SPLIT applied (confirmed late)"
    assert r.m.current_split_index == 1 and c.calls.count("split") == 1


def test_recovery_no_effect_only_logged_when_ls_still_at_s() -> None:
    r = ready_recovery()
    ex, _ = executor()
    out = ex.recover(Seq([0, 0, 0, 0, 0]), r.m, 5.0)
    assert out is not None and out.kind is OutcomeKind.RECOVERY_SKIPPED
    assert "re-split had no effect" in out.lines[1]
    assert "re-splitting" not in out.lines[0] and "recovery not performed" in out.lines[0]
    assert r.m.current_split_index == 0


def test_recovery_unexpected_value_is_not_called_no_effect() -> None:
    r = ready_recovery()
    ex, _ = executor()
    out = ex.recover(Seq([0, 0, -1, -1, -1]), r.m, 5.0)
    assert out is not None and out.kind is OutcomeKind.RECOVERY_SKIPPED
    assert "no effect" not in out.lines[1]


def test_recovery_user_split_raced_follows_livesplit() -> None:
    r = ready_recovery()
    ex, _ = executor()
    c = Seq([0, 0, 2])  # verify shows S+2: a user split slipped in
    out = ex.recover(c, r.m, 5.0)
    assert out is not None and out.kind is OutcomeKind.RECOVERED
    assert "moved ahead during recovery" in out.lines[1] and "advanced to 2" in out.lines[1]
    assert r.m.current_split_index == 2 and "no effect" not in " ".join(out.lines)


def test_recovery_user_split_before_presend_reread_sends_nothing() -> None:
    r = ready_recovery()
    ex, _ = executor()
    c = Seq([0, 1])
    out = ex.recover(c, r.m, 5.0)
    assert out is not None and out.kind is OutcomeKind.RECOVERED
    assert "split" not in c.calls and "nothing sent" in out.lines[0]
    assert r.m.current_split_index == 1


def test_recovery_deferred_then_confirmed_and_deadline() -> None:
    r = ready_recovery()
    ex, clk = executor()
    c = Seq([0, 0, None, None, None, 1])
    out = ex.recover(c, r.m, 5.0)
    assert out is not None and out.kind is OutcomeKind.RECOVERY_SKIPPED and ex.pending
    clk.t += 0.6
    done = ex.pending_step(c, r.m)
    assert done is not None and done.kind is OutcomeKind.RECOVERED and r.m.current_split_index == 1
    r2 = ready_recovery()
    ex2, clk2 = executor()
    c2 = Seq([0, 0, None])
    _ = ex2.recover(c2, r2.m, 5.0)
    final = None
    for _ in range(30):
        clk2.t += 0.5
        final = ex2.pending_step(c2, r2.m)
        if final is not None:
            break
    assert final is not None and final.kind is OutcomeKind.UNCONFIRMED
    assert "could not confirm re-split" in final.lines[0] and c2.calls.count("split") == 1


def test_recovery_racing_user_split_in_stateful_fake() -> None:
    r = ready_recovery()
    ls = SLS(idx=0, count=4)
    ls.arm({3: [SLS.u_split]})  # user splits right before the pre-send re-read
    out = UndoExecutor().recover(ls, r.m, 5.0)
    assert out is not None and ls.idx == 1 and ls.sent == []
    assert r.m.expected_ls_index == 1


# --------------------------------------------------------- wire-level / logs


def test_eof_after_unsplit_exactly_one_unsplit_on_the_wire() -> None:
    script = Script()
    try:
        r = socket_death()
        c = client_for(script, timeout=0.2)
        script.close_after["unsplit"] = 1
        out = UndoExecutor().attempt(c, r.m, 0, 1.0)
        assert out.kind is OutcomeKind.APPLIED and "confirmed late" in out.lines[0]
        assert script.sent() == ["unsplit"] and r.m.current_split_index == 0
        c.close()
    finally:
        script.close()


def test_client_reports_whether_bytes_were_written(monkeypatch: pytest.MonkeyPatch) -> None:
    script = Script()
    try:
        c = client_for(script, timeout=0.2)
        assert c.unsplit() is True and c.last_send_wrote is True

        def eof(_sock: object) -> bool:
            raise ConnectionError("EOF after the write")

        monkeypatch.setattr(c, "_drain", eof)
        assert c.unsplit() is False and c.last_send_wrote is True  # may have been applied
        monkeypatch.undo()
        c.close()
    finally:
        script.close()
    dead = client_for(script, timeout=0.05)
    assert dead.split() is False and dead.last_send_wrote is False  # never left the process


# ----------------------------------------------------- startup text / --check

YAML = (
    "capture: {{fps: 100, stream_url: 'udp://127.0.0.1:9733', width: 64, height: 36}}\n"
    "roi: {{x: 0, y: 0, width: 1, height: 1}}\n"
    "detection: {{required_frames: 2, reset_threshold: 0.5}}\n"
    "colors: {{black: {{rgb: [0, 0, 0], tolerance: 25}}}}\n"
    "splits:\n  - {{id: 1, name: One, color: black, threshold: 0.9}}\n"
    "  - {{id: 2, name: Two, color: black, threshold: 0.9}}\n"
    "livesplit: {{host: '127.0.0.1', port: 1{ls}}}\n"
    "ocr:\n  roi: {{x: 0, y: 0, width: 0.5, height: 0.5}}\n  enabled: {en}\n  mode: undo\n"
    "  patterns: {{'^(\\d+)$': null}}\n"
)


def _cfg(tmp_path: Path, ls: str = ", follow_timer: true", en: str = "true") -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(YAML.format(ls=ls, en=en), encoding="utf-8")
    return p


def _usable(_cfg: object) -> tuple[bool, str]:
    return True, "fake OK"


def test_check_active_with_follow_timer_and_rejected_without(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mainmod, "describe_backend", _usable)
    assert mainmod.main(["--config", str(_cfg(tmp_path)), "--check"]) == 0
    out = capsys.readouterr().out
    assert "ACTIVE (will send unsplit)" in out and "WARN" not in out and "NOTE" not in out
    for ls in ("", ", resync_on_unsplit: true", ", follow_timer: false"):
        assert mainmod.main(["--config", str(_cfg(tmp_path, ls=ls)), "--check"]) == 1
        cap = capsys.readouterr()
        assert 'ocr.mode "undo" requires livesplit.follow_timer: true' in cap.err
        assert "Config OK" not in cap.out


def test_check_says_off_when_ocr_disabled_or_backend_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert mainmod.main(["--config", str(_cfg(tmp_path, en="false")), "--check"]) == 0
    out = capsys.readouterr().out
    assert "ocr undo : OFF (ocr disabled" in out and "ACTIVE" not in out

    def missing(_c: object) -> tuple[bool, str]:
        return False, "no tesseract"

    monkeypatch.setattr(mainmod, "describe_backend", missing)
    assert mainmod.main(["--config", str(_cfg(tmp_path)), "--check"]) == 0
    out = capsys.readouterr().out
    assert "ocr undo : OFF (ocr backend unavailable" in out and "ACTIVE" not in out


def test_fake_client_still_available() -> None:
    assert FakeClient().calls == []


def test_live_with_undo_and_follow_timer_has_no_follow_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from test_undo_live import LsServer, run  # pyright: ignore[reportImplicitRelativeImport]

    srv = LsServer()
    try:
        out = run(tmp_path, monkeypatch, capsys, "death", server=srv, frames=10)
        assert "follow_timer" not in out and "AHEAD of LiveSplit" not in out
    finally:
        srv.close()
