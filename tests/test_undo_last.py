"""Increment 5: undo on the LAST split (apply_to_last_split)."""

from __future__ import annotations

import dataclasses
import math
import re
import time
from pathlib import Path

import numpy as np
import pytest
from test_undo_machine import (  # pyright: ignore[reportImplicitRelativeImport]
    OCR_U,
    UNDO,
    Rig,
    res,
    snapshot,
)

from autosplitter import main as mainmod
from autosplitter.config import ConfigError, OcrConfig, load
from autosplitter.detection import State
from autosplitter.detection import state_machine as sm
from autosplitter.detection.state_machine import SplitEvent
from autosplitter.detection.undo_judge import VerdictKind
from autosplitter.livesplit import (
    Action,
    OutcomeKind,
    TimerFollower,
    UndoExecutor,
    UndoWatch,
)

LAST_UNDO = dataclasses.replace(
    UNDO, apply_to_last_split=True, last_split_extra_settle_seconds=2.0, window_seconds=30.0
)
OCR_LAST = dataclasses.replace(OCR_U, undo=LAST_UNDO)


def last_rig(n: int = 2, ocr: OcrConfig = OCR_LAST, baseline: bool = True) -> Rig:
    """Rig whose LAST split just fired and the machine entered DONE."""
    r = Rig(n, ocr)
    for k in range(n):
        if baseline or k < n - 1:
            r.agree()
        r.m.feed_result(res(1.0))
        ev = r.m.feed_result(res(1.0))
        assert isinstance(ev, SplitEvent)
        r.m.feed_result(res(1.0))  # TRANSITION -> WAITING_FOR_RESET / DONE
        if k < n - 1:
            r.m.feed_result(res(0.0))  # clear
            r.read(str(k + 2))  # genuine: HUD advanced
            r.events()
    assert r.m.state is State.DONE
    return r


def kinds(r: Rig) -> list[VerdictKind]:
    return [v.kind for v in r.events()]


# ------------------------------------------------------------------ judge/machine


def test_window_opens_on_done_only_when_enabled() -> None:
    r = last_rig()
    assert r.m.undo_pending
    off = last_rig(ocr=OCR_U)
    assert not off.m.undo_pending and off.events() == []


def test_window_does_not_self_cancel_after_the_done_epoch_bump() -> None:
    r = last_rig()
    for _ in range(8):  # 2 s of the lingering level number (inside settle)
        r.read("2")
    assert r.m.undo_pending and r.events() == []


def test_death_undo_after_settle_plus_extra() -> None:
    r = last_rig()
    for _ in range(19):  # 4.75 s < settle 3.0 + extra 2.0
        r.read("2")
        assert r.m.undo_target is None
    verdicts: list[VerdictKind] = []
    for _ in range(12):
        r.read("2")
        verdicts += kinds(r)
    assert verdicts == [VerdictKind.UNDO] and r.m.undo_target == 1
    assert r.m.state is State.DONE


def test_extra_delay_boundary_vs_plain_settle() -> None:
    plain = dataclasses.replace(LAST_UNDO, last_split_extra_settle_seconds=0.0)
    r = last_rig(ocr=dataclasses.replace(OCR_U, undo=plain))
    for _ in range(20):
        r.read("2")
    assert r.m.undo_target == 1  # without the extra the same reads already confirm


def test_hud_gone_finishes_without_undo() -> None:
    r = last_rig()
    for _ in range(40):
        r.read("")
    ev = r.events()
    assert [v.kind for v in ev] == [VerdictKind.FINISHED]
    assert ev[0].elapsed >= 5.0  # never before settle + extra
    assert r.m.undo_target is None and not r.m.undo_pending


def test_blank_reads_inside_the_guard_do_not_finish_or_undo() -> None:
    r = last_rig()
    for _ in range(16):  # 4 s of blank (e.g. HUD hidden during the death screen)
        r.read("")
    assert r.events() == [] and r.m.undo_pending
    for _ in range(40):  # then the level number comes back and stays: death
        r.read("2")
    assert [k for k in kinds(r) if k is not VerdictKind.WARN] == [VerdictKind.UNDO]


def test_lingering_hud_longer_than_the_guard_is_wrongly_undone() -> None:
    """Documented limit: a HUD that outlives settle+extra+post_clear+confirm."""
    r = last_rig()
    for _ in range(30):
        r.read("2")
    assert r.m.undo_target == 1  # (a genuine finish whose HUD lingers looks like a death)


def test_hud_vanishing_before_the_guard_wins() -> None:
    r = last_rig()
    for _ in range(12):  # 3 s of lingering number
        r.read("2")
    for _ in range(40):
        r.read("")
    assert [k for k in kinds(r) if k is not VerdictKind.WARN] == [VerdictKind.FINISHED]
    assert r.m.undo_target is None


def test_trusted_read_breaks_the_gone_streak() -> None:
    r = last_rig()
    r.clock.t += 5.5
    r.read("")
    r.read("")
    r.read("2")  # level number flickers back
    r.read("")
    assert not any(v is VerdictKind.FINISHED for v in kinds(r))


def test_baseline_required_for_the_last_window() -> None:
    r = last_rig(baseline=False)
    assert not r.m.undo_pending
    assert [v.kind for v in r.events()] == [VerdictKind.NO_BASELINE]


def test_single_split_run_last_window() -> None:
    r = last_rig(n=1)
    assert r.m.undo_pending
    for _ in range(40):
        r.read("1")
    assert r.m.undo_target == 0


def test_done_ignores_colour_frames_but_judge_is_fed() -> None:
    r = last_rig()
    snap = snapshot(r.m)
    for _ in range(10):
        assert r.m.feed_result(res(1.0)) is None
    assert snapshot(r.m) == snap and r.m.state is State.DONE


def test_reset_and_rearm_cancel_the_window() -> None:
    r = last_rig()
    r.m.reset()
    r.read("2")
    assert [v.kind for v in r.events()] == [VerdictKind.CANCELLED]
    assert not r.m.undo_pending


def test_expiry() -> None:
    r = last_rig()
    r.clock.t += 31.0
    r.read("2")
    assert [v.kind for v in r.events()] == [VerdictKind.EXPIRED]


def test_apply_undo_from_done_leaves_done_and_settles() -> None:
    r = last_rig()
    for _ in range(40):
        r.read("2")
    assert r.m.apply_undo(1, arm_recovery=False)
    assert r.m.state is State.WAITING_FOR_RESET and r.m.current_split_index == 1
    assert r.m.expected_ls_index == 1 and not r.m.finished
    assert not r.m.undo_pending and r.m.undo_target is None
    for _ in range(5):  # the still-lit colour must clear first
        assert r.m.feed_result(res(1.0)) is None
    r.m.feed_result(res(0.0))
    assert r.m.state is State.NORMAL
    r.m.feed_result(res(1.0))
    ev = r.m.feed_result(res(1.0))
    assert isinstance(ev, SplitEvent) and ev.split_id == 2  # waiting for the last transition again


# ------------------------------------------------------------------- executor


class EndedLS:
    """Fake LiveSplit with real Ended/Running rules for the last split."""

    def __init__(self, idx: int = 2, phase: str | None = "Ended") -> None:
        self.idx: int = idx
        self.phase: str | None = phase
        self.calls: list[str] = []
        self.apply: bool = True
        self.unsplit_ok: bool = True
        self.last_send_wrote: bool = True
        self.idx_script: list[int | None] = []

    def get_current_timer_phase(self) -> str | None:
        self.calls.append("phase")
        return self.phase

    def get_split_index(self) -> int | None:
        self.calls.append("idx")
        if self.idx_script:
            return self.idx_script.pop(0)
        return self.idx

    def unsplit(self) -> bool:
        self.calls.append("unsplit")
        if self.apply and self.phase in ("Running", "Ended") and self.idx > 0:
            self.idx -= 1
            self.phase = "Running"
        return self.unsplit_ok

    def split(self) -> bool:
        self.calls.append("split")
        return True


def fast_exec() -> UndoExecutor:
    return UndoExecutor(sleep=lambda _s: None)


def confirmed_last(n: int = 2) -> Rig:
    r = last_rig(n)
    for _ in range(40):
        r.read(str(n))
    assert r.m.undo_target == n - 1 and r.m.state is State.DONE
    return r


def test_last_split_unsplit_from_ended_and_verify_running() -> None:
    r = confirmed_last()
    c = EndedLS()
    out = fast_exec().attempt(c, r.m, 1, 1.0)
    assert out.kind is OutcomeKind.APPLIED and c.calls == ["phase", "idx", "unsplit", "idx"]
    assert c.phase == "Running" and c.idx == 1
    assert r.m.state is State.WAITING_FOR_RESET and r.m.current_split_index == 1
    for _ in range(6):  # no late-HUD recovery after a last-split undo
        r.m.feed_result(res(0.0))
        r.read("2")
    assert r.m.recovery_target is None


@pytest.mark.parametrize("phase", ["Paused", "NotRunning", None])
def test_last_split_other_phases_skipped(phase: str | None) -> None:
    r = confirmed_last()
    c = EndedLS(phase=phase)
    before = snapshot(r.m)
    out = fast_exec().attempt(c, r.m, 1, 1.0)
    assert not out.changed and "unsplit" not in c.calls and snapshot(r.m) == before


def test_last_split_running_at_n_is_not_in_step() -> None:
    r = confirmed_last()
    c = EndedLS(idx=2, phase="Running")
    out = fast_exec().attempt(c, r.m, 1, 1.0)
    assert out.kind is OutcomeKind.SKIPPED and "unsplit" not in c.calls


def test_ended_phase_is_not_accepted_for_a_normal_window() -> None:
    from test_undo_exec import confirmed_death  # pyright: ignore[reportImplicitRelativeImport]

    r = confirmed_death()
    c = EndedLS(idx=1, phase="Ended")
    out = fast_exec().attempt(c, r.m, 0, 1.0)
    assert out.kind is OutcomeKind.SKIPPED and "timer is Ended" in out.lines[0]


def test_last_split_already_unsplit_by_user_rolls_back_only() -> None:
    r = confirmed_last()
    c = EndedLS(idx=1, phase="Running")
    out = fast_exec().attempt(c, r.m, 1, 1.0)
    assert out.kind is OutcomeKind.ALREADY_THERE and "unsplit" not in c.calls


def test_last_split_eof_after_write_applied_is_confirmed_late() -> None:
    r = confirmed_last()
    c = EndedLS()
    c.unsplit_ok = False  # reported undelivered but applied
    out = fast_exec().attempt(c, r.m, 1, 1.0)
    assert out.kind is OutcomeKind.APPLIED and "confirmed late" in out.lines[0]
    assert c.calls.count("unsplit") == 1 and r.m.current_split_index == 1


def test_last_split_late_apply_and_ignored() -> None:
    r = confirmed_last()
    c = EndedLS()
    c.apply = False
    c.idx_script = [2, 2, 2, 2]
    out = fast_exec().attempt(c, r.m, 1, 1.0)
    assert out.kind is OutcomeKind.IGNORED and r.m.state is State.DONE
    r2 = confirmed_last()
    c2 = EndedLS()
    c2.apply = False
    c2.idx_script = [2, 2, 2, 1]  # applied late: seen on the second re-check
    out = fast_exec().attempt(c2, r2.m, 1, 1.0)
    assert out.kind is OutcomeKind.APPLIED and c2.calls.count("unsplit") == 1


def test_last_split_no_reply_pending_then_follower_rollback_no_double() -> None:
    r = confirmed_last()
    clock = [100.0]
    ex = UndoExecutor(sleep=lambda s: clock.__setitem__(0, clock[0] + s), clock=lambda: clock[0])
    c = EndedLS()
    c.idx_script = [2, None, None, None]  # precondition, then silence
    out = ex.attempt(c, r.m, 1, 1.0)
    assert out.kind is OutcomeKind.UNVERIFIED and ex.pending and r.m.state is State.DONE
    # the real follower sees Running at N-1 vs expected N (LiveSplit really applied it)
    follower = TimerFollower()
    arm = follower.observe(
        phase="Ended", ls_index=2, expected=2, n_splits=2, now=900.0, last_split_time=None,
        suppress=False, can_back=True, can_fwd=True, waiting_for_start=False,
    )
    assert arm.action is Action.NONE
    rolled = False
    for i in range(3):
        d = follower.observe(
            phase="Running", ls_index=1, expected=r.m.expected_ls_index, n_splits=2,
            now=1000.0 + i, last_split_time=None, suppress=False,
            can_back=True, can_fwd=True, waiting_for_start=False,
        )
        if d.action is Action.ROLLBACK and d.index is not None:
            rolled = r.m.rollback_to(d.index)
            break
    assert rolled and r.m.current_split_index == 1
    clock[0] += 0.6
    c.idx_script = [1]
    done = ex.pending_step(c, r.m)
    assert done is not None and done.kind is OutcomeKind.SKIPPED and not done.changed
    assert c.calls.count("unsplit") == 1 and r.m.current_split_index == 1


def test_follower_and_undowatch_do_not_misread_the_post_undo_state() -> None:
    r = confirmed_last()
    c = EndedLS()
    assert fast_exec().attempt(c, r.m, 1, 1.0).kind is OutcomeKind.APPLIED
    follower = TimerFollower()
    watch = UndoWatch()
    for i in range(6):
        d = follower.observe(
            phase="Running", ls_index=c.idx, expected=r.m.expected_ls_index, n_splits=2,
            now=1000.0 + 10 * i, last_split_time=None, suppress=False,
            can_back=True, can_fwd=True, waiting_for_start=False,
        )
        assert d.action is Action.NONE  # no RUN_ENDED, no ROLLBACK, no SKIP_FORWARD
        assert watch.observe("Running", c.idx, r.m.expected_ls_index, 1000.0 + 10 * i, None, False) is None


def test_manual_final_split_by_the_user_is_a_genuine_finish() -> None:
    # The user splits the last segment by hand while the detector is not DONE:
    # the follower's RUN_ENDED path ends the run and no last-split window exists.
    r = Rig(2, OCR_LAST)
    follower = TimerFollower()
    for i in range(2):
        d = follower.observe(
            phase="Running", ls_index=1, expected=1, n_splits=2, now=10.0 + i,
            last_split_time=None, suppress=False, can_back=True, can_fwd=True,
            waiting_for_start=False,
        )
        assert d.action is Action.NONE
    acts = [
        follower.observe(
            phase="Ended", ls_index=2, expected=1, n_splits=2, now=20.0 + i,
            last_split_time=None, suppress=False, can_back=True, can_fwd=True,
            waiting_for_start=False,
        ).action
        for i in range(3)
    ]
    assert Action.RUN_ENDED in acts
    assert r.m.advance_to(2) and r.m.state is State.DONE and not r.m.undo_pending


# --------------------------------------------------------------------- config

CFG = (
    "capture: {fps: 60, stream_url: 'udp://127.0.0.1:9733', width: 64, height: 36}\n"
    "roi: {x: 0, y: 0, width: 1, height: 1}\n"
    "detection: {required_frames: 2, reset_threshold: 0.5}\n"
    "colors: {black: {rgb: [0, 0, 0], tolerance: 25}}\n"
    "splits:\n  - {id: 1, name: One, color: black, threshold: 0.9}\n"
    "livesplit: {host: '127.0.0.1', port: 16834, follow_timer: true}\n"
    "ocr:\n  roi: {x: 0, y: 0, width: 0.5, height: 0.5}\n  patterns: {'(\\d+)': null}\n"
)


def load_undo(tmp_path: Path, undo: str) -> object:
    p = tmp_path / "c.yaml"
    p.write_text(CFG + f"  mode: undo\n  undo: {undo}\n", encoding="utf-8")
    return load(p)


def test_last_split_config_defaults_and_values(tmp_path: Path) -> None:
    c = load_undo(tmp_path, "{apply_to_last_split: true}")
    assert isinstance(c, object)
    from autosplitter.config import Config

    assert isinstance(c, Config) and c.ocr is not None and c.ocr.undo is not None
    assert c.ocr.undo.last_split_extra_settle_seconds == 2.0
    c = load_undo(tmp_path, "{apply_to_last_split: true, last_split_extra_settle_seconds: 0}")
    assert isinstance(c, Config) and c.ocr is not None and c.ocr.undo is not None
    assert c.ocr.undo.last_split_extra_settle_seconds == 0.0


@pytest.mark.parametrize(
    ("undo", "msg"),
    [
        ("{last_split_extra_settle_seconds: -1}", "last_split_extra_settle_seconds must be at least 0"),
        ("{last_split_extra_settle_seconds: .inf}", "last_split_extra_settle_seconds must be a finite number"),
        ("{last_split_extra_settle_seconds: .nan}", "last_split_extra_settle_seconds must be a finite number"),
        ("{last_split_extra_settle_seconds: x}", "last_split_extra_settle_seconds must be a number"),
        ("{last_split_extra_settle_seconds: " + "9" * 400 + "}", "last_split_extra_settle_seconds must be a number"),
        ("{apply_to_last_split: true, window_seconds: 6}",
         "window_seconds must be greater than settle_seconds + last_split_extra_settle_seconds + confirm_seconds"),
        ("{apply_to_last_split: true, last_split_extra_settle_seconds: 10}",
         "window_seconds must be greater than settle_seconds + last_split_extra_settle_seconds"),
    ],
)
def test_last_split_config_errors(tmp_path: Path, undo: str, msg: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(msg)):
        _ = load_undo(tmp_path, undo)


def test_window_rule_unchanged_without_apply_to_last_split(tmp_path: Path) -> None:
    c = load_undo(tmp_path, "{window_seconds: 5, last_split_extra_settle_seconds: 10}")
    assert c is not None  # the extra only counts when apply_to_last_split is on


def test_check_mentions_last_split_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def usable(_c: object) -> tuple[bool, str]:
        return True, "fake OK"

    monkeypatch.setattr(mainmod, "describe_backend", usable)
    p = tmp_path / "c.yaml"
    p.write_text(CFG + "  mode: undo\n  undo: {apply_to_last_split: true}\n", encoding="utf-8")
    assert mainmod.main(["--config", str(p), "--check"]) == 0
    out = capsys.readouterr().out
    assert "last-split=ON(+2s settle" in out and "ACTIVE" in out
    p.write_text(CFG + "  mode: undo\n  undo: {}\n", encoding="utf-8")
    assert mainmod.main(["--config", str(p), "--check"]) == 0
    assert "last-split" not in capsys.readouterr().out


# ------------------------------------------------------------ end to end (main)

from test_undo_live import LsServer, Worker, write_cfg  # pyright: ignore[reportImplicitRelativeImport]


class LastHud:
    """Text shown by the fake worker, driven by the fake LiveSplit server."""

    def __init__(self, server: LsServer, linger: float) -> None:
        self.server: LsServer = server
        self.linger: float = linger  # seconds the level number stays after the finish
        self.t_end: float | None = None

    def text(self, _frame_idx: int = 0) -> str:
        s = self.server
        if s.phase != "Ended":
            self.t_end = None
            return str(min(s.idx + 1, s.count))
        if self.t_end is None:
            self.t_end = time.monotonic()
        return str(s.count) if time.monotonic() - self.t_end < self.linger else ""


class Schedule:
    """Black/white frames: three transitions, then white. ``n`` counts reads."""

    exhausted: bool = False
    total: int = 0

    def __init__(self, *_a: object, **_k: object) -> None:
        Schedule.total = 0

    def read(self) -> np.ndarray:
        time.sleep(0.004)
        Schedule.total += 1
        n = Schedule.total
        black = (1 <= n <= 4) or (11 <= n <= 14) or (21 <= n <= 24)
        return np.full((36, 64, 3), 0 if black else 255, np.uint8)

    def close(self) -> None:
        pass


def run_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    linger: float, mode: str = "undo", last: bool = True, loop: bool = False,
    frames: int = 500,
    unsplit_delay: float = 0.0,
) -> tuple[str, LsServer, float]:
    srv = LsServer(idx=0, count=3)
    srv.unsplit_delay = unsplit_delay
    hud = LastHud(srv, linger)

    def fake_build(_c: object) -> tuple[Worker, str]:
        return Worker(hud), "fake OK"  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(sm, "build_ocr_worker", fake_build)
    monkeypatch.setattr(mainmod, "MpegTsFrameSource", Schedule)
    extra = (
        ", apply_to_last_split: true, last_split_extra_settle_seconds: 0.1"
        if last else ""
    )
    p = write_cfg(tmp_path, srv.port, mode, extra, "", 0.2)
    text = p.read_text(encoding="utf-8").replace("window_seconds: 30", "window_seconds: 20")
    p.write_text(text, encoding="utf-8")
    args = ["--config", str(p), "--live", "x.ts", "--max-frames", str(frames), "--livesplit"]
    if loop:
        args.append("--loop")
    t0 = time.monotonic()
    try:
        assert mainmod.main(args) == 0
        wall = time.monotonic() - t0
        out = capsys.readouterr().out
    finally:
        srv.close()
    return out, srv, wall


def test_e2e_death_on_the_last_level_is_unsplit_and_detector_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, _ = run_last(tmp_path, monkeypatch, capsys, linger=1e9)
    assert srv.moves() == ["split", "split", "split", "unsplit"], out
    assert "LiveSplit: UNSPLIT sent -> index 2 (verified)" in out
    assert srv.phase == "Running" and srv.idx == 2
    assert "undo regret" not in out  # no late-HUD recovery after the last split
    assert srv.commands.count("unsplit") == 1
    assert "UNSPLIT detected" not in out and "SKIP/manual split" not in out
    assert Schedule.total == 500  # the process kept running (it did not exit at DONE)


def test_e2e_genuine_finish_with_hud_gone_exits_promptly_without_undo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, wall = run_last(tmp_path, monkeypatch, capsys, linger=0.02)
    assert srv.moves() == ["split", "split", "split"], out
    assert "HUD gone after the last split — run finished, no undo" in out
    assert "DONE: every configured split has fired." in out
    assert Schedule.total < 500, "the process should exit once FINISHED closed the window"
    assert wall < 20.0


def test_e2e_lingering_hud_is_wrongly_undone_documented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, _ = run_last(tmp_path, monkeypatch, capsys, linger=1.5)
    assert srv.moves() == ["split", "split", "split", "unsplit"], out  # the documented wrong undo


def test_e2e_exit_is_held_while_the_window_is_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, _, _ = run_last(tmp_path, monkeypatch, capsys, linger=0.02)
    done_at = out.index("DONE: every configured split has fired.")
    assert out.index("HUD gone after the last split") > done_at  # held until the verdict


def test_e2e_without_apply_to_last_split_exits_at_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, _ = run_last(tmp_path, monkeypatch, capsys, linger=1e9, last=False)
    assert srv.moves() == ["split", "split", "split"], out
    assert Schedule.total < 60 and "HUD gone" not in out


def test_e2e_dry_run_logs_would_undo_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, _ = run_last(tmp_path, monkeypatch, capsys, linger=1e9, mode="verify_only")
    assert 'would UNDO split "Three"' in out
    assert srv.moves() == ["split", "split", "split"]
    assert math.isfinite(Schedule.total)


def test_e2e_loop_does_not_rearm_while_the_window_is_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, _ = run_last(tmp_path, monkeypatch, capsys, linger=1e9, loop=True)
    assert srv.moves() == ["split", "split", "split", "unsplit"], out
    assert out.index("UNSPLIT sent") < (out.index("RUN COMPLETE") if "RUN COMPLETE" in out else 10**9)


# ----------------------------------------------------------- pattern anchoring


def _pat_cfg(tmp_path: Path, pattern: str, ocr_extra: str) -> Path:
    p = tmp_path / "p.yaml"
    p.write_text(
        CFG.replace("patterns: {'(\\d+)': null}", f"patterns: {{\"{pattern}\": null}}")
        + ocr_extra,
        encoding="utf-8",
    )
    return p


def test_unanchored_pattern_warns_with_last_split_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = _pat_cfg(tmp_path, "(\\\\d+)", "  undo: {apply_to_last_split: true}\n")
    assert mainmod.main(["--config", str(p), "--check"]) == 0
    out = capsys.readouterr().out
    assert 'WARN: ocr.patterns "(\\d+)" is not anchored (^...$)' in out
    assert "end-card/menu text" in out


def test_unanchored_pattern_warns_in_mode_undo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = _pat_cfg(tmp_path, "Level (\\\\d+)", "  mode: undo\n")
    assert mainmod.main(["--config", str(p), "--check"]) == 0
    assert "is not anchored" in capsys.readouterr().out


@pytest.mark.parametrize("extra", ["", "  undo: {}\n", "  undo: {apply_to_last_split: false}\n"])
def test_no_anchor_warning_when_flag_off_or_verify_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: str
) -> None:
    p = _pat_cfg(tmp_path, "(\\\\d+)", extra)
    assert mainmod.main(["--config", str(p), "--check"]) == 0
    assert "not anchored" not in capsys.readouterr().out


def test_anchored_pattern_has_no_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = _pat_cfg(tmp_path, "^\\\\s*(\\\\d+)\\\\s*$", "  undo: {apply_to_last_split: true}\n")
    assert mainmod.main(["--config", str(p), "--check"]) == 0
    assert "not anchored" not in capsys.readouterr().out


def test_shipped_start_powerless_has_no_anchor_warning(capsys: pytest.CaptureFixture[str]) -> None:
    path = Path(__file__).parent.parent / "configs" / "start-powerless.yaml"
    assert mainmod.main(["--config", str(path), "--check"]) == 0
    assert "not anchored" not in capsys.readouterr().out


def test_unanchored_pattern_warns_at_startup_and_still_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    p = _pat_cfg(tmp_path, "(\\\\d+)", "  undo: {apply_to_last_split: true}\n")

    class Src:
        exhausted: bool = True

        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def read(self) -> None:
            return None

        def close(self) -> None:
            pass

    class W:
        version: int = 0
        taken_version: int = 0
        status: str = "ok"

        def start(self) -> None:
            pass

        def submit_frame(self, *_a: object) -> None:
            pass

        def take_result(self) -> None:
            return None

        def close(self) -> None:
            pass

    def fake_build(_c: object) -> tuple[W, str]:
        return W(), "fake"

    monkeypatch.setattr(sm, "build_ocr_worker", fake_build)
    monkeypatch.setattr(mainmod, "MpegTsFrameSource", Src)
    assert mainmod.main(["--config", str(p), "--live", "x.ts"]) == 0
    assert "WARN: ocr.patterns" in capsys.readouterr().out


def test_last_split_warn_wording_has_no_nonexistent_next_split() -> None:
    from test_undo_machine import cfg  # pyright: ignore[reportImplicitRelativeImport]

    from autosplitter.detection.undo_judge import UndoVerdict

    c = cfg(2, OCR_LAST)
    last = mainmod._undo_line(UndoVerdict(VerdictKind.WARN, 1, 5), c)  # pyright: ignore[reportPrivateUsage]
    mid = mainmod._undo_line(UndoVerdict(VerdictKind.WARN, 0, 5), c)  # pyright: ignore[reportPrivateUsage]
    assert "(expected 2); ignored" in last and "2 or 3" not in last
    assert "(expected 1 or 2); ignored" in mid


# ------------------------------------------------------- E4: exit vs. a late unsplit


def test_e2e_late_applied_unsplit_does_not_let_the_process_exit_while_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, _ = run_last(tmp_path, monkeypatch, capsys, linger=1e9, unsplit_delay=0.4)
    assert "unsplit had no effect" in out  # the re-checks (~0.3 s) missed the late apply
    assert srv.moves() == ["split", "split", "split", "unsplit"], out
    assert srv.phase == "Running" and srv.idx == 2
    assert "UNSPLIT detected" in out  # follow_timer rolled the detector back
    assert Schedule.total == 500, "the process exited while LiveSplit was Running"
    assert srv.commands.count("unsplit") == 1


def test_e2e_unsplit_applied_within_the_recheck_window_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, _ = run_last(tmp_path, monkeypatch, capsys, linger=1e9, unsplit_delay=0.1)
    assert "UNSPLIT applied (confirmed late)" in out or "UNSPLIT sent -> index 2" in out
    assert srv.moves() == ["split", "split", "split", "unsplit"] and srv.idx == 2
    assert "UNSPLIT detected" not in out


def test_e2e_no_attempt_keeps_todays_exit_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, _ = run_last(tmp_path, monkeypatch, capsys, linger=0.02)  # FINISHED, nothing sent
    assert srv.moves() == ["split", "split", "split"] and Schedule.total < 500
    out2, srv2, _ = run_last(tmp_path, monkeypatch, capsys, linger=1e9, last=False)
    assert srv2.moves() == ["split", "split", "split"] and Schedule.total < 60
    out3, srv3, _ = run_last(tmp_path, monkeypatch, capsys, linger=1e9, mode="verify_only")
    assert srv3.moves() == ["split", "split", "split"] and "would UNDO" in out3
    assert out and out2


def test_fresh_phase_wait_is_bounded_and_skipped_when_unreachable() -> None:
    ex = UndoExecutor()
    poller = mainmod._FollowPoller(3)  # pyright: ignore[reportPrivateUsage]
    wait = mainmod._wait_fresh_phase  # pyright: ignore[reportPrivateUsage]
    assert wait(ex, poller, True, 100.0, 6.0) == (False, False)  # nothing was sent
    ex.last_unsplit_sent = True
    ex.last_unsplit_time = 100.0
    poller.phase_at = 99.0  # the cached phase predates the attempt
    assert wait(ex, poller, True, 101.0, 6.0, 5.0) == (True, False)  # too early to poll again
    assert wait(ex, poller, True, 101.0, 6.0, 0.5) == (True, True)  # past the settle margin
    poller.phase_at = 100.4  # polled after the attempt but inside the late-apply margin
    assert wait(ex, poller, True, 101.0, 6.0, 0.5) == (True, True)
    assert wait(ex, poller, False, 101.0, 6.0) == (False, False)  # unreachable: today's exit
    assert wait(ex, poller, True, 107.0, 6.0) == (False, False)  # hung server: deadline
    poller.phase_at = 100.6  # a phase polled after attempt + settle margin
    assert wait(ex, poller, True, 101.0, 6.0, 0.5) == (False, False)


def test_executor_flags_a_last_split_unsplit_only() -> None:
    r = confirmed_last()
    ex = fast_exec()
    assert not ex.last_unsplit_sent
    c = EndedLS()
    c.apply = False
    c.idx_script = [2, 2, 2, 2]
    _ = ex.attempt(c, r.m, 1, 1.0)
    assert ex.last_unsplit_sent
    from test_undo_exec import confirmed_death  # pyright: ignore[reportImplicitRelativeImport]

    r2 = confirmed_death()
    ex2 = fast_exec()
    c2 = EndedLS(idx=1, phase="Running")
    c2.idx_script = [1, 0]
    _ = ex2.attempt(c2, r2.m, 0, 1.0)
    assert not ex2.last_unsplit_sent  # normal windows never hold the exit
