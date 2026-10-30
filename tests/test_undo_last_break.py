"""Adversarial tests for increment 5 (undo on the LAST split)."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from test_undo_last import (  # pyright: ignore[reportImplicitRelativeImport]
    LAST_UNDO,
    OCR_LAST,
    EndedLS,
    confirmed_last,
    fast_exec,
    kinds,
    last_rig,
    load_undo,
)
from test_undo_live import LsServer, Worker, write_cfg  # pyright: ignore[reportImplicitRelativeImport]
from test_undo_machine import (  # pyright: ignore[reportImplicitRelativeImport]
    FRAME,
    OCR_U,
    Rig,
    res,
)

from autosplitter import main as mainmod
from autosplitter.config import Config, ConfigError
from autosplitter.detection import State
from autosplitter.detection import state_machine as sm
from autosplitter.detection.state_machine import SplitEvent
from autosplitter.detection.undo_judge import VerdictKind
from autosplitter.livesplit import OutcomeKind

# ------------------------------------------------------------------ helpers


def raw(r: Rig, text: str, *, epoch: int | None = None, frame: int | None = None,
        dt: float = 0.25, conf: float = 90.0) -> None:
    """Push one distinct OCR result with explicit epoch/frame (stale-read tests)."""
    r.clock.t += dt
    r.frame += 1
    r.fw.push(text, r.frame if frame is None else frame, r.m.epoch if epoch is None else epoch, conf)
    r.m.check_ocr_resync(FRAME, r.frame)


def lit_then_gone(lit: float, gone: float = 20.0) -> list[VerdictKind]:
    """HUD shows the last level for ``lit`` seconds after DONE, then disappears."""
    r = last_rig()
    out: list[VerdictKind] = []
    for _ in range(round(lit / 0.25)):
        r.read("2")
        out += kinds(r)
    for _ in range(round(gone / 0.25)):
        r.read("")
        out += kinds(r)
    return out


# ------------------------------------------- (1) false undo of a genuine finish


@pytest.mark.parametrize("lit", [0.25, 1.0, 2.0, 3.0, 4.75, 5.0, 5.5, 5.75])
def test_hud_lingering_up_to_the_guard_never_undoes_a_finish(lit: float) -> None:
    got = lit_then_gone(lit)
    assert VerdictKind.UNDO not in got
    assert got[-1:] == [VerdictKind.FINISHED] and got.count(VerdictKind.FINISHED) == 1


def test_lingering_boundary_exactly_settle_extra_plus_confirm() -> None:
    # first eligible trusted read at 5.0 s, confirm span 1.0 s => a HUD lit through
    # 6.0 s is the documented false-undo threshold (and not a tick earlier).
    assert VerdictKind.UNDO in lit_then_gone(6.0)
    assert VerdictKind.UNDO not in lit_then_gone(5.75)


def test_flickering_hud_is_neither_undo_nor_finished_until_expiry() -> None:
    r = last_rig()
    got: list[VerdictKind] = []
    for i in range(110):  # 27 s of strictly alternating level number / blank
        r.read("2" if i % 2 == 0 else "")
        got += kinds(r)
    assert got == []
    for i in range(40):
        r.read("2" if i % 2 == 0 else "")
        got += kinds(r)
    assert got == [VerdictKind.EXPIRED] and r.m.undo_target is None


def test_two_trusted_one_blank_pattern_never_confirms() -> None:
    r = last_rig()
    got: list[VerdictKind] = []
    for i in range(100):
        r.read("" if i % 3 == 2 else "2")
        got += kinds(r)
    assert VerdictKind.UNDO not in got and VerdictKind.FINISHED not in got


def test_low_confidence_level_reads_are_not_trusted_for_undo() -> None:
    r = last_rig()
    got: list[VerdictKind] = []
    for _ in range(60):
        r.read("2", conf=40.0)
        got += kinds(r)
    assert VerdictKind.UNDO not in got and r.m.undo_target is None


def test_wrong_level_read_warns_once_and_breaks_the_undo_streak() -> None:
    r = last_rig(n=3)
    # last_rig(3): HUD "3" = D-1 for the last level; "1" is a stale/other number
    got: list[VerdictKind] = []
    for i in range(80):
        r.read("1" if i % 4 == 3 else "3")
        got += kinds(r)
    assert VerdictKind.UNDO not in got
    assert got.count(VerdictKind.WARN) == 1


def test_stale_epoch_reads_are_never_trusted_for_undo() -> None:
    r = last_rig()
    for _ in range(60):  # reads tagged with the pre-DONE epoch (submitted before the finish)
        raw(r, "2", epoch=r.m.epoch - 1)
    assert VerdictKind.UNDO not in kinds(r) and r.m.undo_target is None


def test_stale_frame_reads_are_never_trusted_for_undo() -> None:
    r = last_rig()
    for _ in range(60):  # right epoch, frame older than max_age_seconds
        raw(r, "2", frame=r.frame - 10**7)
    assert r.m.undo_target is None


def test_stale_pre_finish_reads_do_not_finish_before_the_guard() -> None:
    r = last_rig()
    for _ in range(16):  # 4 s, all untrusted (old epoch): inside the guard
        raw(r, "2", epoch=r.m.epoch - 1)
    assert r.events() == [] and r.m.undo_pending


def test_expiry_wins_over_a_qualifying_streak_on_the_same_read() -> None:
    r = last_rig()
    for _ in range(22):  # 5.5 s: streak under way
        r.read("2")
    r.clock.t += 30.0  # clock jump past window_seconds
    for _ in range(6):
        r.read("2")
    got = kinds(r)
    assert VerdictKind.UNDO not in got and VerdictKind.EXPIRED in got
    assert r.m.undo_target is None


def test_clock_jump_cannot_make_one_read_confirm() -> None:
    r = last_rig()
    r.read("2")
    r.clock.t += 5.5
    r.read("2", dt=0.0)
    r.read("2", dt=0.0)
    r.read("2", dt=0.0)
    r.read("2", dt=0.0)  # many reads, zero wall time between them
    assert r.m.undo_target is None  # confirm_seconds needs real separation


def test_one_verdict_per_window_after_undo_confirmed() -> None:
    r = last_rig()
    for _ in range(80):
        r.read("2")
    ks = kinds(r)
    assert ks.count(VerdictKind.UNDO) == 1
    assert VerdictKind.FINISHED not in ks and VerdictKind.EXPIRED not in ks


def test_finished_then_hud_reappears_gives_no_late_undo() -> None:
    r = last_rig()
    for _ in range(40):
        r.read("")
    assert kinds(r) == [VerdictKind.FINISHED]
    for _ in range(80):  # HUD of the NEXT run's menu shows level 2 again
        r.read("2")
    assert kinds(r) == [] and r.m.undo_target is None and r.m.state is State.DONE


def test_stale_baseline_means_no_window_and_no_hold() -> None:
    r = Rig(2, OCR_LAST)
    r.agree()
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(0.0))
    r.read("2")
    r.read("2")
    r.events()
    r.clock.t += 5.0  # baseline_seconds is 2.0: the pre-fire AGREE is stale
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    assert r.m.state is State.DONE and not r.m.undo_pending
    assert [v.kind for v in r.events()] == [VerdictKind.NO_BASELINE]


def test_pre_fire_reads_taken_before_a_death_screen_do_not_leak_into_the_window() -> None:
    # reads from BEFORE the fire (old epoch) arrive after DONE: they must not
    # count as the trusted D-1 streak even though they show the last level.
    r = last_rig()
    r.clock.t += 5.2  # past the guard
    for _ in range(12):
        raw(r, "2", epoch=r.m.epoch - 2, dt=0.25)
    assert r.m.undo_target is None


def test_n1_hud_gone_finishes_and_hud_lingering_undoes_exactly_once() -> None:
    r = last_rig(n=1)
    got: list[VerdictKind] = []
    for _ in range(30):
        r.read("")
        got += kinds(r)
    assert got == [VerdictKind.FINISHED]


def test_extra_settle_zero_still_uses_plain_settle_for_finished() -> None:
    plain = dataclasses.replace(LAST_UNDO, last_split_extra_settle_seconds=0.0)
    r = last_rig(ocr=dataclasses.replace(OCR_U, undo=plain))
    for _ in range(11):  # 2.75 s < settle 3
        r.read("")
    assert r.events() == []
    for _ in range(12):
        r.read("")
    assert [v.kind for v in r.events()] == [VerdictKind.FINISHED]


def test_blank_before_post_clear_is_ignored_for_finished() -> None:
    cfg = dataclasses.replace(LAST_UNDO, last_split_extra_settle_seconds=0.0, post_clear_seconds=4.0)
    r = last_rig(ocr=dataclasses.replace(OCR_U, undo=cfg))
    for _ in range(14):  # 3.5 s: past settle (3.0) but inside post_clear (4.0)
        r.read("")
    assert r.events() == []
    for _ in range(12):
        r.read("")
    assert [v.kind for v in r.events()] == [VerdictKind.FINISHED]


def test_finished_needs_confirm_span_not_just_reads() -> None:
    r = last_rig()
    for _ in range(21):  # 5.25 s
        r.read("2")
    got: list[VerdictKind] = []
    for _ in range(10):  # many blank reads, all in a 0.0 s burst
        r.read("", dt=0.0)
        got += kinds(r)
    assert got == []


# ----------------------------------------------------- (2) death on last level


def _second_death(r: Rig) -> None:
    """After an undo from DONE: clear colour, HUD back on the last level, fire again."""
    for _ in range(3):
        r.m.feed_result(res(1.0))
    r.m.feed_result(res(0.0))
    assert r.m.state is State.NORMAL and r.m.current_split_index == 1
    for _ in range(4):
        r.read("2", dt=0.1)  # respawned HUD agrees with the detector: new baseline
    r.m.feed_result(res(1.0))
    ev = r.m.feed_result(res(1.0))
    assert isinstance(ev, SplitEvent)
    r.m.feed_result(res(1.0))
    assert r.m.state is State.DONE


def test_second_death_reopens_the_window_and_undoes_again() -> None:
    r = confirmed_last()
    c = EndedLS()
    assert fast_exec().attempt(c, r.m, 1, 1.0).kind is OutcomeKind.APPLIED
    c.phase, c.idx = "Ended", 2  # the second split fire
    _second_death(r)
    assert r.m.undo_pending
    for _ in range(40):
        r.read("2")
    assert r.m.undo_target == 1
    assert fast_exec().attempt(c, r.m, 1, 2.0).kind is OutcomeKind.APPLIED
    assert c.calls.count("unsplit") == 2


def test_real_finish_after_a_death_finishes_without_undo() -> None:
    r = confirmed_last()
    c = EndedLS()
    assert fast_exec().attempt(c, r.m, 1, 1.0).kind is OutcomeKind.APPLIED
    r.events()
    _second_death(r)
    got: list[VerdictKind] = []
    for _ in range(40):
        r.read("")
        got += kinds(r)
    assert got == [VerdictKind.FINISHED]
    assert c.calls.count("unsplit") == 1


def test_second_fire_without_respawn_hud_reads_has_no_baseline_and_no_undo() -> None:
    r = confirmed_last()
    c = EndedLS()
    assert fast_exec().attempt(c, r.m, 1, 1.0).kind is OutcomeKind.APPLIED
    r.events()
    for _ in range(3):
        r.m.feed_result(res(1.0))
    r.m.feed_result(res(0.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    assert r.m.state is State.DONE and not r.m.undo_pending
    assert [v.kind for v in r.events()] == [VerdictKind.NO_BASELINE]


def test_colour_lit_forever_after_undo_never_refires() -> None:
    r = confirmed_last()
    assert fast_exec().attempt(EndedLS(), r.m, 1, 1.0).kind is OutcomeKind.APPLIED
    for _ in range(500):
        assert r.m.feed_result(res(1.0)) is None
    for _ in range(50):
        assert r.m.feed_result(res(0.5)) is None  # == reset_threshold does not clear
    assert r.m.state is State.WAITING_FOR_RESET
    r.m.feed_result(res(0.49))
    assert r.m.state is State.NORMAL


def test_executor_pending_with_ended_never_sends_a_second_unsplit() -> None:
    r = confirmed_last()
    clock = [100.0]
    ex = fast_exec()
    ex2 = type(ex)(sleep=lambda s: clock.__setitem__(0, clock[0] + s), clock=lambda: clock[0])
    c = EndedLS()
    c.idx_script = [2, None, None, None]
    assert ex2.attempt(c, r.m, 1, 1.0).kind is OutcomeKind.UNVERIFIED
    # a second (impossible) verdict for the same window: no new command
    r.m._undo_target = 1  # pyright: ignore[reportPrivateUsage]
    r.m._undo_target_epoch = r.m.epoch  # pyright: ignore[reportPrivateUsage]
    out = ex2.attempt(c, r.m, 1, 2.0)
    assert out.kind is OutcomeKind.SKIPPED and c.calls.count("unsplit") == 1


def test_pending_deadline_unconfirmed_never_double_unsplits() -> None:
    r = confirmed_last()
    clock = [100.0]
    ex = type(fast_exec())(sleep=lambda s: clock.__setitem__(0, clock[0] + s), clock=lambda: clock[0])
    c = EndedLS()
    c.idx_script = [2, None, None, None]
    assert ex.attempt(c, r.m, 1, 1.0).kind is OutcomeKind.UNVERIFIED
    c.idx_script = [None] * 40
    final = None
    for _ in range(40):
        clock[0] += 0.6
        final = ex.pending_step(c, r.m) or final
        if not ex.pending:
            break
    assert final is not None and final.kind is OutcomeKind.UNCONFIRMED
    assert not ex.pending and c.calls.count("unsplit") == 1 and r.m.state is State.DONE


def test_pending_then_user_finishes_again_state_changed() -> None:
    # the reconcile answers after the detector already left DONE: nothing moves
    r = confirmed_last()
    clock = [100.0]
    ex = type(fast_exec())(sleep=lambda s: clock.__setitem__(0, clock[0] + s), clock=lambda: clock[0])
    c = EndedLS()
    c.idx_script = [2, None, None, None]
    assert ex.attempt(c, r.m, 1, 1.0).kind is OutcomeKind.UNVERIFIED
    _ = r.m.rollback_to(1)  # may be refused in DONE: either is fine
    r.m.reset()
    clock[0] += 0.6
    c.idx_script = [1]
    out = ex.pending_step(c, r.m)
    assert out is not None and not out.changed and r.m.current_split_index == 0


def test_executor_ended_ls_equal_target_with_ended_phase() -> None:
    # Ended phase but LiveSplit already reports N-1: roll back, send nothing
    r = confirmed_last()
    c = EndedLS(idx=1, phase="Ended")
    out = fast_exec().attempt(c, r.m, 1, 1.0)
    assert "unsplit" not in c.calls
    assert out.kind in (OutcomeKind.ALREADY_THERE, OutcomeKind.SKIPPED)


def test_unsplit_that_leaves_ended_phase_is_not_applied() -> None:
    # server answers index N-1 but the phase stays Ended (should never happen):
    # executor only checks the index; verify it does not crash or double send.
    r = confirmed_last()
    c = EndedLS()

    def odd_unsplit() -> bool:
        c.calls.append("unsplit")
        c.idx = 1
        return True

    c.unsplit = odd_unsplit
    out = fast_exec().attempt(c, r.m, 1, 1.0)
    assert out.kind is OutcomeKind.APPLIED and c.calls.count("unsplit") == 1


def test_user_unsplit_further_back_rolls_back_to_that() -> None:
    r = confirmed_last(n=3)
    c = EndedLS(idx=3)
    seq: list[int | None] = [3, 1]  # after our unsplit LiveSplit already reports 1 (user also unsplit)
    c.apply = False
    c.idx_script = [3] + seq
    out = fast_exec().attempt(c, r.m, 2, 1.0)
    assert out.kind in (OutcomeKind.USER_UNSPLIT, OutcomeKind.APPLIED)
    assert r.m.current_split_index <= 2 and c.calls.count("unsplit") == 1


# ------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "undo",
    [
        "{last_split_extra_settle_seconds: true}",
        "{last_split_extra_settle_seconds: false}",
        "{last_split_extra_settle_seconds: null}",
        "{last_split_extra_settle_seconds: [1]}",
        "{last_split_extra_settle_seconds: '2'}",
        "{apply_to_last_split: 1}",
        "{apply_to_last_split: 'yes'}",
        "{apply_to_last_split: null}",
        "{apply_to_last_split: 0}",
    ],
)
def test_config_rejects_wrong_types(tmp_path: Path, undo: str) -> None:
    with pytest.raises(ConfigError):
        _ = load_undo(tmp_path, undo)


def test_window_exactly_equal_to_the_sum_is_rejected(tmp_path: Path) -> None:
    # settle 3 + extra 2 + confirm 1 = 6 (defaults) => window 6 must be rejected
    with pytest.raises(ConfigError):
        _ = load_undo(tmp_path, "{apply_to_last_split: true, window_seconds: 6}")


def test_window_rounding_does_not_reject_a_valid_sum(tmp_path: Path) -> None:
    c = load_undo(
        tmp_path,
        "{apply_to_last_split: true, settle_seconds: 0.1, last_split_extra_settle_seconds: 0.2,"
        + " confirm_seconds: 0.7, window_seconds: 1.0000001}",
    )
    assert isinstance(c, Config)


def test_window_rounding_just_below_the_sum_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _ = load_undo(
            tmp_path,
            "{apply_to_last_split: true, settle_seconds: 0.1, last_split_extra_settle_seconds: 0.2,"
            + " confirm_seconds: 0.7, window_seconds: 0.9999999}",
        )


def test_apply_to_last_split_with_verify_only_mode_needs_no_follow_timer(tmp_path: Path) -> None:
    from test_undo_last import CFG  # pyright: ignore[reportImplicitRelativeImport]

    p = tmp_path / "c.yaml"
    p.write_text(CFG + "  mode: verify_only\n  undo: {apply_to_last_split: true}\n", encoding="utf-8")
    from autosplitter.config import load

    assert isinstance(load(p), Config)


# ------------------------------------------------ (3) exit logic / end to end


class Custom:
    """Frame source scripted by frame count + server state."""

    exhausted: bool = False
    eof_at: int | None = None
    total: int = 0
    script: Callable[[int], bool] | None = None  # True => black frame
    on_frame: Callable[[int], None] | None = None

    def __init__(self, *_a: object, **_k: object) -> None:
        Custom.total = 0
        Custom.exhausted = False

    def read(self) -> np.ndarray | None:
        time.sleep(0.004)
        Custom.total += 1
        n = Custom.total
        if Custom.eof_at is not None and n >= Custom.eof_at:
            Custom.exhausted = True
            return None
        if Custom.on_frame is not None:
            Custom.on_frame(n)
        black = Custom.script(n) if Custom.script is not None else False
        return np.full((36, 64, 3), 0 if black else 255, np.uint8)

    def close(self) -> None:
        pass


def first_three(n: int) -> bool:
    return (1 <= n <= 4) or (11 <= n <= 14) or (21 <= n <= 24)


class Hud:
    def __init__(self, server: LsServer, linger: Callable[[], float]) -> None:
        self.server: LsServer = server
        self.linger: Callable[[], float] = linger
        self.t_end: float | None = None

    def text(self, _frame_idx: int = 0) -> str:
        s = self.server
        if s.phase != "Ended":
            self.t_end = None
            return str(max(0, min(s.idx + 1, s.count))) if s.idx >= 0 else ""
        if self.t_end is None:
            self.t_end = time.monotonic()
        return str(s.count) if time.monotonic() - self.t_end < self.linger() else ""


def run_ex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    *, linger: Callable[[], float], script: Callable[[int], bool] = first_three,
    on_frame: Callable[[int, LsServer], None] | None = None, frames: int = 600,
    mode: str = "undo", last: bool = True, loop: bool = False, livesplit: bool = True,
    srv: LsServer | None = None, extra: str = ", apply_to_last_split: true,"
    + " last_split_extra_settle_seconds: 0.1", window: int = 20, ocr_ok: bool = True,
    ls_extra: str = "", eof_at: int | None = None,
) -> tuple[str, LsServer, int]:
    srv = srv or LsServer(idx=0, count=3)
    hud = Hud(srv, linger)

    def fake_build(_c: object) -> tuple[Worker | None, str]:
        return (Worker(hud), "fake OK") if ocr_ok else (None, "unavailable")  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(sm, "build_ocr_worker", fake_build)
    monkeypatch.setattr(mainmod, "MpegTsFrameSource", Custom)
    Custom.script = script
    Custom.eof_at = eof_at
    Custom.on_frame = (lambda n: on_frame(n, srv)) if on_frame is not None else None
    p = write_cfg(tmp_path, srv.port, mode, extra if last else "", ls_extra, 0.2)
    p.write_text(p.read_text(encoding="utf-8").replace("window_seconds: 30", f"window_seconds: {window}"),
                 encoding="utf-8")
    args = ["--config", str(p), "--live", "x.ts", "--max-frames", str(frames)]
    if livesplit:
        args.append("--livesplit")
    if loop:
        args.append("--loop")
    try:
        code = mainmod.main(args)
        out = capsys.readouterr().out
    finally:
        srv.close()
    assert code == 0, out
    return out, srv, Custom.total


def test_e2e_death_then_second_death_then_real_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = {"pulse": 0, "since": 0}
    srv = LsServer(idx=0, count=3)

    def script(n: int) -> bool:
        if first_three(n):
            return True
        return state["since"] != 0 and n - state["since"] < 4

    def on_frame(n: int, s: LsServer) -> None:
        u = s.commands.count("unsplit")
        if (
            n > 30 and s.phase == "Running" and s.idx == 2 and u > state["pulse"]
            and state["since"] + 60 < n
        ):
            state["pulse"] += 1
            state["since"] = n + 40  # let the settle clear (white) first

    out, srv, total = run_ex(
        tmp_path, monkeypatch, capsys, srv=srv, script=script, on_frame=on_frame,
        linger=lambda: 1e9 if srv.commands.count("unsplit") < 2 else 0.02, frames=3000,
    )
    assert srv.moves() == ["split"] * 3 + ["unsplit", "split", "unsplit", "split"], out
    assert srv.phase == "Ended" and "HUD gone after the last split" in out
    assert total < 3000, "must exit after FINISHED"


def test_e2e_ls_ignores_unsplit_no_retry_and_exit_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    srv = LsServer(idx=0, count=3)
    orig = srv._handle  # pyright: ignore[reportPrivateUsage]

    def handle(line: str) -> str | None:
        if line == "unsplit":
            srv.commands.append(line)
            return None  # swallowed: stays Ended
        return orig(line)

    srv._handle = handle  # pyright: ignore[reportPrivateUsage]
    out, srv, total = run_ex(tmp_path, monkeypatch, capsys, srv=srv, linger=lambda: 1e9, frames=2500)
    assert srv.commands.count("unsplit") == 1, out
    assert total < 2500, "exit must be released once the attempt resolved"


def after_ended(delay: float, act: Callable[[LsServer], None]) -> Callable[[int, LsServer], None]:
    """on_frame hook: run ``act`` once, ``delay`` s after the server first reports Ended."""
    st: dict[str, float] = {}

    def hook(_n: int, s: LsServer) -> None:
        if s.phase != "Ended" or "done" in st:
            return
        st.setdefault("t", time.monotonic())
        if time.monotonic() - st["t"] >= delay:
            st["done"] = 1.0
            act(s)

    return hook


def test_e2e_user_resets_timer_during_the_window_sends_no_unsplit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def act(s: LsServer) -> None:
        s.phase, s.idx = "NotRunning", -1

    out, srv, _ = run_ex(tmp_path, monkeypatch, capsys, linger=lambda: 1e9, on_frame=after_ended(0.02, act))
    assert srv.commands.count("unsplit") == 0, out


def test_e2e_user_pauses_timer_during_the_window_sends_no_unsplit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def act(s: LsServer) -> None:
        s.phase = "Paused"

    out, srv, _ = run_ex(tmp_path, monkeypatch, capsys, linger=lambda: 1e9, on_frame=after_ended(0.02, act))
    assert srv.commands.count("unsplit") == 0, out


def test_e2e_user_unsplits_during_the_window_we_send_nothing_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def act(s: LsServer) -> None:
        s.idx, s.phase = 2, "Running"  # user unsplit by hand

    out, srv, _ = run_ex(tmp_path, monkeypatch, capsys, linger=lambda: 1e9, on_frame=after_ended(0.02, act))
    assert srv.commands.count("unsplit") == 0, out


def test_e2e_livesplit_dies_after_the_last_split_bounded_and_no_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def act(s: LsServer) -> None:
        s.stop = True  # server goes away; sockets drop

    t0 = time.monotonic()
    out, srv, total = run_ex(
        tmp_path, monkeypatch, capsys, linger=lambda: 1e9, on_frame=after_ended(0.02, act), frames=3000
    )
    assert srv.commands.count("unsplit") == 0, out
    assert time.monotonic() - t0 < 60 and total < 3000


def test_e2e_ocr_unavailable_exits_at_done_with_no_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, total = run_ex(tmp_path, monkeypatch, capsys, linger=lambda: 1e9, ocr_ok=False, frames=800)
    assert total < 120 and srv.moves() == ["split"] * 3, out


def test_e2e_no_livesplit_flag_dry_run_never_sends_and_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    srv = LsServer(idx=2, count=3)  # HUD reads "3" (= D-1 of the last split) all along
    out, srv, total = run_ex(
        tmp_path, monkeypatch, capsys, srv=srv, linger=lambda: 1e9, livesplit=False, frames=2500
    )
    assert "would UNDO" in out
    assert srv.commands == [], out
    assert total < 2500


def test_e2e_window_expiry_releases_the_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # the HUD flickers forever: neither verdict; the window (4 s) must end the hold
    flip = {"n": 0}

    def linger() -> float:
        flip["n"] += 1
        return 1e9 if flip["n"] % 2 else 0.0

    extra = ", apply_to_last_split: true, last_split_extra_settle_seconds: 0.1"
    t0 = time.monotonic()
    out, srv, total = run_ex(
        tmp_path, monkeypatch, capsys, linger=linger, frames=6000, extra=extra, window=4
    )
    assert srv.commands.count("unsplit") == 0, out
    assert total < 6000 and time.monotonic() - t0 < 30


def test_e2e_keyboard_interrupt_during_the_hold_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def act(_s: LsServer) -> None:
        raise KeyboardInterrupt

    out, srv, _ = run_ex(tmp_path, monkeypatch, capsys, linger=lambda: 1e9, on_frame=after_ended(0.01, act))
    assert srv.commands.count("unsplit") == 0 and "Traceback" not in out, out


def test_e2e_loop_finished_then_rearms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, total = run_ex(tmp_path, monkeypatch, capsys, linger=lambda: 0.02, loop=True, frames=1200)
    assert srv.commands.count("unsplit") == 0, out
    assert "HUD gone after the last split" in out and "RUN COMPLETE" in out
    assert total == 1200


def test_e2e_off_flag_is_byte_identical_exit_at_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, total = run_ex(tmp_path, monkeypatch, capsys, linger=lambda: 1e9, last=False, frames=800)
    assert total < 60 and srv.moves() == ["split"] * 3
    assert "HUD gone" not in out and "would UNDO" not in out and "UNDO" not in out


def test_e2e_max_frames_during_the_hold_returns_cleanly_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, total = run_ex(tmp_path, monkeypatch, capsys, linger=lambda: 1e9, frames=28)
    assert total == 28 and srv.commands.count("unsplit") == 0, out


def test_e2e_eof_during_the_hold_exits_without_unsplit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, srv, _total = run_ex(tmp_path, monkeypatch, capsys, linger=lambda: 1e9, eof_at=29, frames=800)
    assert "Source exhausted" in out and srv.commands.count("unsplit") == 0, out


def test_n1_with_start_gate_undo_from_done_settles_to_normal_not_waiting_for_start() -> None:
    r = Rig(1, OCR_LAST, start=True)
    for _ in range(3):
        r.m.feed_result(res(1.0))  # the start gate fires and settles
    r.m.feed_result(res(0.0))
    assert r.m.state is State.NORMAL and r.m.current_split_index == 0
    r.agree()
    r.m.feed_result(res(1.0))
    assert isinstance(r.m.feed_result(res(1.0)), SplitEvent)
    r.m.feed_result(res(1.0))
    assert r.m.state is State.DONE and r.m.undo_pending
    for _ in range(40):
        r.read("1")
    assert r.m.undo_target == 0
    out = fast_exec().attempt(EndedLS(idx=1), r.m, 0, 1.0)
    assert out.kind is OutcomeKind.APPLIED
    assert r.m.state is State.WAITING_FOR_RESET and not r.m.waiting_for_start
    r.m.feed_result(res(0.0))
    assert r.m.state is State.NORMAL
    r.m.feed_result(res(1.0))
    ev = r.m.feed_result(res(1.0))
    assert isinstance(ev, SplitEvent)  # a SplitEvent, never a second StartEvent
