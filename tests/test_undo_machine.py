"""State-machine integration of the dry-run undo judge (injected fake worker)."""

from __future__ import annotations

import dataclasses
import random
from typing import final

import numpy as np
import pytest

from autosplitter.config import (
    DEFAULT_ROI,
    Capture,
    Color,
    Config,
    Detection,
    LiveSplit,
    OcrConfig,
    OcrEngine,
    OcrFilter,
    OcrPattern,
    OcrUndo,
    Roi,
    Split,
    StartTransition,
)
from autosplitter.detection import DetectionResult, State, TransitionDetector
from autosplitter.detection.ocr_detector import OcrResult
from autosplitter.detection.undo_judge import UndoVerdict, VerdictKind

BASE_OCR = OcrConfig(
    roi=Roi(x=0.0, y=0.0, width=0.5, height=0.5),
    engine=OcrEngine(min_confidence=50.0),
    patterns=(OcrPattern(regex="(\\d+)"),),
    poll_interval_frames=1,
    filter=OcrFilter(consensus_frames=2, jump_consensus_frames=3, max_age_seconds=1.0, min_confidence=50.0),
)


def cfg(n: int = 4, ocr: OcrConfig | None = BASE_OCR, start: bool = False) -> Config:
    return Config(
        capture=Capture(fps=100, stream_url="udp://127.0.0.1:9733", width=64, height=64),
        roi=DEFAULT_ROI,
        detection=Detection(required_frames=2, reset_threshold=0.5),
        colors={"black": Color(name="black", rgb=(0, 0, 0), tolerance=15)},
        splits=tuple(Split(id=i + 1, name=f"S{i + 1}", color="black", threshold=0.9) for i in range(n)),
        livesplit=LiveSplit(host="127.0.0.1", port=16834),
        start_transition=StartTransition(name="go", color="black", threshold=0.9) if start else None,
        ocr=ocr,
    )


@final
class FakeWorker:
    """Duck-typed OcrWorker: results are pushed by the test; no thread."""

    def __init__(self) -> None:
        self.version: int = 0
        self.taken_version: int = 0
        self.status: str = "ok"
        self.latest: OcrResult | None = None

    def push(self, text: str, frame: int, epoch: int, conf: float = 90.0) -> None:
        self.latest = OcrResult(text, conf, None, frame, epoch, 0.0)
        self.version += 1

    def submit_frame(self, _frame: np.ndarray, _frame_idx: int, _epoch: int = 0) -> None:
        pass

    def take_result(self) -> OcrResult | None:
        if self.taken_version == self.version:
            return None
        self.taken_version = self.version
        return self.latest

    def close(self) -> None:
        pass


def snapshot(mc: TransitionDetector) -> tuple[object, ...]:
    return (mc.state, mc.current_split_index, mc.epoch, mc._streak, mc._transition_is_start)  # pyright: ignore[reportPrivateUsage]

FRAME = np.zeros((64, 64, 3), np.uint8)
UNDO = OcrUndo(settle_seconds=3.0, confirm_reads=3, confirm_seconds=1.0, window_seconds=10.0,
               min_confidence=80.0, baseline_seconds=2.0)
OCR_U = dataclasses.replace(
    BASE_OCR,
    filter=OcrFilter(consensus_frames=2, jump_consensus_frames=3, max_age_seconds=100.0, min_confidence=50.0),
    undo=UNDO,
)


def res(ratio: float) -> DetectionResult:
    return DetectionResult(matched_pixels=int(ratio * 100), total_pixels=100, ratio=ratio)


@final
class Clock:
    def __init__(self) -> None:
        self.t: float = 1000.0

    def __call__(self) -> float:
        return self.t


@final
class Rig:
    def __init__(self, n: int = 4, ocr: OcrConfig | None = OCR_U, start: bool = False) -> None:
        self.fw: FakeWorker = FakeWorker()
        self.clock: Clock = Clock()
        self.m: TransitionDetector = TransitionDetector(
            cfg(n, ocr, start), ocr=self.fw, clock=self.clock  # pyright: ignore[reportArgumentType]
        )
        self.frame: int = 0

    def read(self, text: str, conf: float = 90.0, dt: float = 0.25) -> None:
        self.clock.t += dt
        self.frame += 1
        self.fw.push(text, self.frame, self.m.epoch, conf)
        self.m.check_ocr_resync(FRAME, self.frame)

    def agree(self, n: int = 3) -> None:
        for _ in range(n):
            self.read(str(self.m.current_split_index + 1), dt=0.1)

    def fire_and_clear(self) -> None:
        """Colour transition: fire, settle, clear (advances the index)."""
        self.m.feed_result(res(1.0))
        assert self.m.feed_result(res(1.0)) is not None
        self.m.feed_result(res(1.0))
        self.m.feed_result(res(0.0))

    def events(self) -> list[UndoVerdict]:
        out: list[UndoVerdict] = []
        while (v := self.m.take_undo_event()) is not None:
            out.append(v)
        return out


def death_rig() -> Rig:
    r = Rig()
    r.agree()
    r.fire_and_clear()
    assert r.m.state is State.NORMAL and r.m.current_split_index == 1
    return r


def test_no_undo_section_no_judge() -> None:
    r = Rig(ocr=BASE_OCR)
    assert not r.m.undo_judge_active
    r.agree()
    r.fire_and_clear()
    assert not r.m.undo_pending and r.m.undo_target is None and r.m.take_undo_event() is None


def test_window_opens_only_on_advancing_clear() -> None:
    r = Rig()
    r.agree()
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))  # fired
    assert not r.m.undo_pending
    r.m.feed_result(res(1.0))  # WAITING_FOR_RESET
    assert not r.m.undo_pending
    r.m.feed_result(res(0.0))  # clear
    assert r.m.undo_pending and r.m.undo_target is None


def test_start_gate_clear_opens_no_window() -> None:
    r = Rig(start=True)
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(0.0))
    assert r.m.state is State.NORMAL and r.m.current_split_index == 0
    assert not r.m.undo_pending and r.events() == []


def test_last_split_and_single_split_open_no_window() -> None:
    r = Rig(n=2)
    r.agree()
    r.fire_and_clear()
    assert r.m.undo_pending
    r.read("1")  # irrelevant; now fire the last split
    r2 = Rig(n=1)
    r2.agree()
    r2.m.feed_result(res(1.0))
    r2.m.feed_result(res(1.0))
    r2.m.feed_result(res(1.0))
    assert r2.m.state is State.DONE
    assert not r2.m.undo_pending and r2.events() == []
    # last split of a 2-split run: window of split 1 cancelled by the new fire,
    # and no new window is opened for the last split.
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    assert r.m.state is State.DONE
    r.read("2")
    assert not r.m.undo_pending
    assert [v.kind for v in r.events()] == [VerdictKind.CANCELLED]


def test_baseline_latch_and_boundary() -> None:
    r = Rig()
    r.agree()
    r.clock.t += 2.0 - 0.1  # last AGREE was at t0 (just now); age <= baseline
    r.fire_and_clear()
    assert r.m.undo_pending

    r = Rig()
    r.agree()
    r.clock.t += 2.0001
    r.fire_and_clear()
    assert not r.m.undo_pending
    ev = r.events()
    assert [v.kind for v in ev] == [VerdictKind.NO_BASELINE] and ev[0].split_index == 0


def test_baseline_exactly_at_limit_ok() -> None:
    r = Rig()
    r.agree()
    r.clock.t += 2.0
    r.fire_and_clear()
    assert r.m.undo_pending


def test_no_agree_means_no_baseline_and_never_a_verdict() -> None:
    r = Rig()
    r.fire_and_clear()
    assert [v.kind for v in r.events()] == [VerdictKind.NO_BASELINE]
    for _ in range(60):
        r.read("1")
    assert r.events() == [] and r.m.undo_target is None


def test_baseline_for_wrong_index_is_not_baseline() -> None:
    r = Rig()
    r.agree()  # AGREE latched for index 0 only
    r.fire_and_clear()
    r.read("2")  # genuine, closes the first window
    assert [v.kind for v in r.events()] == [VerdictKind.GENUINE]
    r.fire_and_clear()  # split 1 fires without an AGREE for index 1
    assert not r.m.undo_pending
    assert [v.kind for v in r.events()] == [VerdictKind.NO_BASELINE]


def test_require_baseline_false_opens_without_agree() -> None:
    ocr = dataclasses.replace(OCR_U, undo=dataclasses.replace(UNDO, require_baseline=False))
    r = Rig(ocr=ocr)
    r.fire_and_clear()
    assert r.m.undo_pending


def test_death_yields_undo_dry_run_state_untouched() -> None:
    r = death_rig()
    snap = snapshot(r.m)
    verdicts: list[UndoVerdict] = []
    for _ in range(40):
        r.read("1")
        verdicts += r.events()
    assert [v.kind for v in verdicts] == [VerdictKind.UNDO]
    assert verdicts[0].screen_idx == 0 and verdicts[0].split_index == 0
    assert r.m.undo_target == 0 and not r.m.undo_pending
    assert snapshot(r.m) == snap  # nothing changed because of OCR
    r.m.feed_result(res(1.0))  # a new fire changes epoch -> target gone
    r.m.feed_result(res(1.0))
    assert r.m.undo_target is None


def test_lag_guard_then_genuine_logs_lag() -> None:
    r = death_rig()
    for _ in range(8):  # 2.0 s of the old number: inside settle
        r.read("1")
    assert r.events() == [] and r.m.undo_pending
    r.read("2")
    ev = r.events()
    assert [v.kind for v in ev] == [VerdictKind.GENUINE]
    assert 2.0 <= ev[0].elapsed <= 2.6
    assert not r.m.undo_pending and r.m.undo_target is None
    for _ in range(40):
        r.read("1")
    assert r.events() == []


def test_low_confidence_reads_never_count() -> None:
    r = death_rig()
    for _ in range(60):
        r.read("1", conf=79.0)  # passes the filter (50) but not undo's 80
    assert [v.kind for v in r.events()] == [VerdictKind.EXPIRED]  # 15 s > window
    assert r.m.undo_target is None


def test_effective_min_confidence_not_below_engine() -> None:
    ocr = dataclasses.replace(
        OCR_U,
        engine=dataclasses.replace(OCR_U.engine, min_confidence=95.0),
        filter=dataclasses.replace(OCR_U.filter, min_confidence=50.0),
        undo=dataclasses.replace(UNDO, min_confidence=10.0),
    )
    r = Rig(ocr=ocr)
    r.agree()
    r.fire_and_clear()
    for _ in range(60):
        r.read("1", conf=90.0)
    assert r.m.undo_target is None


def test_respawn_flash_cancels() -> None:
    r = death_rig()
    r.read("1")
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))  # new fire: epoch bump
    assert not r.m.undo_pending
    r.read("1")
    assert [v.kind for v in r.events()] == [VerdictKind.CANCELLED]


@pytest.mark.parametrize("action", ["advance", "rollback", "reset"])
def test_follow_actions_close_window(action: str) -> None:
    r = death_rig()
    if action == "advance":
        assert r.m.advance_to(3)
    elif action == "rollback":
        assert r.m.rollback_to(0)
    else:
        r.m.reset()
    assert not r.m.undo_pending
    r.read("1")
    assert [v.kind for v in r.events()] == [VerdictKind.CANCELLED]
    assert r.m.undo_target is None


def test_expiry() -> None:
    r = death_rig()
    r.clock.t += 10.5
    r.read("garbage")
    assert [v.kind for v in r.events()] == [VerdictKind.EXPIRED]
    assert not r.m.undo_pending


def test_stale_and_wrong_epoch_readings_ignored() -> None:
    r = death_rig()
    for _ in range(60):
        r.clock.t += 0.25
        r.frame += 1
        r.fw.push("1", r.frame, r.m.epoch - 1, 99.0)  # wrong epoch
        r.m.check_ocr_resync(FRAME, r.frame)
    assert r.m.undo_target is None
    assert [v.kind for v in r.events()] in ([], [VerdictKind.EXPIRED])


def test_odd_screen_values_warn_only_once_and_never_act() -> None:
    r = death_rig()
    snap = snapshot(r.m)
    for _ in range(6):
        r.read("4")  # S > D
    for _ in range(6):
        r.read("4")
    ev = r.events()
    assert [v.kind for v in ev] == [VerdictKind.WARN]
    assert r.m.undo_target is None and snapshot(r.m) == snap


def test_warn_reading_breaks_undo_streak() -> None:
    r = death_rig()
    r.clock.t += 3.0
    r.read("1")
    r.read("1")
    r.read("4")
    r.read("1")
    r.read("1")
    assert r.m.undo_target is None
    assert [v.kind for v in r.events()] == [VerdictKind.WARN]


def test_hot_config_reload_cannot_break_or_change_judge() -> None:
    r = death_rig()
    r.m.reload_config(cfg(4, BASE_OCR))  # undo: dropped
    r.m.reload_config(cfg(4, None))  # ocr: dropped
    r.m.rollback_to(0)
    r.m.reset()
    r2 = Rig()
    r2.m.reload_config(cfg(4, BASE_OCR))
    r2.m.reload_config(cfg(4, None))
    r2.agree()
    r2.fire_and_clear()  # must not raise
    assert r2.m.undo_pending
    for _ in range(40):  # original settle/confirm thresholds still apply
        r2.read("1")
    assert r2.m.undo_target == 0


def test_idle_state_has_no_window_and_properties_cheap() -> None:
    r = Rig()
    assert r.m.undo_judge_active and not r.m.undo_pending and r.m.undo_target is None


def test_judge_not_built_without_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    from autosplitter.detection import state_machine as sm

    def no_worker(_cfg: OcrConfig) -> tuple[None, str]:
        return None, "no tesseract"

    monkeypatch.setattr(sm, "build_ocr_worker", no_worker)
    m = TransitionDetector(cfg(3, OCR_U))
    assert not m.undo_judge_active


# --------------------------------------------------------------------------
# Twin-machine equivalence: judge on vs OCR absent, randomised feed
# --------------------------------------------------------------------------

WINDOWS: set[VerdictKind] = set()


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("start", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_machine_with_judge_identical_to_machine_without(n: int, start: bool, seed: int) -> None:
    rng = random.Random(seed * 31 + n)
    fw = FakeWorker()
    clock = Clock()
    a = TransitionDetector(cfg(n, OCR_U, start), ocr=fw, clock=clock)  # pyright: ignore[reportArgumentType]
    b = TransitionDetector(cfg(n, None, start))
    mode = "noise"
    for f in range(1, 2500):
        clock.t += 0.2
        if f % 60 == 1:
            mode = rng.choice(["death", "ok", "noise"])
        op = rng.random()
        if op < 0.015 and a.state is State.NORMAL and a.current_split_index < n - 1:
            # scripted death: vouch for the split, fire it, then keep showing the old number
            for _ in range(3):
                fw.push(str(a.current_split_index + 1), f, a.epoch, 95.0)
                a.check_ocr_resync(FRAME, f)
            for r_ in (1.0, 1.0, 1.0, 0.0):
                assert a.feed_result(res(r_)) == b.feed_result(res(r_))
            for _ in range(25):
                clock.t += 0.2
                fw.push(str(a.current_split_index), f, a.epoch, 95.0)
                a.check_ocr_resync(FRAME, f)
                assert a.feed_result(res(0.0)) == b.feed_result(res(0.0))
                assert snapshot(a) == snapshot(b)
                while (v := a.take_undo_event()) is not None:
                    WINDOWS.add(v.kind)
        elif op < 0.02:
            idx = rng.randint(-1, n + 1)
            assert a.advance_to(idx) == b.advance_to(idx)
        elif op < 0.04:
            idx = rng.randint(-1, n + 1)
            assert a.rollback_to(idx) == b.rollback_to(idx)
        elif op < 0.05:
            settle = rng.random() < 0.5
            a.reset(settle=settle)
            b.reset(settle=settle)
        else:
            r = rng.choice([0.0, 0.3, 0.6, 0.95, 1.0, 1.0, 1.0])
            assert a.feed_result(res(r)) == b.feed_result(res(r))
        i = a.current_split_index
        if mode == "death":
            shown = i - 1
        elif mode == "ok":
            shown = i
        else:
            shown = rng.choice([i - 1, i, i + 1, i + 3, -1])
        txt = str(shown + 1) if mode != "noise" else rng.choice([str(shown + 1), "x", ""])
        ep = a.epoch + rng.choice([0, 0, 0, 0, -1])
        fw.push(txt, f - rng.choice([0, 0, 1]), ep, rng.choice([95.0, 95.0, 95.0, 20.0]))
        a.check_ocr_resync(FRAME, f)
        while (v := a.take_undo_event()) is not None:
            WINDOWS.add(v.kind)
        assert snapshot(a) == snapshot(b), f"diverged at frame {f}"


def test_zz_fuzz_exercised_windows() -> None:
    assert {
        VerdictKind.UNDO, VerdictKind.GENUINE, VerdictKind.CANCELLED, VerdictKind.NO_BASELINE,
    } <= WINDOWS, WINDOWS


# --------------------------------------------------------------------------
# main.py: log lines and status column marker
# --------------------------------------------------------------------------


def test_undo_lines_are_one_based_and_say_dry_run() -> None:
    from autosplitter import main as m

    c = cfg(4, OCR_U)
    line = m._undo_line(UndoVerdict(VerdictKind.UNDO, 1, 1, 4.04, 5), c)  # pyright: ignore[reportPrivateUsage]
    assert line == (
        'OCR: would UNDO split "S2" (screen 2, detector 3, fired 4.0s ago, 5 reads)'
        " - dry run: nothing sent"
    )
    g = m._undo_line(UndoVerdict(VerdictKind.GENUINE, 0, 1, 1.234), c)  # pyright: ignore[reportPrivateUsage]
    assert g.startswith("OCR: HUD advanced +1.23s after split")
    assert m._undo_line(UndoVerdict(VerdictKind.CANCELLED, 0), c) == (  # pyright: ignore[reportPrivateUsage]
        "OCR: undo window cancelled: state changed"
    )
    assert m._undo_line(UndoVerdict(VerdictKind.EXPIRED, 0), c) == (  # pyright: ignore[reportPrivateUsage]
        "OCR: undo window expired (never confirmed)"
    )
    assert m._undo_line(UndoVerdict(VerdictKind.NO_BASELINE, 0), c) == (  # pyright: ignore[reportPrivateUsage]
        "OCR: no baseline (OCR unreadable pre-split); undo skipped"
    )
    assert "no action" in m._undo_line(UndoVerdict(VerdictKind.WARN, 0, 6), c)  # pyright: ignore[reportPrivateUsage]


def test_status_column_shows_undo_marker_only_while_window_open() -> None:
    from autosplitter import main as m

    r = death_rig()
    assert m._ocr_column(r.m, False) == "undo?"  # pyright: ignore[reportPrivateUsage]
    assert m._ocr_column(r.m, True) == "ocr=n/a"  # pyright: ignore[reportPrivateUsage]
    r.read("2")  # genuine closes the window
    assert m._ocr_column(r.m, False) != "undo?"  # pyright: ignore[reportPrivateUsage]
