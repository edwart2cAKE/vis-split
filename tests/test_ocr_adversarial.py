"""Adversarial tests for verify_only OCR evidence (increment 1)."""

from __future__ import annotations

import random
import shutil
import subprocess
import threading
import time
import tracemalloc
from pathlib import Path
from typing import cast, final

import cv2
import numpy as np
import pytest

from autosplitter import main as m
from autosplitter.config import (
    DEFAULT_ROI,
    Capture,
    Color,
    Config,
    ConfigError,
    Detection,
    LiveSplit,
    OcrConfig,
    OcrEngine,
    OcrFilter,
    OcrPattern,
    Roi,
    Split,
    StartTransition,
    load,
)
from autosplitter.detection import DetectionResult, State, TransitionDetector
from autosplitter.detection.element_reader import (
    PreprocessParams,
    TesseractCliReader,
    parse_tsv,
    preprocess,
)
from autosplitter.detection.evidence import (
    EvidenceFilter,
    EvidenceKind,
    compile_patterns,
    resolve_index,
)
from autosplitter.detection.ocr_detector import OcrDetector, OcrResult, OcrWorker

RAW = PreprocessParams(scale=1, blur=0, threshold="none", pad=0)
FLT = OcrFilter(consensus_frames=2, jump_consensus_frames=3, max_age_seconds=1.0, min_confidence=50.0)
OCR = OcrConfig(
    roi=Roi(x=0.0, y=0.0, width=0.5, height=0.5),
    engine=OcrEngine(min_confidence=50.0),
    patterns=(OcrPattern(regex="(\\d+)"),),
    poll_interval_frames=1,
    filter=FLT,
)
FRAME = np.zeros((64, 64, 3), np.uint8)
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="no tesseract")


def res(ratio: float) -> DetectionResult:
    return DetectionResult(matched_pixels=int(ratio * 100), total_pixels=100, ratio=ratio)


def cfg(n: int = 4, ocr: OcrConfig | None = OCR, start: bool = False) -> Config:
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


SEEN: set[EvidenceKind] = set()


def snapshot(mc: TransitionDetector) -> tuple[object, ...]:
    return (mc.state, mc.current_split_index, mc.epoch, mc._streak, mc._transition_is_start)  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------------------
# verify_only: twin-machine equivalence under a randomised feed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("start", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_ocr_machine_is_bit_identical_to_machine_without_ocr(n: int, start: bool, seed: int) -> None:
    rng = random.Random(seed * 100 + n)
    fw = FakeWorker()
    a = TransitionDetector(cfg(n, OCR, start), ocr=fw)  # pyright: ignore[reportArgumentType]
    b = TransitionDetector(cfg(n, None, start))
    for f in range(1, 1500):
        op = rng.random()
        if op < 0.04:
            idx = rng.randint(-1, n + 1)
            ra, rb = a.advance_to(idx), b.advance_to(idx)
            assert ra == rb
        elif op < 0.08:
            idx = rng.randint(-1, n + 1)
            assert a.rollback_to(idx) == b.rollback_to(idx)
        elif op < 0.10:
            settle = rng.random() < 0.5
            a.reset(settle=settle)
            b.reset(settle=settle)
        else:
            r = rng.choice([0.0, 0.3, 0.6, 0.95, 1.0, 1.0])
            ea, eb = a.feed_result(res(r)), b.feed_result(res(r))
            assert ea == eb
        # push readings: good epoch, stale epoch, junk text, low conf
        for _ in range(rng.randint(0, 2)):
            ep = a.epoch + rng.choice([0, 0, 0, -1, 1])
            txt = rng.choice(["1", "2", "3", "5", "9", "", "x", "0", "12"])
            for _rep in range(rng.choice([1, 1, 3, 6])):  # bursts reach consensus
                fw.push(txt, f - rng.choice([0, 0, 0, 1, 500]), ep, rng.choice([90.0, 90.0, 90.0, 10.0]))
                a.check_ocr_resync(FRAME, f)
                SEEN.add(a.ocr_evidence.kind)
        a.check_ocr_resync(FRAME, f)
        assert snapshot(a) == snapshot(b), f"diverged at frame {f}"


def test_zz_fuzz_exercised_both_verdicts() -> None:
    assert {EvidenceKind.AGREE, EvidenceKind.MISMATCH} <= SEEN


def test_check_ocr_resync_with_frame_idx_going_backwards_does_not_crash() -> None:
    fw = FakeWorker()
    mc = TransitionDetector(cfg(3), ocr=fw)  # pyright: ignore[reportArgumentType]
    for f in (100, 50, 0, -5, 10_000_000):
        fw.push("1", f, mc.epoch)
        mc.check_ocr_resync(FRAME, f)
    assert mc.state is State.NORMAL


def test_evidence_survives_reset_loop_rearm_only_by_epoch() -> None:
    """After reset() (--loop rearm) a reading stamped pre-reset must be ignored."""
    fw = FakeWorker()
    mc = TransitionDetector(cfg(3), ocr=fw)  # pyright: ignore[reportArgumentType]
    old = mc.epoch
    mc.reset()
    for f in (1, 2, 3):
        fw.push("1", f, old)
        mc.check_ocr_resync(FRAME, f)
    assert mc.ocr_evidence.kind is EvidenceKind.NONE
    for f in (4, 5):
        fw.push("1", f, mc.epoch)
        mc.check_ocr_resync(FRAME, f)
    assert mc.ocr_evidence.kind is EvidenceKind.AGREE


def test_reset_discards_existing_agree() -> None:
    fw = FakeWorker()
    mc = TransitionDetector(cfg(3), ocr=fw)  # pyright: ignore[reportArgumentType]
    for f in (1, 2):
        fw.push("1", f, mc.epoch)
        mc.check_ocr_resync(FRAME, f)
    assert mc.ocr_evidence.kind is EvidenceKind.AGREE
    mc.reset()
    assert mc.ocr_evidence.kind is EvidenceKind.NONE


def test_evidence_ttl_expires() -> None:
    fw = FakeWorker()
    mc = TransitionDetector(cfg(3), ocr=fw)  # pyright: ignore[reportArgumentType]
    for f in (1, 2):
        fw.push("1", f, mc.epoch)
        mc.check_ocr_resync(FRAME, f)
    assert mc.ocr_evidence.kind is EvidenceKind.AGREE
    mc.check_ocr_resync(FRAME, 2 + 100)  # ttl = ceil(max_age_seconds(1.0) * fps(100))
    assert mc.ocr_evidence.kind is EvidenceKind.AGREE
    mc.check_ocr_resync(FRAME, 2 + 101)
    assert mc.ocr_evidence.kind is EvidenceKind.NONE


def test_empty_reading_between_good_ones_breaks_streak() -> None:
    fw = FakeWorker()
    mc = TransitionDetector(cfg(3), ocr=fw)  # pyright: ignore[reportArgumentType]
    for f, txt in ((1, "1"), (2, ""), (3, "1")):
        fw.push(txt, f, mc.epoch)
        mc.check_ocr_resync(FRAME, f)
    assert mc.ocr_evidence.kind is EvidenceKind.NONE


def test_single_split_run() -> None:
    fw = FakeWorker()
    mc = TransitionDetector(cfg(1), ocr=fw)  # pyright: ignore[reportArgumentType]
    for f in (1, 2):
        fw.push("1", f, mc.epoch)
        mc.check_ocr_resync(FRAME, f)
    assert mc.ocr_evidence.kind is EvidenceKind.AGREE
    for f in (3, 4, 5, 6):
        fw.push("2", f, mc.epoch)  # out of range (only 1 split)
        mc.check_ocr_resync(FRAME, f)
    assert mc.state is State.NORMAL


def test_disabled_ocr_with_injected_worker_never_filters() -> None:
    import dataclasses

    fw = FakeWorker()
    c = cfg(3, dataclasses.replace(OCR, enabled=False))
    mc = TransitionDetector(c, ocr=fw)  # pyright: ignore[reportArgumentType]
    for f in (1, 2, 3):
        fw.push("1", f, mc.epoch)
        mc.check_ocr_resync(FRAME, f)
    assert mc.ocr_evidence.kind is EvidenceKind.NONE


# --------------------------------------------------------------------------
# EvidenceFilter / pattern resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["²", "³", "⑧", "9" * 5000])
def test_resolve_index_never_raises_on_odd_digit_text(text: str) -> None:
    """str.isdigit() accepts superscripts / circled digits that int() rejects,
    and Python caps int() at 4300 digits. Untrusted OCR text must not raise."""
    pats = compile_patterns((OcrPattern(regex="(.+)"),))
    resolve_index(pats, text)  # must return int | None, never raise


def test_filter_feed_never_raises_on_odd_text() -> None:
    f = EvidenceFilter(OCR, 4, 100.0)
    from autosplitter.detection.element_reader import Reading

    cfg2 = OcrConfig(roi=OCR.roi, patterns=(OcrPattern(regex="(.+)"),), filter=FLT)
    f = EvidenceFilter(cfg2, 4, 100.0)
    f.feed(Reading("²", 90.0, 1, 0, 0.0), 1, 0, 0)


def test_pattern_edge_cases() -> None:
    pats = compile_patterns(
        (
            OcrPattern(regex="^BOSS$", split_index=3),
            OcrPattern(regex="Split"),  # no group, no explicit index
            OcrPattern(regex="(-?\\d+)"),
        )
    )
    assert resolve_index(pats, "BOSS") == 3
    assert resolve_index(pats, "Split") is None
    assert resolve_index(pats, "-3") is None  # "-3".isdigit() is False
    f = EvidenceFilter(
        OcrConfig(roi=OCR.roi, patterns=(OcrPattern(regex="(\\d+)"),), filter=FLT), 3, 100.0
    )
    from autosplitter.detection.element_reader import Reading

    # "0" -> index -1 -> must be rejected, not wrapped around as list[-1]
    for i in range(1, 6):
        ev = f.feed(Reading("0", 90.0, i, 0, 0.0), i, 0, 2)
        assert ev.kind is EvidenceKind.NONE


def test_filter_streak_requires_consecutive_distinct_indices() -> None:
    from autosplitter.detection.element_reader import Reading

    f = EvidenceFilter(OCR, 5, 100.0)
    seq = ["2", "3", "2", "3", "2", "3"]  # flapping 2/3 never reaches consensus 2
    for i, t in enumerate(seq, 1):
        ev = f.feed(Reading(t, 90.0, i, 0, 0.0), i, 0, 1)
        assert ev.kind is EvidenceKind.NONE


def test_filter_consensus_one_fires_immediately_and_jump_lt_consensus() -> None:
    from autosplitter.detection.element_reader import Reading

    c = OcrConfig(
        roi=OCR.roi,
        patterns=OCR.patterns,
        filter=OcrFilter(consensus_frames=1, jump_consensus_frames=1, max_age_seconds=1.0, min_confidence=0.0),
    )
    f = EvidenceFilter(c, 5, 100.0)
    ev = f.feed(Reading("5", 0.0, 1, 0, 0.0), 1, 0, 0)  # confidence 0.0 == min 0.0 accepted
    assert ev.kind is EvidenceKind.MISMATCH and ev.screen_idx == 4


def test_filter_future_frame_reading_is_accepted_not_crashing() -> None:
    from autosplitter.detection.element_reader import Reading

    f = EvidenceFilter(OCR, 3, 100.0)
    ev = f.feed(Reading("1", 90.0, 10_000, 0, 0.0), 1, 0, 0)
    assert ev.kind in (EvidenceKind.NONE, EvidenceKind.AGREE)


# --------------------------------------------------------------------------
# parse_tsv robustness
# --------------------------------------------------------------------------

HDR = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"


def row(conf: str, text: str, level: str = "5") -> str:
    return f"{level}\t1\t1\t1\t1\t1\t0\t0\t1\t1\t{conf}\t{text}\n"


def test_parse_tsv_variants() -> None:
    assert parse_tsv("") is None
    assert parse_tsv(HDR) is None
    assert parse_tsv(HDR + row("-1", "x", "4")) is None
    assert parse_tsv(HDR + row("abc", "5")) is None
    assert parse_tsv((HDR + row("90", "7")).replace("\n", "\r\n")) == ("7", 90.0)
    assert parse_tsv(HDR + "5\tshort\n") is None
    assert parse_tsv(HDR + row("80", "é中")) == ("é中", 80.0)


def test_parse_tsv_multiword_joined_hazard() -> None:
    """Documents: two words '1' '2' read as '1 2' (regex (\\d+) then says split 1)."""
    out = parse_tsv(HDR + row("95", "1") + row("95", "2"))
    assert out == ("1 2", 95.0)
    pats = compile_patterns(OCR.patterns)
    assert resolve_index(pats, "1 2") == 0  # NOTE: silently truncated number


def test_parse_tsv_mean_confidence_can_hide_junk_word() -> None:
    out = parse_tsv(HDR + row("96", "12") + row("2", "|"))
    assert out is not None and out[1] == pytest.approx(49.0)


# --------------------------------------------------------------------------
# TesseractCliReader: process failures
# --------------------------------------------------------------------------


def test_close_twice_and_before_read_and_after_read() -> None:
    r = TesseractCliReader()
    r.close()
    r.close()


def test_binary_vanishes_after_available_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    r = TesseractCliReader()
    r._avail = (None, "5.5.4")  # pyright: ignore[reportPrivateUsage]

    def boom(*_a: object, **_k: object) -> None:
        raise FileNotFoundError("tesseract")

    monkeypatch.setattr(subprocess, "Popen", boom)
    assert r.read(np.zeros((8, 8), np.uint8)) is None
    assert r.last_error is not None and "FileNotFoundError" in r.last_error


def test_garbage_stdout_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    @final
    class P:
        returncode: int = 0

        def communicate(
            self, _data: bytes | None = None, timeout: float | None = None  # pyright: ignore[reportUnusedParameter]
        ) -> tuple[bytes, bytes]:
            return b"\xff\xfe\x00level\n" + b"\xc3\x28\n", b""

        def kill(self) -> None:
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: P())  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
    r = TesseractCliReader()
    assert r.read(np.zeros((8, 8), np.uint8)) is None
    assert r.last_error is None


@needs_tesseract
def test_real_timeout_leaves_no_child_and_recovers() -> None:
    r = TesseractCliReader(timeout_seconds=0.001, psm=7)
    img = preprocess(np.zeros((36, 64, 3), np.uint8), PreprocessParams())
    assert r.read(img) is None
    assert r.last_error is not None and "timed out" in r.last_error
    assert r._proc is None  # pyright: ignore[reportPrivateUsage]
    r._timeout = 10.0  # pyright: ignore[reportPrivateUsage]
    assert r.read(img) is None and r.last_error is None  # recovers; blank image


@needs_tesseract
def test_real_close_from_other_thread_kills_inflight() -> None:
    r = TesseractCliReader(timeout_seconds=30.0)
    img = preprocess(np.random.default_rng(0).integers(0, 255, (400, 1200, 3), dtype=np.uint8), PreprocessParams())
    out: list[object] = []
    t = threading.Thread(target=lambda: out.append(r.read(img)))
    t.start()
    time.sleep(0.05)
    t0 = time.monotonic()
    r.close()
    t.join(10)
    assert not t.is_alive()
    assert time.monotonic() - t0 < 5


@needs_tesseract
@pytest.mark.parametrize("shape", [(1, 1), (2, 36), (36, 2)])
def test_real_tiny_images_are_blank_not_errors(shape: tuple[int, int]) -> None:
    r = TesseractCliReader(psm=7, whitelist="0123456789")
    for params in (RAW, PreprocessParams()):
        assert r.read(preprocess(np.zeros((*shape, 3), np.uint8), params)) is None
        assert r.last_error is None


# --------------------------------------------------------------------------
# preprocess degenerate input
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fill", [0, 255, 127])
def test_preprocess_uniform_images(fill: int) -> None:
    out = preprocess(np.full((36, 64, 3), fill, np.uint8), PreprocessParams())
    assert out.dtype == np.uint8 and out.ndim == 2


def test_preprocess_grayscale_and_non_contiguous_and_blur_large() -> None:
    g = np.zeros((36, 64), np.uint8)
    assert preprocess(g, PreprocessParams()).ndim == 2
    nc = np.zeros((36, 128, 3), np.uint8)[:, ::2]
    assert not nc.flags["C_CONTIGUOUS"]
    assert preprocess(nc, PreprocessParams()).ndim == 2


def test_preprocess_even_blur_rejected_by_opencv_not_silently() -> None:
    with pytest.raises(cv2.error):
        preprocess(np.zeros((8, 8, 3), np.uint8), PreprocessParams(blur=2))


def test_preprocess_full_frame_roi_is_bounded() -> None:
    big = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    out = preprocess(big, PreprocessParams())
    assert time.perf_counter() - t0 < 2.0
    assert out.shape[0] > 4000  # documents: 4x full-frame upscale (no size cap)


# --------------------------------------------------------------------------
# Real tesseract on harder renderings
# --------------------------------------------------------------------------


def render(txt: str, w: int, h: int, fg: tuple[int, int, int], bg: tuple[int, int, int],
           thick: int = 2, noise: float = 0.0, grad: bool = False) -> np.ndarray:
    img = np.full((h, w, 3), bg, np.uint8)
    if grad:
        g = np.linspace(0, 60, w).astype(np.int16)[None, :, None]
        img = np.clip(img.astype(np.int16) + g, 0, 255).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(txt, font, 1, thick)
    s = min(w * 0.8 / tw, h * 0.7 / th)
    (tw, th), _ = cv2.getTextSize(txt, font, s, thick)
    cv2.putText(img, txt, ((w - tw) // 2, (h + th) // 2), font, s, fg, thick, cv2.LINE_AA)
    if noise:
        rng = np.random.default_rng(1)
        img = np.clip(img.astype(np.int16) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


@needs_tesseract
@pytest.mark.parametrize("psm", [7])
@pytest.mark.parametrize("txt", ["1", "10", "12", "7", "15"])
@pytest.mark.parametrize(
    "fg,bg",
    [((255, 255, 255), (0, 0, 0)), ((0, 200, 255), (30, 30, 30)), ((20, 20, 20), (235, 235, 235))],
)
def test_real_multi_digit_small_roi(psm: int, txt: str, fg: tuple[int, int, int], bg: tuple[int, int, int]) -> None:
    r = TesseractCliReader(psm=psm, whitelist="0123456789")
    out = r.read(preprocess(render(txt, 64, 36, fg, bg, noise=10, grad=True), PreprocessParams()))
    assert out is not None, r.last_error
    assert out[0] == txt and out[1] > 60


@needs_tesseract
def test_real_noise_never_passes_confidence_gate() -> None:
    r = TesseractCliReader(psm=7, whitelist="0123456789")
    rng = np.random.default_rng(0)
    for _ in range(5):
        img = rng.integers(0, 255, (36, 64, 3), dtype=np.uint8)
        out = r.read(preprocess(img, PreprocessParams()))
        assert out is None or out[1] < 60, out


@needs_tesseract
def test_real_fraction_text_partial_read_hazard() -> None:
    """'12/20' low contrast partially read as '20' with conf 77 -> wrong split."""
    r = TesseractCliReader(psm=7)
    out = r.read(preprocess(render("12/20", 160, 48, (140, 140, 140), (110, 110, 110), thick=1, grad=True), PreprocessParams()))
    assert out is None or out[1] < 60 or "12" in out[0], out


@needs_tesseract
def test_real_end_to_end_worker_to_machine_agree() -> None:
    det = OcrDetector(Roi(x=0.0, y=0.0, width=1.0, height=1.0), OcrEngine(whitelist="0123456789"), OCR.patterns)
    worker = OcrWorker(det, 1)
    worker.start()
    try:
        mc = TransitionDetector(cfg(15, OCR), ocr=worker)
        frame = render("12", 64, 36, (255, 255, 255), (0, 0, 0))
        mc.advance_to(11)
        mc.feed_result(res(0.0))  # settle -> NORMAL at index 11
        assert mc.state is State.NORMAL and mc.current_split_index == 11
        deadline = time.monotonic() + 15
        f = 0
        while time.monotonic() < deadline and mc.ocr_evidence.kind is EvidenceKind.NONE:
            f += 1
            mc.check_ocr_resync(frame, f)
            time.sleep(0.02)
        assert mc.ocr_evidence.kind is EvidenceKind.AGREE
        assert mc.current_split_index == 11
    finally:
        worker.close()


# --------------------------------------------------------------------------
# Worker: races, latest-wins, memory, odd frames
# --------------------------------------------------------------------------


@final
class SlowReader:
    name: str = "slow"

    def __init__(self) -> None:
        self.gate: threading.Event = threading.Event()
        self.seen: list[int] = []
        self.entered: threading.Event = threading.Event()

    def available(self) -> str | None:
        return None

    def read(self, img: np.ndarray) -> tuple[str, float] | None:
        self.entered.set()
        _ = self.gate.wait(5)
        self.seen.append(int(cast(int, img.flat[0])))
        return "1", 90.0


DEF_ROI = Roi(0.0, 0.0, 0.5, 0.5)


def mk(reader: object, poll: int = 1, roi: Roi = DEF_ROI) -> OcrWorker:
    det = OcrDetector(roi, OcrEngine(), OCR.patterns, reader=reader)  # pyright: ignore[reportArgumentType]
    return OcrWorker(det, poll, reader=reader, params=RAW)  # pyright: ignore[reportArgumentType]


def test_latest_wins_and_bounded_reads() -> None:
    rd = SlowReader()
    w = mk(rd)
    w.start()
    try:
        fr = np.zeros((64, 64, 3), np.uint8)
        fr[0, 0] = 1
        w.submit_frame(fr, 0, 0)
        assert rd.entered.wait(3)
        for i in range(2, 2000):
            fr[0, 0] = i % 250 + 1
            w.submit_frame(fr, i, 0)
        last = int(cast(int, fr[0, 0, 0]))
        rd.gate.set()
        end = time.monotonic() + 3
        while time.monotonic() < end and (len(rd.seen) < 2):
            time.sleep(0.01)
        time.sleep(0.2)
        assert len(rd.seen) == 2, rd.seen  # first + exactly one coalesced latest
        assert rd.seen[-1] in (last, 255 - last)  # preprocess may invert polarity
        res_ = w.take_result()
        assert res_ is not None and res_.frame_number == 1999
    finally:
        rd.gate.set()
        w.close()


def test_submit_memory_flat_with_blocked_reader() -> None:
    rd = SlowReader()
    w = mk(rd)
    w.start()
    try:
        fr = np.zeros((720, 1280, 3), np.uint8)
        w.submit_frame(fr, 0, 0)
        assert rd.entered.wait(3)
        for i in range(1, 50):
            w.submit_frame(fr, i, 0)
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        for i in range(50, 3000):
            w.submit_frame(fr, i, 0)
        grown = tracemalloc.get_traced_memory()[0] - base
        tracemalloc.stop()
        assert grown < 64 * 1024, grown
    finally:
        rd.gate.set()
        w.close()


def test_changing_frame_shapes_dtypes_and_layouts() -> None:
    @final
    class Rd:
        name: str = "r"

        def available(self) -> str | None:
            return None

        def read(self, _img: np.ndarray) -> tuple[str, float] | None:
            return "1", 90.0

    w = mk(Rd())
    w.start()
    try:
        frames: list[np.ndarray] = [
            np.zeros((64, 64, 3), np.uint8),
            np.zeros((360, 640, 3), np.uint8),
            np.zeros((10, 10), np.uint8),  # grayscale
            np.zeros((64, 128, 3), np.uint8)[:, ::2],  # non-contiguous
            np.zeros((64, 64, 3), np.float32),  # wrong dtype
            np.zeros((1, 1, 3), np.uint8),
            np.zeros((0, 5, 3), np.uint8),
        ]
        for i, fr in enumerate(frames):
            w.submit_frame(fr, i, 0)
            time.sleep(0.05)
        assert w.alive
        assert w.status in ("ok",) or w.status.startswith("error")
        w.submit_frame(np.zeros((64, 64, 3), np.uint8), 99, 0)
        end = time.monotonic() + 3
        while time.monotonic() < end and (w.poll_result() is None or w.poll_result().frame_number != 99):  # pyright: ignore[reportOptionalMemberAccess]
            time.sleep(0.01)
        assert w.status == "ok"
    finally:
        w.close()


@pytest.mark.parametrize("roi", [Roi(0.99, 0.99, 0.01, 0.01), Roi(1.0, 1.0, 0.0, 0.0), Roi(0.0, 0.0, 1.0, 1.0)])
def test_roi_at_edges_tiny_frame(roi: Roi) -> None:
    rd = SlowReader()
    rd.gate.set()
    w = mk(rd, roi=roi)
    w.start()
    try:
        for shape in ((5, 5, 3), (1, 1, 3), (100, 3, 3)):
            w.submit_frame(np.zeros(shape, np.uint8), 0, 0)
        time.sleep(0.2)
        assert w.alive and w.status == "ok"
    finally:
        w.close()


def test_lifecycle_edges() -> None:
    rd = SlowReader()
    rd.gate.set()
    w = mk(rd)
    w.close()  # never started
    w.close()
    w.start()  # after close: must not start
    assert not w.alive
    w.submit_frame(FRAME, 0, 0)  # submit after close must not raise
    w2 = mk(rd)
    w2.start()
    w2.start()  # double start tolerated
    w2.close()
    w2.close()
    assert not w2.alive
    w2.submit_frame(FRAME, 1, 0)


def test_close_never_hangs_with_stuck_reader() -> None:
    rd = SlowReader()  # gate never opens (5 s)
    w = mk(rd)
    w.start()
    w.submit_frame(FRAME, 0, 0)
    assert rd.entered.wait(3)
    t0 = time.monotonic()
    w.close()
    assert time.monotonic() - t0 < 3.5
    rd.gate.set()


def test_concurrent_submit_take_close_stress() -> None:
    @final
    class Rd:
        name: str = "r"

        def available(self) -> str | None:
            return None

        def read(self, _img: np.ndarray) -> tuple[str, float] | None:
            return "3", 99.0

    w = mk(Rd())
    w.start()
    stop = threading.Event()
    errors: list[BaseException] = []

    def producer() -> None:
        try:
            i = 0
            while not stop.is_set():
                w.submit_frame(FRAME, i, 0)
                i += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def consumer() -> None:
        try:
            while not stop.is_set():
                _ = w.take_result()
                _ = w.poll_result()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=producer), threading.Thread(target=consumer), threading.Thread(target=producer)]
    for t in ts:
        t.start()
    time.sleep(0.5)
    w.close()
    time.sleep(0.2)
    stop.set()
    for t in ts:
        t.join(3)
    assert not errors and not w.alive


def test_reading_arriving_after_reset_is_epoch_stale_with_real_worker() -> None:
    rd = SlowReader()
    w = mk(rd)
    w.start()
    try:
        mc = TransitionDetector(cfg(3, OCR), ocr=w)
        mc.check_ocr_resync(FRAME, 1)  # submit stamped with epoch e0
        assert rd.entered.wait(3)
        mc.reset()  # epoch bump while read in flight
        rd.gate.set()
        for f in range(2, 12):
            mc.check_ocr_resync(FRAME, f)
            time.sleep(0.01)
        # Any evidence produced must be from post-reset readings only.
        assert snapshot(mc)[:2] == (State.NORMAL, 0)
    finally:
        rd.gate.set()
        w.close()


def test_unexpected_reader_return_types_do_not_kill_thread() -> None:
    @final
    class Weird:
        name: str = "w"

        def __init__(self) -> None:
            self.n: int = 0

        def available(self) -> str | None:
            return None

        def read(self, _img: np.ndarray) -> tuple[str, float] | None:
            self.n += 1
            choices: list[object] = [None, 5, ("1",), ("1", "x"), ("1", float("nan"))]
            return cast("tuple[str, float] | None", choices[self.n % 5])

    w = mk(Weird())
    w.start()
    try:
        for i in range(10):
            w.submit_frame(FRAME, i, 0)
            time.sleep(0.03)
        assert w.alive
    finally:
        w.close()


# --------------------------------------------------------------------------
# Hot loop cost when idle
# --------------------------------------------------------------------------


def test_idle_check_ocr_resync_is_allocation_free_and_cheap() -> None:
    rd = SlowReader()
    rd.gate.set()
    w = mk(rd, poll=3)
    w.start()
    try:
        mc = TransitionDetector(cfg(3, OCR), ocr=w)
        fr = np.zeros((1080, 1920, 3), np.uint8)
        for i in range(1, 100):  # warm up
            mc.check_ocr_resync(fr, i)
        time.sleep(0.2)
        idle = [i for i in range(100, 3100) if i % 3 != 0]
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        for i in idle:
            mc.check_ocr_resync(fr, i)
        grown = tracemalloc.get_traced_memory()[0] - base
        tracemalloc.stop()
        assert grown < 2048, grown
        t0 = time.perf_counter()
        for i in idle:
            mc.check_ocr_resync(fr, i)
        per = (time.perf_counter() - t0) / len(idle)
        assert per < 20e-6, f"{per * 1e6:.1f} us per idle frame"
    finally:
        w.close()


def test_submit_frame_cost_on_poll_frame_1080p() -> None:
    rd = SlowReader()
    rd.gate.set()
    det = OcrDetector(Roi(0.4, 0.0, 0.1, 0.05), OcrEngine(), OCR.patterns, reader=rd)
    w = OcrWorker(det, 1, reader=rd, params=RAW)
    w.start()
    try:
        fr = np.zeros((1080, 1920, 3), np.uint8)
        t0 = time.perf_counter()
        for i in range(1000):
            w.submit_frame(fr, i, 0)
        per = (time.perf_counter() - t0) / 1000
        assert per < 200e-6, f"{per * 1e6:.1f} us"
    finally:
        w.close()


# --------------------------------------------------------------------------
# Config validation edge cases
# --------------------------------------------------------------------------

BASE = """
capture: {fps: 60, stream_url: "udp://127.0.0.1:9733", width: 1920, height: 1080}
roi: {x: 0.25, y: 0.25, width: 0.5, height: 0.5}
detection: {required_frames: 3, reset_threshold: 0.8}
colors: {black: {rgb: [0, 0, 0], tolerance: 15}}
splits:
  - {id: 1, name: S, color: black, threshold: 0.95}
livesplit: {host: "127.0.0.1", port: 16834}
"""
ROI = "  roi: {x: 0.0, y: 0.0, width: 0.1, height: 0.1}\n"


def load_ocr(tmp_path: Path, body: str) -> Config:
    p = tmp_path / "c.yaml"
    p.write_text(BASE + "ocr:\n" + ROI + body, encoding="utf-8")
    return load(p)


@pytest.mark.parametrize(
    "body",
    [
        "  filter: {consensus_frames: true}\n",
        "  filter: {jump_consensus_frames: 2.5}\n",
        "  filter: {consensus_frames: '3'}\n",
        "  filter: {max_age_seconds: true}\n",
        "  filter: {max_age_seconds: .nan}\n",
        "  filter: {max_age_seconds: -1}\n",
        "  filter: {min_confidence: .nan}\n",
        "  filter: {min_confidence: -0.1}\n",
        "  filter: {min_confidence: true}\n",
        "  preprocess: {scale: true}\n",
        "  preprocess: {scale: 0}\n",
        "  preprocess: {scale: 9}\n",
        "  preprocess: {scale: 2.0}\n",
        "  preprocess: {blur: 2}\n",
        "  preprocess: {blur: -1}\n",
        "  preprocess: {blur: 4}\n",
        "  preprocess: {blur: false}\n",
        "  preprocess: {pad: -1}\n",
        "  preprocess: {pad: 51}\n",
        "  preprocess: {threshold: null}\n",
        "  preprocess: {threshold: adaptive}\n",
        "  preprocess: [1]\n",
        "  engine: {backend: TESSEROCR}\n",
        "  engine: {backend: 1}\n",
        "  engine: {whitelist: ''}\n",
        "  engine: {whitelist: 12}\n",
        "  engine: {timeout_seconds: .nan}\n",
        "  engine: {timeout_seconds: .inf}\n",
        "  engine: {timeout_seconds: true}\n",
        "  engine: {timeout_seconds: -2}\n",
        "  engine: {min_confidence: .nan}\n",
        "  engine: {psm: true}\n",
        "  enabled: 1\n",
        "  enabled: 'yes'\n",
        "  mode: null\n",
        "  mode: resync\n",
        "  mode: Verify_Only\n",
        "  poll_interval_frames: true\n",
        "  poll_interval_frames: 0\n",
    ],
)
def test_invalid_ocr_config_raises_configerror_not_traceback(tmp_path: Path, body: str) -> None:
    with pytest.raises(ConfigError) as ei:
        load_ocr(tmp_path, body)
    assert "c.yaml: " in str(ei.value)


@pytest.mark.parametrize(
    "body",
    [
        "  preprocess: {blur: 0, scale: 1, pad: 0}\n",
        "  preprocess: {blur: 1}\n",
        "  preprocess: {scale: 8, pad: 50, blur: 99}\n",
        "  filter: {consensus_frames: 1, jump_consensus_frames: 1, min_confidence: 0}\n",
        "  filter: {min_confidence: 100}\n",
        "  engine: {backend: tesseract_cli, whitelist: '0123456789', timeout_seconds: 0.5}\n",
        "  preprocess:\n  filter:\n",
        "  mystery_key: 1\n",
        "  preprocess: {mystery: 1}\n",
    ],
)
def test_valid_edge_ocr_config_loads(tmp_path: Path, body: str) -> None:
    assert load_ocr(tmp_path, body).ocr is not None


def test_even_blur_never_reaches_preprocess_via_config(tmp_path: Path) -> None:
    c = load_ocr(tmp_path, "  preprocess: {blur: 99}\n")
    assert c.ocr is not None
    from autosplitter.detection.ocr_detector import params_from

    preprocess(np.zeros((8, 8, 3), np.uint8), params_from(c.ocr.preprocess))


# --------------------------------------------------------------------------
# main.py surfaces
# --------------------------------------------------------------------------


def test_ocr_column_strings() -> None:
    fw = FakeWorker()
    mc = TransitionDetector(cfg(4), ocr=fw)  # pyright: ignore[reportArgumentType]
    assert m._ocr_column(mc, True) == "ocr=n/a"  # pyright: ignore[reportPrivateUsage]
    assert m._ocr_column(mc, False) == "ocr=?"  # pyright: ignore[reportPrivateUsage]
    for f in (1, 2):
        fw.push("1", f, mc.epoch)
        mc.check_ocr_resync(FRAME, f)
    assert m._ocr_column(mc, False) == "ocr=1="  # pyright: ignore[reportPrivateUsage]
    for f in (3, 4):
        fw.push("2", f, mc.epoch)
        mc.check_ocr_resync(FRAME, f)
    assert m._ocr_column(mc, False) == "ocr=2!"  # pyright: ignore[reportPrivateUsage]
    fw.status = "failed"
    assert m._ocr_column(mc, False) == "ocr=err"  # pyright: ignore[reportPrivateUsage]
    fw.status = "error: x"
    assert m._ocr_column(mc, False) == "ocr=2!"  # pyright: ignore[reportPrivateUsage]
    assert len(m._ocr_column(mc, False)) <= 7  # pyright: ignore[reportPrivateUsage]


def test_ocr_column_width_overflow_for_double_digit_splits() -> None:
    fw = FakeWorker()
    mc = TransitionDetector(cfg(20), ocr=fw)  # pyright: ignore[reportArgumentType]
    for f in (1, 2, 3, 4):
        fw.push("12", f, mc.epoch)
        mc.check_ocr_resync(FRAME, f)
    assert len(m._ocr_column(mc, False)) <= 7  # pyright: ignore[reportPrivateUsage]


def _write_cfg(tmp_path: Path, ocr_body: str) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(BASE + "ocr:\n" + ROI + ocr_body, encoding="utf-8")
    return p


def test_check_tesserocr_backend_unavailable_exit_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = _write_cfg(tmp_path, "  engine: {backend: tesserocr}\n")
    assert m.main(["--config", str(p), "--check"]) == 0
    assert "ocr backend: UNAVAILABLE (tesserocr backend not implemented)" in capsys.readouterr().out


def test_check_disabled_does_not_probe_tesseract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("probed")

    monkeypatch.setattr(subprocess, "run", boom)
    p = _write_cfg(tmp_path, "  enabled: false\n")
    assert m.main(["--config", str(p), "--check"]) == 0
    assert "disabled" in capsys.readouterr().out


def test_check_when_version_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(shutil, "which", _which_false)
    p = _write_cfg(tmp_path, "")
    assert m.main(["--config", str(p), "--check"]) == 0
    assert "UNAVAILABLE" in capsys.readouterr().out


def _which_none(_n: str) -> str | None:
    return None


def _which_false(_n: str) -> str | None:
    return "/bin/false"


def _ocr_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "ocr-worker" and t.is_alive()]


@needs_tesseract
def test_detect_with_ocr_section_does_not_leak_worker_thread(tmp_path: Path) -> None:
    img = tmp_path / "f.png"
    assert cv2.imwrite(str(img), np.zeros((36, 64, 3), np.uint8))
    p = _write_cfg(tmp_path, "")
    before = len(_ocr_threads())
    assert m.main(["--config", str(p), "--detect", str(img)]) == 0
    assert len(_ocr_threads()) == before, "--detect leaked an ocr-worker thread"


@needs_tesseract
def test_probe_with_ocr_section_does_not_spawn_worker(tmp_path: Path) -> None:
    img = tmp_path / "f.png"
    assert cv2.imwrite(str(img), np.zeros((36, 64, 3), np.uint8))
    p = _write_cfg(tmp_path, "")
    before = len(_ocr_threads())
    assert m.main(["--config", str(p), "--probe", str(img)]) == 0
    assert len(_ocr_threads()) == before


def test_machine_build_with_unavailable_backend_has_no_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", _which_none)
    before = len(_ocr_threads())
    mc = TransitionDetector(cfg(3, OCR))
    assert mc.ocr_worker_status is None and "not found" in mc.ocr_status
    mc.check_ocr_resync(FRAME, 1)
    mc.shutdown_ocr()
    assert len(_ocr_threads()) == before


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
@pytest.mark.parametrize("tess_present", [True, False])
def test_live_smoke_file_with_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tess_present: bool
) -> None:
    ts = tmp_path / "in.ts"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x36:r=30:d=2",
         "-c:v", "libx264", "-f", "mpegts", str(ts)],
        check=True,
    )
    text = (BASE.replace("width: 1920, height: 1080", "width: 64, height: 36")
            + "ocr:\n" + ROI + "  poll_interval_frames: 1\n")
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    if not tess_present:
        real_which = shutil.which
        def which(n: str) -> str | None:
            return None if n == "tesseract" else real_which(n)

        monkeypatch.setattr(shutil, "which", which)
    before = len(_ocr_threads())
    rc = m.main(["--config", str(p), "--live", str(ts), "--width", "64", "--height", "36", "--max-frames", "40"])
    out = capsys.readouterr().out
    assert rc == 0, out
    if tess_present:
        assert "ocr=" in out and "n/a" not in out
    else:
        assert "WARN: ocr configured but backend unavailable" in out
        assert "ocr=n/a" in out
    assert len(_ocr_threads()) == before, "live run leaked an ocr-worker thread"
