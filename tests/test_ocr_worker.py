"""OcrWorker threading behavior with fake readers (no tesseract needed)."""


from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import override

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
    Roi,
    Split,
)
from autosplitter.detection import TransitionDetector
from autosplitter.detection import ocr_detector as od
from autosplitter.detection.element_reader import ElementReader, PreprocessParams
from autosplitter.detection.evidence import EvidenceKind
from autosplitter.detection.ocr_detector import OcrDetector, OcrWorker, build_ocr_worker

ROI = Roi(x=0.0, y=0.0, width=0.5, height=0.5)
RAW = PreprocessParams(scale=1, blur=0, threshold="none", pad=0)


def make_ocr_cfg(poll: int = 1, **flt: float) -> OcrConfig:
    return OcrConfig(
        roi=ROI,
        engine=OcrEngine(min_confidence=50.0),
        patterns=(OcrPattern(regex="(\\d+)"),),
        poll_interval_frames=poll,
        filter=OcrFilter(
            consensus_frames=int(flt.get("consensus", 2)),
            jump_consensus_frames=6,
            max_age_seconds=float(flt.get("max_age", 5.0)),
            min_confidence=50.0,
        ),
    )


def make_config(ocr: OcrConfig | None, n: int = 4) -> Config:
    return Config(
        capture=Capture(fps=100, stream_url="udp://127.0.0.1:9733", width=64, height=64),
        roi=DEFAULT_ROI,
        detection=Detection(required_frames=3, reset_threshold=0.5),
        colors={"black": Color(name="black", rgb=(0, 0, 0), tolerance=15)},
        splits=tuple(Split(id=i + 1, name=f"S{i + 1}", color="black", threshold=0.9) for i in range(n)),
        livesplit=LiveSplit(host="127.0.0.1", port=16834),
        ocr=ocr,
    )


class FakeReader:
    name: str = "fake"

    def __init__(self, value: str = "1", conf: float = 90.0) -> None:
        self.value: str = value
        self.conf: float = conf
        self.calls: int = 0
        self.delay: float = 0.0

    def available(self) -> str | None:
        return None

    def read(self, img: np.ndarray) -> tuple[str, float] | None:
        _ = img
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return self.value, self.conf


def make_worker(reader: ElementReader, poll: int = 1) -> OcrWorker:
    det = OcrDetector(ROI, OcrEngine(), (OcrPattern(regex="(\\d+)"),), reader=reader)
    return OcrWorker(det, poll, reader=reader, params=RAW)


def _which_true(_name: str) -> str:
    return "/usr/bin/true"


def _which_none(_name: str) -> None:
    return None


def _popen_returning(proc: object) -> Callable[..., object]:
    def popen(*_a: object, **_k: object) -> object:
        return proc

    return popen


def wait_for(cond: Callable[[], bool], timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_late_results_still_consumed() -> None:
    reader = FakeReader("2")
    worker = make_worker(reader, poll=1)
    machine = TransitionDetector(make_config(make_ocr_cfg(poll=1)), ocr=worker)
    worker.start()
    frame = np.zeros((64, 64, 3), np.uint8)
    try:
        saw = False
        for i in range(1, 301):
            reader.delay = 0.05 + 0.15 * ((i % 3) / 2)
            machine.check_ocr_resync(frame, i)
            time.sleep(0.002)
            if machine.ocr_evidence.kind is not EvidenceKind.NONE:
                saw = True
                break
        # detector at index 0; screen says "2" -> index 1 = det+1 -> needs 2 readings
        assert saw
        assert machine.ocr_evidence.kind is EvidenceKind.MISMATCH
        assert machine.current_split_index == 0
    finally:
        worker.close()


def test_buffer_tearing_crop_is_a_copy_at_submit_time() -> None:
    gate = threading.Event()
    got: list[np.ndarray] = []

    class Blocking(FakeReader):
        last_img: np.ndarray = np.zeros(0)

        @override
        def read(self, img: np.ndarray) -> tuple[str, float] | None:
            _ = img
            _ = gate.wait(3.0)
            got.append(img.copy())
            self.last_img = img
            return "1", 90.0

    reader = Blocking()
    worker = make_worker(reader)
    worker.start()
    frame = np.zeros((64, 64, 3), np.uint8)
    try:
        frame[:] = 7
        worker.submit_frame(frame, 1)
        assert wait_for(lambda: worker._pending is None or worker._pending_meta is None)  # pyright: ignore[reportPrivateUsage]
        # worker holds crop of value 7 and is blocked; the producer reuses the buffer
        for v in range(8, 20):
            frame[:] = v
            worker.submit_frame(frame, v)
        gate.set()
        assert wait_for(lambda: len(got) >= 1)
        first = got[0]
        assert np.array_equal(first, np.full_like(first, 7))
        assert not np.shares_memory(reader.last_img, frame)
        # the later frame also arrives intact (latest wins)
        assert wait_for(lambda: len(got) >= 2)
        assert np.array_equal(got[-1], np.full_like(got[-1], int(got[-1][0, 0])))  # pyright: ignore[reportAny]
    finally:
        gate.set()
        worker.close()


def test_exception_survival_and_status(caplog: pytest.LogCaptureFixture) -> None:
    class Flaky(FakeReader):
        @override
        def read(self, img: np.ndarray) -> tuple[str, float] | None:
            _ = img
            self.calls += 1  # pyright: ignore[reportUnannotatedClassAttribute]
            if self.calls <= 2:
                raise RuntimeError("boom")
            return "3", 88.0

    reader = Flaky()
    worker = make_worker(reader)
    worker.start()
    frame = np.zeros((64, 64, 3), np.uint8)
    statuses: list[str] = []
    try:
        with caplog.at_level(logging.WARNING, logger="autosplitter.ocr"):
            for i in range(1, 4):
                before = reader.calls
                worker.submit_frame(frame, i)
                assert wait_for(lambda b=before: reader.calls > b)
                assert wait_for(lambda n=i: worker.status != "ok" or (n == 3 and worker.version > 0))
                statuses.append(worker.status)
                if i < 3:
                    assert worker.alive
        assert statuses[0].startswith("error: RuntimeError: boom")
        assert wait_for(lambda: worker.version == 1)
        assert worker.status == "ok"
        res = worker.take_result()
        assert res is not None and res.text == "3" and res.split_index == 2
        boom_logs = [r for r in caplog.records if "boom" in r.getMessage()]
        assert len(boom_logs) == 1
    finally:
        worker.close()


def test_five_consecutive_errors_failed_but_keeps_trying() -> None:
    class Dead(FakeReader):
        @override
        def read(self, img: np.ndarray) -> tuple[str, float] | None:
            _ = img
            self.calls += 1  # pyright: ignore[reportUnannotatedClassAttribute]
            if self.calls <= 5:
                raise ValueError(f"bad {self.calls}")
            return "4", 90.0

    reader = Dead()
    worker = make_worker(reader)
    worker.start()
    frame = np.zeros((64, 64, 3), np.uint8)
    try:
        for i in range(1, 6):
            worker.submit_frame(frame, i)
            assert wait_for(lambda n=i: reader.calls == n)
            assert wait_for(lambda n=i: worker._errors == n)  # pyright: ignore[reportPrivateUsage]
        assert worker.status == "failed"
        assert worker.alive
        worker.submit_frame(frame, 6)
        assert wait_for(lambda: worker.version == 1)
        assert worker.status == "ok"
    finally:
        worker.close()


def test_reader_last_error_counts_as_error() -> None:
    class Soft(FakeReader):
        last_error: str | None = "exited with 1"

        @override
        def read(self, img: np.ndarray) -> tuple[str, float] | None:
            _ = img
            return None

    worker = make_worker(Soft())
    worker.start()
    try:
        worker.submit_frame(np.zeros((64, 64, 3), np.uint8), 1)
        assert wait_for(lambda: worker.status == "error: exited with 1")
        assert worker.version == 0
    finally:
        worker.close()


def test_backend_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", _which_none)
    worker, status = build_ocr_worker(make_ocr_cfg())
    assert worker is None
    assert status == "tesseract binary not found on PATH"
    before = threading.active_count()
    machine = TransitionDetector(make_config(make_ocr_cfg()))
    assert machine.ocr_status == "tesseract binary not found on PATH"
    assert machine.ocr_worker_status is None
    frame = np.zeros((64, 64, 3), np.uint8)
    for i in range(1, 20):
        machine.check_ocr_resync(frame, i)
    assert machine.ocr_text is None
    assert threading.active_count() == before
    machine.shutdown_ocr()


def test_tesserocr_backend_unavailable_for_now() -> None:
    cfg = make_ocr_cfg()
    cfg = OcrConfig(roi=cfg.roi, engine=OcrEngine(backend="tesserocr"), patterns=cfg.patterns)
    worker, status = build_ocr_worker(cfg)
    assert worker is None and status == "tesserocr backend not implemented"


def test_disabled_builds_no_worker() -> None:
    cfg = OcrConfig(roi=ROI, enabled=False)
    worker, _status = build_ocr_worker(cfg)
    assert worker is None
    machine = TransitionDetector(make_config(cfg))
    assert machine.ocr_worker_status is None


def test_close_while_reader_blocked_kills_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    from autosplitter.detection import element_reader as er

    block = threading.Event()
    killed: list[bool] = []

    class Proc:
        returncode: int = 0

        def communicate(self, *_a: object, **_k: object) -> tuple[bytes, bytes]:
            _ = block.wait(30.0)
            return b"", b""

        def kill(self) -> None:
            killed.append(True)
            block.set()

    monkeypatch.setattr(subprocess, "Popen", _popen_returning(Proc()))
    reader = er.TesseractCliReader(timeout_seconds=30.0)
    worker = make_worker(reader)
    worker.start()
    worker.submit_frame(np.zeros((64, 64, 3), np.uint8), 1)
    assert wait_for(lambda: reader._proc is not None)  # pyright: ignore[reportPrivateUsage]
    t0 = time.monotonic()
    worker.close()
    assert time.monotonic() - t0 < 3.0
    assert killed
    assert not worker.alive


def test_take_result_counts_sticky_result_once() -> None:
    worker = make_worker(FakeReader("5"))
    worker.start()
    try:
        worker.submit_frame(np.zeros((64, 64, 3), np.uint8), 1)
        assert wait_for(lambda: worker.version == 1)
        assert worker.take_result() is not None
        assert worker.take_result() is None
        assert worker.poll_result() is not None  # sticky for HUD
        assert worker.poll_result() is not None
    finally:
        worker.close()


def test_submit_only_on_poll_frames() -> None:
    reader = FakeReader()
    worker = make_worker(reader, poll=5)
    worker.start()
    try:
        frame = np.zeros((64, 64, 3), np.uint8)
        for i in (1, 2, 3, 4):
            worker.submit_frame(frame, i)
        assert worker._pending is None  # pyright: ignore[reportPrivateUsage]
        worker.submit_frame(frame, 5)
        assert wait_for(lambda: worker.version == 1)
    finally:
        worker.close()


def test_does_not_keep_frame_reference() -> None:
    worker = make_worker(FakeReader())
    frame = np.zeros((64, 64, 3), np.uint8)
    worker.submit_frame(frame, 1)
    assert worker._pending is not None  # pyright: ignore[reportPrivateUsage]
    assert not np.shares_memory(worker._pending, frame)  # pyright: ignore[reportPrivateUsage]
    worker.close()


def test_default_reader_is_cli_for_auto() -> None:
    reader = od.make_reader(OcrEngine())
    assert reader is not None and reader.name == "tesseract_cli"
    assert od.make_reader(OcrEngine(backend="tesserocr")) is None


def test_close_between_popen_and_assignment_kills_child(monkeypatch: pytest.MonkeyPatch) -> None:
    from autosplitter.detection import element_reader as er

    killed: list[bool] = []
    holder: dict[str, er.TesseractCliReader] = {}

    class Proc:
        returncode: int = -9

        def communicate(self, *_a: object, **_k: object) -> tuple[bytes, bytes]:
            return b"", b""

        def kill(self) -> None:
            killed.append(True)

    def popen(*_a: object, **_k: object) -> Proc:
        holder["r"].close()  # close() lands inside the Popen -> assignment window
        return Proc()

    monkeypatch.setattr(subprocess, "Popen", popen)
    reader = er.TesseractCliReader()
    holder["r"] = reader
    assert reader.read(np.zeros((8, 8), np.uint8)) is None
    assert killed


def test_evidence_ttl_derived_from_max_age() -> None:
    cfg = make_config(make_ocr_cfg(max_age=0.5))
    m = TransitionDetector(cfg, ocr=FakeReader())  # pyright: ignore[reportArgumentType]
    assert m._evidence_ttl_frames == 50  # pyright: ignore[reportPrivateUsage]  # ceil(0.5 * fps 100)


def test_live_setup_failure_does_not_leak_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autosplitter import main as mainmod

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("setup failed")

    monkeypatch.setattr(mainmod, "LiveSplitClient", boom)
    ts = tmp_path / "x.ts"
    ts.write_bytes(b"")
    cfg_text = (
        "capture: {fps: 60, stream_url: 'udp://127.0.0.1:9733', width: 64, height: 64}\n"
        + "roi: {x: 0, y: 0, width: 1, height: 1}\n"
        + "detection: {required_frames: 3, reset_threshold: 0.8}\n"
        + "colors: {black: {rgb: [0, 0, 0], tolerance: 15}}\n"
        + "splits:\n  - {id: 1, name: S, color: black, threshold: 0.9}\n"
        + "livesplit: {host: '127.0.0.1', port: 16834}\n"
        + "ocr:\n  roi: {x: 0, y: 0, width: 0.5, height: 0.5}\n"
    )
    p = tmp_path / "c.yaml"
    p.write_text(cfg_text, encoding="utf-8")
    before = [t for t in threading.enumerate() if t.name == "ocr-worker"]
    with pytest.raises(RuntimeError):
        _ = mainmod.main(["--config", str(p), "--live", str(ts), "--width", "64", "--height", "64", "--livesplit"])
    after = [t for t in threading.enumerate() if t.name == "ocr-worker" and t.is_alive()]
    assert len(after) == len(before)


def test_rate_cap_respected() -> None:
    reader = FakeReader("1")
    det = OcrDetector(ROI, OcrEngine(), (OcrPattern(regex="(\\d+)"),), reader=reader)
    worker = OcrWorker(det, 1, reader=reader, params=RAW, max_reads_per_second=10.0)
    worker.start()
    frame = np.zeros((64, 64, 3), np.uint8)
    try:
        t0 = time.monotonic()
        i = 0
        while time.monotonic() - t0 < 1.0:
            i += 1
            worker.submit_frame(frame, i)
            time.sleep(0.001)
        # 10/s for ~1 s: first read immediate -> at most ~11-12, far below uncapped (~1000)
        assert 5 <= reader.calls <= 13
    finally:
        worker.close()


def test_uncapped_by_default_for_legacy_constructor() -> None:
    reader = FakeReader("1")
    worker = make_worker(reader)
    worker.start()
    frame = np.zeros((64, 64, 3), np.uint8)
    try:
        for i in range(1, 60):
            worker.submit_frame(frame, i)
            time.sleep(0.005)
        assert reader.calls > 15
    finally:
        worker.close()


def test_close_prompt_while_rate_waiting() -> None:
    reader = FakeReader("1")
    det = OcrDetector(ROI, OcrEngine(), (OcrPattern(regex="(\\d+)"),), reader=reader)
    worker = OcrWorker(det, 1, reader=reader, params=RAW, max_reads_per_second=0.2)  # 5 s interval
    worker.start()
    worker.submit_frame(np.zeros((64, 64, 3), np.uint8), 1)
    assert wait_for(lambda: worker.version == 1)
    time.sleep(0.05)  # now inside the rate wait
    t0 = time.monotonic()
    worker.close()
    assert time.monotonic() - t0 < 1.0
    assert not worker.alive


def test_late_results_consumed_with_rate_cap() -> None:
    reader = FakeReader("2")
    reader.delay = 0.05
    det = OcrDetector(ROI, OcrEngine(), (OcrPattern(regex="(\\d+)"),), reader=reader)
    worker = OcrWorker(det, 1, reader=reader, params=RAW, max_reads_per_second=20.0)
    machine = TransitionDetector(make_config(make_ocr_cfg(poll=1, max_age=30.0)), ocr=worker)
    worker.start()
    frame = np.zeros((64, 64, 3), np.uint8)
    try:
        saw = False
        for i in range(1, 2000):
            machine.check_ocr_resync(frame, i)
            time.sleep(0.002)
            if machine.ocr_evidence.kind is EvidenceKind.MISMATCH:
                saw = True
                break
        assert saw
    finally:
        worker.close()


def test_build_worker_passes_rate_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", _which_true)
    base = make_ocr_cfg()
    cfg = OcrConfig(roi=base.roi, patterns=base.patterns, max_reads_per_second=8.0)
    worker, _status = build_ocr_worker(cfg)
    if worker is not None:  # `true --version` succeeds on most systems
        assert abs(worker._min_interval - 0.125) < 1e-9  # pyright: ignore[reportPrivateUsage]
