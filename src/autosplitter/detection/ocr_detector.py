"""OCR-based split number reading (verify-only screen evidence).

Reads text from a separate ROI of the game frame. Runs on a background
thread so the main detection loop never blocks on OCR. The capture thread
copies the (tiny) ROI crop into a private slot buffer; the worker never sees
the reused frame buffer. The default backend shells out to the system
``tesseract`` binary (``pacman -S tesseract tesseract-data-eng``).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import cast

import numpy as np

from autosplitter.config import (
    OcrConfig,
    OcrEngine,
    OcrPattern,
    OcrPreprocess,
    Roi,
)
from autosplitter.detection.color_detector import roi_bounds
from autosplitter.detection.element_reader import (
    ElementReader,
    PreprocessParams,
    TesseractCliReader,
    preprocess,
    text_bounds,
)
from autosplitter.detection.evidence import compile_patterns, resolve_index

log = logging.getLogger("autosplitter.ocr")

FAILED_AFTER_ERRORS = 5


@dataclass(frozen=True)
class OcrResult:
    """One OCR reading from a frame."""

    text: str
    confidence: float
    split_index: int | None
    frame_number: int
    epoch: int = 0
    t_mono: float = 0.0


def make_reader(engine: OcrEngine) -> ElementReader | None:
    """Reader for ``engine.backend`` (None when the backend is not implemented)."""
    if engine.backend in ("auto", "tesseract_cli"):
        return TesseractCliReader(
            language=engine.language,
            psm=engine.psm,
            whitelist=engine.whitelist,
            timeout_seconds=engine.timeout_seconds,
        )
    return None


def params_from(pre: OcrPreprocess) -> PreprocessParams:
    lo: tuple[int, int, int] | None = None
    hi: tuple[int, int, int] | None = None
    if pre.threshold == "color" and pre.text_color is not None:
        lo, hi = text_bounds(pre.text_color, pre.text_tolerance)
    return PreprocessParams(
        scale=pre.scale, blur=pre.blur, threshold=pre.threshold, pad=pre.pad,
        text_lo=lo, text_hi=hi,
    )


class OcrDetector:
    """Crops an ROI, preprocesses it, reads it and resolves a split index."""

    def __init__(
        self,
        roi: Roi,
        engine: OcrEngine,
        patterns: tuple[OcrPattern, ...],
        preprocess_cfg: OcrPreprocess | None = None,
        reader: ElementReader | None = None,
    ) -> None:
        self._roi: Roi = roi
        self._min_confidence: float = engine.min_confidence
        self._patterns: list[tuple[re.Pattern[str], int | None]] = compile_patterns(patterns)
        self.params: PreprocessParams = params_from(preprocess_cfg or OcrPreprocess())
        self.reader: ElementReader | None = reader if reader is not None else make_reader(engine)

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        return self.crop_roi(frame)

    def crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """Crop the OCR ROI from a BGR frame (a view)."""
        h, w = cast(tuple[int, int], frame.shape[:2])
        x1, y1, x2, y2 = roi_bounds(self._roi, w, h)
        return frame[y1:y2, x1:x2]

    def _resolve_index(self, text: str) -> int | None:
        return self.resolve(text)

    def resolve(self, text: str) -> int | None:
        """Match OCR text against patterns and return a 0-based split index."""
        return resolve_index(self._patterns, text)

    def detect(self, frame: np.ndarray, frame_number: int = 0) -> OcrResult | None:
        """Run OCR synchronously (tests/GUI). None if unavailable or low confidence."""
        reader = self.reader
        if reader is None:
            return None
        crop = self._crop_roi(frame)
        if crop.size == 0:
            return None
        read = reader.read(preprocess(crop, self.params))
        if read is None:
            return None
        text, confidence = read
        if not confidence >= self._min_confidence:
            return None
        return OcrResult(
            text=text,
            confidence=confidence,
            split_index=self._resolve_index(text),
            frame_number=frame_number,
        )


class OcrWorker:
    """Background thread that reads the OCR ROI every ``poll_interval_frames``.

    The capture thread calls ``submit_frame`` per frame (modulo check only on
    idle frames; otherwise a small ROI copy under the lock). The worker waits
    on an Event, swaps buffers under the lock and runs preprocess + reader
    outside it. ``poll_result`` is sticky (HUD text); ``take_result`` yields
    each distinct reading once.
    """

    def __init__(
        self,
        detector: OcrDetector,
        poll_interval_frames: int,
        reader: ElementReader | None = None,
        params: PreprocessParams | None = None,
        max_reads_per_second: float = 0.0,
    ) -> None:
        # 0 = unlimited (legacy constructor); build_ocr_worker passes the config cap.
        self._min_interval: float = 1.0 / max_reads_per_second if max_reads_per_second > 0 else 0.0
        self._detector: OcrDetector = detector
        self._poll_interval: int = poll_interval_frames
        self._reader: ElementReader | None = reader if reader is not None else detector.reader
        self._params: PreprocessParams = params if params is not None else detector.params
        self._lock: threading.Lock = threading.Lock()
        self._wake: threading.Event = threading.Event()
        self._stop: threading.Event = threading.Event()
        self._pending: np.ndarray | None = None
        self._spare: np.ndarray | None = None
        self._pending_meta: tuple[int, int, float] | None = None
        self._latest: OcrResult | None = None
        self._version: int = 0
        self._taken_version: int = 0
        self._errors: int = 0
        self._last_error: str = ""
        self._logged: set[str] = set()
        self._thread: threading.Thread = threading.Thread(
            target=self._run, daemon=True, name="ocr-worker"
        )

    @property
    def version(self) -> int:
        """Bumped on every stored reading (unlocked int read)."""
        return self._version

    @property
    def taken_version(self) -> int:
        """Version handed out by the last ``take_result`` (single consumer)."""
        return self._taken_version

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def status(self) -> str:
        """'ok' | 'error: <msg>' | 'failed' (5+ consecutive errors)."""
        errors = self._errors
        if errors >= FAILED_AFTER_ERRORS:
            return "failed"
        if errors > 0:
            return f"error: {self._last_error}"
        return "ok"

    def start(self) -> None:
        if not self._thread.is_alive() and not self._stop.is_set():
            self._thread.start()

    def submit_frame(self, frame: np.ndarray, frame_idx: int, epoch: int = 0) -> None:
        """Copy the ROI crop into a private buffer; never retains ``frame``."""
        if frame_idx % self._poll_interval != 0:
            return
        crop = self._detector.crop_roi(frame)
        if crop.size == 0:
            return
        now = time.monotonic()
        with self._lock:
            buf = self._pending
            if buf is None or buf.shape != crop.shape or buf.dtype != crop.dtype:
                buf = np.empty_like(crop)
                self._pending = buf
            np.copyto(buf, crop)
            self._pending_meta = (frame_idx, epoch, now)
        self._wake.set()

    def poll_result(self) -> OcrResult | None:
        """Latest reading (sticky; for HUD text)."""
        with self._lock:
            return self._latest

    def take_result(self) -> OcrResult | None:
        """The latest reading only if it is new since the last take."""
        with self._lock:
            if self._version == self._taken_version:
                return None
            self._taken_version = self._version
            return self._latest

    def _note_error(self, msg: str) -> None:
        if self._stop.is_set():
            return  # shutdown killed the child; not a real error
        self._errors += 1
        self._last_error = msg
        if msg not in self._logged:
            self._logged.add(msg)
            log.warning("ocr read error: %s", msg)

    def _process(self, buf: np.ndarray, meta: tuple[int, int, float]) -> None:
        reader = self._reader
        if reader is None:
            self._note_error("no reader")
            return
        idx, epoch, t_mono = meta
        try:
            read = reader.read(preprocess(buf, self._params))
        except Exception as exc:  # noqa: BLE001 - the thread must survive any reader failure
            self._note_error(f"{type(exc).__name__}: {exc}")
            return
        err = cast(str | None, getattr(reader, "last_error", None))
        if read is None and err:
            self._note_error(err)
            return
        self._errors = 0
        text, conf = read if read is not None else ("", 0.0)
        result = OcrResult(
            text=text,
            confidence=conf,
            split_index=self._detector.resolve(text),
            frame_number=idx,
            epoch=epoch,
            t_mono=t_mono,
        )
        with self._lock:
            self._latest = result
            self._version += 1

    def _run(self) -> None:
        while True:
            _ = self._wake.wait()
            if self._stop.is_set():
                return
            self._wake.clear()
            with self._lock:
                buf = self._pending
                meta = self._pending_meta
                self._pending = self._spare
                self._spare = None
                self._pending_meta = None
            if buf is None or meta is None:
                continue
            started = time.monotonic()
            try:
                self._process(buf, meta)
            except Exception as exc:  # noqa: BLE001 - never let the thread die
                self._note_error(f"{type(exc).__name__}: {exc}")
            with self._lock:
                self._spare = buf
            if self._min_interval > 0.0:
                # Rate cap: wait out the rest of the interval; the stop Event
                # keeps close() prompt. Frames submitted meanwhile are
                # latest-wins in the slot.
                remaining = started + self._min_interval - time.monotonic()
                if remaining > 0.0 and self._stop.wait(remaining):
                    return

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        closer = getattr(self._reader, "close", None)
        if callable(closer):
            try:
                _ = closer()
            except Exception as exc:  # noqa: BLE001
                log.debug("ocr reader close failed: %s", exc)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def describe_backend(cfg: OcrConfig) -> tuple[bool, str]:
    """(usable, text) for ``--check``: 'tesseract_cli OK (5.5.4)' or a reason."""
    if cfg.engine.backend == "tesserocr":
        return False, "tesserocr backend not implemented"
    reader = make_reader(cfg.engine)
    if reader is None:  # pragma: no cover - guarded by config validation
        return False, f"unknown backend {cfg.engine.backend}"
    reason = reader.available()
    if reason is not None:
        return False, reason
    version = cast(str, getattr(reader, "version", ""))
    return True, f"{reader.name} OK ({version})"


def build_ocr_worker(cfg_ocr: OcrConfig) -> tuple[OcrWorker | None, str]:
    """(worker or None, status). The worker is returned unstarted."""
    if not cfg_ocr.enabled:
        return None, "ocr disabled in config"
    if cfg_ocr.engine.backend == "tesserocr":
        return None, "tesserocr backend not implemented"
    reader = make_reader(cfg_ocr.engine)
    if reader is None:  # pragma: no cover
        return None, f"unknown backend {cfg_ocr.engine.backend}"
    reason = reader.available()
    if reason is not None:
        return None, reason
    detector = OcrDetector(
        cfg_ocr.roi, cfg_ocr.engine, cfg_ocr.patterns, cfg_ocr.preprocess, reader
    )
    worker = OcrWorker(
        detector,
        cfg_ocr.poll_interval_frames,
        max_reads_per_second=cfg_ocr.max_reads_per_second,
    )
    return worker, f"{reader.name} OK"
