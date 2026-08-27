"""OCR-based split number detection (augments color detection).

Reads text from a separate ROI on the game frame and resolves the current
split index via regex patterns.  Runs on a background thread to avoid
blocking the main detection loop.

Requires the system ``tesseract`` package (``pacman -S tesseract tesseract-data-eng``).
The ``tesserocr`` Python binding is imported lazily — configs without ``ocr:``
never trigger the import.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from autosplitter.config import OcrEngine, OcrPattern, Roi
from autosplitter.detection.color_detector import roi_bounds


@dataclass(frozen=True)
class OcrResult:
    """One OCR reading from a frame."""

    text: str
    confidence: float
    split_index: int | None
    frame_number: int


class OcrDetector:
    """Crops an ROI, runs Tesseract, and resolves a split index via regex."""

    def __init__(self, roi: Roi, engine: OcrEngine, patterns: tuple[OcrPattern, ...]) -> None:
        self._roi: Roi = roi
        self._language: str = engine.language
        self._psm: int = engine.psm
        self._min_confidence: float = engine.min_confidence
        self._patterns: list[tuple[re.Pattern[str], int | None]] = [
            (re.compile(p.regex), p.split_index) for p in patterns
        ]

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """Crop the OCR ROI from a BGR frame."""
        h, w = cast(tuple[int, int], frame.shape[:2])
        x1, y1, x2, y2 = roi_bounds(self._roi, w, h)
        return frame[y1:y2, x1:x2]

    def _resolve_index(self, text: str) -> int | None:
        """Match OCR text against patterns and return a 0-based split index."""
        for compiled, split_index in self._patterns:
            m = compiled.search(text)
            if m is None:
                continue
            if split_index is not None:
                return split_index
            # Extract first digit group as 1-based split id
            for group in m.groups():
                if group is not None and group.isdigit():
                    return int(group) - 1
        return None

    def detect(self, frame: np.ndarray, frame_number: int = 0) -> OcrResult | None:
        """Run OCR synchronously.  Returns None if tesserocr unavailable."""
        try:
            import tesserocr  # pyright: ignore[reportMissingImports]
        except ImportError:
            return None

        crop = self._crop_roi(frame)
        if crop.size == 0:
            return None

        # Convert BGR → grayscale for Tesseract
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # Enlarge 2x for better accuracy on small text
        enlarged = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        api = tesserocr.PyTessBaseAPI(lang=self._language)
        try:
            api.SetPageSegMode(self._psm)
            api.SetImage(enlarged.tolist(), enlarged.shape[1], enlarged.shape[0], 1, enlarged.strides[0])
            text = api.GetUTF8Text().strip()
            confidence = api.MeanTextConf()
        finally:
            api.End()

        if confidence < self._min_confidence:
            return None

        split_index = self._resolve_index(text)
        return OcrResult(
            text=text,
            confidence=confidence,
            split_index=split_index,
            frame_number=frame_number,
        )


class OcrWorker:
    """Background thread that runs OCR at a configurable interval.

    The main thread calls ``submit_frame`` each frame and ``poll_result``
    to get the latest reading.  Thread safety: only ``_latest`` is shared
    (guarded by ``_lock``).
    """

    def __init__(self, detector: OcrDetector, poll_interval_frames: int) -> None:
        self._detector: OcrDetector = detector
        self._poll_interval: int = poll_interval_frames
        self._frame: np.ndarray | None = None
        self._frame_idx: int = -1
        self._last_processed: int = -1
        self._latest: OcrResult | None = None
        self._lock: threading.Lock = threading.Lock()
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(
            target=self._run, daemon=True, name="ocr-worker"
        )

    def start(self) -> None:
        self._thread.start()

    def submit_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        """Store a reference for the worker to process."""
        with self._lock:
            self._frame = frame
            self._frame_idx = frame_idx

    def poll_result(self) -> OcrResult | None:
        """Return the latest OCR result (or None)."""
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                frame = self._frame
                idx = self._frame_idx

            if frame is not None and idx > self._last_processed and idx % self._poll_interval == 0:
                self._last_processed = idx
                result = self._detector.detect(frame, idx)
                if result is not None:
                    with self._lock:
                        self._latest = result

            time.sleep(0.001)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
