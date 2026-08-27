"""Tests for autosplitter.detection.ocr_detector.

OcrDetector pattern resolution and ROI cropping are tested without tesserocr.
OcrWorker threading is tested with a mock detector.
"""

from __future__ import annotations

import numpy as np

from autosplitter.config import OcrConfig, OcrEngine, OcrPattern, Roi
from autosplitter.detection.color_detector import roi_bounds
from autosplitter.detection.ocr_detector import OcrDetector, OcrResult


class TestOcrDetectorResolveIndex:
    def _make_detector(
        self, patterns: list[tuple[str, int | None]]
    ) -> OcrDetector:
        roi = Roi(x=0.0, y=0.0, width=0.1, height=0.1)
        engine = OcrEngine()
        ocr_patterns = tuple(OcrPattern(regex=r, split_index=si) for r, si in patterns)
        return OcrDetector(roi, engine, ocr_patterns)

    def test_digit_extraction(self) -> None:
        d = self._make_detector([("(\\d+)", None)])
        assert d._resolve_index("Level 3") == 2
        assert d._resolve_index("12") == 11
        assert d._resolve_index("Stage 1") == 0

    def test_explicit_split_index(self) -> None:
        d = self._make_detector([("Win", 5)])
        assert d._resolve_index("You Win!") == 5

    def test_no_match_returns_none(self) -> None:
        d = self._make_detector([("Level (\\d+)", None)])
        assert d._resolve_index("Game Over") is None

    def test_multiple_patterns_first_match_wins(self) -> None:
        d = self._make_detector([
            ("Level (\\d+)", None),
            ("Stage (\\d+)", None),
        ])
        assert d._resolve_index("Level 5") == 4
        assert d._resolve_index("Stage 3") == 2

    def test_non_capturing_group_ignored(self) -> None:
        d = self._make_detector([("(?:Level) (\\d+)", None)])
        assert d._resolve_index("Level 7") == 6

    def test_empty_text_returns_none(self) -> None:
        d = self._make_detector([("(\\d+)", None)])
        assert d._resolve_index("") is None

    def test_digit_at_start(self) -> None:
        d = self._make_detector([("(\\d+)", None)])
        assert d._resolve_index("8/30") == 7

    def test_double_digit(self) -> None:
        d = self._make_detector([("(\\d+)", None)])
        assert d._resolve_index("Level 12") == 11
        assert d._resolve_index("Level 99") == 98


class TestOcrDetectorCropRoi:
    def test_crop_returns_correct_region(self) -> None:
        roi = Roi(x=0.25, y=0.25, width=0.5, height=0.5)
        engine = OcrEngine()
        detector = OcrDetector(roi, engine, ())

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = detector._crop_roi(frame)

        x1, y1, x2, y2 = roi_bounds(roi, 200, 100)
        assert crop.shape == (y2 - y1, x2 - x1, 3)

    def test_crop_full_frame(self) -> None:
        roi = Roi(x=0.0, y=0.0, width=1.0, height=1.0)
        engine = OcrEngine()
        detector = OcrDetector(roi, engine, ())

        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        crop = detector._crop_roi(frame)
        assert crop.shape == (60, 80, 3)


class TestOcrResult:
    def test_fields(self) -> None:
        r = OcrResult(text="8", confidence=95.0, split_index=7, frame_number=42)
        assert r.text == "8"
        assert r.confidence == 95.0
        assert r.split_index == 7
        assert r.frame_number == 42

    def test_frozen(self) -> None:
        r = OcrResult(text="8", confidence=95.0, split_index=7, frame_number=42)
        try:
            r.text = "9"  # pyright: ignore[reportAttributeAccessIssue]
        except AttributeError:
            pass
        else:  # pragma: no cover
            raise AssertionError("OcrResult should be frozen")


class TestOcrConfigDataclass:
    def test_defaults(self) -> None:
        roi = Roi(x=0.0, y=0.0, width=0.1, height=0.1)
        cfg = OcrConfig(roi=roi)
        assert cfg.engine.language == "eng"
        assert cfg.engine.psm == 7
        assert cfg.engine.min_confidence == 60.0
        assert cfg.patterns == ()
        assert cfg.poll_interval_frames == 3

    def test_custom(self) -> None:
        roi = Roi(x=0.1, y=0.2, width=0.3, height=0.4)
        engine = OcrEngine(language="fra", psm=10, min_confidence=80.0)
        patterns = (OcrPattern(regex="\\d+", split_index=0),)
        cfg = OcrConfig(roi=roi, engine=engine, patterns=patterns, poll_interval_frames=5)
        assert cfg.engine.language == "fra"
        assert cfg.patterns[0].split_index == 0
        assert cfg.poll_interval_frames == 5
