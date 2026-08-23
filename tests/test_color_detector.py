"""Tests for autosplitter.detection.color_detector (spec sections 4-6, 16).

Uses static PNG fixtures under tests/fixtures/images/ plus small synthetic
numpy frames for boundary cases.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from autosplitter.config import Color, Config, Roi, load
from autosplitter.detection import ColorDetector, build_detector, roi_bounds

FIXTURES = Path(__file__).parent / "fixtures"
IMAGES = FIXTURES / "images"

BLACK = Color(name="black", rgb=(0, 0, 0), tolerance=15.0)
WHITE = Color(name="white", rgb=(255, 255, 255), tolerance=15.0)
RED = Color(name="red", rgb=(255, 0, 0), tolerance=0.0)
NEAR_BLACK_TIGHT = Color(name="black", rgb=(0, 0, 0), tolerance=5.0)

MIDDLE_50 = Roi(x=0.25, y=0.25, width=0.50, height=0.50)


def load_png(name: str) -> np.ndarray:
    data = np.fromfile(IMAGES / name, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    assert img is not None, f"fixture {name} failed to decode"
    return img


# ---------------------------------------------------------------------------
# ROI geometry
# ---------------------------------------------------------------------------


class TestRoiBounds:
    def test_full_frame(self) -> None:
        assert roi_bounds(Roi(0.0, 0.0, 1.0, 1.0), 1920, 1080) == (0, 0, 1920, 1080)

    def test_middle_50_on_even_frame(self) -> None:
        assert roi_bounds(MIDDLE_50, 200, 200) == (50, 50, 150, 150)

    def test_rounds_to_nearest_pixel(self) -> None:
        # x2 = round(0.75 * 99) = round(74.25) = 74
        roi = Roi(1 / 3, 0.0, 1 / 3, 1.0)
        x1, _, x2, _ = roi_bounds(roi, 99, 10)
        assert (x1, x2) == (33, 66)

    def test_clamps_to_frame(self) -> None:
        roi = Roi(0.90, 0.90, 0.20, 0.20)
        assert roi_bounds(roi, 100, 100) == (90, 90, 100, 100)


# ---------------------------------------------------------------------------
# Synthetic-frame detection semantics
# ---------------------------------------------------------------------------


class TestDetectSynthetic:
    def test_tolerance_is_inclusive_at_boundary(self) -> None:
        detector = ColorDetector(MIDDLE_50, BLACK)
        at_limit = np.full((10, 10, 3), 15, dtype=np.uint8)
        over = np.full((10, 10, 3), 16, dtype=np.uint8)

        assert detector.detect(at_limit).ratio == 1.0
        assert detector.detect(over).ratio == 0.0

    def test_counts_known_mix(self) -> None:
        frame = np.full((4, 4, 3), 255, dtype=np.uint8)
        frame[0, 0] = 0
        frame[1, 1] = 0
        frame[2, 2] = 0
        frame[3, 3] = 3  # within tolerance 15 of black

        result = ColorDetector(Roi(0.0, 0.0, 1.0, 1.0), BLACK).detect(frame)

        assert (result.matched_pixels, result.total_pixels) == (4, 16)
        assert result.ratio == pytest.approx(0.25)

    def test_channel_order_is_bgr_not_rgb(self) -> None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:] = (0, 0, 255)  # pure red in BGR

        red_result = ColorDetector(Roi(0.0, 0.0, 1.0, 1.0), RED).detect(frame)
        white_result = ColorDetector(Roi(0.0, 0.0, 1.0, 1.0), WHITE).detect(frame)

        assert red_result.ratio == 1.0
        assert white_result.ratio == 0.0

    def test_accepts_alpha_channel_using_first_three(self) -> None:
        frame = np.zeros((10, 10, 4), dtype=np.uint8)
        frame[..., 3] = 255

        assert ColorDetector(MIDDLE_50, BLACK).detect(frame).ratio == 1.0

    @pytest.mark.parametrize("shape", [(4, 4), (4,), (4, 4, 2)])
    def test_rejects_invalid_frames(self, shape: tuple[int, ...]) -> None:
        frame = np.zeros(shape, dtype=np.uint8)

        with pytest.raises(ValueError, match="expected an HxWx3"):
            ColorDetector(MIDDLE_50, BLACK).detect(frame)

    def test_degenerate_roi_reports_zero(self) -> None:
        tiny = Roi(0.5, 0.5, 1e-9, 1e-9)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        result = ColorDetector(tiny, BLACK).detect(frame)

        assert (result.matched_pixels, result.total_pixels, result.ratio) == (0, 0, 0.0)


# ---------------------------------------------------------------------------
# Static-image fixtures
# ---------------------------------------------------------------------------


class TestDetectFixtures:
    def test_solid_black_is_full_match(self) -> None:
        result = ColorDetector(MIDDLE_50, BLACK).detect(load_png("solid_black.png"))

        assert result.total_pixels == 100 * 100
        assert result.ratio == 1.0

    def test_near_black_respects_tolerance(self) -> None:
        img = load_png("near_black.png")

        assert ColorDetector(MIDDLE_50, BLACK).detect(img).ratio == 1.0
        assert ColorDetector(MIDDLE_50, NEAR_BLACK_TIGHT).detect(img).ratio == 0.0

    def test_black_block_exactly_filling_default_roi(self) -> None:
        """The block covers rows/cols 50:150 of a 200x200 image — precisely
        the middle-50% ROI — so the ratio must be exactly 1.0."""
        result = ColorDetector(MIDDLE_50, BLACK).detect(load_png("roi_black_exact.png"))

        assert result.ratio == 1.0

    def test_white_block_inverts_the_result(self) -> None:
        img = load_png("roi_white_exact.png")

        assert ColorDetector(MIDDLE_50, BLACK).detect(img).ratio == 0.0
        assert ColorDetector(MIDDLE_50, WHITE).detect(img).ratio == 1.0

    def test_red_fixture_distinguishes_channels(self) -> None:
        img = load_png("solid_red.png")

        blue_probe = Color(name="blue", rgb=(0, 80, 255), tolerance=20.0)
        assert ColorDetector(Roi(0.0, 0.0, 1.0, 1.0), RED).detect(img).ratio == 1.0
        assert ColorDetector(Roi(0.0, 0.0, 1.0, 1.0), blue_probe).detect(img).ratio == 0.0


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestBuildDetector:
    def test_resolves_split_color_and_uses_configured_roi(self) -> None:
        cfg = load_config()
        split = cfg.splits[0]  # color: black, threshold: 0.95

        detector = build_detector(cfg, split)
        result = detector.detect(load_png("roi_black_exact.png"))

        assert detector.name == "black"
        assert result.ratio >= split.threshold


def load_config() -> Config:
    return load(FIXTURES / "valid_config.yaml")
