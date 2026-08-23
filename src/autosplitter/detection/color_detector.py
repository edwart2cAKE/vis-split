"""Color-percentage detection over a normalized ROI.

Implements spec sections 4-6 and 16: crop the ROI first, then run a
vectorized per-channel tolerance comparison. Frames are BGR uint8 with
shape (H, W, 3) — the OpenCV convention used across the whole pipeline.
Configuration colors are RGB (config.yaml ``rgb:``) and are converted to
BGR once at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from autosplitter.config import Color, Config, Roi, Split


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of analyzing one frame's ROI against one target color."""

    matched_pixels: int
    total_pixels: int
    ratio: float

    @property
    def percent(self) -> float:
        return self.ratio * 100.0


def roi_bounds(roi: Roi, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a normalized ROI to integer pixel bounds (x1, y1, x2, y2).

    Endpoints are rounded to nearest pixel and clamped to the frame. A
    degenerate (empty) rectangle is possible only for extreme values; the
    detector reports ratio 0.0 for it instead of failing.
    """
    x1 = min(max(round(roi.x * width), 0), width)
    y1 = min(max(round(roi.y * height), 0), height)
    x2 = min(max(round((roi.x + roi.width) * width), 0), width)
    y2 = min(max(round((roi.y + roi.height) * height), 0), height)
    return x1, y1, x2, y2


class ColorDetector:
    """Computes what fraction of ROI pixels match a configured color.

    A pixel matches when every channel is within ``color.tolerance`` of the
    target (inclusive), i.e. an axis-aligned cube around the target color in
    BGR space. Only the cropped ROI is processed; no full-frame work.
    """

    def __init__(self, roi: Roi, color: Color) -> None:
        self._roi: Roi = roi
        self._name: str = color.name
        # config stores RGB; pipeline frames are BGR.
        self._target_bgr: tuple[int, int, int] = (
            color.rgb[2],
            color.rgb[1],
            color.rgb[0],
        )
        self._tolerance: float = color.tolerance

    @property
    def name(self) -> str:
        return self._name

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(
                f"expected an HxWx3 (or HxWx4) BGR frame, got shape {frame.shape}"
            )

        # numpy's bare ndarray exposes .shape as loosely typed; pin it here.
        height, width = cast(tuple[int, int], frame.shape[:2])
        x1, y1, x2, y2 = roi_bounds(self._roi, width, height)
        crop = frame[y1:y2, x1:x2, :3]

        rows, cols = cast(tuple[int, int], crop.shape[:2])
        total = rows * cols
        if total == 0:
            return DetectionResult(matched_pixels=0, total_pixels=0, ratio=0.0)

        # uint8 arithmetic wraps on negative results; widen before subtracting.
        diff = np.abs(crop.astype(np.int16) - np.array(self._target_bgr, dtype=np.int16))
        mask = np.all(diff <= self._tolerance, axis=-1)

        matched = int(mask.sum())
        return DetectionResult(
            matched_pixels=matched,
            total_pixels=total,
            ratio=matched / total,
        )


def build_detector(config: Config, split: Split) -> ColorDetector:
    """Resolve a split's named color against the config and build its detector."""
    return ColorDetector(config.roi, config.color(split.color))
