"""Diagnostics overlay (spec §14): draws ROI rectangle + HUD on frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from autosplitter.config import Config
from autosplitter.detection.state_machine import TransitionDetector


@dataclass(frozen=True)
class OverlayConfig:
    """Derived overlay parameters from Config."""
    roi_px: tuple[int, int, int, int]  # x, y, w, h in pixels
    font: int
    font_scale: float
    font_thickness: int
    text_color: tuple[int, int, int]
    outline_color: tuple[int, int, int]
    line_height: int
    margin: int


def _derive_overlay_cfg(frame: np.ndarray, cfg: Config) -> OverlayConfig:
    h, w = frame.shape[:2]
    roi = cfg.roi
    roi_x = int(roi.x * w)
    roi_y = int(roi.y * h)
    roi_w = int(roi.width * w)
    roi_h = int(roi.height * h)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)
    outline_color = (0, 0, 0)
    (_, text_h), _ = cv2.getTextSize("Mg", font, font_scale, font_thickness)
    line_height = text_h + 4
    margin = 8

    return OverlayConfig(
        roi_px=(roi_x, roi_y, roi_w, roi_h),
        font=font,
        font_scale=font_scale,
        font_thickness=font_thickness,
        text_color=text_color,
        outline_color=outline_color,
        line_height=line_height,
        margin=margin,
    )


def _draw_text_with_outline(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    ocfg: OverlayConfig,
) -> None:
    cv2.putText(
        img,
        text,
        org,
        ocfg.font,
        ocfg.font_scale,
        ocfg.outline_color,
        ocfg.font_thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        org,
        ocfg.font,
        ocfg.font_scale,
        ocfg.text_color,
        ocfg.font_thickness,
        cv2.LINE_AA,
    )


def _get_active_split_info(machine: TransitionDetector, cfg: Config) -> tuple[str, float, str]:
    """Return (split_name, threshold_pct, color_name) for HUD."""
    if machine.finished:
        if cfg.splits:
            first = cfg.splits[0]
            color_name = cfg.colors[first.color].name
            return f"{first.name} (next)", first.threshold * 100, color_name
        return "—", 0.0, "—"
    split = machine.current_split
    if split is None:
        return "—", 0.0, "—"
    color_name = cfg.colors[split.color].name
    return split.name, split.threshold * 100, color_name


def annotate(
    frame: np.ndarray,
    cfg: Config,
    machine: TransitionDetector,
    ls_connected: bool,
    last_split_age: float | None,
    fps: float,
    match_ratio: float,
) -> np.ndarray:
    """Draw ROI rectangle + HUD on a copy of the frame.

    Args:
        frame: BGR uint8 (H, W, 3) — caller should pass a copy if original needed.
        cfg: Full configuration.
        machine: Current state machine.
        ls_connected: LiveSplit connection status.
        last_split_age: Seconds since last SplitEvent, or None.
        fps: Current rolling fps.
        match_ratio: Current match ratio (0.0-1.0) for active split.

    Returns:
        Annotated frame (same array, modified in place).
    """
    ocfg = _derive_overlay_cfg(frame, cfg)

    # ROI rectangle
    x, y, w, h = ocfg.roi_px
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # HUD lines
    split_name, threshold_pct, color_name = _get_active_split_info(machine, cfg)

    if machine.finished:
        state_str = "DONE"
    else:
        state_str = machine.state.name

    ls_str = "CONNECTED" if ls_connected else "DISCONNECTED"
    last_str = f"{last_split_age:.1f}s" if last_split_age is not None else "—"
    split_idx = machine.current_split_index + 1
    total_splits = len(cfg.splits)
    split_str = f"{split_idx}/{total_splits}" if not machine.finished else "—"

    lines = [
        f"fps: {fps:.1f}",
        f"match: {match_ratio * 100:.2f}%",
        f"target: {split_name} ({color_name})",
        f"threshold: {threshold_pct:.1f}%",
        f"state: {state_str}",
        f"split: {split_str}",
        f"last: {last_str}",
        f"ls: {ls_str}",
    ]

    x0 = ocfg.margin
    y0 = ocfg.margin + ocfg.line_height
    for i, line in enumerate(lines):
        org = (x0, y0 + i * ocfg.line_height)
        _draw_text_with_outline(frame, line, org, ocfg)

    return frame