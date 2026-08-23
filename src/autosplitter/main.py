"""Command-line entry point.

Phase 1: configuration validation (--check).
Phase 2/3: color detection over static images (--detect IMAGE ...).
Phase 4: live capture over MPEG-TS (--live [SOURCE]).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from .capture import MpegTsFrameSource
from .config import DEFAULT_ROI, Config, ConfigError, load
from .detection import DetectionResult, TransitionDetector, build_detector


def _print_summary(label: str, cfg: Config) -> None:
    print(f"Config OK: {label}")
    capture = cfg.capture
    print(
        f"  capture  : {capture.fps:g} fps @ {capture.width}x{capture.height}"
        + f" via {capture.stream_url}"
    )
    roi = cfg.roi
    default_note = " (default: middle 50%)" if roi == DEFAULT_ROI else ""
    print(f"  ROI      : x={roi.x:g} y={roi.y:g} w={roi.width:g} h={roi.height:g}{default_note}")
    detection = cfg.detection
    reset_pct = detection.reset_threshold * 100
    print(
        f"  detection: {detection.required_frames} frame(s) debounce, "
        + f"reset below {reset_pct:.1f}%"
    )
    colors = " | ".join(
        f"{c.name} rgb({','.join(str(v) for v in c.rgb)}) tol={c.tolerance:g}"
        for c in cfg.colors.values()
    )
    print(f"  colors   : {colors}")
    print(f"  splits   : {len(cfg.splits)}")
    for split in cfg.splits:
        threshold_pct = split.threshold * 100
        print(
            f"    {split.id}. {split.name}  color={split.color} threshold={threshold_pct:.1f}%"
        )
    print(f"  livesplit: {cfg.livesplit.host}:{cfg.livesplit.port}")


def _load_image(image_path: str) -> np.ndarray | None:
    data = np.fromfile(Path(image_path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _frame_ratio(cfg: Config, machine: TransitionDetector, frame: np.ndarray) -> float:
    """Match ratio of the currently active split against one frame."""
    split = machine.current_split
    if split is None:
        return 0.0
    return build_detector(cfg, split).detect(frame).ratio


def _detect_images(cfg: Config, image_paths: list[str]) -> int:
    """Feed a sequence of prerecorded frames through the state machine and
    print the per-frame timeline (Phase 3 demo)."""
    machine = TransitionDetector(cfg)
    name_width = max((len(Path(p).name) for p in image_paths), default=5)

    print(f"Frames: {len(image_paths)}  (required_frames={cfg.detection.required_frames}, "
          + f"reset_threshold={cfg.detection.reset_threshold:.2f})")
    print(f"{'#':>4}  {'Image':<{name_width}}  Match     State               Event")
    for index, path in enumerate(image_paths, start=1):
        img = _load_image(path)
        if img is None:
            print(f"INVALID IMAGE: {path}", file=sys.stderr)
            return 1

        ratio = _frame_ratio(cfg, machine, img)
        event = machine.feed_result(res_from_ratio(ratio))
        event_text = f"SPLIT id={event.split_id} \"{event.name}\"" if event else "-"
        print(
            f"{index:>4}  {Path(path).name:<{name_width}}  {ratio * 100:6.2f}%   "
            + f"{machine.state.name:<19} {event_text}"
        )
    return 0


def res_from_ratio(ratio: float) -> DetectionResult:
    return DetectionResult(matched_pixels=round(ratio * 100), total_pixels=100, ratio=ratio)


def _run_live(
    cfg: Config,
    source_override: str | None,
    max_frames: int | None,
    width_override: int | None,
    height_override: int | None,
) -> int:
    """Decode the sidecar stream and feed frames through the state machine,
    printing a rolling status line (Phase 4 demo)."""
    url = source_override or cfg.capture.stream_url
    width = width_override or cfg.capture.width
    height = height_override or cfg.capture.height
    try:
        source = MpegTsFrameSource(url, width, height)
    except RuntimeError as exc:
        print(f"CAPTURE ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Live from {url} @ {width}x{height}"
        + f" (required_frames={cfg.detection.required_frames},"
        + f" reset_threshold={cfg.detection.reset_threshold:.2f})"
    )
    active = cfg.splits[0]
    print(f"Waiting for: {active.id}. \"{active.name}\" (threshold {active.threshold * 100:.1f}%)")

    machine = TransitionDetector(cfg)
    frame_index = 0
    stamps: deque[float] = deque(maxlen=60)
    unavailable = False

    try:
        while max_frames is None or frame_index < max_frames:
            if machine.finished:
                print("\nDONE: every configured split has fired.")
                break

            frame = source.read()
            if frame is None:
                if source.exhausted:
                    print("\nSource exhausted (EOF).")
                    break
                if not unavailable:
                    print("\ncapture unavailable, reconnecting...", flush=True)
                    unavailable = True
                continue

            now = time.monotonic()
            stamps.append(now)
            frame_index += 1
            unavailable = False

            ratio = _frame_ratio(cfg, machine, frame)
            event = machine.feed_result(res_from_ratio(ratio))

            fps = 0.0
            if len(stamps) >= 2 and stamps[-1] > stamps[0]:
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            line = (
                f"\r#{frame_index:<6} {fps:5.1f} fps  match {ratio * 100:6.2f}%"
                + f"  {machine.state.name:<17}"
            )
            if event is not None:
                print(line, end="", flush=True)
                print(
                    f"\nSPLIT fired: id={event.split_id} \"{event.name}\""
                    + f" at frame {frame_index}"
                )
            else:
                print(line, end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        source.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autosplitter",
        description="Roblox visual autosplitter: OBS capture -> color transition detection -> LiveSplit splits",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the configuration and print a summary",
    )
    parser.add_argument(
        "--detect",
        metavar="IMAGE",
        nargs="+",
        help="feed one or more images through the split state machine and print the timeline",
    )
    parser.add_argument(
        "--live",
        metavar="SOURCE",
        nargs="?",
        const="",
        default=None,
        help="decode the sidecar MPEG-TS stream live (optionally overridden SOURCE, e.g. a .ts file)",
    )
    parser.add_argument(
        "--max-frames",
        metavar="N",
        type=int,
        default=None,
        help="with --live: stop after N frames instead of running until Ctrl-C or DONE",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="with --live: frame width override (default: capture.width from config)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="with --live: frame height override (default: capture.height from config)",
    )
    args = parser.parse_args(argv)

    # argparse.Namespace attributes are untyped; validate/cast once, here.
    config_path = cast(str, args.config)
    check_enabled = cast(bool, args.check)
    detect_paths = cast("list[str] | None", args.detect)
    live_arg = cast("str | None", args.live)
    max_frames = cast("int | None", args.max_frames)
    width_override = cast("int | None", args.width)
    height_override = cast("int | None", args.height)

    for name, value in (("width", width_override), ("height", height_override)):
        if value is not None and value <= 0:
            parser.error(f"--{name} must be a positive integer")

    if not check_enabled and not detect_paths and live_arg is None:
        parser.print_help()
        return 0

    try:
        cfg = load(config_path)
    except ConfigError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    exit_code = 0
    if check_enabled:
        _print_summary(config_path, cfg)
    if detect_paths:
        exit_code = _detect_images(cfg, detect_paths) or exit_code
    if live_arg is not None:
        source_override = live_arg or None
        exit_code = _run_live(cfg, source_override, max_frames, width_override, height_override) or exit_code
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
