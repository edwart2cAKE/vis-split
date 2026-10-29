"""Command-line entry point.

Phase 1: configuration validation (--check).
Phase 2/3: color detection over static images (--detect IMAGE ...).
Phase 4: live capture over MPEG-TS (--live [SOURCE]).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from .capture import MpegTsFrameSource
from .config import DEFAULT_ROI, Config, ConfigError, Record, Split, load
from .detection import DetectionResult, StartEvent, State, TransitionDetector
from .diagnostics import annotate
from .livesplit import Action, Decision, LiveSplitClient, TimerFollower, UndoWatch
from .record import resolve_path


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
    print(
        f"  livesplit: {cfg.livesplit.host}:{cfg.livesplit.port}"
        + f" (timeout {cfg.livesplit.timeout_seconds:g}s,"
        + f" reconnect {cfg.livesplit.reconnect_seconds:g}s)"
    )
    if cfg.start_transition is not None:
        st = cfg.start_transition
        roi_note = f" (roi override: x={st.roi_override.x:g} y={st.roi_override.y:g} w={st.roi_override.width:g} h={st.roi_override.height:g})" if st.roi_override else ""
        auto_start_note = " (auto-start timer)" if st.auto_start_timer else ""
        print(
            f"  start    : {st.name}  color={st.color} threshold={st.threshold * 100:.1f}%{roi_note}{auto_start_note}"
        )
    if cfg.ocr is not None:
        ocr = cfg.ocr
        print(
            f"  ocr      : roi x={ocr.roi.x:g} y={ocr.roi.y:g} w={ocr.roi.width:g} h={ocr.roi.height:g}"
            + f" engine={ocr.engine.language} psm={ocr.engine.psm}"
            + f" poll={ocr.poll_interval_frames} patterns={len(ocr.patterns)}"
        )


def _load_image(image_path: str) -> np.ndarray | None:
    data = np.fromfile(Path(image_path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _frame_ratio(machine: TransitionDetector, frame: np.ndarray) -> float:
    """Match ratio of the currently active split against one frame."""
    if machine.waiting_for_start:
        detector = machine.start_detector
    else:
        detector = machine.current_detector
    if detector is None:
        return 0.0
    return detector.detect(frame).ratio


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

        ratio = _frame_ratio(machine, img)
        event = machine.feed_result(res_from_ratio(ratio))
        if isinstance(event, StartEvent):
            event_text = f"START \"{event.name}\""
        elif event is not None:
            event_text = f"SPLIT id={event.split_id} \"{event.name}\""
        else:
            event_text = "-"
        print(
            f"{index:>4}  {Path(path).name:<{name_width}}  {ratio * 100:6.2f}%   "
            + f"{machine.state.name:<19} {event_text}"
        )
    return 0


def _probe_image(cfg: Config, image_path: str) -> int:
    """Load an image, crop to ROI, and print average RGB in the ROI for color tuning."""
    img = _load_image(image_path)
    if img is None:
        print(f"INVALID IMAGE: {image_path}", file=sys.stderr)
        return 1

    h, w = img.shape[:2]
    roi = cfg.roi
    # Parse ROI (normalized 0.0-1.0) to pixel coordinates
    x = int(roi.x * w)
    y = int(roi.y * h)
    roi_w = int(roi.width * w)
    roi_h = int(roi.height * h)

    # Clamp to image bounds
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    roi_w = max(1, min(roi_w, w - x))
    roi_h = max(1, min(roi_h, h - y))

    roi_img = img[y:y + roi_h, x:x + roi_w]
    # Image is BGR (OpenCV), convert to RGB for display
    avg_bgr = roi_img.mean(axis=(0, 1))
    avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))

    print(f"Image: {image_path} ({w}x{h})")
    print(f"ROI: x={x} y={y} w={roi_w} h={roi_h} (normalized: {roi.x:.3f}, {roi.y:.3f}, {roi.width:.3f}, {roi.height:.3f})")
    print(f"Average RGB in ROI: {avg_rgb}")
    print("Suggested config:")
    print("  colors:")
    print("    your_color_name:")
    print(f"      rgb: [{avg_rgb[0]}, {avg_rgb[1]}, {avg_rgb[2]}]")
    print("      tolerance: 25  # adjust based on color consistency")
    return 0


def res_from_ratio(ratio: float) -> DetectionResult:
    return DetectionResult(matched_pixels=round(ratio * 100), total_pixels=100, ratio=ratio)


def _convert_recording(record_path: str, convert_to: str) -> None:
    """Convert a recording to another format (best-effort, never crashes)."""
    src = Path(record_path)
    if not src.exists() or src.stat().st_size == 0:
        print(f"WARNING: recording {record_path} missing or empty, skipping conversion")
        return
    dst = src.with_suffix(f".{convert_to}")
    print(f"Converting {src.name} -> {dst.name} ...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(src), "-c", "copy", str(dst)],
            capture_output=True,
            check=False,
            timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            print(f"WARNING: conversion failed (ffmpeg exit {result.returncode}): {stderr}")
        else:
            print(f"Converted: {dst}")
    except FileNotFoundError:
        print("WARNING: ffmpeg not found, skipping conversion")
    except subprocess.TimeoutExpired:
        print("WARNING: conversion timed out after 120s")
    except OSError as exc:
        print(f"WARNING: conversion failed: {exc}")


def _reload_config(
    path: Path, machine: TransitionDetector
) -> Config | None:
    """Reload config from file and apply hot-swappable changes to machine."""
    try:
        new_cfg = load(path)
        machine.reload_config(new_cfg)
        print("\nConfig reloaded (colors/thresholds/ROI/detection)")
        return new_cfg
    except ConfigError as e:
        print(f"\nConfig reload failed: {e}")
        return None


@dataclass(frozen=True)
class _FollowPollResult:
    phase: str | None
    name: str | None
    ok: bool
    suppress: bool
    message: str | None = None
    clear_last_split: bool = False
    # False for PAUSE/RESUME: they must not start the undo hold-off.
    touch_last_split: bool = True


# Consecutive failed polls while paused before the detector is thawed.
_THAW_AFTER_FAILURES = 3


class _FollowPoller:
    """One follow_timer poll: queries, follower decision, machine update.

    Lives outside ``_run_live`` so the hot loop's control flow stays small.
    Called at most every ``livesplit.poll_seconds``, never per frame.
    """

    def __init__(self, n_splits: int, loop: bool = False, pause: bool = False) -> None:
        self._n_splits: int = n_splits
        self._loop: bool = loop
        self._follower: TimerFollower = TimerFollower(pause)
        # Plain attribute mirror of the follower's paused flag: the frame loop
        # reads it once per frame to freeze detection.
        self.paused: bool = False
        self._failures: int = 0
        self._count_checked: bool = False
        self._skip_enabled: bool = True
        self._prev_index: int | None = None

    def disarm(self) -> None:
        self._follower.disarm()
        self.paused = False
        self._failures = 0

    def connection_lost(self) -> None:
        self._count_checked = False
        self._skip_enabled = True
        # A fresh connection may be an idle LiveSplit: not a RESET.
        self._follower.disarm()
        self.paused = False
        self._failures = 0

    def _abort(
        self,
        machine: TransitionDetector,
        phase: str | None,
        cached_name: str | None,
        suppress: bool,
    ) -> _FollowPollResult:
        """Failed query: clear candidates, still apply phase-only actions.

        PAUSE/RESUME depend only on the phase, so they take effect even when
        the index query failed. After ``_THAW_AFTER_FAILURES`` consecutive
        failed polls while paused the detector is thawed (safety valve).
        """
        decision = self._clear_pending(phase, None)
        message: str | None = None
        if decision.action in (Action.PAUSE, Action.RESUME):
            message = _apply_decision(machine, decision)
            self._failures = 0
        else:
            self._failures += 1
            if self._follower.paused and self._failures >= _THAW_AFTER_FAILURES:
                self._follower.unpause()
                machine.clear_streak()
                self._failures = 0
                message = "LiveSplit: no answers while paused — detector unfrozen"
        self.paused = self._follower.paused
        return _FollowPollResult(
            phase, cached_name, False, suppress, message, touch_last_split=False
        )

    def _clear_pending(self, phase: str | None, index: int | None) -> Decision:
        """Feed a failed poll so 'consecutive polls' really are consecutive."""
        return self._follower.observe(
            phase=phase,
            ls_index=index,
            expected=0,
            n_splits=self._n_splits,
            now=time.monotonic(),
            last_split_time=None,
            suppress=True,
            can_back=False,
            can_fwd=False,
            waiting_for_start=True,
        )

    def poll(
        self,
        client: LiveSplitClient,
        machine: TransitionDetector,
        cached_name: str | None,
        last_split_time: float | None,
        suppress: bool,
    ) -> _FollowPollResult:
        phase = client.get_current_timer_phase()
        if phase is None:
            return self._abort(machine, None, cached_name, suppress)
        if phase == "NotRunning":
            self._failures = 0
            # No index/name queries, but observe still runs so RESET can fire.
            decision = self._follower.observe(
                phase=phase,
                ls_index=None,
                expected=machine.expected_ls_index,
                n_splits=self._n_splits,
                now=time.monotonic(),
                last_split_time=last_split_time,
                suppress=suppress,
                can_back=False,
                can_fwd=False,
                waiting_for_start=machine.waiting_for_start,
            )
            message = _apply_decision(
                machine, decision, allow_rewind_done=self._loop
            )
            self.paused = self._follower.paused
            if message is not None:
                return _FollowPollResult(
                    phase, cached_name, False, False, message, clear_last_split=True
                )
            return _FollowPollResult(phase, cached_name, False, suppress)
        index = client.get_split_index()
        if index is None:
            return self._abort(machine, phase, cached_name, suppress)
        self._failures = 0
        if not self._count_checked and phase in ("Running", "Paused", "Ended"):
            self._count_checked = True
            total = client.get_split_count()
            if total is None:
                print(
                    "\nLiveSplit: split count unavailable; skip-forward runs"
                    + " without validating against len(splits)"
                )
            elif total != self._n_splits:
                self._skip_enabled = False
                print(
                    f"\nWARN: LiveSplit has {total} segments but config has"
                    + f" {self._n_splits} splits; skip-forward following disabled"
                )
        expected = machine.expected_ls_index
        name = cached_name
        if index != expected or index != self._prev_index or name is None:
            name = client.get_current_split_name()
        self._prev_index = index
        if index == expected:
            suppress = False
        state = machine.state
        decision = self._follower.observe(
            phase=phase,
            ls_index=index,
            expected=expected,
            n_splits=self._n_splits,
            now=time.monotonic(),
            last_split_time=last_split_time,
            suppress=suppress,
            can_back=state in (State.NORMAL, State.WAITING_FOR_RESET, State.DONE),
            can_fwd=self._skip_enabled
            and state in (State.WAITING_FOR_START, State.NORMAL, State.WAITING_FOR_RESET),
            waiting_for_start=machine.waiting_for_start,
        )
        message = _apply_decision(machine, decision)
        self.paused = self._follower.paused
        touch = decision.action not in (Action.PAUSE, Action.RESUME)
        if message is not None and touch:
            suppress = False
        return _FollowPollResult(
            phase, name, True, suppress, message, touch_last_split=touch
        )


def _apply_decision(
    machine: TransitionDetector, decision: Decision, *, allow_rewind_done: bool = True
) -> str | None:
    """Apply a follower decision to the machine; log line if it took effect.

    ``allow_rewind_done`` lets a timer RESET rearm a finished machine (only
    meaningful with --loop; without it the run exits at DONE).
    """
    action = decision.action
    if action is Action.PAUSE:
        machine.clear_streak()
        return "LiveSplit: timer PAUSED — detector frozen"
    if action is Action.RESUME:
        machine.clear_streak()
        return "LiveSplit: timer RESUMED"
    if action is Action.RESET:
        if machine.finished and not allow_rewind_done:
            return None
        machine.reset(settle=True)
        return "LiveSplit: timer RESET — detector rewound"
    index = decision.index
    if index is None:
        return None
    if action is Action.MANUAL_START:
        if not machine.advance_to(index):
            return None
        return "LiveSplit: timer started — detector left the start gate"
    if action is Action.RUN_ENDED:
        if not machine.advance_to(index):
            return None
        return "LiveSplit: run ended — detector DONE"
    current: Split | None = None
    if action is Action.ROLLBACK:
        if not machine.rollback_to(index):
            return None
        current = machine.current_split
        verb = "UNSPLIT detected — detector rolled back to"
    elif action is Action.SKIP_FORWARD:
        if not machine.advance_to(index):
            return None
        current = machine.current_split
        verb = "SKIP/manual split detected — detector advanced to"
    else:
        return None
    name = current.name if current is not None else ""
    return f'LiveSplit: {verb} "{name}" (index {index})'


def _run_live(
    cfg: Config,
    source_override: str | None,
    max_frames: int | None,
    width_override: int | None,
    height_override: int | None,
    snap_frames: int | None,
    livesplit_enabled: bool = False,
    loop_enabled: bool = False,
    overlay_enabled: bool = False,
    record_path: str | None = None,
    record_cfg: Record | None = None,
    config_path: Path | None = None,
    config_poll_interval: float = 1.0,
) -> int:
    """Decode the sidecar stream and feed frames through the state machine,
    printing a rolling status line (Phase 4 demo).

    With ``snap_frames`` set, the Nth decoded frame is written to snap.png
    and the run ends — pairs with a local ``testsrc`` sender to measure
    end-to-end latency against its burned-in clock.

    For live (udp/tcp/srt) sources, once the rolling fps window is full it is
    compared against ``capture.fps``; a sustained deviation >15% prints a
    one-time HINT (debounce is frame-counted, so a stream/config rate
    mismatch silently changes split timing).

    With ``livesplit_enabled``, split events are forwarded to LiveSplit
    Server (splits only); connection state changes are reported inline.
    Every couple of seconds the detector also polls LiveSplit for its timer
    phase and current split name: the status line shows them, and a one-time
    WARN prints when LiveSplit's active segment disagrees with the split the
    detector expects next (only compared while LiveSplit reports Running).

    With ``loop_enabled``, DONE is no longer terminal: once the next run's
    first split color stays below ``reset_threshold`` for
    ``detection.restart_quiet_seconds`` (frame-counted at ``capture.fps``),
    the machine rearms for another pass.

    With ``overlay_enabled``, a GUI window shows the ROI rectangle and HUD
    (throttled to ~capture.fps/10 Hz). Press ESC/q or close the window to exit.

    With ``record_path`` set, ffmpeg also writes a codec-copy .mkv to that
    path while decoding frames for detection — single UDP consumer, no
    extra socket needed.
    """
    url = source_override or cfg.capture.stream_url
    width = width_override or cfg.capture.width
    height = height_override or cfg.capture.height
    live_stream = url.startswith(("udp://", "tcp://", "srt://"))
    try:
        source = MpegTsFrameSource(url, width, height, record_path=record_path)
    except RuntimeError as exc:
        print(f"CAPTURE ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Live from {url} @ {width}x{height}"
        + f" (required_frames={cfg.detection.required_frames},"
        + f" reset_threshold={cfg.detection.reset_threshold:.2f})"
    )
    if record_path is not None:
        print(f"Recording: {record_path}")
    if snap_frames is not None:
        print(f"Snap mode: saving frame {snap_frames} -> snap.png")
    if loop_enabled:
        quiet = cfg.detection.restart_quiet_seconds
        print(
            f"Loop: rearms after {quiet:g}s of quiet"
            + " (you reset your own timer between runs)"
        )
    if cfg.start_transition is not None:
        st = cfg.start_transition
        print(
            f"Waiting for start: \"{st.name}\" (color={st.color}, threshold {st.threshold * 100:.1f}%)"
        )
    else:
        active = cfg.splits[0]
        print(f"Waiting for: {active.id}. \"{active.name}\" (threshold {active.threshold * 100:.1f}%)")

    machine = TransitionDetector(cfg)
    first_detector = machine.first_detector
    quiet_needed = machine.quiet_frames_needed
    quiet_frames = 0
    frame_index = 0
    stamps: deque[float] = deque(maxlen=60)
    unavailable = False
    last_draw: float | None = None
    fps_hint_shown = False
    last_split_time: float | None = None
    _done_time: float | None = None
    _done_grace_print: float | None = None

    client: LiveSplitClient | None = None
    ls_was_connected = False
    if livesplit_enabled:
        ls = cfg.livesplit
        client = LiveSplitClient(
            ls.host,
            ls.port,
            timeout_seconds=ls.timeout_seconds,
            reconnect_seconds=ls.reconnect_seconds,
        )
        print(f"LiveSplit: {ls.host}:{ls.port} (splits only)")

    # Low-rate LiveSplit sync polling (wall-clock gated, not per-frame):
    # each query can stall up to livesplit.timeout_seconds, so poll both
    # phase and split name at most every couple of seconds.
    ls_poll_interval = cfg.livesplit.poll_seconds
    ls_next_poll = 0.0
    ls_phase: str | None = None
    ls_name: str | None = None
    # Last (local, remote) pair already warned about; cleared when they
    # agree again so a later drift re-warns instead of staying silent.
    sync_warned: tuple[str, str] | None = None
    # Unsplit rollback (opt-in): confirm across polls; suppress while a split
    # went undelivered (LiveSplit is legitimately behind the detector).
    undo_watch = UndoWatch()
    undo_suppress = False
    # follow_timer replaces the legacy undo path: rollback plus skip-forward.
    follow_enabled = cfg.livesplit.follow_timer
    done_nudged = False
    follow_poller = _FollowPoller(
        len(cfg.splits), loop_enabled, cfg.livesplit.follow_pause
    )
    if cfg.ocr is not None and (cfg.livesplit.resync_on_unsplit or follow_enabled):
        print(
            "WARN: ocr and livesplit.resync_on_unsplit/follow_timer are both enabled; OCR"
            + " resync and unsplit rollback can disagree and flip-flop the index"
        )

    # Overlay setup — force xcb on pure Wayland (OpenCV Qt backend has no Wayland plugin).
    if "QT_QPA_PLATFORM" not in os.environ and overlay_enabled:
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    overlay_window_name = "autosplitter"
    overlay_enabled_actual = False
    overlay_throttle = max(1, cfg.capture.fps // 10)
    if overlay_enabled:
        try:
            cv2.namedWindow(overlay_window_name, cv2.WINDOW_NORMAL)
            overlay_enabled_actual = True
        except cv2.error:
            print("\nHINT: install opencv-python (not -headless) for ROI overlay")
            overlay_enabled_actual = False
    # Sustained-rate baseline for the HINT: latched after the join phase so
    # bursts around startup cannot skew the measurement.
    steady_base_index: int | None = None
    steady_base_time: float = 0.0

    has_ocr = cfg.ocr is not None
    if loop_enabled or max_frames is not None or has_ocr:
        hdr = f"  {'frame':>6}  {'fps':>5}  {'match':>7}  {'state':<17}"
        if has_ocr:
            hdr += f"  {'ocr':>6}"
        if livesplit_enabled:
            hdr += f"  {'ls':>5}"
        print(hdr)

    # Config reload polling (only if config_path provided)
    config_poll_interval = config_poll_interval if config_path is not None else 0.0
    last_mtime = config_path.stat().st_mtime if config_path is not None else 0.0
    last_config_check = 0.0

    try:
        while max_frames is None or frame_index < max_frames:
            now = time.monotonic()

            # Check for config file changes
            if config_path is not None and now - last_config_check >= config_poll_interval:
                last_config_check = now
                try:
                    mtime = config_path.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        new_cfg = _reload_config(config_path, machine)
                        if new_cfg is not None:
                            cfg = new_cfg
                except OSError:
                    pass  # file temporarily unavailable

            frame = source.read()
            if frame is None:
                if source.exhausted:
                    print("\nSource exhausted (EOF).")
                    break
                if not unavailable:
                    print("\ncapture unavailable, reconnecting...", flush=True)
                    unavailable = True
                continue

            if machine.finished and not loop_enabled:
                if _done_time is None:
                    _done_time = now
                    print("\nDONE: every configured split has fired.")
                # Don't break until both conditions are met:
                # 1) post_run_seconds have elapsed (or no recording configured)
                # 2) LiveSplit timer is no longer Running (if connected)
                timer_ready = (
                    record_cfg is None
                    or record_cfg.post_run_seconds <= 0
                    or now - _done_time >= record_cfg.post_run_seconds
                )
                livesplit_done = (
                    client is None
                    or not client.connected
                    or (ls_phase is not None and ls_phase != "Running")
                )
                if timer_ready and livesplit_done:
                    print()
                    break
                # Periodic status during grace period
                if _done_grace_print is None or now - _done_grace_print >= 1.0:
                    remaining = max(0.0, record_cfg.post_run_seconds - (now - _done_time)) if record_cfg else 0.0
                    ls_status = f"  ls={ls_phase}" if ls_phase is not None else ""
                    print(f"\rDONE — recording {remaining:.0f}s remaining{ls_status}  ", end="", flush=True)
                    _done_grace_print = now
            stamps.append(now)
            frame_index += 1
            unavailable = False

            ratio = _frame_ratio(machine, frame)
            # follow_pause: the timer is paused, detection is frozen (the
            # ratio is still computed for the HUD/status).
            if follow_poller.paused:
                event = None
            else:
                event = machine.feed_result(res_from_ratio(ratio))
            machine.check_ocr_resync(frame, frame_index)
            # follow_timer: poll LiveSplit right away once the run completes
            # so the non-loop exit does not wait out poll_seconds.
            if follow_enabled:
                if machine.finished:
                    if not done_nudged:
                        done_nudged = True
                        ls_next_poll = 0.0
                else:
                    done_nudged = False

            # --loop rearm watch: once finished, track the NEXT run's first
            # split color (or start color if start_transition configured);
            # a sustained quiet stretch rewinds the machine.
            if loop_enabled and machine.finished:
                if cfg.start_transition is not None and machine.start_detector is not None:
                    ratio = machine.start_detector.detect(frame).ratio
                    waiting_for = cfg.start_transition.name
                elif first_detector is not None:
                    ratio = first_detector.detect(frame).ratio
                    waiting_for = cfg.splits[0].name
                else:
                    waiting_for = "unknown"
                if ratio >= cfg.detection.reset_threshold:
                    quiet_frames = 0
                else:
                    quiet_frames += 1
                    if quiet_frames >= quiet_needed:
                        machine.reset()
                        quiet_frames = 0
                        undo_watch.disarm()
                        follow_poller.disarm()
                        undo_suppress = False
                        print(f"\nRUN COMPLETE — rearmed, watching for {waiting_for}")

            if client is not None:
                connected_now = client.poll_connection()
                if connected_now != ls_was_connected:
                    ls_was_connected = connected_now
                    if not connected_now:
                        follow_poller.connection_lost()
                    print(
                        "\nLiveSplit: "
                        + ("CONNECTED" if connected_now else "DISCONNECTED (retrying)")
                    )
                if event is not None:
                    if isinstance(event, StartEvent):
                        if cfg.start_transition is not None and cfg.start_transition.auto_start_timer:
                            delivered = client.start()
                            if not delivered:
                                print("\nLiveSplit: start NOT DELIVERED (server unreachable)")
                    else:
                        delivered = client.split()
                        if not delivered:
                            print("\nLiveSplit: split NOT DELIVERED (server unreachable)")
                            if not undo_suppress:
                                print(
                                    "LiveSplit: unsplit detection paused until"
                                    + " LiveSplit catches up"
                                )
                            undo_suppress = True
                    last_split_time = time.monotonic()

                if now >= ls_next_poll:
                    ls_next_poll = now + ls_poll_interval
                    if client.poll_connection():
                        poll_ok = True
                        if follow_enabled:
                            fres = follow_poller.poll(
                                client, machine, ls_name, last_split_time, undo_suppress
                            )
                            ls_phase = fres.phase
                            ls_name = fres.name
                            poll_ok = fres.ok
                            undo_suppress = fres.suppress
                            if fres.message is not None:
                                print("\n" + fres.message)
                                sync_warned = None
                                quiet_frames = 0
                                if fres.clear_last_split:
                                    last_split_time = None
                                elif fres.touch_last_split:
                                    last_split_time = time.monotonic()
                        else:
                            ls_phase = client.get_current_timer_phase()
                            ls_name = client.get_current_split_name()
                        # Compare only while LiveSplit is actively running;
                        # NotRunning/Paused/Ended carry no segment meaning.
                        if poll_ok and ls_phase == "Running" and isinstance(ls_name, str):
                            local_split = machine.current_split
                            local_next = (
                                local_split.name if local_split is not None else ""
                            )
                            # Partial match: check if autosplitter name is contained in LiveSplit name
                            if local_next and local_next not in ls_name:
                                if sync_warned != (local_next, ls_name):
                                    sync_warned = (local_next, ls_name)
                                    print(
                                        "\nWARN: LiveSplit desync: detector expects"
                                        f' "{local_next}"'
                                        f' but LiveSplit is at "{ls_name}"'
                                    )
                            else:
                                sync_warned = None
                        if not follow_enabled and cfg.livesplit.resync_on_unsplit:
                            ls_index = client.get_split_index()
                            expected_idx = machine.expected_ls_index
                            if ls_index == expected_idx:
                                undo_suppress = False
                            target = undo_watch.observe(
                                ls_phase,
                                ls_index,
                                expected_idx,
                                time.monotonic(),
                                last_split_time,
                                undo_suppress,
                                machine.state
                                in (State.NORMAL, State.WAITING_FOR_RESET, State.DONE),
                            )
                            if target is not None and machine.rollback_to(target):
                                rolled = machine.current_split
                                rolled_name = rolled.name if rolled is not None else ""
                                print(
                                    "\nLiveSplit: UNSPLIT detected — detector rolled"
                                    + f' back to "{rolled_name}" (index {target})'
                                )
                                sync_warned = None
                                quiet_frames = 0
                                undo_suppress = False

            # Redraw at most 10 Hz: printing to a terminal every frame stalls
            # the read loop, backs up the ffmpeg pipe, and triggers UDP
            # overruns. SPLIT events always draw immediately.
            fps = 0.0
            if len(stamps) >= 2 and stamps[-1] > stamps[0]:
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            if (
                live_stream
                and not fps_hint_shown
                # Judge only sustained rate over >= 1 s past the join phase;
                # short rolling windows misread mpegts delivery bursts.
                and len(stamps) == stamps.maxlen
            ):
                if steady_base_index is None:
                    steady_base_index = frame_index
                    steady_base_time = time.monotonic()
                else:
                    span = now - steady_base_time
                    if span >= 2.0:
                        sustained_fps = (frame_index - steady_base_index) / span
                        if abs(sustained_fps - cfg.capture.fps) > cfg.capture.fps * 0.15:
                            fps_hint_shown = True
                            print(
                                f"\nHINT: stream sustains {sustained_fps:.0f} fps"
                                + f" but capture.fps declares {cfg.capture.fps:g}"
                                + " — align OBS's Common FPS Value with config.yaml"
                                + " (debounce is frame-counted and scales with fps)"
                            )

            # Overlay drawing (throttled to ~capture.fps/10 Hz)
            if overlay_enabled_actual and frame_index % overlay_throttle == 0:
                try:
                    last_split_age = (
                        time.monotonic() - last_split_time
                        if last_split_time is not None
                        else None
                    )
                    overlay_frame = annotate(
                        frame.copy(),
                        cfg,
                        machine,
                        ls_was_connected,
                        last_split_age,
                        fps,
                        ratio,
                    )
                    cv2.imshow(overlay_window_name, overlay_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):  # ESC or q
                        print("\nOverlay window closed by user.")
                        break
                    # Defer visibility check: on Wayland+Qt the window reports
                    # invisible until first paint; give it a couple of frames.
                    if frame_index > overlay_throttle * 2 and cv2.getWindowProperty(overlay_window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("\nOverlay window closed.")
                        break
                except cv2.error:
                    print("\nHINT: overlay window error; disabling overlay.")
                    overlay_enabled_actual = False

            line = (
                f"\r#{frame_index:<6} {fps:5.1f} fps  match {ratio * 100:6.2f}%"
                + f"  {machine.state.name:<17}"
            )
            if has_ocr:
                ocr_txt = machine.ocr_text
                ocr_col = f"ocr={ocr_txt}" if ocr_txt is not None else ""
                line += f"  {ocr_col:>6}"
            if loop_enabled and machine.finished:
                line += f"  quiet {quiet_frames}/{quiet_needed}"
            if follow_poller.paused:
                line += "  FROZEN"
            if livesplit_enabled and ls_phase is not None:
                shown_name = (ls_name or "").strip()
                line += f'  ls={ls_phase}:"{shown_name[:24]}"'
            due = last_draw is None or now - last_draw >= 0.1
            if event is not None or due:
                print(line, end="", flush=True)
                last_draw = now
            if event is not None:
                if isinstance(event, StartEvent):
                    print(f"\nSTART \"{event.name}\" at frame {frame_index}")
                else:
                    print(
                        f"\nSPLIT fired: id={event.split_id} \"{event.name}\""
                        + f" at frame {frame_index}"
                    )
            if snap_frames is not None and frame_index >= snap_frames:
                out = Path("snap.png")
                if not cv2.imwrite(str(out), frame):
                    print(f"\nSNAP FAILED: could not write {out.resolve()}", file=sys.stderr)
                    return 1
                print(f"\nSaved frame {frame_index} -> {out.resolve()}")
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        machine.shutdown_ocr()
        source.close()
        if client is not None:
            client.close()
        if overlay_enabled_actual:
            try:
                cv2.destroyWindow(overlay_window_name)
            except cv2.error:
                pass
        # Post-run conversion: .mkv → target format (best-effort, never crashes).
        if record_path is not None and record_cfg is not None and record_cfg.convert_to is not None:
            _convert_recording(record_path, record_cfg.convert_to)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autosplitter",
        description="Visual autosplitter: OBS capture -> color transition detection -> LiveSplit splits",
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
    parser.add_argument(
        "--snap",
        metavar="N",
        type=int,
        default=None,
        help="with --live: save the Nth decoded frame to snap.png and exit",
    )
    parser.add_argument(
        "--livesplit",
        action="store_true",
        help=(
            "with --live: forward split events to LiveSplit Server"
            " (splits only unless the config sets start_transition.auto_start_timer);"
            " also polls LiveSplit every ~2s for timer phase + current split"
            " and warns on desync"
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "with --live: after DONE, rearm automatically once the next run's"
            " first split color stays quiet for detection.restart_quiet_seconds"
            " (reset your LiveSplit timer between runs yourself)"
        ),
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="with --live: show ROI rectangle + HUD overlay window (throttled ~10 Hz; ESC/q to close)",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        default=None,
        help="with --live: record a codec-copy .mkv to PATH while detecting (single ffmpeg, no extra UDP consumer)",
    )
    parser.add_argument(
        "--config-poll-interval",
        type=float,
        default=1.0,
        help="with --live: config file poll interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--probe",
        metavar="IMAGE",
        help="load an image, crop to ROI from config, and print average RGB in the ROI (for color tuning)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="launch the PySide6 GUI instead of the CLI live loop",
    )
    args = parser.parse_args(argv)

    # argparse.Namespace attributes are untyped; validate/cast once, here.
    config_path = cast(str, args.config)
    check_enabled = cast(bool, args.check)
    detect_paths = cast("list[str] | None", args.detect)
    live_arg = cast("str | None", args.live)
    probe_path = cast("str | None", args.probe)
    max_frames = cast("int | None", args.max_frames)
    width_override = cast("int | None", args.width)
    height_override = cast("int | None", args.height)
    snap_frames = cast("int | None", args.snap)
    livesplit_enabled = cast(bool, args.livesplit)
    loop_enabled = cast(bool, args.loop)
    overlay_enabled = cast(bool, args.overlay)
    record_path = cast("str | None", args.record)
    config_poll_interval = cast(float, args.config_poll_interval)
    gui_enabled = cast(bool, args.gui)

    for name, value in (("width", width_override), ("height", height_override), ("snap", snap_frames)):
        if value is not None and value <= 0:
            parser.error(f"--{name} must be a positive integer")

    if gui_enabled:
        from .gui import launch_gui
        return launch_gui(config_path)

    if not check_enabled and not detect_paths and live_arg is None and probe_path is None:
        parser.print_help()
        return 0

    try:
        cfg = load(config_path)
    except ConfigError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    # Resolve record_path: CLI --record wins; otherwise use config template.
    # Resolve once at startup so reconnects use the same file (with -y to overwrite
    # empty files from failed connection attempts).
    if record_path is None and cfg.record is not None:
        record_path = resolve_path(cfg.record.path_template, config_path)
    record_cfg = cfg.record if record_path is not None else None

    exit_code = 0
    if check_enabled:
        _print_summary(config_path, cfg)
    if probe_path is not None:
        exit_code = _probe_image(cfg, probe_path) or exit_code
    if detect_paths:
        exit_code = _detect_images(cfg, detect_paths) or exit_code
    if live_arg is not None:
        source_override = live_arg or None
        exit_code = (
            _run_live(
                cfg,
                source_override,
                max_frames,
                width_override,
                height_override,
                snap_frames,
                livesplit_enabled,
                loop_enabled,
                overlay_enabled,
                record_path,
                record_cfg,
                Path(config_path),
                config_poll_interval,
            )
            or exit_code
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
