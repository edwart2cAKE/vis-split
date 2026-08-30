"""Pipeline worker — runs capture→detect→LiveSplit in a background thread."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage

from ..capture import MpegTsFrameSource
from ..config import Config, ConfigError, load
from ..detection import SplitEvent, StartEvent, TransitionDetector
from ..livesplit import LiveSplitClient
from ..record import resolve_path

log = logging.getLogger(__name__)


class PipelineWorker(QThread):
    """Runs capture→detect→LiveSplit in a background thread.

    Standard QThread pattern:
        worker = PipelineWorker()
        worker.set_params(config_path, record_path, ...)
        worker.start()  # run() executes on new thread
    """

    # --- Outgoing signals (to GUI, auto-queued to main thread) ---
    frame_ready = Signal(QImage)            # throttled, for video display
    state_changed = Signal(str)             # machine.state.name
    split_fired = Signal(int, str)          # split_id, split_name
    start_fired = Signal(str)               # start name
    match_ratio_updated = Signal(float)     # 0.0–1.0
    fps_updated = Signal(float)             # rolling fps
    split_index_updated = Signal(int, int)  # current_index, total
    ls_connection_changed = Signal(bool)    # connected
    ls_status_updated = Signal(str, str)    # phase, split_name
    ocr_text_updated = Signal(str)          # latest OCR text or ""
    recording_changed = Signal(bool, str)   # is_recording, path
    log_message = Signal(str)               # status line for log panel
    log_error = Signal(str)                 # error message for log panel
    pipeline_error = Signal(str)            # fatal error message
    pipeline_finished = Signal()            # source exhausted or DONE
    loop_rearmed = Signal(str)              # waiting_for name

    # --- Incoming signals (queued to worker thread) ---
    split_requested = Signal()
    start_requested = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        # Pipeline params (set via set_params, read in run())
        self._config_path_str: str = ""
        self._record_path: str = ""
        self._livesplit_enabled: bool = False
        self._loop_enabled: bool = False
        self._overlay_enabled: bool = False
        self._config_poll_interval: float = 1.0

        # Pipeline state (set in run(), accessed from worker thread only)
        self._source: MpegTsFrameSource | None = None
        self._machine: TransitionDetector | None = None
        self._livesplit: LiveSplitClient | None = None
        self._cfg: Config | None = None

        self._running = False

        # Frame timing
        self._frame_index = 0
        self._frame_stamps: list[float] = []

        # Split timing
        self._last_split_time = 0.0
        self._last_split_name = ""

        # Config reload
        self._config_path: Path | None = None
        self._last_config_mtime = 0.0

        # Loop rearm state
        self._quiet_frames = 0

        # Connect incoming signals (these fire on worker thread after start())
        self.split_requested.connect(self._do_split)
        self.start_requested.connect(self._do_start)

    def set_params(
        self,
        config_path: str,
        record_path: str,
        livesplit_enabled: bool,
        loop_enabled: bool,
        overlay_enabled: bool,
        config_poll_interval: float,
    ) -> None:
        """Store pipeline parameters. Call before start()."""
        self._config_path_str = config_path
        self._record_path = record_path
        self._livesplit_enabled = livesplit_enabled
        self._loop_enabled = loop_enabled
        self._overlay_enabled = overlay_enabled
        self._config_poll_interval = config_poll_interval

    def run(self) -> None:
        """QThread entry point — runs on the worker thread."""
        config_path = self._config_path_str
        record_path = self._record_path

        try:
            cfg = load(config_path)
        except ConfigError as exc:
            self.log_error.emit(f"Config error: {exc}")
            self.pipeline_error.emit(str(exc))
            return

        self._cfg = cfg
        self._config_path = Path(config_path)

        # Resolve recording path
        rec_path = record_path if record_path else None
        if rec_path is None and cfg.record is not None:
            rec_path = resolve_path(cfg.record.path_template, config_path)

        # Create capture source
        try:
            self._source = MpegTsFrameSource(
                cfg.capture.stream_url,
                cfg.capture.width,
                cfg.capture.height,
                record_path=rec_path,
            )
        except RuntimeError as exc:
            self.log_error.emit(f"Capture error: {exc}")
            self.pipeline_error.emit(str(exc))
            return

        # Create detection machine
        self._machine = TransitionDetector(cfg)

        # Create LiveSplit client
        if self._livesplit_enabled:
            self._livesplit = LiveSplitClient(
                cfg.livesplit.host,
                cfg.livesplit.port,
                timeout_seconds=cfg.livesplit.timeout_seconds,
                reconnect_seconds=cfg.livesplit.reconnect_seconds,
            )
        else:
            self._livesplit = None

        # Emit initial state
        self.log_message.emit(f"Pipeline started: {cfg.capture.stream_url}")
        if rec_path:
            self.recording_changed.emit(True, rec_path)
        self._emit_initial_state()

        # Run the loop
        self._running = True
        self._run_loop(cfg)

    def _run_loop(self, cfg: Config) -> None:
        """Main capture→detect→emit loop. Runs on worker thread."""
        source = self._source
        assert source is not None
        machine = self._machine
        assert machine is not None

        target_interval = 1.0 / cfg.capture.fps
        last_emit_time = 0.0
        emit_interval = 1.0 / 10.0  # 10Hz to GUI

        while self._running:
            frame_start = time.monotonic()

            # Config reload check (every ~1s)
            if (
                self._config_path is not None
                and self._frame_index % max(1, int(cfg.capture.fps)) == 0
            ):
                try:
                    mtime = self._config_path.stat().st_mtime
                    if mtime > self._last_config_mtime:
                        self._last_config_mtime = mtime
                        if self._last_config_mtime > 0:
                            self._try_config_reload()
                except OSError:
                    pass

            # Read frame
            frame = source.read()
            if frame is None:
                if source.exhausted:
                    self.log_message.emit("Source exhausted")
                    self.pipeline_finished.emit()
                    self._running = False
                    break
                time.sleep(0.001)
                continue

            self._frame_index += 1
            now = time.monotonic()
            self._frame_stamps.append(now)
            while self._frame_stamps and self._frame_stamps[0] < now - 2.0:
                self._frame_stamps.pop(0)

            # Detection
            result = machine.current_detector.detect(frame) if machine.current_detector is not None else None
            match_ratio = result.ratio if result is not None else 0.0

            event = machine.feed_result(result) if result is not None else None

            # OCR resync
            if cfg.ocr is not None:
                machine.check_ocr_resync(frame, self._frame_index)

            # Loop rearm
            if self._loop_enabled and machine.finished:
                self._handle_loop_rearm(cfg, match_ratio)

            # LiveSplit
            if self._livesplit is not None:
                self._handle_livesplit(event)

            # Emit frame + status (throttled)
            if now - last_emit_time >= emit_interval:
                self._emit_frame(frame)
                self._emit_status(cfg, match_ratio)
                last_emit_time = now

            # Emit events
            if event is not None:
                if isinstance(event, SplitEvent):
                    self._last_split_time = now
                    self._last_split_name = event.name
                    self.split_fired.emit(event.split_id, event.name)
                    self.log_message.emit(f"SPLIT: #{event.split_id} {event.name}")
                else:
                    self.start_fired.emit(event.name)
                    self.log_message.emit(f"START: {event.name}")

            # Yield CPU
            elapsed = time.monotonic() - frame_start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        self._cleanup()

    def _emit_frame(self, frame: np.ndarray) -> None:
        """Convert BGR frame to QImage and emit for display."""
        rgb = cast(np.ndarray, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        height, width = cast(tuple[int, int], rgb.shape[:2])
        bytes_per_line = width * 3
        qimg = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self.frame_ready.emit(qimg)

    def _emit_status(self, cfg: Config, match_ratio: float) -> None:
        """Emit current detection status."""
        machine = self._machine
        if machine is None:
            return

        self.state_changed.emit(machine.state.name)
        self.match_ratio_updated.emit(match_ratio)

        if len(self._frame_stamps) >= 2:
            fps = len(self._frame_stamps) / (self._frame_stamps[-1] - self._frame_stamps[0])
            self.fps_updated.emit(fps)

        total = len(cfg.splits)
        self.split_index_updated.emit(machine.current_split_index, total)

        if self._livesplit is not None:
            self.ls_connection_changed.emit(self._livesplit.connected)
            if self._frame_index % max(1, int(cfg.capture.fps * 2)) == 0:
                phase = self._livesplit.get_current_timer_phase()
                name = self._livesplit.get_current_split_name()
                self.ls_status_updated.emit(phase or "Unknown", name or "-")

        ocr_text = machine.ocr_text
        if ocr_text:
            self.ocr_text_updated.emit(ocr_text)

    def _emit_initial_state(self) -> None:
        """Emit initial state after pipeline starts."""
        cfg = self._cfg
        if cfg is None:
            return
        self.state_changed.emit("WAITING_FOR_START")
        self.match_ratio_updated.emit(0.0)
        self.fps_updated.emit(0.0)
        total = len(cfg.splits)
        self.split_index_updated.emit(0, total)
        if self._livesplit is not None:
            self.ls_connection_changed.emit(False)
            self.ls_status_updated.emit("NotRunning", "-")

    def _handle_livesplit(self, event: SplitEvent | StartEvent | None) -> None:
        """Handle LiveSplit connection and split events."""
        if self._livesplit is None:
            return

        self._livesplit.poll_connection()

        if event is not None and self._livesplit.connected:
            if isinstance(event, SplitEvent):
                self._livesplit.split()
            else:
                self._livesplit.start()

    def _handle_loop_rearm(self, cfg: Config, match_ratio: float) -> None:
        """Handle loop rearm: wait for quiet period on first split color."""
        machine = self._machine
        if machine is None:
            return

        if machine.first_detector is None:
            return

        reset_threshold = cfg.detection.reset_threshold
        if match_ratio < reset_threshold:
            self._quiet_frames += 1
            quiet_needed = machine.quiet_frames_needed
            if self._quiet_frames >= quiet_needed:
                machine.reset()
                self._quiet_frames = 0
                self.log_message.emit("Loop rearmed: waiting for start")
                start_name = cfg.start_transition.name if cfg.start_transition else "Run Start"
                self.loop_rearmed.emit(start_name)
        else:
            self._quiet_frames = 0

    def _try_config_reload(self) -> None:
        """Try to reload config from disk."""
        if self._config_path is None or self._machine is None:
            return
        try:
            new_cfg = load(str(self._config_path))
            self._machine.reload_config(new_cfg)
            self._cfg = new_cfg
            self.log_message.emit("Config reloaded")
        except Exception as exc:  # noqa: BLE001
            self.log_error.emit(f"Config reload failed: {exc}")

    def _cleanup(self) -> None:
        """Clean up resources when pipeline stops."""
        if self._machine is not None:
            self._machine.shutdown_ocr()
            self._machine = None
        if self._source is not None:
            self._source.close()
            self._source = None
        if self._livesplit is not None:
            self._livesplit.close()
            self._livesplit = None
        self.log_message.emit("Pipeline stopped")

    @Slot()
    def _do_split(self) -> None:
        """Manual split request."""
        if self._livesplit is not None and self._livesplit.connected:
            self._livesplit.split()
            self.log_message.emit("Manual split sent")

    @Slot()
    def _do_start(self) -> None:
        """Manual start timer request."""
        if self._livesplit is not None and self._livesplit.connected:
            self._livesplit.start()
            self.log_message.emit("Manual start sent")
