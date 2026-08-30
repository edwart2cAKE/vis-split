"""Main window — zenmap-style layout."""

from __future__ import annotations

import logging
from typing import override

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QImage
from PySide6.QtWidgets import (
    QGroupBox,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .app import setup_app
from .config_editor import ConfigEditor
from .log_panel import LogPanel
from .status_panels import StatusPanels
from .toolbar import Toolbar
from .video_widget import VideoWidget
from .workers import PipelineWorker

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Zenmap-style main window.

    Layout::

        ┌──────────────────────────────┬──────────────────┐
        │                              │  Status Panels    │
        │     Video Preview            │  (Detection,      │
        │     (VideoWidget)            │   LiveSplit,      │
        │                              │   Recording,      │
        │                              │   OCR, Splits)    │
        ├──────────────────────────────┴──────────────────┤
        │  Log Panel (terminal-style)                      │
        ├─────────────────────────────────────────────────┤
        │  Config Editor (collapsible)                     │
        ├─────────────────────────────────────────────────┤
        │  Toolbar (Start/Stop/Record/LS/Loop/Reload/Split)│
        └─────────────────────────────────────────────────┘
    """

    def __init__(
        self, config_path: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._config_path = config_path
        self._pipeline_running = False
        self._livesplit_enabled = False
        self._loop_enabled = False
        self._record_enabled = False

        self._worker: PipelineWorker | None = None

        setup_app()
        self._setup_ui()
        self._setup_worker()
        self._load_config()

    def _setup_ui(self) -> None:
        """Build the zenmap-style layout."""
        self.setWindowTitle("AutoSplitter")
        self.setMinimumSize(900, 600)
        self.resize(1200, 700)

        # Dark theme
        self.setStyleSheet(
            "QMainWindow { background-color: #0f172a; }"
            "QGroupBox {"
            "  border: 1px solid #334155;"
            "  border-radius: 4px;"
            "  margin-top: 8px;"
            "  padding-top: 8px;"
            "  color: #94a3b8;"
            "  font-weight: bold;"
            "}"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            "QLabel { color: #e2e8f0; }"
            "QPushButton {"
            "  background-color: #1e293b;"
            "  color: #e2e8f0;"
            "  border: 1px solid #334155;"
            "  border-radius: 4px;"
            "  padding: 4px 8px;"
            "}"
            "QPushButton:hover { background-color: #334155; }"
            "QPushButton:pressed { background-color: #475569; }"
            "QPushButton:disabled { color: #475569; }"
            "QPushButton:checked { background-color: #1d4ed8; border-color: #3b82f6; }"
            "QProgressBar {"
            "  border: 1px solid #334155;"
            "  border-radius: 4px;"
            "  text-align: center;"
            "  color: #e2e8f0;"
            "  background-color: #1e293b;"
            "}"
            "QProgressBar::chunk { background-color: #3b82f6; border-radius: 3px; }"
            "QScrollArea { border: none; }"
        )

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Top splitter: video + status panels
        top_splitter = QSplitter()

        # Video widget (left, stretch=3)
        video_group = QGroupBox("Preview")
        video_layout = QVBoxLayout(video_group)
        video_layout.setContentsMargins(4, 4, 4, 4)
        self._video = VideoWidget()
        video_layout.addWidget(self._video)
        top_splitter.addWidget(video_group)

        # Status panels (right, stretch=1)
        self._status = StatusPanels()
        top_splitter.addWidget(self._status)

        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 1)

        # Bottom splitter: log + config
        bottom_splitter = QSplitter()

        # Log panel
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(4, 4, 4, 4)
        self._log = LogPanel()
        log_layout.addWidget(self._log)
        bottom_splitter.addWidget(log_group)

        # Config editor
        config_group = QGroupBox("Config")
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(4, 4, 4, 4)
        self._config_editor = ConfigEditor()
        config_layout.addWidget(self._config_editor)
        bottom_splitter.addWidget(config_group)

        bottom_splitter.setStretchFactor(0, 2)
        bottom_splitter.setStretchFactor(1, 1)

        # Vertical splitter: top + bottom
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.addWidget(top_splitter)
        v_splitter.addWidget(bottom_splitter)
        v_splitter.setStretchFactor(0, 3)
        v_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(v_splitter, stretch=1)

        # Toolbar
        self._toolbar = Toolbar()
        main_layout.addWidget(self._toolbar)

        # Connect toolbar signals
        self._toolbar.start_clicked.connect(self._on_start)
        self._toolbar.stop_clicked.connect(self._on_stop)
        self._toolbar.record_clicked.connect(self._on_record_toggle)
        self._toolbar.livesplit_clicked.connect(self._on_livesplit_toggle)
        self._toolbar.loop_clicked.connect(self._on_loop_toggle)
        self._toolbar.reload_clicked.connect(self._on_reload)
        self._toolbar.split_clicked.connect(self._on_manual_split)
        self._toolbar.start_timer_clicked.connect(self._on_manual_start)

        # Connect config editor signals
        self._config_editor.config_reloaded.connect(self._on_config_reloaded)
        self._config_editor.config_error.connect(self._on_config_error)

    def _setup_worker(self) -> None:
        """Create the pipeline worker (QThread subclass) and connect signals."""
        self._worker = PipelineWorker()

        # Worker → GUI connections (auto-queued because worker is on its own thread)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.state_changed.connect(self._status.update_state)
        self._worker.match_ratio_updated.connect(self._status.update_match)
        self._worker.fps_updated.connect(self._status.update_fps)
        self._worker.split_index_updated.connect(
            self._status.update_split_index
        )
        self._worker.ls_connection_changed.connect(
            self._status.update_ls_connection
        )
        self._worker.ls_status_updated.connect(
            self._status.update_ls_status
        )
        self._worker.recording_changed.connect(
            self._status.update_recording
        )
        self._worker.ocr_text_updated.connect(self._status.update_ocr)
        self._worker.log_message.connect(self._log.append_line)
        self._worker.log_error.connect(self._log.append_error)
        self._worker.split_fired.connect(self._on_split_fired)
        self._worker.start_fired.connect(self._on_start_fired)
        self._worker.pipeline_finished.connect(self._on_pipeline_finished)
        self._worker.pipeline_error.connect(self._on_pipeline_error)
        self._worker.loop_rearmed.connect(self._on_loop_rearmed)

    def _load_config(self) -> None:
        """Load initial config and populate UI."""
        self._config_editor.load_config(self._config_path)
        self._log.append_line(f"Loaded config: {self._config_path}")

        # Set ROI on video widget
        from ..config import ConfigError, load

        try:
            cfg = load(self._config_path)
            self._video.set_roi(
                cfg.roi.x, cfg.roi.y, cfg.roi.width, cfg.roi.height
            )

            # Set splits list
            names = [s.name for s in cfg.splits]
            colors = [s.color for s in cfg.splits]
            thresholds = [s.threshold for s in cfg.splits]
            self._status.set_splits(names, colors, thresholds)

            # Set initial toolbar states
            self._toolbar.set_livesplit(True)
            self._toolbar.set_loop(True)
            self._toolbar.set_recording(cfg.record is not None)
        except ConfigError as exc:
            self._log.append_error(f"Failed to load config: {exc}")

    # --- Slots: Pipeline events ---

    def _on_frame(self, image: QImage) -> None:
        """Update video display."""
        self._video.set_frame(image)

    def _on_split_fired(self, split_id: int, name: str) -> None:
        """Handle split event."""
        self._status.highlight_split(split_id)
        self._log.append_split(split_id, name)

    def _on_start_fired(self, name: str) -> None:
        """Handle start event."""
        self._log.append_start(name)

    def _on_pipeline_finished(self) -> None:
        """Handle pipeline finished."""
        self._status.show_done()
        self._toolbar.set_running(False)
        self._pipeline_running = False
        self._log.append_line("Pipeline finished")

    def _on_pipeline_error(self, error: str) -> None:
        """Handle pipeline error."""
        self._toolbar.set_running(False)
        self._pipeline_running = False
        self._log.append_error(error)

    def _on_loop_rearmed(self, name: str) -> None:
        """Handle loop rearm."""
        self._log.append_line(f"Loop rearmed: waiting for {name}")

    # --- Slots: Toolbar actions ---

    def _on_start(self) -> None:
        """Start the pipeline."""
        if self._pipeline_running:
            return
        self._pipeline_running = True
        self._toolbar.set_running(True)
        self._log.append_line("Starting pipeline...")

        # Resolve record path
        record_path = ""
        from ..config import ConfigError, load

        try:
            cfg = load(self._config_path)
            if self._record_enabled and cfg.record is not None:
                from ..record import resolve_path

                record_path = resolve_path(
                    cfg.record.path_template, self._config_path
                )
        except (ConfigError, OSError):
            pass  # best-effort; pipeline starts regardless

        # Store params, then start thread (run() executes on new thread)
        assert self._worker is not None
        self._worker.set_params(
            self._config_path,
            record_path,
            self._livesplit_enabled,
            self._loop_enabled,
            False,  # overlay_enabled (not used in GUI mode)
            1.0,    # config_poll_interval
        )
        self._worker.start()

    def _on_stop(self) -> None:
        """Stop the pipeline."""
        if not self._pipeline_running:
            return
        assert self._worker is not None
        self._worker._running = False
        self._worker.wait(3000)  # Wait up to 3s for clean shutdown
        self._toolbar.set_running(False)
        self._pipeline_running = False

    def _on_record_toggle(self) -> None:
        """Toggle recording."""
        self._record_enabled = not self._record_enabled
        self._toolbar.set_recording(self._record_enabled)
        status = "enabled" if self._record_enabled else "disabled"
        self._log.append_line(f"Recording: {status}")

    def _on_livesplit_toggle(self) -> None:
        """Toggle LiveSplit."""
        self._livesplit_enabled = not self._livesplit_enabled
        self._toolbar.set_livesplit(self._livesplit_enabled)
        status = "enabled" if self._livesplit_enabled else "disabled"
        self._log.append_line(f"LiveSplit: {status}")

    def _on_loop_toggle(self) -> None:
        """Toggle loop mode."""
        self._loop_enabled = not self._loop_enabled
        self._toolbar.set_loop(self._loop_enabled)
        status = "enabled" if self._loop_enabled else "disabled"
        self._log.append_line(f"Loop: {status}")

    def _on_reload(self) -> None:
        """Reload config."""
        self._config_editor.load_config(self._config_path)
        self._log.append_line("Config reloaded")

    def _on_manual_split(self) -> None:
        """Manual split — emit signal to worker thread."""
        assert self._worker is not None
        self._worker.split_requested.emit()

    def _on_manual_start(self) -> None:
        """Manual start timer — emit signal to worker thread."""
        assert self._worker is not None
        self._worker.start_requested.emit()

    # --- Slots: Config editor events ---

    def _on_config_reloaded(self, path: str) -> None:
        """Handle config reload from editor."""
        self._config_path = path
        self._log.append_line(f"Config reloaded: {path}")

        # Update ROI
        from ..config import ConfigError, load

        try:
            cfg = load(path)
            self._video.set_roi(
                cfg.roi.x, cfg.roi.y, cfg.roi.width, cfg.roi.height
            )

            # Update splits list
            names = [s.name for s in cfg.splits]
            colors = [s.color for s in cfg.splits]
            thresholds = [s.threshold for s in cfg.splits]
            self._status.set_splits(names, colors, thresholds)
        except ConfigError as exc:
            self._log.append_error(f"Failed to update UI: {exc}")

    def _on_config_error(self, error: str) -> None:
        """Handle config error from editor."""
        self._log.append_error(f"Config error: {error}")

    @override
    def closeEvent(self, event: QCloseEvent) -> None:
        """Clean up on window close."""
        if self._pipeline_running and self._worker is not None:
            self._worker._running = False
            self._worker.wait(3000)
        super().closeEvent(event)
