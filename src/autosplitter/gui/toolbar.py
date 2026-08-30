"""Bottom toolbar with control buttons."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QWidget,
)


class Toolbar(QWidget):
    """Bottom toolbar: Start/Stop, Record, LiveSplit, Loop, Reload, Manual Split."""

    start_clicked = Signal()
    stop_clicked = Signal()
    record_clicked = Signal()
    livesplit_clicked = Signal()
    loop_clicked = Signal()
    reload_clicked = Signal()
    split_clicked = Signal()
    start_timer_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "Toolbar { border-top: 1px solid #334155; }"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Start/Stop
        self._start_btn = QPushButton("\u25b6 Start")
        self._start_btn.setFixedWidth(80)
        self._start_btn.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("\u23f9 Stop")
        self._stop_btn.setFixedWidth(80)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self._stop_btn)

        # Separator
        layout.addWidget(self._make_separator())

        # Record
        self._record_btn = QPushButton("\U0001f534 Record")
        self._record_btn.setFixedWidth(90)
        self._record_btn.setCheckable(True)
        self._record_btn.clicked.connect(self.record_clicked.emit)
        layout.addWidget(self._record_btn)

        # LiveSplit
        self._ls_btn = QPushButton("\u2194 LiveSplit")
        self._ls_btn.setFixedWidth(100)
        self._ls_btn.setCheckable(True)
        self._ls_btn.clicked.connect(self.livesplit_clicked.emit)
        layout.addWidget(self._ls_btn)

        # Loop
        self._loop_btn = QPushButton("\u21bb Loop")
        self._loop_btn.setFixedWidth(80)
        self._loop_btn.setCheckable(True)
        self._loop_btn.clicked.connect(self.loop_clicked.emit)
        layout.addWidget(self._loop_btn)

        # Separator
        layout.addWidget(self._make_separator())

        # Reload
        self._reload_btn = QPushButton("\u27f3 Reload")
        self._reload_btn.setFixedWidth(80)
        self._reload_btn.clicked.connect(self.reload_clicked.emit)
        layout.addWidget(self._reload_btn)

        # Manual Split
        self._split_btn = QPushButton("\u23ed Split")
        self._split_btn.setFixedWidth(80)
        self._split_btn.setEnabled(False)
        self._split_btn.clicked.connect(self.split_clicked.emit)
        layout.addWidget(self._split_btn)

        # Manual Start Timer
        self._start_timer_btn = QPushButton("\u25b6 Start Timer")
        self._start_timer_btn.setFixedWidth(100)
        self._start_timer_btn.setEnabled(False)
        self._start_timer_btn.clicked.connect(self.start_timer_clicked.emit)
        layout.addWidget(self._start_timer_btn)

        layout.addStretch()

    def set_running(self, running: bool) -> None:
        """Update button states based on pipeline running state."""
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._split_btn.setEnabled(running)
        self._start_timer_btn.setEnabled(running)

    def set_recording(self, recording: bool) -> None:
        """Update recording button state."""
        self._record_btn.setChecked(recording)
        if recording:
            self._record_btn.setText("\u23f9 Stop Rec")
        else:
            self._record_btn.setText("\U0001f534 Record")

    def set_livesplit(self, enabled: bool) -> None:
        """Update LiveSplit button state."""
        self._ls_btn.setChecked(enabled)

    def set_loop(self, enabled: bool) -> None:
        """Update Loop button state."""
        self._loop_btn.setChecked(enabled)

    @staticmethod
    def _make_separator() -> QWidget:
        """Create a vertical separator line."""
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setFixedHeight(24)
        sep.setStyleSheet("background-color: #334155;")
        return sep
