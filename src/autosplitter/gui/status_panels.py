"""Status panels — Detection, LiveSplit, Recording, OCR, Splits progress."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class StatusPanels(QWidget):
    """Container for all status panels, stacked vertically."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMaximumWidth(320)
        self.setMinimumWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Detection panel
        self._detection = _DetectionPanel()
        layout.addWidget(self._detection)

        # LiveSplit panel
        self._livesplit = _LiveSplitPanel()
        layout.addWidget(self._livesplit)

        # Recording panel
        self._recording = _RecordingPanel()
        layout.addWidget(self._recording)

        # OCR panel
        self._ocr = _OcrPanel()
        layout.addWidget(self._ocr)

        # Splits progress
        self._splits_scroll = QScrollArea()
        self._splits_scroll.setWidgetResizable(True)
        self._splits_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._splits = _SplitsPanel()
        self._splits_scroll.setWidget(self._splits)
        layout.addWidget(self._splits_scroll, stretch=1)

        layout.addStretch()

    # --- Public update methods (called from MainWindow via signals) ---

    def update_state(self, state: str) -> None:
        self._detection.update_state(state)

    def update_match(self, ratio: float) -> None:
        self._detection.update_match(ratio)

    def update_fps(self, fps: float) -> None:
        self._detection.update_fps(fps)

    def update_split_index(self, current: int, total: int) -> None:
        self._detection.update_split_index(current, total)
        self._splits.update_current(current)

    def update_ls_connection(self, connected: bool) -> None:
        self._livesplit.update_connection(connected)

    def update_ls_status(self, phase: str, split_name: str) -> None:
        self._livesplit.update_status(phase, split_name)

    def update_recording(self, active: bool, path: str) -> None:
        self._recording.update_status(active, path)

    def update_ocr(self, text: str) -> None:
        self._ocr.update_text(text)

    def set_splits(
        self, names: list[str], colors: list[str], thresholds: list[float]
    ) -> None:
        """Set the splits list from config."""
        self._splits.set_splits(names, colors, thresholds)

    def highlight_split(self, split_id: int) -> None:
        """Highlight the split that just fired."""
        self._splits.highlight_fired(split_id)

    def show_done(self) -> None:
        """Mark all splits as done."""
        self._splits.show_done()


class _DetectionPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Detection", parent)
        layout = QVBoxLayout(self)

        self._state_label = QLabel("State: \u2014")
        self._state_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._state_label)

        self._target_label = QLabel("Target: \u2014")
        layout.addWidget(self._target_label)

        # Match ratio bar
        match_row = QHBoxLayout()
        self._match_bar = QProgressBar()
        self._match_bar.setRange(0, 100)
        self._match_bar.setValue(0)
        self._match_bar.setTextVisible(True)
        self._match_bar.setFormat("%p%")
        match_row.addWidget(self._match_bar, stretch=1)
        self._match_threshold = QLabel("\u2265 \u2014")
        self._match_threshold.setMinimumWidth(60)
        match_row.addWidget(self._match_threshold)
        layout.addLayout(match_row)

        self._split_label = QLabel("Split: \u2014")
        layout.addWidget(self._split_label)

        self._fps_label = QLabel("FPS: \u2014")
        layout.addWidget(self._fps_label)

    def update_state(self, state: str) -> None:
        color_map = {
            "NORMAL": "#4ade80",
            "TRANSITION": "#facc15",
            "WAITING_FOR_RESET": "#fb923c",
            "WAITING_FOR_START": "#94a3b8",
            "DONE": "#a78bfa",
        }
        color = color_map.get(state, "#ffffff")
        self._state_label.setText(f"State: {state}")
        self._state_label.setStyleSheet(f"font-weight: bold; color: {color};")

    def update_match(self, ratio: float) -> None:
        self._match_bar.setValue(int(ratio * 100))

    def update_fps(self, fps: float) -> None:
        self._fps_label.setText(f"FPS: {fps:.1f}")

    def update_split_index(self, current: int, total: int) -> None:
        if current < total:
            self._split_label.setText(f"Split: {current + 1}/{total}")
        else:
            self._split_label.setText("Split: DONE")


class _LiveSplitPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("LiveSplit", parent)
        layout = QVBoxLayout(self)

        self._connection_label = QLabel("\u25cb Disconnected")
        self._connection_label.setStyleSheet("color: #94a3b8;")
        layout.addWidget(self._connection_label)

        self._phase_label = QLabel("Phase: \u2014")
        layout.addWidget(self._phase_label)

        self._segment_label = QLabel("Segment: \u2014")
        layout.addWidget(self._segment_label)

    def update_connection(self, connected: bool) -> None:
        if connected:
            self._connection_label.setText("\u25cf Connected")
            self._connection_label.setStyleSheet(
                "color: #4ade80; font-weight: bold;"
            )
        else:
            self._connection_label.setText("\u25cb Disconnected")
            self._connection_label.setStyleSheet("color: #94a3b8;")

    def update_status(self, phase: str, split_name: str) -> None:
        self._phase_label.setText(f"Phase: {phase}")
        self._segment_label.setText(f"Segment: {split_name}")


class _RecordingPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Recording", parent)
        layout = QVBoxLayout(self)

        self._status_label = QLabel("\u25cb Idle")
        self._status_label.setStyleSheet("color: #94a3b8;")
        layout.addWidget(self._status_label)

        self._path_label = QLabel("")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("font-size: 10px; color: #64748b;")
        layout.addWidget(self._path_label)

    def update_status(self, active: bool, path: str) -> None:
        from pathlib import Path

        if active:
            self._status_label.setText("\u25cf Recording")
            self._status_label.setStyleSheet("color: #ef4444; font-weight: bold;")
            self._path_label.setText(Path(path).name)
        else:
            self._status_label.setText("\u25cb Idle")
            self._status_label.setStyleSheet("color: #94a3b8;")
            self._path_label.setText("")


class _OcrPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("OCR", parent)
        layout = QVBoxLayout(self)

        self._text_label = QLabel("Text: \u2014")
        layout.addWidget(self._text_label)

        self._conf_label = QLabel("")
        self._conf_label.setStyleSheet("font-size: 10px; color: #64748b;")
        layout.addWidget(self._conf_label)

    def update_text(self, text: str) -> None:
        if text:
            self._text_label.setText(f"Text: {text}")
            self._text_label.setStyleSheet("font-weight: bold;")
        else:
            self._text_label.setText("Text: \u2014")
            self._text_label.setStyleSheet("")


class _SplitsPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Splits", parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(2)

        self._rows: list[_SplitRow] = []
        self._current_index = -1

    def set_splits(
        self, names: list[str], colors: list[str], thresholds: list[float]
    ) -> None:
        """Set the splits list from config."""
        # Clear existing
        for row in self._rows:
            row.setParent(None)
        self._rows.clear()

        for i, (name, color, threshold) in enumerate(
            zip(names, colors, thresholds)
        ):
            row = _SplitRow(i + 1, name, color, threshold)
            self._rows.append(row)
            self._layout.addWidget(row)

        self._layout.addStretch()

    def update_current(self, index: int) -> None:
        """Update which split is currently active."""
        if index == self._current_index:
            return
        # Unmark old
        if 0 <= self._current_index < len(self._rows):
            self._rows[self._current_index].set_active(False)
        # Mark new
        self._current_index = index
        if 0 <= index < len(self._rows):
            self._rows[index].set_active(True)

    def highlight_fired(self, split_id: int) -> None:
        """Mark a split as fired (check mark)."""
        idx = split_id - 1
        if 0 <= idx < len(self._rows):
            self._rows[idx].set_fired(True)

    def show_done(self) -> None:
        """Mark all splits as done."""
        for row in self._rows:
            row.set_fired(True)


class _SplitRow(QWidget):
    """A single split row showing: number, name, color dot, threshold."""

    def __init__(
        self,
        number: int,
        name: str,
        color_name: str,
        threshold: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._number = number
        self._fired = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        self._status = QLabel(f"{number}.")
        self._status.setFixedWidth(24)
        layout.addWidget(self._status)

        self._name_label = QLabel(name)
        layout.addWidget(self._name_label, stretch=1)

        # Color dot (using Unicode circle)
        color_map = {
            "red": "#ef4444",
            "blue": "#3b82f6",
            "green": "#22c55e",
            "yellow": "#eab308",
            "purple": "#a855f7",
            "orange": "#f97316",
            "white": "#ffffff",
            "black": "#000000",
            "gray": "#6b7280",
        }
        dot_color = color_map.get(color_name.lower(), "#6b7280")
        self._color_dot = QLabel("\u25cf")
        self._color_dot.setStyleSheet(
            f"color: {dot_color}; font-size: 12px;"
        )
        self._color_dot.setFixedWidth(16)
        layout.addWidget(self._color_dot)

        self._threshold_label = QLabel(f"{threshold:.0%}")
        self._threshold_label.setFixedWidth(36)
        self._threshold_label.setStyleSheet(
            "font-size: 10px; color: #64748b;"
        )
        layout.addWidget(self._threshold_label)

    def set_active(self, active: bool) -> None:
        """Highlight as current split."""
        if active:
            self.setStyleSheet(
                "background-color: rgba(59, 130, 246, 0.3);"
                " border-radius: 3px;"
            )
            self._name_label.setStyleSheet("font-weight: bold; color: #60a5fa;")
        else:
            self.setStyleSheet("")
            self._name_label.setStyleSheet("")

    def set_fired(self, fired: bool) -> None:
        """Mark as completed."""
        self._fired = fired
        if fired:
            self._status.setText("\u2713")
            self._status.setStyleSheet(
                "color: #4ade80; font-weight: bold;"
            )
            self._name_label.setStyleSheet(
                "color: #64748b; text-decoration: line-through;"
            )
        else:
            self._status.setText(f"{self._number}.")
            self._status.setStyleSheet("")
            self._name_label.setStyleSheet("")
