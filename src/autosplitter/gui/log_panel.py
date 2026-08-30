"""Terminal-style log panel."""

from __future__ import annotations

from datetime import UTC, datetime

from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit, QWidget


class LogPanel(QPlainTextEdit):
    """Read-only terminal-style log. Auto-scrolls to bottom.

    Captures status lines the CLI prints. Max 10,000 lines (ring buffer).
    """

    MAX_LINES = 10_000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)

        # Monospace font
        font = QFont("monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        self.setFont(font)

        # Dark theme
        self.setStyleSheet(
            "QPlainTextEdit {"
            "  background-color: #0f172a;"
            "  color: #e2e8f0;"
            "  border: 1px solid #334155;"
            "  padding: 4px;"
            "}"
        )

    def append_line(self, text: str) -> None:
        """Append a normal log line with timestamp."""
        timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")
        self.appendHtml(
            f'<span style="color:#64748b;">[{timestamp}]</span> {text}'
        )
        self._auto_scroll()

    def append_error(self, text: str) -> None:
        """Append an error line in red."""
        timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")
        self.appendHtml(
            f'<span style="color:#64748b;">[{timestamp}]</span> '
            f'<span style="color:#ef4444; font-weight:bold;">ERROR: {text}</span>'
        )
        self._auto_scroll()

    def append_split(self, split_id: int, name: str) -> None:
        """Append a split event line in green."""
        timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")
        self.appendHtml(
            f'<span style="color:#64748b;">[{timestamp}]</span> '
            f'<span style="color:#4ade80; font-weight:bold;">'
            f"SPLIT #{split_id}: {name}</span>"
        )
        self._auto_scroll()

    def append_start(self, name: str) -> None:
        """Append a start event line in yellow."""
        timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")
        self.appendHtml(
            f'<span style="color:#64748b;">[{timestamp}]</span> '
            f'<span style="color:#facc15; font-weight:bold;">START: {name}</span>'
        )
        self._auto_scroll()

    def _auto_scroll(self) -> None:
        """Scroll to bottom."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
