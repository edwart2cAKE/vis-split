"""YAML config editor with reload."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ConfigEditor(QWidget):
    """Simple YAML text editor with reload button.

    Shows the current config.yaml content. Edit + Reload loads changes.
    Shows validation errors inline.
    """

    config_reloaded = Signal(str)  # new config path
    config_error = Signal(str)  # error message

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        self._file_label = QLabel("Config: \u2014")
        self._file_label.setStyleSheet("font-weight: bold; color: #94a3b8;")
        header.addWidget(self._file_label, stretch=1)

        self._reload_btn = QPushButton("\u27f3 Reload")
        self._reload_btn.setFixedWidth(80)
        self._reload_btn.clicked.connect(self._on_reload)
        header.addWidget(self._reload_btn)
        layout.addLayout(header)

        # Editor
        self._editor = QTextEdit()
        font = QFont("monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        self._editor.setFont(font)
        self._editor.setStyleSheet(
            "QTextEdit {"
            "  background-color: #1e293b;"
            "  color: #e2e8f0;"
            "  border: 1px solid #334155;"
            "}"
        )
        layout.addWidget(self._editor, stretch=1)

        # Error label
        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #ef4444; font-weight: bold;")
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        layout.addWidget(self._error_label)

    def load_config(self, path: str) -> None:
        """Load config file content into editor."""
        self._config_path = Path(path)
        self._file_label.setText(f"Config: {self._config_path.name}")
        try:
            content = self._config_path.read_text()
            self._editor.setPlainText(content)
            self._error_label.hide()
        except OSError as exc:
            self._error_label.setText(f"Failed to load: {exc}")
            self._error_label.show()

    def _on_reload(self) -> None:
        """Validate and reload config."""
        if self._config_path is None:
            return

        # Write editor content back to file
        try:
            self._config_path.write_text(self._editor.toPlainText())
        except OSError as exc:
            self._error_label.setText(f"Failed to write: {exc}")
            self._error_label.show()
            self.config_error.emit(str(exc))
            return

        # Validate by loading
        from ..config import ConfigError, load

        try:
            load(str(self._config_path))
            self._error_label.hide()
            self._editor.setStyleSheet(
                "QTextEdit {"
                "  background-color: #1e293b;"
                "  color: #e2e8f0;"
                "  border: 1px solid #334155;"
                "}"
            )
            self.config_reloaded.emit(str(self._config_path))
        except ConfigError as exc:
            self._error_label.setText(f"Invalid config: {exc}")
            self._error_label.show()
            self._editor.setStyleSheet(
                "QTextEdit {"
                "  background-color: #1e293b;"
                "  color: #e2e8f0;"
                "  border: 2px solid #ef4444;"
                "}"
            )
            self.config_error.emit(str(exc))
