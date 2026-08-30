"""QApplication setup and utilities."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def setup_app() -> None:
    """Apply any global Qt settings (high-DPI, style, etc.)."""
    # High-DPI scaling is enabled by default in Qt6.
    # Force fusion style for consistent cross-platform look.
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if isinstance(app, QApplication):
        app.setStyle("Fusion")
