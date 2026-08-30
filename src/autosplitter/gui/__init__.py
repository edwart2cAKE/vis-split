"""GUI package — PySide6-based zenmap-style interface."""

from __future__ import annotations

import sys


def launch_gui(config_path: str) -> int:
    """Create QApplication, build MainWindow, exec event loop.

    Returns the QApplication exit code.
    """
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "PySide6 is required for the GUI.\nInstall it with: uv sync --extra gui",
            file=sys.stderr,
        )
        return 1

    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("AutoSplitter")
    app.setApplicationDisplayName("AutoSplitter")

    window = MainWindow(config_path)
    window.show()

    return app.exec()
