"""Live video display widget with ROI overlay."""

from __future__ import annotations

from typing import override

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QWidget


class VideoWidget(QWidget):
    """Displays the latest frame as a scaled QImage, with optional ROI rectangle overlay.

    Frames arrive as QImage (already RGB, already .copy()'d by worker).
    Scaled to fit the widget while preserving aspect ratio.
    ROI overlay is drawn as a QPainter rectangle.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._roi: tuple[float, float, float, float] | None = None  # normalized x,y,w,h
        self.setMinimumSize(320, 180)
        self.setStyleSheet("background-color: #1a1a2e;")

    def set_frame(self, image: QImage) -> None:
        """Slot: receives a copied QImage, converts to pixmap for display."""
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def set_roi(self, x: float, y: float, w: float, h: float) -> None:
        """Update ROI overlay coordinates (normalized 0.0–1.0)."""
        self._roi = (x, y, w, h)
        self.update()

    @override
    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the frame scaled to fit, then overlay ROI rectangle."""
        if self._pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Scale pixmap to fit widget, preserving aspect ratio, centered
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Center in widget
        x_offset = (self.width() - scaled.width()) // 2
        y_offset = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x_offset, y_offset, scaled)

        # Draw ROI overlay
        if self._roi is not None and self._pixmap.width() > 0:
            rx, ry, rw, rh = self._roi
            # Map normalized coords to scaled pixmap coordinates
            px = int(x_offset + rx * scaled.width())
            py = int(y_offset + ry * scaled.height())
            pw = int(rw * scaled.width())
            ph = int(rh * scaled.height())

            pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 255, 0, 20))  # semi-transparent fill
            painter.drawRect(px, py, pw, ph)

        painter.end()
