"""Detection logic, independent of any capture or LiveSplit concerns."""

from autosplitter.detection.color_detector import (
    ColorDetector,
    DetectionResult,
    build_detector,
    roi_bounds,
)
from autosplitter.detection.ocr_detector import OcrDetector, OcrResult, OcrWorker
from autosplitter.detection.state_machine import (
    SplitEvent,
    StartEvent,
    State,
    TransitionDetector,
)

__all__ = [
    "ColorDetector",
    "DetectionResult",
    "OcrDetector",
    "OcrResult",
    "OcrWorker",
    "SplitEvent",
    "StartEvent",
    "State",
    "TransitionDetector",
    "build_detector",
    "roi_bounds",
]
