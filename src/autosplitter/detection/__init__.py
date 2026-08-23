"""Detection logic, independent of any capture or LiveSplit concerns."""

from autosplitter.detection.color_detector import (
    ColorDetector,
    DetectionResult,
    build_detector,
    roi_bounds,
)
from autosplitter.detection.state_machine import (
    SplitEvent,
    State,
    TransitionDetector,
)

__all__ = [
    "ColorDetector",
    "DetectionResult",
    "SplitEvent",
    "State",
    "TransitionDetector",
    "build_detector",
    "roi_bounds",
]
