"""Transition state machine (spec sections 7-10).

State flow:

    NORMAL --(streak of required_frames at/above threshold)--> TRANSITION
    TRANSITION --(next frame)--> WAITING_FOR_RESET
    WAITING_FOR_RESET --(ratio strictly below reset_threshold)--> NORMAL
    ... after the last split clears --> DONE (terminal)

Semantics contract:
- In NORMAL, a frame at or above the active split's threshold extends the
  consecutive streak; any miss resets it to zero. The split fires on the
  frame where the streak reaches ``detection.required_frames``; that frame
  reports state TRANSITION and yields the SplitEvent.
- In WAITING_FOR_RESET only a ratio strictly below ``detection.reset_threshold``
  clears the target. Ratios inside the band [reset_threshold, threshold) are
  hysteresis dead-zone: they neither clear nor re-trigger.
- One continuous transition therefore emits exactly one split.
- After the last split clears, the machine enters terminal DONE and ignores
  further frames until :meth:`reset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from autosplitter.config import Config, Split
from autosplitter.detection.color_detector import DetectionResult, build_detector


class State(Enum):
    NORMAL = auto()
    TRANSITION = auto()
    WAITING_FOR_RESET = auto()
    DONE = auto()


@dataclass(frozen=True)
class SplitEvent:
    """Emitted once when a split's debounce completes."""

    split_id: int
    name: str


class TransitionDetector:
    """Tracks the split sequence across a stream of per-frame detections."""

    def __init__(self, config: Config) -> None:
        self._config: Config = config
        self._index: int = 0
        self._state: State = State.NORMAL
        self._streak: int = 0

    @property
    def state(self) -> State:
        return self._state

    @property
    def current_split_index(self) -> int:
        """Zero-based index into config.splits."""
        return self._index

    @property
    def current_split(self) -> Split | None:
        """The split being waited for, or None once every split has fired."""
        if self._index >= len(self._config.splits):
            return None
        return self._config.splits[self._index]

    @property
    def finished(self) -> bool:
        return self._state is State.DONE

    def reset(self) -> None:
        """Restart from the first split in NORMAL state."""
        self._index = 0
        self._state = State.NORMAL
        self._streak = 0

    def feed_frame(self, frame: np.ndarray) -> SplitEvent | None:
        """Detect the active split's color on this frame, then advance."""
        split = self.current_split
        if split is None:
            return None
        result = build_detector(self._config, split).detect(frame)
        return self.feed_result(result)

    def feed_result(self, result: DetectionResult) -> SplitEvent | None:
        if self.finished or self.current_split is None:
            return None

        if self._state is State.TRANSITION:
            # The split fired on the previous frame; settle into waiting.
            self._state = State.WAITING_FOR_RESET

        if self._state is State.NORMAL:
            return self._feed_normal(result)
        return self._feed_waiting(result)

    def _feed_normal(self, result: DetectionResult) -> SplitEvent | None:
        split = self._config.splits[self._index]
        if result.ratio >= split.threshold:
            self._streak += 1
        else:
            self._streak = 0
            return None

        if self._streak < self._config.detection.required_frames:
            return None

        self._state = State.TRANSITION
        self._streak = 0
        return SplitEvent(split_id=split.id, name=split.name)

    def _feed_waiting(self, result: DetectionResult) -> SplitEvent | None:
        if result.ratio >= self._config.detection.reset_threshold:
            return None

        self._index += 1
        self._streak = 0
        self._state = (
            State.DONE if self._index >= len(self._config.splits) else State.NORMAL
        )
        return None
