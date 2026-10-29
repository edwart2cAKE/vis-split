"""Transition state machine (spec sections 7-10).

State flow (with optional start_transition):

    WAITING_FOR_START --(streak of required_frames at/above threshold)--> TRANSITION
    TRANSITION --(next frame)--> WAITING_FOR_RESET --(clears)--> NORMAL (index unchanged)
    NORMAL --(streak of required_frames at/above threshold)--> TRANSITION
    TRANSITION --(next frame, not last split)--> WAITING_FOR_RESET
    TRANSITION --(next frame, last split)--> DONE (terminal)
    WAITING_FOR_RESET --(ratio strictly below reset_threshold)--> NORMAL (index advances)

If no start_transition is configured, the machine starts in NORMAL directly.

Semantics contract:
- In WAITING_FOR_START / NORMAL, a frame at or above the active split's threshold
  extends the consecutive streak; any miss resets it to zero. The split fires on
  the frame where the streak reaches ``detection.required_frames``; that frame
  reports state TRANSITION and yields the SplitEvent (or StartEvent for start).
- In WAITING_FOR_RESET only a ratio strictly below ``detection.reset_threshold``
  clears the target. Ratios inside the band [reset_threshold, threshold) are
  hysteresis dead-zone: they neither clear nor re-trigger.
- One continuous transition therefore emits exactly one split.
- The final split skips the reset wait entirely: on the next frame the
  machine enters terminal DONE. Hysteresis exists to shield a later split
  from re-triggering on the same color; after the last one there is nothing
  left to protect, and a persistent victory screen must not stall the run.
- Once in DONE the machine ignores further frames until :meth:`reset`.
- If start_transition is configured, ``reset()`` returns to WAITING_FOR_START
  instead of NORMAL, and the loop rearm watches for the start color again.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from autosplitter.config import Config, Split
from autosplitter.detection.color_detector import (
    ColorDetector,
    DetectionResult,
    build_detector,
)
from autosplitter.detection.ocr_detector import OcrDetector, OcrResult, OcrWorker


class State(Enum):
    WAITING_FOR_START = auto()
    NORMAL = auto()
    TRANSITION = auto()
    WAITING_FOR_RESET = auto()
    DONE = auto()


@dataclass(frozen=True)
class SplitEvent:
    """Emitted once when a split's debounce completes."""

    split_id: int
    name: str


@dataclass(frozen=True)
class StartEvent:
    """Emitted once when the start transition's debounce completes."""

    name: str


class TransitionDetector:
    """Tracks the split sequence across a stream of per-frame detections."""

    def __init__(self, config: Config) -> None:
        self._config: Config = config
        # Built once: per-frame detector construction would dominate the
        # hot loop at capture rate (~60 fps).
        self._detectors: list[ColorDetector] = [
            build_detector(config, split) for split in config.splits
        ]
        # Start transition detector (if configured)
        self._start_detector: ColorDetector | None = None
        if config.start_transition is not None:
            self._start_detector = build_detector(config, config.start_transition)

        # State machine initialization
        if config.start_transition is not None:
            self._state: State = State.WAITING_FOR_START
        else:
            self._state: State = State.NORMAL
        self._index: int = 0
        self._streak: int = 0
        # True while TRANSITION was entered from the start gate (vs a split):
        # decides whether settling waits for the gate color and keeps the
        # split index, or advances / ends the run. It also means "settle
        # without advance" for :meth:`rollback_to`, which reuses that path.
        self._transition_is_start: bool = False

        # OCR worker (optional — augments color detection by re-syncing index)
        self._ocr_worker: OcrWorker | None = None
        self._ocr_latest: OcrResult | None = None
        if config.ocr is not None:
            ocr_detector = OcrDetector(
                config.ocr.roi, config.ocr.engine, config.ocr.patterns
            )
            self._ocr_worker = OcrWorker(ocr_detector, config.ocr.poll_interval_frames)
            self._ocr_worker.start()

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
    def current_detector(self) -> ColorDetector | None:
        """Detector for the split being waited for (None when finished)."""
        if self._index >= len(self._detectors):
            return None
        return self._detectors[self._index]

    @property
    def first_detector(self) -> ColorDetector | None:
        """Detector for the first split, even while finished.

        ``--loop`` uses it to watch for the quiet period that rearms the
        machine after DONE.
        """
        if not self._detectors:
            return None
        return self._detectors[0]

    @property
    def start_detector(self) -> ColorDetector | None:
        """Detector for the start transition (if configured)."""
        return self._start_detector

    @property
    def quiet_frames_needed(self) -> int:
        """Frame-counted equivalent of ``detection.restart_quiet_seconds``.

        Debounce is frame-based everywhere, so the ``--loop`` quiet period
        counts frames at capture rate too; zero seconds still requires one
        quiet frame.
        """
        seconds = self._config.detection.restart_quiet_seconds
        return max(1, round(seconds * self._config.capture.fps))

    @property
    def finished(self) -> bool:
        return self._state is State.DONE

    @property
    def waiting_for_start(self) -> bool:
        return self._state is State.WAITING_FOR_START

    def reset(self) -> None:
        """Restart from the beginning (start transition or first split)."""
        self._index = 0
        self._streak = 0
        self._transition_is_start = False
        if self._config.start_transition is not None:
            self._state = State.WAITING_FOR_START
        else:
            self._state = State.NORMAL

    @property
    def expected_ls_index(self) -> int:
        """Split index LiveSplit should report given what the detector fired.

        A split in flight (TRANSITION / WAITING_FOR_RESET) has already been
        sent but ``_index`` only advances on clear, so LiveSplit is one ahead.
        Start-gate settling and DONE (index exhausted) need no adjustment.
        """
        if (
            self._state in (State.TRANSITION, State.WAITING_FOR_RESET)
            and not self._transition_is_start
        ):
            return self._index + 1
        return self._index

    def rollback_to(self, index: int) -> bool:
        """Roll back after the user undid split(s) in LiveSplit.

        Acts only in NORMAL / WAITING_FOR_RESET / DONE and only for an index
        below :attr:`expected_ls_index`. From WAITING_FOR_RESET / DONE the
        machine re-enters WAITING_FOR_RESET without advancing, so a color
        that is still lit must drop below ``reset_threshold`` first (no
        duplicate split) and any pending advance is cancelled.
        """
        if self._state not in (State.NORMAL, State.WAITING_FOR_RESET, State.DONE):
            return False
        if not 0 <= index < len(self._config.splits):
            return False
        if index >= self.expected_ls_index:
            return False
        self._index = index
        self._streak = 0
        if self._state is not State.NORMAL:
            self._transition_is_start = True
            self._state = State.WAITING_FOR_RESET
        return True

    @property
    def ocr_text(self) -> str | None:
        """Latest raw OCR text (or None if OCR not configured / no reading yet)."""
        if self._ocr_latest is None:
            return None
        return self._ocr_latest.text or None

    def check_ocr_resync(self, frame: np.ndarray, frame_idx: int) -> None:
        """Submit frame to OCR worker and re-sync index if needed.

        Called once per frame from the hot loop.  Cost: <3µs (two lock ops).
        """
        if self._ocr_worker is None:
            return
        self._ocr_worker.submit_frame(frame, frame_idx)
        result = self._ocr_worker.poll_result()
        if result is None:
            return
        self._ocr_latest = result
        if result.split_index is None:
            return
        if result.frame_number != frame_idx:
            return
        self._resync_index(result.split_index)

    def _resync_index(self, new_index: int) -> None:
        """Force-set the split index when OCR disagrees with color detection."""
        if self._state in (State.TRANSITION, State.WAITING_FOR_RESET, State.DONE):
            return
        if not 0 <= new_index < len(self._config.splits):
            return
        if new_index == self._index:
            return
        self._index = new_index
        self._streak = 0

    def shutdown_ocr(self) -> None:
        """Stop the OCR worker thread."""
        if self._ocr_worker is not None:
            self._ocr_worker.close()

    def reload_config(self, new_cfg: Config) -> None:
        """Hot-swap color/threshold/ROI/detection params without restart.

        Rebuilds color detectors and start detector. Updates config reference
        so detection params (required_frames, reset_threshold, etc.) are read
        from the new config on the next frame.

        Does NOT handle: adding/removing splits, adding/removing start_transition,
        OCR config changes, or capture/livesplit/record changes.
        """
        # Rebuild color detectors for all splits
        self._detectors = [build_detector(new_cfg, s) for s in new_cfg.splits]

        # Rebuild start detector if it exists (modify only, not add/remove)
        if new_cfg.start_transition is not None and self._start_detector is not None:
            self._start_detector = build_detector(new_cfg, new_cfg.start_transition)

        # Update config reference — detection params read per-frame from here
        self._config = new_cfg

    def feed_frame(self, frame: np.ndarray) -> SplitEvent | StartEvent | None:
        """Detect the active split's color on this frame, then advance."""
        if self._state is State.WAITING_FOR_START:
            detector = self._start_detector
        else:
            detector = self.current_detector
        if detector is None:
            return None
        return self.feed_result(detector.detect(frame))

    def feed_result(self, result: DetectionResult) -> SplitEvent | StartEvent | None:
        if self.finished:
            return None

        # If we're waiting for start, current_split is None but that's expected
        if self._state is not State.WAITING_FOR_START and self.current_split is None:
            return None

        if self._state is State.TRANSITION:
            # Something fired on the previous frame and must settle before
            # the next target is watched. Three origins, three destinations:
            # - start gate: wait for ITS color to clear, then watch
            #   splits[0] (the index must NOT advance — the start precedes
            #   the first split);
            # - last split: DONE immediately. Hysteresis exists to shield a
            #   LATER target from re-triggering on the same color; after the
            #   final split there is none, so a persistent victory screen
            #   must not stall the run in the dead zone forever;
            # - any other split: WAITING_FOR_RESET as usual.
            if self._transition_is_start:
                self._state = State.WAITING_FOR_RESET
            elif self._index >= len(self._config.splits) - 1:
                # Exhaust the index so DONE keeps its invariant that
                # current_split/current_detector are None (HUD, --probe,
                # and --loop all rely on it).
                self._index = len(self._config.splits)
                self._state = State.DONE
                # Done on this very frame: do not fall through to the
                # waiting handlers below.
                return None
            else:
                self._state = State.WAITING_FOR_RESET

        if self._state is State.WAITING_FOR_START:
            return self._feed_start(result)
        if self._state is State.NORMAL:
            return self._feed_normal(result)
        return self._feed_waiting(result)

    def _feed_start(self, result: DetectionResult) -> StartEvent | None:
        assert self._config.start_transition is not None
        split = self._config.start_transition
        if result.ratio >= split.threshold:
            self._streak += 1
        else:
            self._streak = 0
            return None

        if self._streak < self._config.detection.required_frames:
            return None

        self._transition_is_start = True
        self._state = State.TRANSITION
        self._streak = 0
        return StartEvent(name=split.name)

    def _feed_normal(self, result: DetectionResult) -> SplitEvent | None:
        split = self._config.splits[self._index]
        if result.ratio >= split.threshold:
            self._streak += 1
        else:
            self._streak = 0
            return None

        if self._streak < self._config.detection.required_frames:
            return None

        self._transition_is_start = False
        self._state = State.TRANSITION
        self._streak = 0
        return SplitEvent(split_id=split.id, name=split.name)

    def _feed_waiting(self, result: DetectionResult) -> SplitEvent | None:
        if result.ratio >= self._config.detection.reset_threshold:
            return None

        self._streak = 0
        if self._transition_is_start:
            # The start gate's color cleared: begin watching splits[0] with
            # the index unchanged (the gate precedes the first split).
            self._transition_is_start = False
            self._state = State.NORMAL
            return None

        # The final split never reaches WAITING_FOR_RESET (it goes straight
        # to DONE from TRANSITION), so clearing here always yields a next
        # split to watch.
        self._index += 1
        self._state = State.NORMAL
        return None
