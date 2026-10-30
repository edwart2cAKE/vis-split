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
- ``rollback_to`` / ``advance_to`` (LiveSplit Unsplit / skip-follow) land on an
  arbitrary index by entering WAITING_FOR_RESET with ``_transition_is_start``
  set ("settle without advance"): the target's color must read strictly below
  ``reset_threshold`` before NORMAL, so a lit color never double-fires.
- If start_transition is configured, ``reset()`` returns to WAITING_FOR_START
  instead of NORMAL, and the loop rearm watches for the start color again.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from autosplitter.config import Config, OcrUndo, Split
from autosplitter.detection.color_detector import (
    ColorDetector,
    DetectionResult,
    build_detector,
)
from autosplitter.detection.element_reader import Reading
from autosplitter.detection.evidence import (
    NO_EVIDENCE,
    EvidenceEvent,
    EvidenceFilter,
    EvidenceKind,
)
from autosplitter.detection.ocr_detector import OcrResult, OcrWorker, build_ocr_worker
from autosplitter.detection.undo_judge import UndoJudge, UndoVerdict, VerdictKind

_MAX_UNDO_EVENTS = 32  # bounded in case nobody drains (GUI)


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

    def __init__(
        self,
        config: Config,
        ocr: OcrWorker | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config: Config = config
        self._clock: Callable[[], float] = clock
        # Bumped on EVERY state or index change; OCR readings are stamped with
        # it at capture time so evidence from before a change is discarded.
        self.epoch: int = 0
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
        # without advance" for :meth:`rollback_to` and :meth:`advance_to`,
        # which reuse that path to land on an arbitrary index.
        self._transition_is_start: bool = False

        # OCR worker (optional, read-only screen evidence). An injected worker
        # is used as given (the caller starts it); otherwise build from config.
        self._ocr_worker: OcrWorker | None = ocr
        self._ocr_latest: OcrResult | None = None
        self.ocr_status: str = "ocr not configured"
        self._ocr_version: int = 0
        self._evidence: EvidenceEvent = NO_EVIDENCE
        self._evidence_epoch: int = -1
        self._evidence_frame: int = 0
        self._frame_now: int = 0
        self._evidence_ttl_frames: int = 1
        if config.ocr is not None:
            self._evidence_ttl_frames = max(
                1, math.ceil(config.ocr.filter.max_age_seconds * config.capture.fps)
            )
        self._evidence_filter: EvidenceFilter | None = None
        if config.ocr is not None and config.ocr.enabled:
            self._evidence_filter = EvidenceFilter(
                config.ocr, len(config.splits), float(config.capture.fps)
            )
            if ocr is not None:
                self.ocr_status = "ocr injected"
            else:
                worker, self.ocr_status = build_ocr_worker(config.ocr)
                self._ocr_worker = worker
                if worker is not None:
                    worker.start()
        elif ocr is not None:
            self.ocr_status = "ocr injected"
            self._evidence_filter = None

        # Dry-run undo judge (only with ocr.undo + a live evidence filter).
        # Read-only: it never touches state, index or epoch.
        # The OcrUndo is captured here and never re-read from self._config, so
        # a hot config reload (which does not rebuild the judge) cannot crash
        # or half-apply: ocr.undo changes need a restart.
        self._undo_cfg: OcrUndo | None = None
        self._judge: UndoJudge | None = None
        if (
            config.ocr is not None
            and config.ocr.undo is not None
            and self._evidence_filter is not None
            and self._ocr_worker is not None
        ):
            self._undo_cfg = config.ocr.undo
            self._judge = UndoJudge(
                config.ocr.undo,
                max(config.ocr.undo.min_confidence, config.ocr.engine.min_confidence),
                clock,
            )
        # apply_to_last_split: open a window on TRANSITION -> DONE (captured once).
        self._undo_last_on: bool = (
            self._undo_cfg is not None and self._undo_cfg.apply_to_last_split
        )
        self._undo_open: bool = False
        self._undo_epoch: int = -1
        self._undo_target: int | None = None
        self._undo_target_epoch: int = -1
        self._undo_events: list[UndoVerdict] = []
        self._fire_time: float = 0.0
        self._fire_baseline_ok: bool = False
        self._agree_idx: int = -1
        self._agree_time: float = 0.0
        self._agree_epoch: int = -1
        # Late-HUD recovery (ocr.mode: undo): captured once, like the judge's
        # OcrUndo. Armed only by apply_undo(..., arm_recovery=True).
        self._regret_enabled: bool = (
            self._undo_cfg is not None
            and config.ocr is not None
            and config.ocr.mode == "undo"
            and self._undo_cfg.recover_late_hud
        )
        self._regret_seconds: float = (
            self._undo_cfg.recover_seconds if self._undo_cfg is not None else 0.0
        )
        self._regret_min_conf: float = (
            max(self._undo_cfg.min_confidence, config.ocr.engine.min_confidence)
            if self._undo_cfg is not None and config.ocr is not None
            else 0.0
        )
        # Own consensus for the (timer-writing) recovery: every read of the
        # streak must clear the UNDO confidence floor, not just the last one.
        self._regret_need: int = (
            config.ocr.filter.consensus_frames if config.ocr is not None else 1
        )
        self._regret_streak: int = 0
        self._regret_streak_t0: float = 0.0
        # ...and the streak must SPAN this much wall time (undo.confirm_seconds).
        self._regret_span: float = (
            self._undo_cfg.confirm_seconds if self._undo_cfg is not None else 0.0
        )
        self._regret_armed: bool = False
        self._regret_epoch: int = -1
        self._regret_screen: int = -1
        self._regret_t0: float = 0.0
        self._regret_hit: float | None = None

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

    def reset(self, *, settle: bool = False) -> None:
        """Restart from the beginning (start transition or first split).

        ``settle=True`` (follow_timer: LiveSplit's timer was reset) without a
        start gate lands in WAITING_FOR_RESET at index 0 with "settle without
        advance", so a still-lit split-0 color cannot fire into an idle
        timer; with a start gate it is identical to the default.
        """
        self.epoch += 1
        self._index = 0
        self._streak = 0
        self._transition_is_start = False
        if self._config.start_transition is not None:
            self._state = State.WAITING_FOR_START
        elif settle:
            self._transition_is_start = True
            self._state = State.WAITING_FOR_RESET
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
        self.epoch += 1
        self._index = index
        self._streak = 0
        if self._state is not State.NORMAL:
            self._transition_is_start = True
            self._state = State.WAITING_FOR_RESET
        return True

    def apply_undo(self, index: int, *, arm_recovery: bool = True) -> bool:
        """Roll back because the OCR undo executor unsplit LiveSplit.

        Same guards as :meth:`rollback_to` (refused in TRANSITION /
        WAITING_FOR_START, for out-of-range indexes and for ``index >=
        expected_ls_index``), but ALWAYS ends in the settle state
        (WAITING_FOR_RESET, "settle without advance") even from NORMAL, so a
        transition colour that is still lit must read strictly below
        ``reset_threshold`` before the target split is watched again. Closes
        the judge window, drops the confirmed target and the AGREE latch.
        ``arm_recovery`` starts the late-HUD watch (mode undo only).
        """
        if not self.rollback_to(index):
            return False
        self._transition_is_start = True
        self._state = State.WAITING_FOR_RESET
        self._undo_target = None
        self._undo_open = False
        if self._judge is not None:
            self._judge.discard()
        self._agree_idx = -1
        self._regret_armed = False
        self._regret_hit = None
        if arm_recovery and self._regret_enabled:
            self._regret_armed = True
            self._regret_streak = 0
            self._regret_epoch = self.epoch
            self._regret_screen = index + 1
            self._regret_t0 = self._clock()
        return True

    def dismiss_undo_target(self) -> None:
        """Consume the confirmed UNDO verdict (one executor attempt per event)."""
        self._undo_target = None

    @property
    def recovery_target(self) -> int | None:
        """0-based index to re-advance to when the HUD showed up late, else None.

        Valid only in the epoch the recovery was armed in (the settle clear
        is the one allowed epoch bump) and while the machine is in NORMAL.
        """
        if (
            self._regret_hit is None
            or self._regret_epoch != self.epoch
            or self._state is not State.NORMAL
        ):
            return None
        return self._regret_screen

    @property
    def recovery_lag(self) -> float:
        """Seconds between the undo and the late HUD read (0.0 if none)."""
        if self._regret_hit is None:
            return 0.0
        return self._regret_hit - self._regret_t0

    def dismiss_recovery(self) -> None:
        """Consume/cancel the late-HUD watch (at most one recovery per undo)."""
        self._regret_armed = False
        self._regret_hit = None

    def advance_to(self, index: int) -> bool:
        """Jump forward because LiveSplit is ahead (skip / manual split).

        Allowed in WAITING_FOR_START / NORMAL / WAITING_FOR_RESET; refused in
        TRANSITION and DONE, for ``index <= expected_ls_index`` (any
        non-negative index in WAITING_FOR_START) and for ``index`` beyond the
        split count. ``index == len(splits)`` ends the run (DONE); otherwise
        the machine settles in WAITING_FOR_RESET without advancing, so the
        target split's color must read strictly below ``reset_threshold``
        before it is watched (no immediate re-fire; a pending +1 advance is
        cancelled).
        """
        n = len(self._config.splits)
        if self._state not in (
            State.WAITING_FOR_START,
            State.NORMAL,
            State.WAITING_FOR_RESET,
        ):
            return False
        floor = 0 if self._state is State.WAITING_FOR_START else self.expected_ls_index + 1
        if index < floor or index > n:
            return False
        self.epoch += 1
        self._index = index
        self._streak = 0
        if index >= n:
            self._transition_is_start = False
            self._state = State.DONE
        else:
            self._transition_is_start = True
            self._state = State.WAITING_FOR_RESET
        return True

    def clear_streak(self) -> None:
        """Drop the debounce streak without changing state."""
        self._streak = 0

    @property
    def ocr_text(self) -> str | None:
        """Latest raw OCR text (or None if OCR not configured / no reading yet)."""
        if self._ocr_latest is None:
            return None
        return self._ocr_latest.text or None

    @property
    def ocr_evidence(self) -> EvidenceEvent:
        """Latest AGREE/MISMATCH verdict, or NO_EVIDENCE once stale.

        Valid only while the machine is still in the epoch it was computed in
        (any state/index change discards it) and for ``filter.max_age_seconds`` of frames.
        """
        if self._evidence.kind is EvidenceKind.NONE:
            return NO_EVIDENCE
        if self._evidence_epoch != self.epoch:
            return NO_EVIDENCE
        if self._frame_now - self._evidence_frame > self._evidence_ttl_frames:
            return NO_EVIDENCE
        return self._evidence

    @property
    def ocr_worker_status(self) -> str | None:
        """Worker health ('ok' | 'error: ..' | 'failed') or None without a worker."""
        if self._ocr_worker is None:
            return None
        return self._ocr_worker.status

    @property
    def undo_judge_active(self) -> bool:
        """True when ``ocr.undo`` is configured and a judge exists."""
        return self._judge is not None

    @property
    def undo_pending(self) -> bool:
        """An undo event window is open (and still in its own epoch)."""
        return self._undo_open and self._undo_epoch == self.epoch

    @property
    def undo_target(self) -> int | None:
        """0-based screen index of a confirmed UNDO verdict, else None.

        Valid only while the machine is still in the epoch it was confirmed in.
        """
        if self._undo_target is None or self._undo_target_epoch != self.epoch:
            return None
        return self._undo_target

    def take_undo_event(self) -> UndoVerdict | None:
        """Pop the oldest unreported judge verdict (for logging), or None."""
        if not self._undo_events:
            return None
        return self._undo_events.pop(0)

    def _undo_note_fire(self) -> None:
        """Record fire time and whether OCR vouched for the split beforehand."""
        undo = self._undo_cfg
        assert self._judge is not None and undo is not None
        now = self._clock()
        self._fire_time = now
        self._fire_baseline_ok = (not undo.require_baseline) or (
            self._agree_idx == self._index
            and self._agree_epoch == self.epoch
            and now - self._agree_time <= undo.baseline_seconds
        )

    def _undo_open_window(self, fired_index: int, last: bool = False) -> None:
        judge = self._judge
        assert judge is not None
        self._undo_target = None
        verdict = judge.open(
            fired_index, self._fire_time, self.epoch, self._fire_baseline_ok, self._clock(),
            last,
        )
        self._undo_open = judge.is_open
        self._undo_epoch = self.epoch
        if verdict is not None:
            self._undo_record(verdict)

    def _undo_record(self, verdict: UndoVerdict) -> None:
        judge = self._judge
        assert judge is not None
        self._undo_open = judge.is_open
        if verdict.kind is VerdictKind.UNDO:
            self._undo_target = verdict.screen_idx
            self._undo_target_epoch = self.epoch
        if len(self._undo_events) >= _MAX_UNDO_EVENTS:
            del self._undo_events[0]
        self._undo_events.append(verdict)

    def check_ocr_resync(self, frame: np.ndarray, frame_idx: int) -> None:
        """Feed the OCR worker and update read-only screen evidence.

        Called once per frame from the hot loop. Idle cost: one modulo and one
        unlocked int compare. NEVER changes ``_index`` or the state
        (verify_only): the verdict is exposed through :attr:`ocr_evidence`.
        """
        worker = self._ocr_worker
        if worker is None:
            return
        self._frame_now = frame_idx
        if self._regret_armed and (
            self._regret_epoch != self.epoch
            or self._clock() - self._regret_t0 > self._regret_seconds
        ):
            self.dismiss_recovery()
        judge = self._judge
        if self._undo_open and judge is not None:
            closed = judge.check(self.epoch)
            if closed is not None:
                self._undo_record(closed)
        worker.submit_frame(frame, frame_idx, self.epoch)
        if worker.version == self._ocr_version:
            return
        result = worker.take_result()
        if result is None:
            return
        self._ocr_version = worker.taken_version
        self._ocr_latest = result
        flt = self._evidence_filter
        if flt is None:
            return
        if self._state is not State.NORMAL:
            if self._state is State.DONE and self._undo_open and judge is not None:
                done_reading = Reading(
                    value=result.text,
                    confidence=result.confidence,
                    frame_idx=result.frame_number,
                    epoch=result.epoch,
                    t_mono=result.t_mono,
                )
                closing = judge.feed(
                    flt.screen_index(done_reading, frame_idx, self.epoch),
                    done_reading.confidence,
                )
                if closing is not None:
                    self._undo_record(closing)
            return
        reading = Reading(
            value=result.text,
            confidence=result.confidence,
            frame_idx=result.frame_number,
            epoch=result.epoch,
            t_mono=result.t_mono,
        )
        if self._regret_armed and self._regret_hit is None:
            shown = flt.screen_index(reading, frame_idx, self.epoch)
            if (
                shown == self._regret_screen
                and reading.confidence >= self._regret_min_conf
                and self._regret_epoch == self.epoch
            ):
                now = self._clock()
                if self._regret_streak == 0:
                    self._regret_streak_t0 = now
                self._regret_streak += 1
                if (
                    self._regret_streak >= self._regret_need
                    and now - self._regret_streak_t0 >= self._regret_span
                ):
                    self._regret_hit = now
            else:
                self._regret_streak = 0
        event = flt.feed(reading, frame_idx, self.epoch, self._index)
        if event.kind is not EvidenceKind.NONE:
            self._evidence = event
            self._evidence_epoch = self.epoch
            self._evidence_frame = frame_idx
            if judge is not None and event.kind is EvidenceKind.AGREE:
                self._agree_idx = self._index
                self._agree_epoch = self.epoch
                self._agree_time = self._clock()
        if self._undo_open and judge is not None:
            verdict = judge.feed(
                flt.screen_index(reading, frame_idx, self.epoch), reading.confidence
            )
            if verdict is not None:
                self._undo_record(verdict)

    def _resync_index(self, new_index: int) -> None:
        """Force-set the split index when OCR disagrees with color detection."""
        if self._state in (State.TRANSITION, State.WAITING_FOR_RESET, State.DONE):
            return
        if not 0 <= new_index < len(self._config.splits):
            return
        if new_index == self._index:
            return
        self.epoch += 1
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
            self.epoch += 1
            if self._transition_is_start:
                self._state = State.WAITING_FOR_RESET
            elif self._index >= len(self._config.splits) - 1:
                # Exhaust the index so DONE keeps its invariant that
                # current_split/current_detector are None (HUD, --probe,
                # and --loop all rely on it).
                self._index = len(self._config.splits)
                self._state = State.DONE
                if self._judge is not None and self._undo_last_on:
                    # AFTER the epoch bump above, so the window does not
                    # self-cancel; the DONE entry is its "clear" time.
                    self._undo_open_window(self._index - 1, True)
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

        self.epoch += 1
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

        if self._judge is not None:
            # Before the epoch bump: the baseline AGREE must come from this
            # same NORMAL stretch (any reset/rollback/advance/clear bumped it).
            self._undo_note_fire()
        self.epoch += 1
        self._transition_is_start = False
        self._state = State.TRANSITION
        self._streak = 0
        return SplitEvent(split_id=split.id, name=split.name)

    def _feed_waiting(self, result: DetectionResult) -> SplitEvent | None:
        if result.ratio >= self._config.detection.reset_threshold:
            return None

        self._streak = 0
        self.epoch += 1
        if self._transition_is_start:
            # The start gate's color cleared: begin watching splits[0] with
            # the index unchanged (the gate precedes the first split).
            self._transition_is_start = False
            self._state = State.NORMAL
            if self._regret_armed and self._regret_epoch == self.epoch - 1:
                self._regret_epoch = self.epoch  # the undo's own settle clear
            return None

        # The final split never reaches WAITING_FOR_RESET (it goes straight
        # to DONE from TRANSITION), so clearing here always yields a next
        # split to watch.
        fired = self._index
        self._index += 1
        self._state = State.NORMAL
        if self._judge is not None:
            self._undo_open_window(fired)
        return None
