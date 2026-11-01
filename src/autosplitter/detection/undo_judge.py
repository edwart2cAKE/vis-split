"""Dry-run "go then undo" judge (pure logic: no sockets, threads or OCR).

The colour detector fires a split on the colour transition. A death looks
identical, so after the split ``k`` fired the detector sits at ``D = k + 1``
while the HUD may still show level ``k`` (screen index ``D - 1``). The judge
watches one trusted HUD reading per DISTINCT OCR result and decides, per event
window, whether the HUD advanced (GENUINE) or stayed behind long enough and
stably enough that the split would be undone (UNDO).

The HUD changes AFTER the colour event with a game-dependent lag, so a premature
UNDO during that lag is the main risk: ``settle_seconds`` must exceed the
largest observed lag (the GENUINE verdict logs the measured lag for tuning).

The judge never acts. It returns verdicts; the caller logs them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

from autosplitter.config import OcrUndo


class VerdictKind(Enum):
    GENUINE = auto()      # HUD advanced to D: never undo in this window
    UNDO = auto()         # HUD stayed at D-1, stable past settle
    WARN = auto()         # trusted reading neither D-1 nor D (once per pair)
    CANCELLED = auto()    # epoch changed (new transition, rollback, reset, ...)
    EXPIRED = auto()      # window_seconds elapsed without a verdict
    NO_BASELINE = auto()  # no pre-split AGREE: window never opened
    FINISHED = auto()     # last-split window: the HUD is gone, the run is finished


@dataclass(frozen=True)
class UndoVerdict:
    kind: VerdictKind
    split_index: int                # k: the split that fired (0-based)
    screen_idx: int | None = None   # trusted S that triggered the verdict
    elapsed: float = 0.0            # seconds since the fire
    reads: int = 0                  # streak length (UNDO)


class UndoJudge:
    """One event window at a time; at most one closing verdict per window."""

    def __init__(
        self,
        cfg: OcrUndo,
        min_confidence: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg: OcrUndo = cfg
        self._min_conf: float = min_confidence
        self._clock: Callable[[], float] = clock
        self._open: bool = False
        self._k: int = 0
        self._epoch: int = -1
        self._fire_t: float = 0.0
        self._clear_t: float = 0.0
        self._streak: int = 0
        self._streak_t0: float = 0.0
        self._warned: set[tuple[int, int]] = set()
        # Last-split windows (apply_to_last_split): extra settle, and a "HUD
        # gone" streak of blank/untrusted reads that closes as FINISHED.
        self._last: bool = False
        self._gone: int = 0
        self._gone_t0: float = 0.0

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def split_index(self) -> int:
        return self._k

    @property
    def epoch(self) -> int:
        return self._epoch

    def open(
        self,
        split_index: int,
        fire_time: float,
        epoch: int,
        baseline_ok: bool,
        clear_time: float | None = None,
        last: bool = False,
    ) -> UndoVerdict | None:
        """Start a window for the split that just fired.

        Without a baseline no window opens and NO_BASELINE is returned. Any
        previous window is silently replaced (the caller closes via ``check``).
        """
        self._streak = 0
        self._gone = 0
        self._last = last
        self._warned.clear()
        if not baseline_ok:
            self._open = False
            return UndoVerdict(VerdictKind.NO_BASELINE, split_index)
        self._open = True
        self._k = split_index
        self._epoch = epoch
        self._fire_t = fire_time
        self._clear_t = fire_time if clear_time is None else clear_time
        return None

    def discard(self) -> None:
        """Close any open window silently (the executor rolled the machine back)."""
        self._open = False
        self._streak = 0
        self._gone = 0

    def _close(self, kind: VerdictKind, screen: int | None, elapsed: float) -> UndoVerdict:
        reads = self._gone if kind is VerdictKind.FINISHED else self._streak
        self._open = False
        self._streak = 0
        self._gone = 0
        return UndoVerdict(kind, self._k, screen, elapsed, reads)

    def check(self, epoch: int) -> UndoVerdict | None:
        """Per-frame housekeeping: cancel on epoch change, expire on timeout."""
        if not self._open:
            return None
        elapsed = self._clock() - self._fire_t
        if epoch != self._epoch:
            return self._close(VerdictKind.CANCELLED, None, elapsed)
        if elapsed >= self._cfg.window_seconds:
            return self._close(VerdictKind.EXPIRED, None, elapsed)
        return None

    def feed(self, screen_idx: int | None, confidence: float) -> UndoVerdict | None:
        """One trusted screen index (None = untrusted) per DISTINCT OCR result."""
        if not self._open:
            return None
        now = self._clock()
        elapsed = now - self._fire_t
        if elapsed >= self._cfg.window_seconds:
            return self._close(VerdictKind.EXPIRED, None, elapsed)
        settle = self._cfg.settle_seconds + (
            self._cfg.last_split_extra_settle_seconds if self._last else 0.0
        )
        if screen_idx is None or not confidence >= self._min_conf:
            self._streak = 0
            if self._last and elapsed >= settle and now - self._clear_t >= self._cfg.post_clear_seconds:
                # Blank / unreadable HUD after the guard: the level number is gone.
                if self._gone == 0:
                    self._gone_t0 = now
                self._gone += 1
                if (
                    self._gone >= self._cfg.confirm_reads
                    and now - self._gone_t0 >= self._cfg.confirm_seconds
                ):
                    return self._close(VerdictKind.FINISHED, None, elapsed)
            else:
                self._gone = 0
            return None
        self._gone = 0
        d = self._k + 1
        if screen_idx == d:
            return self._close(VerdictKind.GENUINE, screen_idx, elapsed)
        if screen_idx == d - 1:
            if (
                elapsed < settle
                or now - self._clear_t < self._cfg.post_clear_seconds
            ):
                self._streak = 0
                return None  # HUD-lag guard
            if self._streak == 0:
                self._streak_t0 = now
            self._streak += 1
            if (
                self._streak >= self._cfg.confirm_reads
                and now - self._streak_t0 >= self._cfg.confirm_seconds
            ):
                return self._close(VerdictKind.UNDO, screen_idx, elapsed)
            return None
        self._streak = 0
        pair = (screen_idx, d)
        if pair in self._warned:
            return None
        self._warned.add(pair)
        return UndoVerdict(VerdictKind.WARN, self._k, screen_idx, elapsed)
