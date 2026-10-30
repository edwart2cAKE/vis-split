"""Screen-evidence filter: turns noisy OCR readings into trusted comparisons.

Pure logic (no threads, sockets or OCR). A reading only counts after it
passes range, confidence, staleness and epoch checks, and after N consecutive
distinct readings resolve to the same split index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from autosplitter.config import OcrConfig, OcrPattern
from autosplitter.detection.element_reader import Reading


class EvidenceKind(Enum):
    NONE = auto()
    AGREE = auto()
    MISMATCH = auto()


@dataclass(frozen=True)
class EvidenceEvent:
    kind: EvidenceKind
    screen_idx: int | None = None
    text: str = ""
    confidence: float = 0.0


NO_EVIDENCE = EvidenceEvent(EvidenceKind.NONE)


MAX_DIGITS = 9  # longer digit runs are noise, never a split number


def compile_patterns(
    patterns: tuple[OcrPattern, ...],
) -> list[tuple[re.Pattern[str], int | None]]:
    return [(re.compile(p.regex), p.split_index) for p in patterns]


def resolve_index(
    patterns: list[tuple[re.Pattern[str], int | None]], text: str
) -> int | None:
    """Match text against patterns; return a 0-based split index or None."""
    for compiled, split_index in patterns:
        m = compiled.search(text)
        if m is None:
            continue
        if split_index is not None:
            return split_index
        for group in m.groups():
            if (
                group is not None
                and group.isascii()
                and group.isdecimal()
                and len(group) <= MAX_DIGITS
            ):
                return int(group) - 1
    return None


def compare(screen_idx: int, detector_idx: int) -> EvidenceKind:
    return EvidenceKind.AGREE if screen_idx == detector_idx else EvidenceKind.MISMATCH


class EvidenceFilter:
    """Feed one call per DISTINCT reading (see ``OcrWorker.take_result``)."""

    def __init__(self, cfg: OcrConfig, n_splits: int, fps: float) -> None:
        self._patterns: list[tuple[re.Pattern[str], int | None]] = compile_patterns(cfg.patterns)
        self._n_splits: int = n_splits
        self._max_age_frames: float = cfg.filter.max_age_seconds * fps
        self._min_conf: float = cfg.filter.min_confidence
        self._consensus: int = cfg.filter.consensus_frames
        self._jump: int = cfg.filter.jump_consensus_frames
        self._streak_idx: int | None = None
        self._streak: int = 0
        self._streak_epoch: int = -1

    def reset(self) -> None:
        self._streak_idx = None
        self._streak = 0

    def _reject(self) -> EvidenceEvent:
        self._streak_idx = None
        self._streak = 0
        return NO_EVIDENCE

    def screen_index(
        self, reading: Reading, now_frame_idx: int, machine_epoch: int
    ) -> int | None:
        """Pure trust check: the 0-based split index a reading shows, or None.

        Applies the epoch, age, confidence and label-set/range checks only; no
        consensus streak and no state change.
        """
        if reading.epoch != machine_epoch:
            return None
        if now_frame_idx - reading.frame_idx > self._max_age_frames:
            return None
        if not reading.confidence >= self._min_conf:
            return None
        idx = resolve_index(self._patterns, reading.value)
        if idx is None or not 0 <= idx < self._n_splits:
            return None
        return idx

    def feed(
        self, reading: Reading, now_frame_idx: int, machine_epoch: int, detector_idx: int
    ) -> EvidenceEvent:
        idx = self.screen_index(reading, now_frame_idx, machine_epoch)
        if idx is None:
            return self._reject()

        if idx == self._streak_idx and reading.epoch == self._streak_epoch:
            self._streak += 1
        else:
            self._streak_idx = idx
            self._streak_epoch = reading.epoch
            self._streak = 1

        need = self._consensus if idx in (detector_idx, detector_idx + 1) else self._jump
        if self._streak < need:
            return NO_EVIDENCE
        return EvidenceEvent(
            compare(idx, detector_idx), idx, reading.value, reading.confidence
        )
