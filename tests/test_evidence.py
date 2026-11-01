"""EvidenceFilter: consensus, staleness, epoch and range rules (pure logic)."""


from __future__ import annotations

from autosplitter.config import OcrConfig, OcrFilter, OcrPattern, Roi
from autosplitter.detection.element_reader import Reading
from autosplitter.detection.evidence import (
    EvidenceEvent,
    EvidenceFilter,
    EvidenceKind,
    compare,
)

FPS = 100.0


def make_filter(
    n_splits: int = 10,
    consensus: int = 3,
    jump: int = 6,
    max_age: float = 0.5,
    min_conf: float = 60.0,
    patterns: tuple[OcrPattern, ...] | None = None,
) -> EvidenceFilter:
    if patterns is None:
        patterns = (OcrPattern(regex="(\\d+)"),)
    cfg = OcrConfig(
        roi=Roi(x=0.0, y=0.0, width=0.1, height=0.1),
        patterns=patterns,
        filter=OcrFilter(
            consensus_frames=consensus,
            jump_consensus_frames=jump,
            max_age_seconds=max_age,
            min_confidence=min_conf,
        ),
    )
    return EvidenceFilter(cfg, n_splits, FPS)


def rd(value: str, conf: float = 90.0, frame: int = 100, epoch: int = 0) -> Reading:
    return Reading(value=value, confidence=conf, frame_idx=frame, epoch=epoch, t_mono=0.0)


def feed(
    f: EvidenceFilter, r: Reading, now: int = 100, epoch: int = 0, det: int = 2
) -> EvidenceEvent:
    return f.feed(r, now, epoch, det)


class TestConsensus:
    def test_agree_needs_n(self) -> None:
        f = make_filter()
        assert feed(f, rd("3")).kind is EvidenceKind.NONE
        assert feed(f, rd("3")).kind is EvidenceKind.NONE
        ev = feed(f, rd("3"))
        assert ev.kind is EvidenceKind.AGREE
        assert ev.screen_idx == 2

    def test_n_minus_one_is_none(self) -> None:
        f = make_filter(consensus=5)
        for _ in range(4):
            assert feed(f, rd("3")).kind is EvidenceKind.NONE
        assert feed(f, rd("3")).kind is EvidenceKind.AGREE

    def test_differing_value_restarts(self) -> None:
        f = make_filter()
        feed(f, rd("3"))
        feed(f, rd("3"))
        assert feed(f, rd("4")).kind is EvidenceKind.NONE  # restart at 1 (idx 3)
        assert feed(f, rd("3")).kind is EvidenceKind.NONE  # restart at 1 again
        assert feed(f, rd("3")).kind is EvidenceKind.NONE
        assert feed(f, rd("3")).kind is EvidenceKind.AGREE

    def test_rejected_reading_does_not_extend(self) -> None:
        f = make_filter()
        feed(f, rd("3"))
        feed(f, rd("3"))
        assert feed(f, rd("")).kind is EvidenceKind.NONE
        assert feed(f, rd("3")).kind is EvidenceKind.NONE  # streak restarted, 1
        assert feed(f, rd("3")).kind is EvidenceKind.NONE
        assert feed(f, rd("3")).kind is EvidenceKind.AGREE

    def test_low_confidence_does_not_extend(self) -> None:
        f = make_filter()
        feed(f, rd("3"))
        feed(f, rd("3"))
        assert feed(f, rd("3", conf=10.0)).kind is EvidenceKind.NONE
        assert feed(f, rd("3")).kind is EvidenceKind.NONE

    def test_epoch_change_restarts_streak(self) -> None:
        f = make_filter()
        feed(f, rd("3"))
        feed(f, rd("3"))
        # machine moved to epoch 1; next readings carry the new epoch
        assert feed(f, rd("3", epoch=1), epoch=1).kind is EvidenceKind.NONE
        assert feed(f, rd("3", epoch=1), epoch=1).kind is EvidenceKind.NONE
        assert feed(f, rd("3", epoch=1), epoch=1).kind is EvidenceKind.AGREE


class TestMismatchThresholds:
    def test_plus_one_needs_n(self) -> None:
        f = make_filter()
        # detector at 2, screen idx 3 (value "4")
        feed(f, rd("4"))
        feed(f, rd("4"))
        ev = feed(f, rd("4"))
        assert ev.kind is EvidenceKind.MISMATCH
        assert ev.screen_idx == 3
        assert ev.text == "4"
        assert ev.confidence == 90.0

    def test_other_jump_needs_jump_n(self) -> None:
        f = make_filter()
        for _ in range(5):
            assert feed(f, rd("8")).kind is EvidenceKind.NONE
        assert feed(f, rd("8")).kind is EvidenceKind.MISMATCH

    def test_backward_jump_needs_jump_n(self) -> None:
        f = make_filter()
        for _ in range(5):
            assert feed(f, rd("1")).kind is EvidenceKind.NONE
        assert feed(f, rd("1")).kind is EvidenceKind.MISMATCH

    def test_compare(self) -> None:
        assert compare(2, 2) is EvidenceKind.AGREE
        assert compare(3, 2) is EvidenceKind.MISMATCH


class TestRejection:
    def test_min_confidence_boundary(self) -> None:
        f = make_filter(consensus=1, min_conf=60.0)
        assert feed(f, rd("3", conf=60.0)).kind is EvidenceKind.AGREE
        f = make_filter(consensus=1, min_conf=60.0)
        assert feed(f, rd("3", conf=59.999)).kind is EvidenceKind.NONE

    def test_max_age_boundary(self) -> None:
        f = make_filter(consensus=1, max_age=0.5)  # 50 frames at 100 fps
        assert feed(f, rd("3", frame=50), now=100).kind is EvidenceKind.AGREE
        f = make_filter(consensus=1, max_age=0.5)
        assert feed(f, rd("3", frame=49), now=100).kind is EvidenceKind.NONE

    def test_late_reading_is_not_matched_by_frame_equality(self) -> None:
        f = make_filter(consensus=1)
        assert feed(f, rd("3", frame=90), now=100).kind is EvidenceKind.AGREE

    def test_epoch_mismatch(self) -> None:
        f = make_filter(consensus=1)
        assert feed(f, rd("3", epoch=0), epoch=1).kind is EvidenceKind.NONE

    def test_out_of_range(self) -> None:
        f = make_filter(consensus=1, n_splits=5)
        assert feed(f, rd("6")).kind is EvidenceKind.NONE  # idx 5 >= 5
        assert feed(f, rd("0")).kind is EvidenceKind.NONE  # idx -1
        assert feed(f, rd("4")).kind is EvidenceKind.MISMATCH  # idx 3 = det+1, in range

    def test_no_regex_match(self) -> None:
        f = make_filter(consensus=1)
        assert feed(f, rd("abc")).kind is EvidenceKind.NONE

    def test_explicit_split_index_pattern(self) -> None:
        f = make_filter(consensus=1, patterns=(OcrPattern(regex="WIN", split_index=2),))
        assert feed(f, rd("WIN")).kind is EvidenceKind.AGREE
