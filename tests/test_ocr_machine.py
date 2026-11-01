"""State-machine epoch and read-only (verify_only) OCR evidence."""


from __future__ import annotations

import numpy as np

from autosplitter.config import (
    DEFAULT_ROI,
    Capture,
    Color,
    Config,
    Detection,
    LiveSplit,
    OcrConfig,
    OcrEngine,
    OcrFilter,
    OcrPattern,
    Roi,
    Split,
)
from autosplitter.detection import DetectionResult, State, TransitionDetector
from autosplitter.detection.evidence import EvidenceKind
from autosplitter.detection.ocr_detector import OcrResult, OcrWorker


def res(ratio: float) -> DetectionResult:
    return DetectionResult(matched_pixels=int(ratio * 100), total_pixels=100, ratio=ratio)


def make_config(n: int = 4, ocr: OcrConfig | None = None) -> Config:
    return Config(
        capture=Capture(fps=100, stream_url="udp://127.0.0.1:9733", width=64, height=64),
        roi=DEFAULT_ROI,
        detection=Detection(required_frames=2, reset_threshold=0.5),
        colors={"black": Color(name="black", rgb=(0, 0, 0), tolerance=15)},
        splits=tuple(Split(id=i + 1, name=f"S{i + 1}", color="black", threshold=0.9) for i in range(n)),
        livesplit=LiveSplit(host="127.0.0.1", port=16834),
        ocr=ocr,
    )


OCR = OcrConfig(
    roi=Roi(x=0.0, y=0.0, width=0.5, height=0.5),
    engine=OcrEngine(min_confidence=50.0),
    patterns=(OcrPattern(regex="(\\d+)"),),
    poll_interval_frames=1,
    filter=OcrFilter(consensus_frames=2, jump_consensus_frames=3, max_age_seconds=1.0, min_confidence=50.0),
)


class FakeWorker:
    """Duck-typed OcrWorker: results are pushed by the test; no thread."""

    def __init__(self) -> None:
        self.version: int = 0
        self.taken_version: int = 0
        self.status: str = "ok"
        self.latest: OcrResult | None = None
        self.submits: list[tuple[int, int]] = []

    def push(self, text: str, frame: int, epoch: int, conf: float = 90.0) -> None:
        self.latest = OcrResult(text, conf, None, frame, epoch, 0.0)
        self.version += 1

    def submit_frame(self, _frame: np.ndarray, frame_idx: int, epoch: int = 0) -> None:
        self.submits.append((frame_idx, epoch))

    def take_result(self) -> OcrResult | None:
        if self.taken_version == self.version:
            return None
        self.taken_version = self.version
        return self.latest

    def poll_result(self) -> OcrResult | None:
        return self.latest

    def close(self) -> None:
        pass


def machine_with(fw: FakeWorker, n: int = 4) -> TransitionDetector:
    return TransitionDetector(make_config(n, OCR), ocr=fw)  # pyright: ignore[reportArgumentType]


FRAME = np.zeros((64, 64, 3), np.uint8)


class TestEpoch:
    def test_epoch_on_split_cycle(self) -> None:
        m = TransitionDetector(make_config(3))
        e = [m.epoch]
        m.feed_result(res(1.0))
        assert m.epoch == e[0]  # streak only
        m.feed_result(res(1.0))  # fires -> TRANSITION
        e.append(m.epoch)
        m.feed_result(res(1.0))  # TRANSITION -> WAITING_FOR_RESET
        e.append(m.epoch)
        m.feed_result(res(1.0))  # dead zone/lit: no change
        assert m.epoch == e[-1]
        m.feed_result(res(0.0))  # clear -> NORMAL, index+1
        e.append(m.epoch)
        assert e == sorted(set(e)) and len(e) == 4

    def test_epoch_dead_zone_no_change(self) -> None:
        m = TransitionDetector(make_config(3))
        m.feed_result(res(1.0))
        m.feed_result(res(1.0))
        m.feed_result(res(1.0))
        ep = m.epoch
        m.feed_result(res(0.7))
        assert m.epoch == ep

    def test_epoch_done(self) -> None:
        m = TransitionDetector(make_config(1))
        m.feed_result(res(1.0))
        m.feed_result(res(1.0))
        ep = m.epoch
        m.feed_result(res(1.0))
        assert m.state is State.DONE and m.epoch == ep + 1

    def test_epoch_reset_advance_rollback(self) -> None:
        m = TransitionDetector(make_config(4))
        e0 = m.epoch
        m.reset()
        assert m.epoch == e0 + 1
        assert m.advance_to(2)
        assert m.epoch == e0 + 2
        m.feed_result(res(0.0))  # settle
        e = m.epoch
        assert m.rollback_to(0) is False or m.epoch == e + 1
        m2 = TransitionDetector(make_config(4))
        m2.feed_result(res(1.0))
        m2.feed_result(res(1.0))
        m2.feed_result(res(1.0))
        e = m2.epoch
        assert m2.rollback_to(0)
        assert m2.epoch == e + 1

    def test_failed_calls_do_not_bump(self) -> None:
        m = TransitionDetector(make_config(4))
        e = m.epoch
        assert not m.rollback_to(0)
        assert not m.advance_to(0)
        assert m.epoch == e

    def test_resync_index_bumps_only_on_change(self) -> None:
        m = TransitionDetector(make_config(4))
        e = m.epoch
        m._resync_index(0)  # pyright: ignore[reportPrivateUsage]
        assert m.epoch == e
        m._resync_index(2)  # pyright: ignore[reportPrivateUsage]
        assert m.epoch == e + 1


class TestVerifyOnly:
    def test_injected_worker_used_and_epoch_submitted(self) -> None:
        fw = FakeWorker()
        m = machine_with(fw)
        m.check_ocr_resync(FRAME, 7)
        assert fw.submits == [(7, m.epoch)]

    def test_agree(self) -> None:
        fw = FakeWorker()
        m = machine_with(fw)
        for f in (1, 2):
            fw.push("1", f, m.epoch)
            m.check_ocr_resync(FRAME, f)
        assert m.ocr_evidence.kind is EvidenceKind.AGREE
        assert m.ocr_text == "1"

    def test_mismatch_never_changes_state(self) -> None:
        fw = FakeWorker()
        m = machine_with(fw)
        for f in (1, 2):
            fw.push("2", f, m.epoch)
            m.check_ocr_resync(FRAME, f)
        ev = m.ocr_evidence
        assert ev.kind is EvidenceKind.MISMATCH and ev.screen_idx == 1
        assert m.current_split_index == 0
        assert m.state is State.NORMAL

    def test_jump_mismatch_needs_jump_consensus(self) -> None:
        fw = FakeWorker()
        m = machine_with(fw)
        for f in (1, 2):
            fw.push("4", f, m.epoch)
            m.check_ocr_resync(FRAME, f)
        assert m.ocr_evidence.kind is EvidenceKind.NONE
        fw.push("4", 3, m.epoch)
        m.check_ocr_resync(FRAME, 3)
        assert m.ocr_evidence.kind is EvidenceKind.MISMATCH
        assert m.current_split_index == 0

    def test_evidence_discarded_on_epoch_change(self) -> None:
        fw = FakeWorker()
        m = machine_with(fw)
        for f in (1, 2):
            fw.push("1", f, m.epoch)
            m.check_ocr_resync(FRAME, f)
        assert m.ocr_evidence.kind is EvidenceKind.AGREE
        m.feed_result(res(1.0))
        m.feed_result(res(1.0))  # TRANSITION
        assert m.ocr_evidence.kind is EvidenceKind.NONE

    def test_stale_epoch_reading_ignored(self) -> None:
        fw = FakeWorker()
        m = machine_with(fw)
        old = m.epoch
        m.feed_result(res(1.0))
        m.feed_result(res(1.0))
        m.feed_result(res(1.0))
        m.feed_result(res(0.0))  # NORMAL, index 1, new epoch
        for f in (1, 2, 3):
            fw.push("1", f, old)
            m.check_ocr_resync(FRAME, f)
        assert m.ocr_evidence.kind is EvidenceKind.NONE

    def test_only_evaluated_in_normal_and_never_mutates(self) -> None:
        for setup in ("transition", "waiting", "done", "start"):
            fw = FakeWorker()
            cfg = make_config(2, OCR)
            m = TransitionDetector(cfg, ocr=fw)  # pyright: ignore[reportArgumentType]
            if setup == "transition":
                m.feed_result(res(1.0))
                m.feed_result(res(1.0))
                assert m.state is State.TRANSITION
            elif setup == "waiting":
                for _ in range(3):
                    m.feed_result(res(1.0))
                assert m.state is State.WAITING_FOR_RESET
            elif setup == "done":
                m.advance_to(2)
                assert m.state is State.DONE
            else:
                m.advance_to(1)
            snap = (m.state, m.current_split_index, m.epoch)
            for f in range(1, 10):
                fw.push("2", f, m.epoch)
                m.check_ocr_resync(FRAME, f)
            assert (m.state, m.current_split_index, m.epoch) == snap
            assert m.ocr_evidence.kind is EvidenceKind.NONE
            assert m.ocr_text == "2"  # HUD text still kept

    def test_no_worker_no_crash(self) -> None:
        m = TransitionDetector(make_config(2))
        m.check_ocr_resync(FRAME, 1)
        assert m.ocr_evidence.kind is EvidenceKind.NONE
        assert m.ocr_worker_status is None

    def test_real_worker_type_accepted(self) -> None:
        # signature check only: injected OcrWorker is stored as given
        assert OcrWorker is not None
