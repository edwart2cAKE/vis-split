"""UndoJudge (pure, fake clock) and EvidenceFilter.screen_index parity."""

from __future__ import annotations

import random

import pytest

from autosplitter.config import OcrConfig, OcrFilter, OcrPattern, OcrUndo, Roi
from autosplitter.detection.element_reader import Reading
from autosplitter.detection.evidence import EvidenceFilter
from autosplitter.detection.undo_judge import UndoJudge, UndoVerdict, VerdictKind

K = 3  # split that fired (0-based): D = 4, death screen shows index 3


class Clock:
    def __init__(self) -> None:
        self.t: float = 100.0

    def __call__(self) -> float:
        return self.t


def make(
    clock: Clock, *, baseline_ok: bool = True, min_conf: float = 80.0, **kw: float
) -> UndoJudge:
    cfg = OcrUndo(
        settle_seconds=kw.get("settle", 3.0),
        confirm_reads=int(kw.get("reads", 3)),
        confirm_seconds=kw.get("confirm", 1.0),
        window_seconds=kw.get("window", 10.0),
    )
    j = UndoJudge(cfg, min_conf, clock)
    assert j.open(K, clock.t, 7, baseline_ok) is None
    return j


def feed_at(j: UndoJudge, c: Clock, dt: float, idx: int | None, conf: float = 90.0) -> UndoVerdict | None:
    c.t += dt
    return j.feed(idx, conf)


def test_positive_death_gives_exactly_one_undo() -> None:
    c = Clock()
    j = make(c)
    verdicts: list[UndoVerdict] = []
    for _ in range(40):
        v = feed_at(j, c, 0.25, K)
        if v is not None:
            verdicts.append(v)
    assert [v.kind for v in verdicts] == [VerdictKind.UNDO]
    assert verdicts[0].screen_idx == K and verdicts[0].reads == 5  # 3.0s..4.0s at 4 reads/s
    assert not j.is_open


def test_genuine_closes_and_logs_lag_and_never_undoes_afterwards() -> None:
    c = Clock()
    j = make(c)
    assert feed_at(j, c, 1.0, K) is None
    v = feed_at(j, c, 0.5, K + 1)
    assert v is not None and v.kind is VerdictKind.GENUINE
    assert v.elapsed == pytest.approx(1.5)
    assert not j.is_open
    for _ in range(30):
        assert feed_at(j, c, 0.5, K) is None


def test_hud_lag_guard_ignores_before_settle() -> None:
    c = Clock()
    j = make(c)
    for _ in range(11):  # 2.75 s of D-1 reads: all inside settle
        assert feed_at(j, c, 0.25, K) is None
    assert j.is_open


def test_lagged_hud_flipping_after_settle_but_before_confirm_no_undo() -> None:
    c = Clock()
    j = make(c)
    for _ in range(12):
        assert feed_at(j, c, 0.25, K) is None  # t=3.0 exactly: streak 1
    assert feed_at(j, c, 0.25, K) is None  # 2 reads
    v = feed_at(j, c, 0.25, K + 1)
    assert v is not None and v.kind is VerdictKind.GENUINE


def test_fast_hud_no_undo() -> None:
    c = Clock()
    j = make(c)
    v = feed_at(j, c, 0.1, K + 1)
    assert v is not None and v.kind is VerdictKind.GENUINE


def test_settle_boundary_exactly_at_settle_counts() -> None:
    c = Clock()
    j = make(c, reads=1, confirm=0.0)
    assert feed_at(j, c, 2.999, K) is None
    c.t = 100.0 + 3.0
    v = j.feed(K, 90.0)
    assert v is not None and v.kind is VerdictKind.UNDO


def test_confirm_reads_boundary() -> None:
    c = Clock()
    j = make(c, reads=3, confirm=0.0)
    c.t += 3.0
    assert j.feed(K, 90.0) is None
    assert j.feed(K, 90.0) is None
    v = j.feed(K, 90.0)
    assert v is not None and v.kind is VerdictKind.UNDO and v.reads == 3


def test_confirm_seconds_boundary() -> None:
    c = Clock()
    j = make(c, reads=2, confirm=1.0)
    c.t += 3.0
    assert j.feed(K, 90.0) is None
    c.t += 0.999
    assert j.feed(K, 90.0) is None  # span 0.999 < 1.0
    c.t += 0.001
    v = j.feed(K, 90.0)
    assert v is not None and v.kind is VerdictKind.UNDO


def test_fewer_reads_than_confirm_none() -> None:
    c = Clock()
    j = make(c, reads=3, confirm=0.0)
    c.t += 3.0
    assert j.feed(K, 90.0) is None
    assert j.feed(K, 90.0) is None
    assert j.is_open


def test_span_too_short_none_even_with_many_reads() -> None:
    c = Clock()
    j = make(c, reads=3, confirm=2.0)
    c.t += 3.0
    for _ in range(5):
        c.t += 0.2
        assert j.feed(K, 90.0) is None


def test_confidence_boundary() -> None:
    c = Clock()
    j = make(c, reads=1, confirm=0.0, min_conf=80.0)
    c.t += 3.0
    assert j.feed(K, 79.999) is None
    v = j.feed(K, 80.0)
    assert v is not None and v.kind is VerdictKind.UNDO


@pytest.mark.parametrize("breaker", [None, "low", K + 1 + 1, 0])
def test_untrusted_or_odd_reading_breaks_streak(breaker: int | str | None) -> None:
    c = Clock()
    j = make(c, reads=3, confirm=0.0)
    c.t += 3.0
    assert j.feed(K, 90.0) is None
    assert j.feed(K, 90.0) is None
    if breaker is None:
        assert j.feed(None, 0.0) is None
    elif breaker == "low":
        assert j.feed(K, 10.0) is None
    else:
        v = j.feed(int(breaker), 90.0)
        assert v is None or v.kind is VerdictKind.WARN
    assert j.feed(K, 90.0) is None  # streak restarted: 1
    assert j.feed(K, 90.0) is None  # 2
    v = j.feed(K, 90.0)
    assert v is not None and v.kind is VerdictKind.UNDO


def test_flapping_never_undoes() -> None:
    c = Clock()
    j = make(c, reads=3, confirm=0.0, window=1000.0)
    c.t += 3.0
    seq = [K, K, None, K, K, 9, K, K, None]
    for s in seq * 5:
        v = j.feed(s, 90.0)
        assert v is None or v.kind is VerdictKind.WARN
    assert j.is_open


def test_warn_once_per_pair_for_greater_and_lower() -> None:
    c = Clock()
    j = make(c, window=1000.0)
    c.t += 4.0
    kinds = [j.feed(9, 90.0), j.feed(9, 90.0), j.feed(0, 90.0), j.feed(0, 90.0), j.feed(9, 90.0)]
    assert [None if v is None else v.kind for v in kinds] == [
        VerdictKind.WARN, None, VerdictKind.WARN, None, None,
    ]
    assert j.is_open  # warn-only: no verdict closes the window


def test_one_verdict_per_window() -> None:
    c = Clock()
    j = make(c, reads=1, confirm=0.0)
    c.t += 3.0
    assert j.feed(K, 90.0) is not None
    for _ in range(5):
        assert j.feed(K, 90.0) is None
        assert j.feed(K + 1, 90.0) is None
    assert j.check(7) is None


def test_cancel_on_epoch_change() -> None:
    c = Clock()
    j = make(c)
    assert j.check(7) is None
    v = j.check(8)
    assert v is not None and v.kind is VerdictKind.CANCELLED
    assert j.check(8) is None and j.feed(K, 90.0) is None


def test_expiry_after_window_seconds() -> None:
    c = Clock()
    j = make(c, window=10.0)
    c.t += 9.999
    assert j.check(7) is None
    c.t += 0.001
    v = j.check(7)
    assert v is not None and v.kind is VerdictKind.EXPIRED


def test_feed_after_expiry_does_not_undo() -> None:
    c = Clock()
    j = make(c, reads=1, confirm=0.0, window=10.0)
    c.t += 10.0
    v = j.feed(K, 90.0)
    assert v is not None and v.kind is VerdictKind.EXPIRED


def test_no_baseline_window_never_yields_verdict() -> None:
    c = Clock()
    j = UndoJudge(OcrUndo(), 80.0, c)
    v = j.open(K, c.t, 7, False)
    assert v is not None and v.kind is VerdictKind.NO_BASELINE and v.split_index == K
    assert not j.is_open
    c.t += 5
    for _ in range(10):
        assert j.feed(K, 99.0) is None
    assert j.check(7) is None


def test_reopen_resets_streak_and_warnings() -> None:
    c = Clock()
    j = make(c, reads=2, confirm=0.0)
    c.t += 3.0
    assert j.feed(K, 90.0) is None
    assert j.open(K + 1, c.t, 9, True) is None
    c.t += 3.0
    assert j.feed(K + 1, 90.0) is None
    v = j.feed(K + 1, 90.0)
    assert v is not None and v.kind is VerdictKind.UNDO and v.split_index == K + 1


@pytest.mark.parametrize("seed", range(20))
def test_random_stream_at_most_one_closing_verdict(seed: int) -> None:
    rng = random.Random(seed)
    c = Clock()
    j = make(c, reads=rng.randint(1, 4), confirm=rng.choice([0.0, 0.5]), window=20.0)
    closing = 0
    for _ in range(300):
        c.t += rng.random() * 0.4
        v = j.feed(rng.choice([None, K, K, K + 1, 0, 9]), rng.choice([10.0, 90.0]))
        if v is not None and v.kind is not VerdictKind.WARN:
            closing += 1
    assert closing <= 1


# ---------------------------------------------------------------------------
# EvidenceFilter.screen_index
# ---------------------------------------------------------------------------


def _filter(n: int = 6) -> EvidenceFilter:
    cfg = OcrConfig(
        roi=Roi(x=0.0, y=0.0, width=0.1, height=0.1),
        patterns=(OcrPattern(regex="^\\s*(\\d+)\\s*$"),),
        filter=OcrFilter(consensus_frames=2, jump_consensus_frames=3, max_age_seconds=0.5, min_confidence=60.0),
    )
    return EvidenceFilter(cfg, n, 10.0)  # max age 5 frames


def rd(value: str, conf: float = 90.0, frame: int = 10, epoch: int = 1) -> Reading:
    return Reading(value=value, confidence=conf, frame_idx=frame, epoch=epoch, t_mono=0.0)


def test_screen_index_checks() -> None:
    f = _filter()
    assert f.screen_index(rd("3"), 10, 1) == 2
    assert f.screen_index(rd("3", epoch=0), 10, 1) is None  # wrong epoch
    assert f.screen_index(rd("3"), 15, 1) == 2  # age exactly max
    assert f.screen_index(rd("3"), 16, 1) is None  # stale
    assert f.screen_index(rd("3", conf=60.0), 10, 1) == 2  # conf boundary
    assert f.screen_index(rd("3", conf=59.9), 10, 1) is None
    assert f.screen_index(rd("6"), 10, 1) == 5
    assert f.screen_index(rd("7"), 10, 1) is None  # out of range
    assert f.screen_index(rd("0"), 10, 1) is None
    assert f.screen_index(rd("x"), 10, 1) is None
    assert f.screen_index(rd(""), 10, 1) is None
    assert f.screen_index(rd("1234567890"), 10, 1) is None


def test_screen_index_is_pure_and_feed_parity() -> None:
    rng = random.Random(3)
    a, b = _filter(), _filter()
    for i in range(400):
        r = rd(rng.choice(["1", "2", "3", "7", "x", ""]), rng.choice([30.0, 90.0]),
               frame=i - rng.choice([0, 0, 9]), epoch=rng.choice([1, 1, 2]))
        det = rng.randint(0, 5)
        before = (b._streak, b._streak_idx)  # pyright: ignore[reportPrivateUsage]
        _ = b.screen_index(r, i, 1)
        assert (b._streak, b._streak_idx) == before  # pyright: ignore[reportPrivateUsage]
        assert a.feed(r, i, 1, det) == b.feed(r, i, 1, det)
