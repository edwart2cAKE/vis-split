"""Review fixes: NaN confidence, huge ints, post_clear_seconds, confirm rule."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

BASE = """
capture: {fps: 60, stream_url: "udp://127.0.0.1:9733", width: 1920, height: 1080}
roi: {x: 0.25, y: 0.25, width: 0.5, height: 0.5}
detection: {required_frames: 3, reset_threshold: 0.8}
colors: {black: {rgb: [0, 0, 0], tolerance: 15}}
splits:
  - {id: 1, name: S, color: black, threshold: 0.95}
livesplit: {host: "127.0.0.1", port: 16834}
"""
ROI = "  roi: {x: 0.0, y: 0.0, width: 0.1, height: 0.1}\n"
PAT = '  patterns: {"^\\\\s*(\\\\d+)\\\\s*$": null}\n'

from autosplitter.config import ConfigError, OcrUndo, load
from autosplitter.detection.element_reader import Reading, parse_tsv
from autosplitter.detection.evidence import EvidenceFilter
from autosplitter.detection.undo_judge import UndoJudge, VerdictKind

HUGE = "1" + "0" * 400


class Clk:
    def __init__(self) -> None:
        self.t: float = 50.0

    def __call__(self) -> float:
        return self.t


def _cfg_text(extra: str, ocr_extra: str = "") -> str:
    return BASE + "ocr:\n" + ROI + PAT + ocr_extra + extra


def _load(tmp_path: Path, text: str) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(text, encoding="utf-8")
    _ = load(p)


@pytest.mark.parametrize(
    "extra",
    [
        f"  max_reads_per_second: {HUGE}\n",
        f"  engine: {{timeout_seconds: {HUGE}}}\n",
        f"  engine: {{min_confidence: {HUGE}}}\n",
        f"  filter: {{max_age_seconds: {HUGE}}}\n",
        f"  filter: {{min_confidence: {HUGE}}}\n",
        f"  undo: {{post_clear_seconds: {HUGE}}}\n",
        f"  undo: {{settle_seconds: {HUGE}}}\n",
    ],
)
def test_huge_ints_in_other_ocr_keys_are_config_errors(tmp_path: Path, extra: str) -> None:
    with pytest.raises(ConfigError, match="must be a number"):
        _load(tmp_path, _cfg_text(extra))


@pytest.mark.parametrize(
    "section",
    [
        f"capture: {{fps: {HUGE}, stream_url: x, width: 1, height: 1}}\n",
        f"livesplit: {{host: h, port: 1, timeout_seconds: {HUGE}}}\n",
        f"livesplit: {{host: h, port: 1, poll_seconds: {HUGE}}}\n",
    ],
)
def test_huge_ints_elsewhere_are_config_errors(tmp_path: Path, section: str) -> None:
    key = section.split(":")[0]
    text = "\n".join(
        section if line.startswith(key + ":") else line for line in BASE.splitlines()
    )
    with pytest.raises(ConfigError):
        _load(tmp_path, text + "\n")


def test_nan_confidence_untrusted_everywhere() -> None:
    assert parse_tsv("h\n5\t1\t1\t1\t1\t1\t0\t0\t1\t1\tnan\t8\n") is None
    assert parse_tsv("h\n5\t1\t1\t1\t1\t1\t0\t0\t1\t1\tinf\t8\n") is None
    from autosplitter.config import OcrConfig, OcrFilter, OcrPattern, Roi

    cfg = OcrConfig(
        roi=Roi(x=0.0, y=0.0, width=0.1, height=0.1),
        patterns=(OcrPattern(regex="(\\d+)"),),
        filter=OcrFilter(max_age_seconds=1.0, min_confidence=60.0),
    )
    f = EvidenceFilter(cfg, 5, 10.0)
    assert f.screen_index(Reading("2", math.nan, 1, 0, 0.0), 1, 0) is None
    assert f.screen_index(Reading("2", 90.0, 1, 0, 0.0), 1, 0) == 1


def test_post_clear_guard_and_boundary() -> None:
    c = Clk()
    cfg = OcrUndo(settle_seconds=3.0, confirm_reads=1, confirm_seconds=0.5,
                  window_seconds=30.0, post_clear_seconds=1.0)
    j = UndoJudge(cfg, 80.0, c)
    fire = c.t
    c.t += 3.5  # long-lit transition: cleared 3.5 s after the fire
    assert j.open(0, fire, 1, True, c.t) is None
    c.t += 0.999
    assert j.feed(0, 90.0) is None  # past settle but < post_clear since clear
    c.t += 0.001
    assert j.feed(0, 90.0) is None  # exactly post_clear: streak starts (span 0 < 0.5)
    c.t += 0.5
    v = j.feed(0, 90.0)
    assert v is not None and v.kind is VerdictKind.UNDO


def test_post_clear_does_not_change_default_short_transition() -> None:
    c = Clk()
    j = UndoJudge(OcrUndo(confirm_reads=1, confirm_seconds=0.5), 80.0, c)
    fire = c.t
    c.t += 0.3
    assert j.open(0, fire, 1, True, c.t) is None
    c.t += 2.7  # exactly settle from fire, 2.7 s after clear
    assert j.feed(0, 90.0) is None  # streak starts
    c.t += 0.5
    v = j.feed(0, 90.0)
    assert v is not None and v.kind is VerdictKind.UNDO


def test_early_reads_are_ignored_not_breaking() -> None:
    c = Clk()
    cfg = OcrUndo(settle_seconds=1.0, confirm_reads=2, confirm_seconds=0.0001,
                  window_seconds=30.0, post_clear_seconds=2.0)
    j = UndoJudge(cfg, 80.0, c)
    assert j.open(0, c.t, 1, True, c.t) is None
    c.t += 1.5
    assert j.feed(0, 90.0) is None  # settled but inside post_clear
    c.t += 0.6
    assert j.feed(0, 90.0) is None  # exactly 2.1 s: streak 1
    c.t += 0.1
    v = j.feed(0, 90.0)
    assert v is not None and v.kind is VerdictKind.UNDO


def test_post_clear_config(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(_cfg_text("  undo: {post_clear_seconds: 0}\n"), encoding="utf-8")
    c = load(p)
    assert c.ocr is not None and c.ocr.undo is not None and c.ocr.undo.post_clear_seconds == 0.0
    p.write_text(_cfg_text("  undo: {}\n"), encoding="utf-8")
    c = load(p)
    assert c.ocr is not None and c.ocr.undo is not None and c.ocr.undo.post_clear_seconds == 1.0
    for bad, _msg in (("-1", "at least 0"), (".nan", "finite"), ("x", "a number")):
        p.write_text(_cfg_text(f"  undo: {{post_clear_seconds: {bad}}}\n"), encoding="utf-8")
        with pytest.raises(ConfigError, match="ocr.undo.post_clear_seconds must be (at least 0|a finite|a number)"):
            _ = load(p)


def test_single_read_without_span_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(_cfg_text("  undo: {confirm_reads: 1, confirm_seconds: 0}\n"), encoding="utf-8")
    with pytest.raises(
        ConfigError,
        match="ocr.undo.confirm_reads of 1 requires ocr.undo.confirm_seconds greater than 0",
    ):
        _ = load(p)
    p.write_text(_cfg_text("  undo: {confirm_reads: 2, confirm_seconds: 0}\n"), encoding="utf-8")
    _ = load(p)
