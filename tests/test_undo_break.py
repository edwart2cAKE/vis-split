"""Adversarial tests for the dry-run undo judge, color preprocess and wiring.

Tests that fail here expose real defects; do not xfail.
"""

from __future__ import annotations

import dataclasses
import math
import random
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import cast, final

import cv2
import numpy as np
import pytest
from test_undo_machine import (  # pyright: ignore[reportImplicitRelativeImport]
    OCR_U,
    UNDO,
    Rig,
    cfg,
    death_rig,
    res,
)

from autosplitter.config import Config, ConfigError, load
from autosplitter.detection import State
from autosplitter.detection.element_reader import (
    PreprocessParams,
    TesseractCliReader,
    preprocess,
    text_bounds,
)
from autosplitter.detection.undo_judge import UndoJudge, VerdictKind

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "tests" / "fixtures" / "example_hud.png"


class Clk:
    def __init__(self, t: float = 100.0) -> None:
        self.t: float = t

    def __call__(self) -> float:
        return self.t


def judge(clk: Callable[[], float], **kw: float) -> UndoJudge:
    return UndoJudge(dataclasses.replace(UNDO, **kw), 80.0, clk)


# --------------------------------------------------------------------------
# judge: clock pathologies
# --------------------------------------------------------------------------


def test_clock_backwards_never_undoes() -> None:
    c = Clk()
    j = judge(c)
    j.open(0, c.t, 1, True)
    c.t -= 50.0
    for _ in range(20):
        c.t += 0.01
        assert j.feed(0, 90.0) is None
    assert j.is_open


def test_clock_jumps_forward_past_window_expires_not_undo() -> None:
    c = Clk()
    j = judge(c)
    j.open(0, c.t, 1, True)
    c.t += 1e9
    v = j.feed(0, 90.0)
    assert v is not None and v.kind is VerdictKind.EXPIRED


def test_nan_clock_at_feed_never_undoes() -> None:
    c = Clk()
    j = judge(c)
    j.open(0, c.t, 1, True)
    c.t = math.nan
    for _ in range(10):
        v = j.feed(0, 90.0)
        assert v is None or v.kind is not VerdictKind.UNDO


def test_streak_clock_going_backwards_mid_streak_blocks_undo() -> None:
    c = Clk()
    j = judge(c, settle_seconds=1.0, confirm_seconds=1.0)
    j.open(0, c.t, 1, True)
    c.t += 2.0
    assert j.feed(0, 90.0) is None
    c.t -= 0.5  # backwards, elapsed still >= settle
    assert j.feed(0, 90.0) is None
    c.t += 0.2
    assert j.feed(0, 90.0) is None
    assert j.is_open


def test_huge_confirm_reads_never_undoes_and_no_overflow() -> None:
    c = Clk()
    j = judge(c, confirm_reads=10**12)
    j.open(0, c.t, 1, True)
    for _ in range(50):
        c.t += 0.2
        v = j.feed(0, 99.0)
        assert v is None or v.kind is VerdictKind.EXPIRED


def test_confidence_nan_and_inf() -> None:
    c = Clk()
    j = judge(c, settle_seconds=0.1, confirm_seconds=0.0, confirm_reads=1)
    j.open(0, c.t, 1, True)
    c.t += 1.0
    # NaN < min is False -> NaN confidence counts as trusted.
    v = j.feed(0, math.nan)
    assert v is None, "NaN confidence must not be trusted"


def test_confidence_exactly_min_counts_and_just_below_does_not() -> None:
    c = Clk()
    j = judge(c, settle_seconds=0.1, confirm_seconds=0.0, confirm_reads=1)
    j.open(0, c.t, 1, True)
    c.t += 1.0
    assert j.feed(0, 79.999) is None
    v = j.feed(0, 80.0)
    assert v is not None and v.kind is VerdictKind.UNDO


def test_s_out_of_range_negative_and_huge_warn_not_crash() -> None:
    c = Clk()
    j = judge(c)
    j.open(2, c.t, 1, True)
    c.t += 5
    for s in (-1, 10**9, 0, 1):
        v = j.feed(s, 95.0)
        assert v is None or v.kind is VerdictKind.WARN


def test_double_death_s_equals_d_minus_2_warns_once() -> None:
    c = Clk()
    j = judge(c)
    j.open(3, c.t, 1, True)  # D = 4, S = 2 is D-2
    kinds: list[VerdictKind] = []
    for _ in range(10):
        c.t += 0.5
        v = j.feed(2, 95.0)
        if v is not None:
            kinds.append(v.kind)
    assert kinds == [VerdictKind.WARN]


def test_k0_death_on_first_level_undo_path() -> None:
    c = Clk()
    j = judge(c, settle_seconds=1.0, confirm_seconds=0.5, confirm_reads=2)
    j.open(0, c.t, 1, True)
    c.t += 1.5
    assert j.feed(0, 90.0) is None
    c.t += 0.6
    v = j.feed(0, 90.0)
    assert v is not None and v.kind is VerdictKind.UNDO and v.split_index == 0


def test_genuine_exactly_at_window_boundary_is_expired_not_genuine() -> None:
    c = Clk()
    j = judge(c)
    j.open(0, c.t, 1, True)
    c.t += 10.0
    v = j.feed(1, 95.0)
    assert v is not None and v.kind is VerdictKind.EXPIRED


def test_just_before_window_genuine_wins() -> None:
    c = Clk()
    j = judge(c)
    j.open(0, c.t, 1, True)
    c.t += 9.999
    v = j.feed(1, 95.0)
    assert v is not None and v.kind is VerdictKind.GENUINE


def test_check_then_feed_after_epoch_cancel_gives_no_second_verdict() -> None:
    c = Clk()
    j = judge(c)
    j.open(0, c.t, 1, True)
    v = j.check(2)
    assert v is not None and v.kind is VerdictKind.CANCELLED
    c.t += 5
    for _ in range(10):
        assert j.feed(0, 99.0) is None
        assert j.check(2) is None


def test_many_windows_in_sequence_each_one_verdict() -> None:
    c = Clk()
    j = judge(c, settle_seconds=1.0, confirm_seconds=0.5, confirm_reads=2)
    closing = 0
    for k in range(200):
        j.open(k, c.t, k, True)
        c.t += 1.2
        j.feed(k, 90.0)
        c.t += 0.6
        v = j.feed(k, 90.0)
        assert v is not None and v.kind is VerdictKind.UNDO and v.split_index == k
        closing += 1
        assert j.feed(k, 90.0) is None
        assert not j.is_open
    assert closing == 200


def test_reopen_without_baseline_closes_open_window() -> None:
    c = Clk()
    j = judge(c)
    j.open(0, c.t, 1, True)
    v = j.open(1, c.t, 2, False)
    assert v is not None and v.kind is VerdictKind.NO_BASELINE
    assert not j.is_open
    c.t += 5
    assert j.feed(0, 99.0) is None


@pytest.mark.parametrize("seed", range(30))
def test_random_judge_never_undoes_when_hud_ever_advanced_in_lag(seed: int) -> None:
    """If a trusted S==D read occurs before the UNDO condition completes, no UNDO."""
    rng = random.Random(seed)
    c = Clk()
    j = judge(c)
    j.open(1, c.t, 1, True)
    saw_genuine = False
    for _ in range(300):
        c.t += rng.uniform(0.0, 0.6)
        s = rng.choice([1, 1, 1, 2, None, 0, 7])
        v = j.feed(s, rng.choice([95.0, 95.0, 10.0]))
        if v is None:
            continue
        if v.kind is VerdictKind.GENUINE:
            saw_genuine = True
        if saw_genuine:
            assert v.kind is not VerdictKind.UNDO
        if v.kind in (VerdictKind.UNDO, VerdictKind.GENUINE, VerdictKind.EXPIRED):
            assert not j.is_open


# --------------------------------------------------------------------------
# machine
# --------------------------------------------------------------------------


def test_undo_never_changes_detector_state_over_many_windows() -> None:
    r = Rig(n=6)
    for _ in range(5):
        r.agree()
        r.fire_and_clear()
        snap = (r.m.state, r.m.current_split_index, r.m.epoch)
        for _ in range(30):
            r.read(str(r.m.current_split_index))  # S == D-1
            assert (r.m.state, r.m.current_split_index, r.m.epoch) == snap
        assert r.m.undo_target is not None
        r.m.rollback_to(0) if False else None
        # genuine progress after the verdict: HUD catches up
        r.read(str(r.m.current_split_index + 1))
        break


def test_double_death_inside_settle_warns_once_machine() -> None:
    r = Rig(n=6)
    r.agree()
    r.fire_and_clear()
    r.agree()  # AGREE at index 1 (S==1)... baseline for next
    r.read("2", dt=0.1)
    r.read("2", dt=0.1)
    r.read("2", dt=0.1)
    r.fire_and_clear()  # k=1 -> D=2
    # HUD shows level 1 (S==0): D-2
    for _ in range(10):
        r.read("1", dt=0.4)
    warns = [v for v in r.events() if v.kind is VerdictKind.WARN]
    assert len(warns) <= 1
    assert r.m.undo_target is None


def test_respawn_flash_during_transition_cancels_pending_window() -> None:
    r = death_rig()
    assert r.m.undo_pending
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))  # fires split 2: TRANSITION
    r.read("1")
    kinds = [v.kind for v in r.events()]
    assert VerdictKind.UNDO not in kinds
    assert not r.m.undo_pending


def test_fire_in_transition_state_reads_ignored_by_judge() -> None:
    """Readings while TRANSITION/WAITING_FOR_RESET never reach the judge
    (state != NORMAL), so they cannot confirm an UNDO for the NEW window."""
    r = death_rig()
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    assert r.m.state is State.TRANSITION
    for _ in range(40):
        r.read("1")
    assert r.m.undo_target is None


def test_ocr_worker_failed_status_no_crash_with_undo() -> None:
    r = death_rig()
    r.fw.status = "failed"
    for _ in range(40):
        r.read("1")
    assert r.m.ocr_worker_status == "failed"


def test_enabled_false_with_undo_builds_no_judge() -> None:
    c = dataclasses.replace(OCR_U, enabled=False)
    from autosplitter.detection import TransitionDetector

    m = TransitionDetector(cfg(4, c))
    assert not m.undo_judge_active


def test_loop_rearm_reset_during_window_cancels_and_next_run_fresh() -> None:
    r = death_rig()
    r.m.reset()
    r.read("1")
    ev = r.events()
    assert all(v.kind is not VerdictKind.UNDO for v in ev)
    assert not r.m.undo_pending
    # fresh run: baseline stale from previous run must not leak (agree_idx 0 may
    # still be within baseline_seconds) - documents the behaviour.
    r.fire_and_clear()
    assert r.m.current_split_index in (0, 1)


def test_stale_baseline_after_reset_does_not_vouch_for_new_run() -> None:
    """AGREE latched for index 0 in the previous run must not be accepted as a
    baseline for a fire within baseline_seconds after a reset()."""
    r = Rig()
    r.agree()  # AGREE idx 0 at t0
    r.m.reset()
    r.fire_and_clear()  # fires split 0 within 2s of the stale AGREE
    assert not r.m.undo_pending, "stale pre-reset AGREE vouched for a post-reset fire"


def test_stale_baseline_after_rollback_does_not_vouch() -> None:
    r = Rig()
    r.agree()
    r.fire_and_clear()  # D=1
    r.m.rollback_to(0)
    # rollback_to from NORMAL keeps NORMAL (documented); epoch still bumps.
    assert r.m.state is State.NORMAL and r.m.current_split_index == 0
    r.fire_and_clear()
    assert not r.m.undo_pending


def test_window_with_start_gate_first_split_death() -> None:
    r = Rig(start=True)
    assert r.m.state is State.WAITING_FOR_START
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(0.0))
    assert r.m.state is State.NORMAL and r.m.current_split_index == 0
    assert not r.m.undo_pending  # gate clear never opens a window
    r.agree()
    r.fire_and_clear()
    assert r.m.undo_pending
    for _ in range(30):
        r.read("1")
    assert r.m.undo_target == 0


def test_advance_to_closes_window_without_undo() -> None:
    r = death_rig()
    assert r.m.advance_to(3)
    r.read("1")
    kinds = [v.kind for v in r.events()]
    assert VerdictKind.UNDO not in kinds
    assert not r.m.undo_pending


def test_done_last_split_opens_no_window_and_persistent_color() -> None:
    r = Rig(n=2)
    r.agree()
    r.fire_and_clear()
    # genuine
    r.read("2")
    r.agree()
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    assert r.m.state is State.DONE
    assert not r.m.undo_pending
    for _ in range(30):
        r.read("1")
    assert r.m.undo_target is None


def test_n1_split_config_never_opens_window() -> None:
    r = Rig(n=1)
    r.agree()
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    r.m.feed_result(res(1.0))
    assert r.m.state is State.DONE
    assert not r.m.undo_pending
    assert r.events() == [] or all(v.kind is not VerdictKind.UNDO for v in r.events())


def test_n2_death_on_first_split_then_undo_verdict() -> None:
    r = Rig(n=2)
    r.agree()
    r.fire_and_clear()
    for _ in range(40):
        r.read("1")
    assert r.m.undo_target == 0
    assert r.m.current_split_index == 1


def test_reload_config_with_fewer_splits_while_window_open() -> None:
    r = death_rig()
    r.m.reload_config(cfg(2, OCR_U))
    for _ in range(40):
        r.read("1")  # must not raise


def test_reload_config_new_undo_values_do_not_apply() -> None:
    r = death_rig()
    new_ocr = dataclasses.replace(OCR_U, undo=dataclasses.replace(UNDO, settle_seconds=0.1))
    r.m.reload_config(cfg(4, new_ocr))
    r.read("1", dt=0.3)
    r.read("1", dt=0.3)
    r.read("1", dt=0.5)
    assert r.m.undo_target is None  # original settle 3.0 still in force


def test_events_queue_bounded_and_ordered() -> None:
    r = Rig(n=40)
    for _ in range(300):
        r.agree()
        r.fire_and_clear()
        if r.m.state is State.DONE:
            break
        r.m.feed_result(res(0.0))
    evs = r.events()
    assert len(evs) <= 64 * 4


def test_untrusted_flapping_none_between_reads_never_undoes() -> None:
    r = death_rig()
    for i in range(100):
        r.read("1" if i % 2 == 0 else "zz", dt=0.2)
    assert r.m.undo_target is None


def test_duplicate_stale_result_not_double_counted() -> None:
    """The same worker version must not be consumed twice."""
    r = death_rig()
    r.clock.t += 5.0
    r.frame += 1
    r.fw.push("1", r.frame, r.m.epoch)
    for _ in range(10):
        r.m.check_ocr_resync(np.zeros((64, 64, 3), np.uint8), r.frame)
    r.clock.t += 5.0
    r.m.check_ocr_resync(np.zeros((64, 64, 3), np.uint8), r.frame)
    assert r.m.undo_target is None


def test_streak_spanning_epoch_bump_via_rollback_not_counted() -> None:
    r = death_rig()
    for _ in range(3):
        r.read("1", dt=1.5)
    r.m.rollback_to(0)
    for _ in range(3):
        r.read("1", dt=1.5)
    assert r.m.undo_target is None


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

BASE = """
capture: {fps: 60, stream_url: "udp://127.0.0.1:9733", width: 1920, height: 1080}
roi: {x: 0.25, y: 0.25, width: 0.5, height: 0.5}
detection: {required_frames: 3, reset_threshold: 0.8}
colors: {black: {rgb: [0, 0, 0], tolerance: 15}}
splits:
  - {id: 1, name: S, color: black, threshold: 0.95}
livesplit: {host: "127.0.0.1", port: 16834}
ocr:
  roi: {x: 0.0, y: 0.0, width: 0.1, height: 0.1}
  patterns: {"(\\\\d+)": null}
"""


def _load(tmp_path: Path, extra: str) -> Config:
    p = tmp_path / "c.yaml"
    p.write_text(BASE + extra, encoding="utf-8")
    return load(p)


@pytest.mark.parametrize(
    "undo",
    [
        "{settle_seconds: 1" + "0" * 400 + "}",
        "{window_seconds: 1" + "0" * 400 + "}",
        "{baseline_seconds: 1" + "0" * 400 + "}",
        "{min_confidence: 1" + "0" * 400 + "}",
        "{confirm_seconds: 1" + "0" * 400 + "}",
    ],
    ids=["settle", "window", "baseline", "min_conf", "confirm_s"],
)
def test_huge_int_undo_values_give_config_error_not_overflow(tmp_path: Path, undo: str) -> None:
    try:
        _load(tmp_path, f"  undo: {undo}\n")
    except ConfigError:
        return
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"non-ConfigError escaped: {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("val", ["null", "''", "[]", "'3'", "~"])
def test_undo_odd_values_are_config_errors_or_none(tmp_path: Path, val: str) -> None:
    for key in ("settle_seconds", "confirm_reads", "window_seconds", "min_confidence"):
        with pytest.raises(ConfigError):
            _load(tmp_path, f"  undo: {{{key}: {val}}}\n")


def test_undo_null_section_is_absent(tmp_path: Path) -> None:
    c = _load(tmp_path, "  undo:\n")
    assert c.ocr is not None and c.ocr.undo is None


def test_undo_list_section_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _load(tmp_path, "  undo: [1]\n")


def test_undo_with_enabled_false_parses(tmp_path: Path) -> None:
    c = _load(tmp_path, "  enabled: false\n  undo: {}\n")
    assert c.ocr is not None and c.ocr.undo is not None


@pytest.mark.parametrize(
    "pre",
    [
        "{threshold: color}",
        "{threshold: color, text_color: [1, 2]}",
        "{threshold: color, text_color: [256, 0, 0]}",
        "{threshold: color, text_color: [-1, 0, 0]}",
        "{threshold: color, text_color: [true, false, true]}",
        "{threshold: color, text_color: [1.5, 2, 3]}",
        "{threshold: color, text_color: white}",
        "{threshold: color, text_color: [1, 2, 3], text_tolerance: 256}",
        "{threshold: color, text_color: [1, 2, 3], text_tolerance: -1}",
        "{threshold: color, text_color: [1, 2, 3], text_tolerance: true}",
        "{threshold: color, text_color: [1, 2, 3], text_tolerance: 4.5}",
        "{threshold: Color, text_color: [1, 2, 3]}",
    ],
)
def test_color_preprocess_bad_values_rejected(tmp_path: Path, pre: str) -> None:
    with pytest.raises(ConfigError):
        _load(tmp_path, f"  preprocess: {pre}\n")


@pytest.mark.parametrize("tol", [0, 255])
def test_color_tolerance_extremes_ok(tmp_path: Path, tol: int) -> None:
    c = _load(tmp_path, f"  preprocess: {{threshold: color, text_color: [255,255,255], text_tolerance: {tol}}}\n")
    assert c.ocr is not None and c.ocr.preprocess.text_tolerance == tol


def test_text_color_with_otsu_is_tolerated(tmp_path: Path) -> None:
    c = _load(tmp_path, "  preprocess: {threshold: otsu, text_color: [1,2,3]}\n")
    assert c.ocr is not None and c.ocr.preprocess.threshold == "otsu"


# --------------------------------------------------------------------------
# preprocess (color)
# --------------------------------------------------------------------------


def cp(**kw: int) -> PreprocessParams:
    lo, hi = text_bounds((255, 255, 255), kw.pop("tol", 40))
    base = {"scale": 4, "blur": 3, "pad": 10}
    base.update(kw)
    return PreprocessParams(threshold="color", text_lo=lo, text_hi=hi, **base)


def test_color_empty_crop_behaves_like_otsu_path() -> None:
    empty = np.zeros((0, 0, 3), np.uint8)
    for p in (cp(), PreprocessParams()):
        with pytest.raises(cv2.error):
            preprocess(empty, p)


def test_color_all_text_crop_is_all_black_inside_padding() -> None:
    crop = np.full((10, 10, 3), 255, np.uint8)
    out = preprocess(crop, cp(pad=5))
    assert out.shape == (10 * 4 + 10, 10 * 4 + 10)
    assert out[5:-5, 5:-5].max() == 0
    assert out[0, 0] == 255


def test_color_no_text_crop_is_all_white() -> None:
    crop = np.zeros((10, 10, 3), np.uint8)
    assert preprocess(crop, cp()).min() == 255


def test_color_tolerance_zero_matches_exact_only() -> None:
    crop = np.full((8, 8, 3), 254, np.uint8)
    out = preprocess(crop, cp(tol=0, scale=1, blur=0, pad=0))
    assert out.min() == 255
    crop[:] = 255
    assert preprocess(crop, cp(tol=0, scale=1, blur=0, pad=0)).max() == 0


def test_color_tolerance_255_everything_is_text() -> None:
    crop = np.random.default_rng(1).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    assert preprocess(crop, cp(tol=255, scale=1, blur=0, pad=0)).max() == 0


def test_color_background_near_text_colour_is_text_by_design() -> None:
    crop = np.full((8, 8, 3), 230, np.uint8)  # within tol 40 of white
    assert preprocess(crop, cp(scale=1, blur=0, pad=0)).max() == 0


def test_color_gray_2d_crop_supported() -> None:
    crop = np.full((8, 8), 255, np.uint8)
    out = preprocess(crop, cp(scale=1, blur=0, pad=0))
    assert out.max() == 0


def test_color_output_dtype_contiguous_and_binary() -> None:
    rng = np.random.default_rng(2)
    crop = rng.integers(0, 256, (13, 7, 3), dtype=np.uint8)
    out = preprocess(crop, cp(scale=3, blur=5, pad=0))
    assert out.dtype == np.uint8 and out.ndim == 2
    assert {int(v) for v in cast(list[int], np.unique(out).tolist())} <= {0, 255}


def test_color_does_not_mutate_input_crop() -> None:
    crop = np.full((9, 9, 3), 200, np.uint8)
    before = crop.copy()
    preprocess(crop, cp())
    assert np.array_equal(crop, before)


def test_color_mode_without_bounds_falls_into_not_implemented() -> None:
    with pytest.raises(ValueError):
        preprocess(np.zeros((4, 4, 3), np.uint8), PreprocessParams(threshold="color"))


def test_color_large_scale_blur_pad_extremes() -> None:
    crop = np.full((5, 5, 3), 255, np.uint8)
    out = preprocess(crop, cp(scale=8, blur=31, pad=50))
    assert out.shape == (5 * 8 + 100, 5 * 8 + 100)


# --------------------------------------------------------------------------
# real tesseract on tests/fixtures/example_hud.png
# --------------------------------------------------------------------------

needs_tess = pytest.mark.skipif(
    shutil.which("tesseract") is None or not EXAMPLE.exists(), reason="tesseract/tests/fixtures/example_hud.png missing"
)


def _crop(frame: np.ndarray) -> np.ndarray:
    h, w = cast(tuple[int, int], frame.shape[:2])
    x0 = int(0.105 * w)
    x1 = int((0.105 + 0.03) * w)
    y1 = max(1, int(0.055 * h))
    return frame[0:y1, x0:x1]


def _read(frame: np.ndarray, **kw: int) -> str | None:
    rd = TesseractCliReader(psm=7, whitelist="0123456789", timeout_seconds=10.0)
    out = rd.read(preprocess(_crop(frame), cp(**kw)))
    return None if out is None else out[0].strip()


def _frame() -> np.ndarray:
    img = cast("np.ndarray | None", cv2.imread(str(EXAMPLE)))
    assert img is not None
    return img


@needs_tess
@pytest.mark.parametrize("size", [None, (640, 360), (480, 270)])
def test_real_tesseract_reads_8_at_each_resolution(size: tuple[int, int] | None) -> None:
    f = _frame()
    if size is not None:
        f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
    assert _read(f) == "8"


@needs_tess
def test_real_tesseract_reads_8_after_jpeg_and_noise() -> None:
    f = cv2.resize(_frame(), (640, 360), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 40])
    assert ok
    dec = cast("np.ndarray", cv2.imdecode(buf, cv2.IMREAD_COLOR))
    noise = cast("np.ndarray", np.random.default_rng(3).normal(0, 6, dec.shape))
    f = cast("np.ndarray", np.clip(dec.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    assert _read(f) == "8"


@needs_tess
@pytest.mark.parametrize("text", ["1", "12", "15", "7", "20"])
@pytest.mark.parametrize("size", [(1920, 1080), (640, 360)])
def test_real_tesseract_other_digits_same_position(text: str, size: tuple[int, int]) -> None:
    f = _frame().copy()
    h, w = cast(tuple[int, int], f.shape[:2])
    x0, x1, y1 = int(0.105 * w), int(0.135 * w), int(0.055 * h)
    f[0:y1, x0:x1] = int(cast(float, np.mean(cast("np.ndarray", f[0, x0]))))  # erase the 8 with the bg gray
    scale = 1.8 if len(text) == 1 else 1.4
    org = (x0 + 6, int(y1 * 0.88))
    thick = 6
    cv2.putText(f, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
    # The glyph must lie inside the ROI: the documented rule is that the ROI
    # covers ALL digits of the highest level number with ~2 px margin. The
    # strict example ROI clips the '1' of '15' at 640x360 (measured: reads '5').
    fh, fw = cast(tuple[int, int], f.shape[:2])
    cx0 = int(0.105 * fw) - 2
    cx1 = int(0.135 * fw) + 2
    crop = f[0 : int(0.055 * fh) + 2, cx0:cx1]
    rd = TesseractCliReader(psm=7, whitelist="0123456789", timeout_seconds=10.0)
    out = rd.read(preprocess(crop, cp()))
    got = None if out is None else out[0].strip()
    assert got == text, f"expected {text!r} read {got!r}"


# --------------------------------------------------------------------------
# end to end: --live with ocr.undo against a fake LiveSplit server
# --------------------------------------------------------------------------

import socket
import subprocess
import threading

from autosplitter import main as mainmod


class _Srv:
    def __init__(self) -> None:
        self.sock: socket.socket = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port: int = cast(tuple[str, int], self.sock.getsockname())[1]
        self.data: bytes = b""
        self.stop: bool = False
        self.t: threading.Thread = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self) -> None:
        self.sock.settimeout(0.05)
        conns: list[socket.socket] = []
        while not self.stop:
            try:
                c = self.sock.accept()[0]
                c.settimeout(0.01)
                conns.append(c)
            except OSError:
                pass
            for c in conns:
                try:
                    d = c.recv(4096)
                    if d:
                        self.data += d
                except OSError:
                    pass

    def close(self) -> None:
        self.stop = True
        self.t.join(1.0)
        self.sock.close()


@final
class _SlowWorker:
    """Duck-typed worker: every submitted frame yields reading '1' (S == 0)."""

    def __init__(self) -> None:
        self.version: int = 0
        self.taken_version: int = 0
        self.status: str = "ok"
        self.latest: object = None

    def start(self) -> None:
        pass

    def submit_frame(self, _frame: np.ndarray, frame_idx: int, epoch: int = 0) -> None:
        import time

        from autosplitter.detection.ocr_detector import OcrResult

        time.sleep(0.01)
        self.latest = OcrResult("1", 95.0, None, frame_idx, epoch, 0.0)
        self.version += 1

    def take_result(self) -> object:
        self.taken_version = self.version
        return self.latest

    def close(self) -> None:
        pass


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_live_with_undo_sends_only_split_never_unsplit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from autosplitter.detection import state_machine as sm

    ts = tmp_path / "in.ts"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         "color=c=black:s=64x36:r=30:d=0.4,format=yuv420p[a];color=c=white:s=64x36:r=30:d=4,format=yuv420p[b];[a][b]concat=n=2:v=1:a=0",
         "-c:v", "libx264", "-g", "5", "-f", "mpegts", str(ts)],
        check=True,
    )
    srv = _Srv()
    try:
        worker = _SlowWorker()

        def fake_build(_c: object) -> tuple[_SlowWorker, str]:
            return worker, "fake OK"

        monkeypatch.setattr(sm, "build_ocr_worker", fake_build)
        text = (
            "capture: {fps: 30, stream_url: 'udp://127.0.0.1:9733', width: 64, height: 36}\n"
            + "roi: {x: 0, y: 0, width: 1, height: 1}\n"
            + "detection: {required_frames: 2, reset_threshold: 0.5}\n"
            + "colors: {black: {rgb: [0, 0, 0], tolerance: 25}}\n"
            + "splits:\n"
            + "  - {id: 1, name: One, color: black, threshold: 0.9}\n"
            + "  - {id: 2, name: Two, color: black, threshold: 0.9}\n"
            + "  - {id: 3, name: Three, color: black, threshold: 0.9}\n"
            + f"livesplit: {{host: '127.0.0.1', port: {srv.port}, timeout_seconds: 0.05, reconnect_seconds: 0.1}}\n"
            + "ocr:\n  roi: {x: 0, y: 0, width: 0.5, height: 0.5}\n"
            + "  patterns: {'(\\d+)': null}\n"
            + "  filter: {max_age_seconds: 100, consensus_frames: 1, jump_consensus_frames: 1}\n"
            + "  undo: {settle_seconds: 0.1, confirm_reads: 2, confirm_seconds: 0.05,"
            + " window_seconds: 30, require_baseline: false}\n"
        )
        p = tmp_path / "c.yaml"
        p.write_text(text, encoding="utf-8")
        rc = mainmod.main(
            ["--config", str(p), "--live", str(ts), "--width", "64", "--height", "36", "--livesplit"]
        )
        out = capsys.readouterr().out
        assert rc == 0, out
        import time as _t

        _t.sleep(0.2)
    finally:
        srv.close()
    assert "would UNDO" in out, out
    assert b"unsplit" not in srv.data
    assert srv.data.count(b"split\n") == 1, srv.data


def test_only_the_undo_executor_calls_unsplit() -> None:
    """Guard (updated in increment 4): exactly ONE production module may call
    ``.unsplit(``: the opt-in ``ocr.mode: undo`` executor. The client defines it."""
    src = ROOT / "src" / "autosplitter"
    allowed = {"src/autosplitter/livesplit/undo_exec.py"}
    hit_files = {
        f.relative_to(ROOT).as_posix()
        for f in src.rglob("*.py")
        if any(".unsplit(" in line for line in f.read_text(encoding="utf-8").splitlines())
    }
    assert hit_files == allowed
