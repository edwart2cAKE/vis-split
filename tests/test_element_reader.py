"""preprocess() and TesseractCliReader."""


from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

from autosplitter.config import OcrPreprocess
from autosplitter.detection.element_reader import (
    PreprocessParams,
    TesseractCliReader,
    parse_tsv,
    preprocess,
    text_bounds,
)
from autosplitter.detection.ocr_detector import params_from

TSV_HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def tsv(*rows: tuple[int, float, str]) -> str:
    lines = [TSV_HEADER]
    for level, conf, text in rows:
        lines.append(f"{level}\t1\t1\t1\t1\t1\t0\t0\t10\t10\t{conf}\t{text}")
    return "\n".join(lines) + "\n"


class TestPreprocess:
    def test_scale_and_pad_shape(self) -> None:
        crop = np.zeros((10, 20, 3), np.uint8)
        out = preprocess(crop, PreprocessParams(scale=4, blur=3, threshold="otsu", pad=10))
        assert out.shape == (10 * 4 + 20, 20 * 4 + 20)
        assert out.ndim == 2

    def test_no_scale_no_pad(self) -> None:
        crop = np.zeros((10, 20), np.uint8)
        out = preprocess(crop, PreprocessParams(scale=1, blur=0, threshold="none", pad=0))
        assert out.shape == (10, 20)

    def test_pad_is_white(self) -> None:
        crop = np.zeros((10, 10, 3), np.uint8)
        out = preprocess(crop, PreprocessParams(scale=1, blur=0, pad=5))
        assert out[0, 0] == 255 and out[4, 9] == 255 and out[9, 4] == 255

    @pytest.mark.parametrize("white_text", [True, False])
    @pytest.mark.parametrize("thr", ["otsu", "none"])
    def test_polarity_text_dark(self, white_text: bool, thr: str) -> None:
        bg, fg = (0, 255) if white_text else (255, 0)
        crop = np.full((40, 40, 3), bg, np.uint8)
        crop[15:25, 15:25] = fg
        out = preprocess(crop, PreprocessParams(scale=1, blur=0, threshold=thr, pad=0))
        assert out[20, 20] < 64  # text dark
        assert out[2, 2] > 192  # background light

    def test_threshold_color_rejected(self) -> None:
        with pytest.raises(ValueError):
            preprocess(np.zeros((4, 4, 3), np.uint8), PreprocessParams(threshold="color"))


class TestParseTsv:
    def test_words_joined_and_mean_conf(self) -> None:
        r = parse_tsv(tsv((4, -1, ""), (5, 90.0, "Level"), (5, 70.0, "3")))
        assert r == ("Level 3", 80.0)

    def test_empty(self) -> None:
        assert parse_tsv(tsv((4, -1, ""))) is None
        assert parse_tsv("") is None

    def test_blank_word_skipped(self) -> None:
        assert parse_tsv(tsv((5, 95.0, "  "))) is None


class FakeProc:
    def __init__(self, out: str = "", rc: int = 0, block: threading.Event | None = None) -> None:
        self.out: str = out
        self.returncode: int = rc
        self.block: threading.Event | None = block
        self.killed: bool = False
        self.stdin_data: bytes = b""
        self.n_comm: int = 0

    def communicate(self, data: bytes | None = None, timeout: float | None = None) -> tuple[bytes, bytes]:
        self.n_comm += 1
        if data is not None:
            self.stdin_data = data
        if self.block is not None and not self.killed and not self.block.wait(timeout):
            raise subprocess.TimeoutExpired("tesseract", timeout or 0)
        return self.out.encode(), b""

    def kill(self) -> None:
        self.killed = True
        if self.block is not None:
            self.block.set()


@dataclass
class Seen:
    cmd: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def patch_popen(monkeypatch: pytest.MonkeyPatch, proc: FakeProc) -> Seen:
    seen = Seen()

    def fake_popen(cmd: list[str], **kw: object) -> FakeProc:
        seen.cmd = cmd
        seen.env = cast(dict[str, str], kw.get("env"))
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return seen


def _which_none(_name: str) -> None:
    return None


def _which_nice(name: str) -> str | None:
    return "/usr/bin/nice" if name == "nice" else None


IMG = np.full((20, 20), 255, np.uint8)


class TestTesseractCliReaderMocked:
    def test_parses_tsv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = FakeProc(tsv((5, 91.0, "7")))
        seen = patch_popen(monkeypatch, proc)
        r = TesseractCliReader(psm=10, whitelist="0123456789")
        assert r.read(IMG) == ("7", 91.0)
        assert proc.stdin_data.startswith(b"\x89PNG")
        cmd = seen.cmd
        assert "--psm" in cmd and cmd[cmd.index("--psm") + 1] == "10"
        assert "tessedit_char_whitelist=0123456789" in cmd
        assert cmd[-1] == "tsv"
        assert r.last_error is None

    def test_env_and_nice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = patch_popen(monkeypatch, FakeProc(tsv((5, 91.0, "7"))))
        monkeypatch.setattr(shutil, "which", _which_nice)
        _ = TesseractCliReader().read(IMG)
        assert seen.env["OMP_THREAD_LIMIT"] == "1"
        assert seen.cmd[:3] == ["/usr/bin/nice", "-n", "10"]
        assert seen.cmd[3] == "tesseract"

    def test_no_nice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = patch_popen(monkeypatch, FakeProc(tsv((5, 91.0, "7"))))
        monkeypatch.setattr(shutil, "which", _which_none)
        _ = TesseractCliReader().read(IMG)
        assert seen.cmd[0] == "tesseract"

    def test_empty_is_none_without_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ = patch_popen(monkeypatch, FakeProc(tsv((4, -1, ""))))
        r = TesseractCliReader()
        assert r.read(IMG) is None
        assert r.last_error is None

    def test_nonzero_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ = patch_popen(monkeypatch, FakeProc("", rc=1))
        r = TesseractCliReader()
        assert r.read(IMG) is None
        assert r.last_error is not None and "exited" in r.last_error

    def test_timeout_kills_child(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = FakeProc(block=threading.Event())
        _ = patch_popen(monkeypatch, proc)
        r = TesseractCliReader(timeout_seconds=0.05)
        assert r.read(IMG) is None
        assert proc.killed
        assert r.last_error is not None and "timed out" in r.last_error

    def test_popen_oserror_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise OSError("no exec")

        monkeypatch.setattr(subprocess, "Popen", boom)
        r = TesseractCliReader()
        assert r.read(IMG) is None
        assert r.last_error is not None and "no exec" in r.last_error

    def test_available_missing_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", _which_none)
        assert TesseractCliReader().available() == "tesseract binary not found on PATH"

    def test_close_kills_inflight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = FakeProc(block=threading.Event())
        _ = patch_popen(monkeypatch, proc)
        r = TesseractCliReader(timeout_seconds=30.0)
        t = threading.Thread(target=lambda: r.read(IMG))
        t.start()
        for _i in range(200):
            if r._proc is not None:  # pyright: ignore[reportPrivateUsage]
                break
            threading.Event().wait(0.01)
        r.close()
        t.join(3.0)
        assert not t.is_alive()
        assert proc.killed


def render_digit(digit: str, white_on_dark: bool, size: int = 40) -> np.ndarray:
    bg, fg = (20, 245) if white_on_dark else (240, 15)
    img = np.full((size, size, 3), bg, np.uint8)
    cv2.putText(
        img, digit, (size // 4, size * 3 // 4), cv2.FONT_HERSHEY_SIMPLEX,
        size / 30.0, (fg, fg, fg), max(1, size // 15), cv2.LINE_AA,
    )
    return img


def jpeg_roundtrip(img: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    return np.asarray(cv2.imdecode(buf, cv2.IMREAD_COLOR), dtype=np.uint8)


needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed"
)


@needs_tesseract
class TestTesseractIntegration:
    @pytest.mark.parametrize("white_on_dark", [True, False])
    @pytest.mark.parametrize("scale", [3, 4, 5])
    @pytest.mark.parametrize("digit", ["3", "7", "8"])
    def test_reads_digit(self, digit: str, scale: int, white_on_dark: bool) -> None:
        reader = TesseractCliReader(psm=10, whitelist="0123456789")
        assert reader.available() is None
        img = preprocess(render_digit(digit, white_on_dark), PreprocessParams(scale=scale))
        out = reader.read(img)
        assert out is not None, reader.last_error
        assert out[0] == digit

    @pytest.mark.parametrize("white_on_dark", [True, False])
    def test_reads_digit_after_noise_and_jpeg(self, white_on_dark: bool) -> None:
        rng = np.random.default_rng(1234)
        reader = TesseractCliReader(psm=10, whitelist="0123456789")
        for digit in ("3", "7"):
            base = render_digit(digit, white_on_dark)
            noisy = np.clip(
                base.astype(np.int16) + rng.normal(0, 12, base.shape), 0, 255
            ).astype(np.uint8)
            img = preprocess(jpeg_roundtrip(noisy), PreprocessParams(scale=4))
            out = reader.read(img)
            assert out is not None, reader.last_error
            assert out[0] == digit

    def test_blank_image_is_none(self) -> None:
        reader = TesseractCliReader(psm=10, whitelist="0123456789")
        blank = preprocess(np.full((40, 40, 3), 30, np.uint8), PreprocessParams())
        assert reader.read(blank) is None
        assert reader.last_error is None


# --------------------------------------------------------------------------
# preprocess threshold="color"
# --------------------------------------------------------------------------


def color_params(
    rgb: tuple[int, int, int] = (255, 255, 255), tol: int = 40, **kw: int
) -> PreprocessParams:
    lo, hi = text_bounds(rgb, tol)
    return PreprocessParams(
        scale=kw.get("scale", 1), blur=kw.get("blur", 0), threshold="color",
        pad=kw.get("pad", 0), text_lo=lo, text_hi=hi,
    )


class TestPreprocessColor:
    def test_bounds_are_bgr_and_clamped(self) -> None:
        assert text_bounds((10, 20, 250), 40) == ((210, 0, 0), (255, 60, 50))
        assert text_bounds((255, 255, 255), 0) == ((255, 255, 255), (255, 255, 255))

    def test_params_from_config_converts_once(self) -> None:
        p = params_from(OcrPreprocess(threshold="color", text_color=(200, 100, 50), text_tolerance=10))
        assert p.text_lo == (40, 90, 190) and p.text_hi == (60, 110, 210)
        assert params_from(OcrPreprocess()).text_lo is None

    def test_polarity_text_black_on_white(self) -> None:
        crop = np.full((20, 20, 3), 128, np.uint8)
        crop[5:15, 5:15] = 255
        out = preprocess(crop, color_params())
        assert out[10, 10] == 0 and out[1, 1] == 255 and out.ndim == 2

    def test_tolerance_boundary_inclusive(self) -> None:
        # text colour (200, 100, 50) RGB, tol 10 -> R in [190, 210] inclusive.
        p = color_params((200, 100, 50), 10)
        for r, expect_text in ((190, True), (210, True), (189, False), (211, False)):
            crop = np.zeros((4, 4, 3), np.uint8)
            crop[:] = (50, 100, r)  # BGR
            out = preprocess(crop, p)
            assert (out[0, 0] == 0) == expect_text, r
        # each channel independently
        crop = np.zeros((2, 2, 3), np.uint8)
        crop[:] = (50, 111, 200)  # G out by one
        assert preprocess(crop, p)[0, 0] == 255
        crop[:] = (61, 100, 200)  # B out by one
        assert preprocess(crop, p)[0, 0] == 255

    def test_ignores_polarity_text_may_be_the_majority(self) -> None:
        crop = np.full((20, 20, 3), 255, np.uint8)
        crop[0:3, :] = 0
        out = preprocess(crop, color_params())
        assert out[10, 10] == 0 and out[1, 1] == 255

    def test_pad_and_scale_shape(self) -> None:
        crop = np.zeros((10, 20, 3), np.uint8)
        out = preprocess(crop, color_params(scale=4, blur=3, pad=10))
        assert out.shape == (10 * 4 + 20, 20 * 4 + 20)
        assert out[0, 0] == 255 and out[-1, -1] == 255

    def test_no_text_crop_is_blank(self) -> None:
        crop = np.full((12, 12, 3), 90, np.uint8)
        out = preprocess(crop, color_params(scale=2, blur=3, pad=5))
        assert out.max() == 255 and out.min() == 255

    def test_gray_input_accepted(self) -> None:
        crop = np.zeros((6, 6), np.uint8)
        crop[2:4, 2:4] = 255
        out = preprocess(crop, color_params())
        assert out[2, 2] == 0 and out[0, 0] == 255

    def test_color_without_bounds_still_rejected(self) -> None:
        with pytest.raises(ValueError):
            preprocess(np.zeros((4, 4, 3), np.uint8), PreprocessParams(threshold="color"))

    def test_otsu_and_none_unchanged_by_color_fields(self) -> None:
        crop = np.zeros((40, 40, 3), np.uint8)
        crop[15:25, 15:25] = 255
        a = preprocess(crop, PreprocessParams(scale=1, blur=0, pad=0))
        b = preprocess(crop, PreprocessParams(scale=1, blur=0, pad=0, text_lo=(1, 1, 1), text_hi=(2, 2, 2)))
        assert np.array_equal(a, b)


def _digit_scene(digit: str = "8") -> np.ndarray:
    """Black side bar | mid-gray game area with a white HUD digit (BGR)."""
    img = np.full((60, 90, 3), 110, np.uint8)
    img[:, :24] = 0  # black bar edge inside the crop (flips Otsu polarity)
    cv2.putText(img, digit, (46, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4, cv2.LINE_AA)
    return img


needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract missing")


def _read_digit(img: np.ndarray, params: PreprocessParams) -> str | None:
    rd = TesseractCliReader(psm=7, whitelist="0123456789")
    got = rd.read(preprocess(img, params))
    return None if got is None else got[0].strip()


@needs_tesseract
def test_color_reads_digit_where_otsu_fails_on_black_bar() -> None:
    img = _digit_scene("8")
    otsu = _read_digit(img, PreprocessParams(scale=4, blur=3, threshold="otsu", pad=10))
    assert otsu != "8"
    col = _read_digit(img, color_params(scale=4, blur=3, pad=10))
    assert col == "8"


EXAMPLE = Path(__file__).resolve().parent / "fixtures" / "example_hud.png"


@pytest.mark.skipif(not EXAMPLE.exists(), reason="tests/fixtures/example_hud.png not present")
@needs_tesseract
@pytest.mark.parametrize("size", [(1920, 1080), (640, 360)])
def test_example_png_level_digit(size: tuple[int, int]) -> None:
    img = cv2.imread(str(EXAMPLE))
    assert img is not None
    if size != (1920, 1080):
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    from autosplitter.config import OcrEngine, OcrPattern, Roi
    from autosplitter.detection.ocr_detector import OcrDetector

    det = OcrDetector(
        Roi(x=0.105, y=0.0, width=0.03, height=0.055),
        OcrEngine(whitelist="0123456789", min_confidence=60.0),
        (OcrPattern(regex="^\\s*(\\d+)\\s*$"),),
        OcrPreprocess(threshold="color", text_color=(255, 255, 255)),
    )
    got = det.detect(img)
    assert got is not None
    assert got.text.strip() == "8" and got.split_index == 7
    assert got.confidence >= 80
