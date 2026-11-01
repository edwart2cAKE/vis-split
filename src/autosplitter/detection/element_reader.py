"""Screen-element readers: image preprocessing and OCR backends.

``preprocess`` is shared by every backend so they all see the same image.
The CLI backend pipes a PNG to the system ``tesseract`` binary; no Python
OCR binding is required.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Reading:
    """One read of a screen element, stamped at capture time."""

    value: str
    confidence: float
    frame_idx: int
    epoch: int
    t_mono: float


class ElementReader(Protocol):
    """Backend that turns a preprocessed image into ``(text, confidence)``."""

    name: str

    def available(self) -> str | None:
        """Return a reason string when unusable, None when OK."""
        ...

    def read(self, img: np.ndarray) -> tuple[str, float] | None:
        """Read an already-preprocessed image; None means nothing recognized."""
        ...


@dataclass(frozen=True)
class PreprocessParams:
    scale: int = 4
    blur: int = 3
    threshold: str = "otsu"
    pad: int = 10
    # ``threshold == "color"`` only: inclusive per-channel BGR bounds of the
    # text colour, precomputed once (see ``text_bounds``), never per call.
    text_lo: tuple[int, int, int] | None = None
    text_hi: tuple[int, int, int] | None = None


def text_bounds(
    rgb: tuple[int, int, int], tolerance: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Inclusive BGR (lo, hi) bounds for an RGB text colour +/- tolerance."""
    r, g, b = rgb
    lo = (max(0, b - tolerance), max(0, g - tolerance), max(0, r - tolerance))
    hi = (min(255, b + tolerance), min(255, g + tolerance), min(255, r + tolerance))
    return lo, hi


def _preprocess_color(crop: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """Upscale, blur, mask the text colour, invert (black text on white), pad."""
    assert params.text_lo is not None and params.text_hi is not None
    img = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop
    if params.scale > 1:
        img = cv2.resize(
            img, None, fx=params.scale, fy=params.scale, interpolation=cv2.INTER_CUBIC
        )
    if params.blur > 0:
        img = cv2.GaussianBlur(img, (params.blur, params.blur), 0)
    mask = cv2.inRange(img, params.text_lo, params.text_hi)
    out = cv2.bitwise_not(mask)
    if params.pad > 0:
        out = cv2.copyMakeBorder(
            out, params.pad, params.pad, params.pad, params.pad,
            cv2.BORDER_CONSTANT, value=255,
        )
    return out


def preprocess(crop: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """Gray, upscale, blur, optional Otsu, text-dark polarity, white pad.

    ``threshold="color"`` (needs ``text_lo``/``text_hi``) instead masks the
    text colour and ignores backgrounds and polarity entirely.

    Polarity: the minority pixel class (by Otsu split) is taken as the text
    and the image is inverted when needed so text is dark on light.
    """
    if params.threshold == "color" and params.text_lo is not None and params.text_hi is not None:
        return _preprocess_color(crop, params)
    if params.threshold not in ("otsu", "none"):
        raise ValueError(f"preprocess threshold {params.threshold!r} is not implemented")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if params.scale > 1:
        gray = cv2.resize(
            gray, None, fx=params.scale, fy=params.scale, interpolation=cv2.INTER_CUBIC
        )
    if params.blur > 0:
        gray = cv2.GaussianBlur(gray, (params.blur, params.blur), 0)
    _t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_is_text = int(np.count_nonzero(binary)) * 2 < binary.size
    if params.threshold == "otsu":
        out = binary
    else:
        out = gray
    if white_is_text:
        out = cv2.bitwise_not(out)
    if params.pad > 0:
        out = cv2.copyMakeBorder(
            out, params.pad, params.pad, params.pad, params.pad,
            cv2.BORDER_CONSTANT, value=255,
        )
    return out


def parse_tsv(tsv: str) -> tuple[str, float] | None:
    """Join level-5 (word) rows; confidence is the mean of their conf values."""
    words: list[str] = []
    confs: list[float] = []
    for line in tsv.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 12 or cols[0] != "5":
            continue
        text = cols[11].strip()
        if not text:
            continue
        try:
            conf = float(cols[10])
        except ValueError:
            continue
        if not math.isfinite(conf):
            continue
        words.append(text)
        confs.append(conf)
    if not words:
        return None
    return " ".join(words), sum(confs) / len(confs)


class TesseractCliReader:
    """Runs ``tesseract stdin stdout ... tsv`` per read (PNG on stdin)."""

    name: str = "tesseract_cli"

    def __init__(
        self,
        language: str = "eng",
        psm: int = 7,
        whitelist: str | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._language: str = language
        self._psm: int = psm
        self._whitelist: str | None = whitelist
        self._timeout: float = timeout_seconds
        self._avail: tuple[str | None, str] | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._proc_lock: threading.Lock = threading.Lock()
        self._closing: bool = False
        self.last_error: str | None = None

    def _probe(self) -> tuple[str | None, str]:
        if self._avail is not None:
            return self._avail
        exe = shutil.which("tesseract")
        if exe is None:
            self._avail = ("tesseract binary not found on PATH", "")
            return self._avail
        try:
            proc = subprocess.run(
                [exe, "--version"], capture_output=True, text=True, timeout=5.0, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._avail = (f"tesseract --version failed: {exc}", "")
            return self._avail
        out = (proc.stdout or proc.stderr or "").strip()
        first = out.splitlines()[0] if out else ""
        if proc.returncode != 0 or not first:
            self._avail = (f"tesseract --version exited with {proc.returncode}", "")
            return self._avail
        parts = first.split()
        version = parts[1] if len(parts) > 1 else first
        self._avail = (None, version)
        return self._avail

    def available(self) -> str | None:
        return self._probe()[0]

    @property
    def version(self) -> str:
        return self._probe()[1]

    def _command(self) -> list[str]:
        cmd = [
            "tesseract", "stdin", "stdout", "-l", self._language, "--psm", str(self._psm),
        ]
        if self._whitelist:
            cmd += ["-c", f"tessedit_char_whitelist={self._whitelist}"]
        cmd.append("tsv")
        nice = shutil.which("nice")
        if nice is not None:
            cmd = [nice, "-n", "10", *cmd]
        return cmd

    def read(self, img: np.ndarray) -> tuple[str, float] | None:
        """Never raises; failures set ``last_error`` and return None."""
        self.last_error = None
        try:
            ok, png = cv2.imencode(".png", img)
            if not ok:
                self.last_error = "png encode failed"
                return None
            env = dict(os.environ)
            env["OMP_THREAD_LIMIT"] = "1"
            proc = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            with self._proc_lock:
                self._proc = proc
                closing = self._closing
            if closing:
                # close() ran between Popen and the assignment: kill now.
                proc.kill()
            try:
                out, _ = proc.communicate(png.tobytes(), timeout=self._timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                _ = proc.communicate()
                self.last_error = f"tesseract timed out after {self._timeout:g}s"
                return None
            finally:
                with self._proc_lock:
                    self._proc = None
            if proc.returncode != 0:
                self.last_error = f"tesseract exited with {proc.returncode}"
                return None
            return parse_tsv(out.decode("utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001 - reader contract: never raise
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def close(self) -> None:
        """Kill any in-flight tesseract child (called from another thread)."""
        with self._proc_lock:
            self._closing = True
            proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except OSError as exc:
                self.last_error = f"kill failed: {exc}"
