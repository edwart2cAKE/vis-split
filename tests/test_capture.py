"""Tests for autosplitter.capture.frame_source.

RawPipeReader is exercised purely through synthetic byte streams so partial-
read assembly needs no ffmpeg. The integration tests run a real decode over a
committed tiny MPEG-TS fixture and are skipped when ffmpeg is absent.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import override

import numpy as np
import pytest

from autosplitter.capture import MpegTsFrameSource, RawPipeReader

FIXTURES = Path(__file__).parent / "fixtures"
TS_FIXTURE = FIXTURES / "testsrc_64x64.ts"

WIDTH = 4
HEIGHT = 3


def solid_frames(count: int, width: int = WIDTH, height: int = HEIGHT) -> bytes:
    payload = bytearray()
    for index in range(count):
        frame = np.full((height, width, 3), index * 16 % 256, dtype=np.uint8)
        payload.extend(frame.tobytes())
    return bytes(payload)


class Trickle(io.BytesIO):
    """BytesIO that yields at most 3 bytes per read call."""

    @override
    def read(self, size: int | None = -1, /) -> bytes:
        if size is None or size < 0:
            return super().read()
        return super().read(min(size, 3))


class TestRawPipeReader:
    def test_reads_exact_frames_from_contiguous_stream(self) -> None:
        reader = RawPipeReader(WIDTH, HEIGHT, io.BytesIO(solid_frames(3)))

        first = reader.read_frame()
        second = reader.read_frame()

        assert first is not None and second is not None
        assert first.shape == (HEIGHT, WIDTH, 3)
        assert first.dtype == np.uint8
        assert np.array_equal(first, np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
        assert np.array_equal(second, np.full((HEIGHT, WIDTH, 3), 16, dtype=np.uint8))
        assert reader.read_frame() is not None

    def test_partial_reads_are_assembled_into_whole_frames(self) -> None:
        reader = RawPipeReader(WIDTH, HEIGHT, Trickle(solid_frames(4)))

        frames = [reader.read_frame(), reader.read_frame()]

        assert frames[0] is not None and frames[1] is not None
        assert np.array_equal(frames[0], np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
        assert np.array_equal(frames[1], np.full((HEIGHT, WIDTH, 3), 16, dtype=np.uint8))

    def test_clean_eof_returns_none(self) -> None:
        reader = RawPipeReader(WIDTH, HEIGHT, io.BytesIO(solid_frames(1)))

        assert reader.read_frame() is not None
        assert reader.read_frame() is None

    def test_truncated_final_frame_returns_none(self) -> None:
        payload = solid_frames(2)
        truncated = payload[: len(payload) - 7]
        reader = RawPipeReader(WIDTH, HEIGHT, io.BytesIO(truncated))

        assert reader.read_frame() is not None
        assert reader.read_frame() is None


class TestMpegTsFrameSource:
    def test_missing_binary_raises_at_construction(self) -> None:
        with pytest.raises(RuntimeError, match="not found on PATH"):
            MpegTsFrameSource(
                "udp://127.0.0.1:9733", WIDTH, HEIGHT, ffmpeg_bin="no-such-ffmpeg"
            )

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_decodes_fixture_to_bgr_frames(self) -> None:
        source = MpegTsFrameSource(str(TS_FIXTURE), 64, 64)

        try:
            frames: list[np.ndarray] = []
            for _ in range(64):
                frame = source.read()
                if frame is None:
                    break
                frames.append(frame)

            assert len(frames) >= 20
            assert all(f.shape == (64, 64, 3) for f in frames)
            assert all(f.dtype == np.uint8 for f in frames)
            assert any(not np.array_equal(frames[0], f) for f in frames[1:])
        finally:
            source.close()

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_file_source_stops_after_eof_without_replaying(self) -> None:
        source = MpegTsFrameSource(str(TS_FIXTURE), 64, 64)

        try:
            while source.read() is not None:
                pass

            for _ in range(5):
                assert source.read() is None
        finally:
            source.close()
