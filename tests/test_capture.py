"""Tests for autosplitter.capture.frame_source.

RawPipeReader is exercised purely through synthetic byte streams so partial-
read assembly needs no ffmpeg. The integration tests run a real decode over a
committed tiny MPEG-TS fixture and are skipped when ffmpeg is absent.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, override

import numpy as np
import pytest

from autosplitter.capture import MpegTsFrameSource, RawPipeReader

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

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
    """BytesIO that accepts at most 3 bytes per readinto call."""

    @override
    def readinto(self, b: WriteableBuffer, /) -> int:
        return super().readinto(memoryview(b)[:3])


class TestRawPipeReader:
    def test_reads_exact_frames_from_contiguous_stream(self) -> None:
        reader = RawPipeReader(WIDTH, HEIGHT, io.BytesIO(solid_frames(3)))

        first = reader.read_frame()
        assert first is not None
        assert first.shape == (HEIGHT, WIDTH, 3)
        assert first.dtype == np.uint8
        assert np.array_equal(first, np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))

        second = reader.read_frame()
        assert second is not None
        assert np.array_equal(second, np.full((HEIGHT, WIDTH, 3), 16, dtype=np.uint8))

        assert reader.read_frame() is not None

    def test_partial_reads_are_assembled_into_whole_frames(self) -> None:
        reader = RawPipeReader(WIDTH, HEIGHT, Trickle(solid_frames(4)))

        first = reader.read_frame()
        assert first is not None
        assert np.array_equal(first, np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))

        second = reader.read_frame()
        assert second is not None
        assert np.array_equal(second, np.full((HEIGHT, WIDTH, 3), 16, dtype=np.uint8))

    def test_frames_reuse_one_preallocated_buffer(self) -> None:
        """Contract pin: consecutive frames share storage — copy to retain."""
        reader = RawPipeReader(WIDTH, HEIGHT, io.BytesIO(solid_frames(2)))

        first = reader.read_frame()
        second = reader.read_frame()

        assert first is not None and second is not None
        assert first is second

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

    def test_spawn_command_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Locks the ffmpeg invocation: input flags before -i, raw BGR24 out."""
        captured: list[list[str]] = []
        captured_kwargs: list[dict[str, object]] = []

        class FakeProc:
            stdout: object | None = None

            def kill(self) -> None:
                return None

        def fake_popen(cmd: list[str], **kwargs: object) -> FakeProc:
            captured.append(list(cmd))
            captured_kwargs.append(kwargs)
            return FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        url = "udp://127.0.0.1:9733"
        source = MpegTsFrameSource(url, 1920, 1080)
        assert source.read() is None

        (cmd,) = captured
        (popen_kwargs,) = captured_kwargs
        assert popen_kwargs == {"stdout": subprocess.PIPE}
        i_index = cmd.index("-i")
        assert cmd[i_index + 1] == url
        for expected in (
            "-nostdin",
            "-probesize",
            "32768",
            "-analyzeduration",
            "0",
            "-fflags",
            "+nobuffer+discardcorrupt",
            "-flags",
            "low_delay",
            "-err_detect",
            "ignore_err",
        ):
            assert expected in cmd[1:i_index]
        assert cmd[i_index + 2 : i_index + 4] == ["-fps_mode", "passthrough"]
        assert cmd[i_index + 4 :] == ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_decodes_fixture_to_bgr_frames(self) -> None:
        """Frames arrive as (64, 64, 3) uint8 and are not all identical.

        Content is hashed per read because the reader reuses its buffer.
        """
        source = MpegTsFrameSource(str(TS_FIXTURE), 64, 64)

        try:
            count = 0
            distinct = 0
            first: bytes | None = None
            shapes_ok = True
            for _ in range(64):
                frame = source.read()
                if frame is None:
                    break
                count += 1
                shapes_ok &= frame.shape == (64, 64, 3) and frame.dtype == np.uint8
                data = frame.tobytes()
                if first is None:
                    first = data
                elif data != first:
                    distinct += 1

            assert count >= 20
            assert shapes_ok
            assert distinct > 0
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
