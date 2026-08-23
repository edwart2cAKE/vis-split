"""Frame acquisition over MPEG-TS (Phase 4).

Pipeline:

    OBS sidecar output (Recording -> Custom Output (FFmpeg), mpegts)
      -> udp:// or tcp:// localhost URL
      -> ffmpeg subprocess decodes to raw BGR24 on stdout
      -> RawPipeReader assembles fixed-size (H, W, 3) uint8 frames

OBS-side setup (one time):

    Settings -> Output -> Recording -> Output Type: Custom Output (FFmpeg)
      Container format:           mpegts
      File path or URL:           udp://127.0.0.1:9733?pkt_size=1316
      Encoder:                    x264 (tune zerolatency) or hardware NVENC
      Output (scaled) Resolution: must equal capture.width x capture.height
                                  in config.yaml (pinned 1920x1080 regardless
                                  of canvas size)

The OBS sidecar runs alongside Streaming; nothing here touches the stream.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import BinaryIO, Protocol, cast

import numpy as np


class FrameSource(Protocol):
    """A swappable source of BGR uint8 frames shaped (H, W, 3)."""

    def read(self) -> np.ndarray | None:
        """Return the next frame, or None if none is available right now."""
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...


class RawPipeReader:
    """Assembles fixed-size BGR frames from any binary byte stream.

    Deliberately subprocess-free: partial-read assembly is unit-testable
    with e.g. io.BytesIO. Returns None on EOF, whether clean or mid-frame.
    """

    def __init__(self, width: int, height: int, stream: BinaryIO) -> None:
        self._width: int = width
        self._height: int = height
        self._stream: BinaryIO = stream
        self._frame_bytes: int = width * height * 3

    def read_frame(self) -> np.ndarray | None:
        needed = self._frame_bytes
        buf = bytearray(needed)
        filled = 0
        while filled < needed:
            chunk = self._stream.read(needed - filled)
            if not chunk:
                return None
            end = filled + len(chunk)
            buf[filled:end] = chunk
            filled = end
        flat = np.frombuffer(buf, dtype=np.uint8)
        return flat.reshape(self._height, self._width, 3)


class MpegTsFrameSource:
    """Decodes an MPEG-TS location into raw BGR frames via one ffmpeg child.

    ``location`` is anything ffmpeg accepts as -i input:

    - ``udp://`` / ``tcp://`` URLs are treated as live: if the decoder exits
      or the stream ends, it is respawned with exponential backoff (0.5 s
      doubling to a 2 s cap) until :meth:`close`.
    - Anything else (e.g. a file path) plays once; after EOF, :meth:`read`
      keeps returning None instead of replaying in a loop.

    Raises RuntimeError at construction if the ffmpeg binary is missing.
    """

    def __init__(
        self,
        location: str,
        width: int,
        height: int,
        *,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        resolved = shutil.which(ffmpeg_bin)
        if resolved is None:
            raise RuntimeError(f"ffmpeg binary '{ffmpeg_bin}' not found on PATH")
        self._bin: str = resolved
        self._location: str = location
        self._width: int = width
        self._height: int = height
        self._is_live: bool = location.startswith(("udp://", "tcp://"))
        self._ended: bool = False
        self._backoff: float = 0.0
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: RawPipeReader | None = None

    @property
    def exhausted(self) -> bool:
        """True once a non-live source has played out (no further reads)."""
        return not self._is_live and self._ended

    def read(self) -> np.ndarray | None:
        if not self._is_live and self._ended:
            return None
        if (self._proc is None or self._reader is None) and not self._spawn():
            return None
        reader = self._reader
        if reader is None:
            return None
        frame = reader.read_frame()
        if frame is None:
            self._teardown()
            return None
        self._backoff = 0.0
        return frame

    def close(self) -> None:
        self._teardown()

    def _spawn(self) -> bool:
        if self._is_live and self._backoff > 0.0:
            time.sleep(self._backoff)
        self._backoff = min(max(self._backoff * 2.0, 0.5), 2.0)

        cmd = [
            self._bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "+nobuffer",
            "-flags",
            "low_delay",
            "-i",
            self._location,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        except OSError:
            return False

        stream = proc.stdout
        if stream is None:
            proc.kill()
            return False
        self._proc = proc
        # Popen types stdout loosely; stdout=PIPE guarantees a bytes stream.
        self._reader = RawPipeReader(self._width, self._height, cast(BinaryIO, stream))
        return True

    def _teardown(self) -> None:
        self._reader = None
        proc = self._proc
        self._proc = None
        if not self._is_live:
            self._ended = True
        if proc is None:
            return
        if proc.stdout is not None:
            proc.stdout.close()
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
