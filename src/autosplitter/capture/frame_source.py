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
      Video Encoder:              libx264 or h264_nvenc (tick "Show all
                                  codecs" if only mpeg2video is listed)
      Video Encoder Settings:     tune=zerolatency bf=0   (libx264)
                                  bf=0                   (h264_nvenc)
      Output (scaled) Resolution: must equal capture.width x capture.height
                                  in config.yaml (pinned 1920x1080 regardless
                                  of canvas size)

The OBS sidecar runs alongside Streaming; nothing here touches the stream.
"""

from __future__ import annotations

import select
import shutil
import subprocess
import time
from typing import Protocol, cast

import numpy as np


class ByteStream(Protocol):
    """A readinto-capable byte stream (e.g. io.BytesIO, BufferedReader)."""

    def readinto(self, buffer: memoryview, /) -> int:
        """Read up to len(buffer) bytes into buffer; 0 signals EOF."""
        ...


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

    Hot-path contract: every frame is decoded into one preallocated buffer
    via ``readinto`` — no per-frame allocation, no intermediate copies.
    The returned array is REUSED by the next call, so copy it (``.copy()``
    or ``np.array(frame)``) if it must outlive the following read.
    """

    def __init__(self, width: int, height: int, stream: ByteStream) -> None:
        self._width: int = width
        self._height: int = height
        self._stream: ByteStream = stream
        self._frame_bytes: int = width * height * 3
        self._buf: np.ndarray = np.empty(self._frame_bytes, dtype=np.uint8)
        # One stable view per reader: readinto fills the flat buffer, callers
        # always see (and reuse) this same shaped ndarray.
        self._frame: np.ndarray = self._buf.reshape(height, width, 3)
        self._view: memoryview = memoryview(self._buf)

    def read_frame(self) -> np.ndarray | None:
        view = self._view
        filled = 0
        while filled < len(view):
            written = self._stream.readinto(view[filled:])
            if written == 0:
                return None
            filled += written
        return self._frame


class MpegTsFrameSource:
    """Decodes an MPEG-TS location into raw BGR frames via one ffmpeg child.

    ``location`` is anything ffmpeg accepts as -i input:

    - ``udp://`` / ``tcp://`` URLs are treated as live: if the decoder exits
      or the stream ends, it is respawned with exponential backoff (0.5 s
      doubling to a 2 s cap) until :meth:`close`.
    - Anything else (e.g. a file path) plays once; after EOF, :meth:`read`
      keeps returning None instead of replaying in a loop.

    When ``record_path`` is provided, ffmpeg outputs TWO destinations:

    1. **Recording**: codec copy (``-c:v copy``) at full source resolution
       to the specified file. No re-encoding — quality is preserved.
    2. **Detection**: scaled to ``width x height`` (``-vf scale=W:H``)
       and output as raw BGR24 to stdout for the detection pipeline.

    This lets a single UDP consumer serve both detection (at detection
    resolution) and recording (at full source resolution) — no second
    socket needed.

    When ``record_path`` is NOT provided, ffmpeg outputs raw BGR24 at
    source resolution to stdout for detection only.

    Raises RuntimeError at construction if the ffmpeg binary is missing.
    """

    def __init__(
        self,
        location: str,
        width: int,
        height: int,
        *,
        record_path: str | None = None,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        resolved = shutil.which(ffmpeg_bin)
        if resolved is None:
            raise RuntimeError(f"ffmpeg binary '{ffmpeg_bin}' not found on PATH")
        self._bin: str = resolved
        self._location: str = location
        self._width: int = width
        self._height: int = height
        self._record_path: str | None = record_path
        self._is_live: bool = location.startswith(("udp://", "tcp://"))
        self._ended: bool = False
        self._backoff: float = 0.0
        self._next_spawn: float = 0.0
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: RawPipeReader | None = None
        self._logged_dims: bool = False

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
        proc = self._proc
        if reader is None or proc is None:
            return None
        # Guard against a hung ffmpeg: if the pipe has no data within the
        # timeout, treat it as a stall and reconnect.  Only for live sources
        # (files legitimately block until EOF).
        if self._is_live and proc.stdout is not None:
            try:
                fd = proc.stdout.fileno()
                readable, _, _ = select.select([fd], [], [], 2.0)
                if not readable:
                    self._teardown()
                    return None
            except (OSError, ValueError):
                self._teardown()
                return None
        frame = reader.read_frame()
        if frame is None:
            self._teardown()
            return None
        if not self._logged_dims:
            self._logged_dims = True
            h, w = frame.shape[:2]
            print(f"Decoded: {w}x{h} BGR")
        self._backoff = 0.0
        return frame

    def close(self) -> None:
        self._teardown()

    def _spawn(self) -> bool:
        now = time.monotonic()
        if self._is_live and now < self._next_spawn:
            return False

        cmd = [
            self._bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            # Sniff as little as possible so spawns/resyncs start decoding
            # fast; 32 KiB is enough to find stream headers even when joining
            # an ongoing broadcast mid-stream.
            "-probesize",
            "32768",
            "-analyzeduration",
            "0",
            # Ride out UDP bursts: never buffer up frames, drop corrupt input
            # instead of dying, and keep decoding through demux errors.
            "-fflags",
            "+nobuffer+discardcorrupt",
            "-flags",
            "low_delay",
            "-err_detect",
            "ignore_err",
            "-i",
            self._location,
            # Never let ffmpeg resample the timeline: joining mid-GOP gives
            # it a bogus framerate estimate and CFR would then duplicate or
            # drop frames to "fix" it. Passthrough emits every decoded frame
            # exactly once, preserving true capture cadence.
            "-fps_mode",
            "passthrough",
        ]
        if self._record_path is not None:
            # Output 1: codec-copy recording at full source resolution.
            # Use -y to overwrite any empty file from a failed connection attempt.
            cmd += ["-y", "-c:v", "copy", self._record_path]
        # Output 2: raw BGR frames for detection.
        # When recording, scale to detection resolution (width x height).
        # When not recording, output at source resolution (no filter needed).
        if self._record_path is not None:
            cmd += ["-vf", f"scale={self._width}:{self._height}"]
        cmd += [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            self._backoff = min(max(self._backoff * 2.0, 0.5), 2.0)
            self._next_spawn = now + self._backoff
            return False

        stream = proc.stdout
        if stream is None:
            proc.kill()
            self._backoff = min(max(self._backoff * 2.0, 0.5), 2.0)
            self._next_spawn = now + self._backoff
            return False
        self._proc = proc
        # Popen types stdout loosely as IO[Any], which does not structurally
        # overlap the readinto Protocol; hop through object per repo rules.
        self._reader = RawPipeReader(
            self._width, self._height, cast(ByteStream, cast(object, stream))
        )
        # Prepare backoff for the next reconnect attempt (only used if this
        # process dies).  Reset in read() on successful frame.
        if self._is_live:
            self._backoff = min(max(self._backoff * 2.0, 0.5), 2.0)
        return True

    def _teardown(self) -> None:
        self._reader = None
        proc = self._proc
        self._proc = None
        if not self._is_live:
            self._ended = True
        if proc is None:
            return
        # Stop the writer before closing its pipe: closing stdout first makes
        # ffmpeg race us and spew EPIPE diagnostics on the terminal.
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except (OSError, ValueError):
                pass
        if proc.stderr is not None:
            try:
                err = proc.stderr.read(4096).decode("utf-8", errors="replace").strip()
                if err:
                    first_line = err.split("\n", 1)[0]
                    print(f"\nffmpeg: {first_line}")
            except (OSError, ValueError):
                pass
            try:
                proc.stderr.close()
            except (OSError, ValueError):
                pass
