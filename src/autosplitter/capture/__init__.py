"""Frame acquisition, independent of detection logic."""

from autosplitter.capture.frame_source import (
    FrameSource,
    MpegTsFrameSource,
    RawPipeReader,
)

__all__ = [
    "FrameSource",
    "MpegTsFrameSource",
    "RawPipeReader",
]
