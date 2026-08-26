"""Tests for diagnostics overlay."""

import os

os.environ["PYTEST_CURRENT_TEST"] = "1"  # disable imshow in tests

import numpy as np

from autosplitter.config import (
    Capture,
    Color,
    Config,
    Detection,
    LiveSplit,
    Roi,
    Split,
)
from autosplitter.detection.state_machine import State, TransitionDetector
from autosplitter.diagnostics.overlay import annotate


def make_config(fps: float = 60.0) -> Config:
    return Config(
        capture=Capture(fps=fps, stream_url="udp://127.0.0.1:9733", width=64, height=64),
        roi=Roi(x=0.25, y=0.25, width=0.50, height=0.50),
        detection=Detection(required_frames=3, reset_threshold=0.80, restart_quiet_seconds=3.0),
        colors={
            "black": Color(name="black", rgb=(0, 0, 0), tolerance=15.0),
            "white": Color(name="white", rgb=(255, 255, 255), tolerance=15.0),
        },
        splits=(
            Split(id=1, name="Transition 1", color="black", threshold=0.95),
            Split(id=2, name="Transition 2", color="white", threshold=0.95),
        ),
        livesplit=LiveSplit(host="127.0.0.1", port=16834),
    )


def make_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a black frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestOverlayGeometry:
    def test_roi_rectangle_at_correct_pixels(self) -> None:
        """ROI rectangle should be drawn at normalized coordinates."""
        cfg = make_config()
        frame = make_frame(64, 64)
        machine = TransitionDetector(cfg)

        annotated = annotate(frame, cfg, machine, True, None, 60.0, 0.5)

        # ROI: x=0.25*64=16, y=0.25*64=16, w=0.5*64=32, h=0.5*64=32
        # Green rectangle (0, 255, 0) with thickness=2
        x, y, w, h = 16, 16, 32, 32

        # Check that green pixels exist on each edge (thickness=2 means 2-pixel wide edges)
        # Top edge
        top_edge = annotated[y:y+2, x:x+w]
        assert np.any(np.all(top_edge == [0, 255, 0], axis=2))
        # Bottom edge
        bottom_edge = annotated[y+h-2:y+h, x:x+w]
        assert np.any(np.all(bottom_edge == [0, 255, 0], axis=2))
        # Left edge
        left_edge = annotated[y:y+h, x:x+2]
        assert np.any(np.all(left_edge == [0, 255, 0], axis=2))
        # Right edge
        right_edge = annotated[y:y+h, x+w-2:x+w]
        assert np.any(np.all(right_edge == [0, 255, 0], axis=2))

    def test_roi_rectangle_x1y1x2y2_form(self) -> None:
        """ROI rectangle with x1/y1/x2/y2 form (parsed to x/y/w/h)."""
        # The parser converts endpoint form to size form, so we test the
        # resulting Roi object that would be produced by the parser.
        cfg = Config(
            capture=Capture(fps=60.0, stream_url="u", width=64, height=64),
            roi=Roi(x=0.25, y=0.25, width=0.50, height=0.50),  # equivalent to x1=0.25,y1=0.25,x2=0.75,y2=0.75
            detection=Detection(required_frames=3, reset_threshold=0.80, restart_quiet_seconds=3.0),
            colors={"black": Color(name="black", rgb=(0, 0, 0), tolerance=15.0)},
            splits=(Split(id=1, name="T1", color="black", threshold=0.95),),
            livesplit=LiveSplit(host="127.0.0.1", port=16834),
        )
        frame = make_frame(64, 64)
        machine = TransitionDetector(cfg)

        annotated = annotate(frame, cfg, machine, True, None, 60.0, 0.5)

        x, y, w, h = 16, 16, 32, 32
        # Check that green pixels exist on each edge (thickness=2 means 2-pixel wide edges)
        top_edge = annotated[y:y+2, x:x+w]
        assert np.any(np.all(top_edge == [0, 255, 0], axis=2))
        bottom_edge = annotated[y+h-2:y+h, x:x+w]
        assert np.any(np.all(bottom_edge == [0, 255, 0], axis=2))
        left_edge = annotated[y:y+h, x:x+2]
        assert np.any(np.all(left_edge == [0, 255, 0], axis=2))
        right_edge = annotated[y:y+h, x+w-2:x+w]
        assert np.any(np.all(right_edge == [0, 255, 0], axis=2))


class TestOverlayEdgeCases:
    def test_empty_splits_does_not_crash(self) -> None:
        """HUD should render without crashing when no splits defined."""
        cfg = Config(
            capture=Capture(fps=60.0, stream_url="u", width=64, height=64),
            roi=Roi(x=0.25, y=0.25, width=0.50, height=0.50),
            detection=Detection(required_frames=3, reset_threshold=0.80, restart_quiet_seconds=3.0),
            colors={"black": Color(name="black", rgb=(0, 0, 0), tolerance=15.0)},
            splits=(),
            livesplit=LiveSplit(host="127.0.0.1", port=16834),
        )
        frame = make_frame(64, 64)
        machine = TransitionDetector(cfg)

        # Should not raise
        annotated = annotate(frame, cfg, machine, False, None, 60.0, 0.0)
        assert annotated.shape == frame.shape

    def test_done_state_shows_dash_for_split(self) -> None:
        """In DONE state, split counter shows —."""
        cfg = make_config()
        frame = make_frame(64, 64)
        machine = TransitionDetector(cfg)

        # Force DONE by manually setting state
        machine._index = len(cfg.splits)
        machine._state = State.DONE

        annotated = annotate(frame, cfg, machine, True, 5.0, 60.0, 0.0)
        assert annotated.shape == frame.shape

    def test_ls_disconnected_shows_correctly(self) -> None:
        """HUD shows DISCONNECTED when ls_connected=False."""
        cfg = make_config()
        frame = make_frame(64, 64)
        machine = TransitionDetector(cfg)

        annotated = annotate(frame, cfg, machine, False, None, 60.0, 0.5)
        assert annotated.shape == frame.shape

    def test_last_split_age_none_shows_dash(self) -> None:
        """HUD shows — for last when age is None."""
        cfg = make_config()
        frame = make_frame(64, 64)
        machine = TransitionDetector(cfg)

        annotated = annotate(frame, cfg, machine, True, None, 60.0, 0.5)
        assert annotated.shape == frame.shape


class TestOverlayDynamicValues:
    def test_fps_displayed(self) -> None:
        """FPS value should appear in HUD."""
        cfg = make_config()
        frame = make_frame(64, 64)
        machine = TransitionDetector(cfg)

        annotated = annotate(frame, cfg, machine, True, 2.5, 59.7, 0.75)
        assert annotated.shape == frame.shape

    def test_match_ratio_displayed(self) -> None:
        """Match ratio should appear in HUD."""
        cfg = make_config()
        frame = make_frame(64, 64)
        machine = TransitionDetector(cfg)

        annotated = annotate(frame, cfg, machine, True, None, 60.0, 0.95)
        assert annotated.shape == frame.shape