"""Tests for autosplitter.detection.state_machine (spec sections 7-10).

Logic tests use synthetic DetectionResults; the integration test feeds real
PNG fixture frames through feed_frame, per spec section 19 ("test entirely
using prerecorded frames").
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from autosplitter.config import (
    DEFAULT_ROI,
    Capture,
    Color,
    Config,
    Detection,
    LiveSplit,
    Split,
    load,
)
from autosplitter.detection import (
    DetectionResult,
    SplitEvent,
    State,
    TransitionDetector,
)

FIXTURES = Path(__file__).parent / "fixtures"
IMAGES = FIXTURES / "images"

BLACK = Color(name="black", rgb=(0, 0, 0), tolerance=15.0)
WHITE = Color(name="white", rgb=(255, 255, 255), tolerance=15.0)


def make_config(
    splits: tuple[Split, ...] | None = None,
    required_frames: int = 3,
    reset_threshold: float = 0.80,
) -> Config:
    if splits is None:
        splits = (
            Split(id=1, name="First", color="black", threshold=0.95),
            Split(id=2, name="Second", color="black", threshold=0.95),
        )
    return Config(
        capture=Capture(fps=60.0, stream_url="udp://127.0.0.1:9733", width=1920, height=1080),
        roi=DEFAULT_ROI,
        detection=Detection(required_frames=required_frames, reset_threshold=reset_threshold),
        colors={"black": BLACK, "white": WHITE},
        splits=splits,
        livesplit=LiveSplit(host="127.0.0.1", port=16834),
    )


def res(ratio: float) -> DetectionResult:
    return DetectionResult(matched_pixels=int(ratio * 100), total_pixels=100, ratio=ratio)


def events_of(machine: TransitionDetector, ratios: list[float]) -> list[SplitEvent]:
    out: list[SplitEvent] = []
    for ratio in ratios:
        event = machine.feed_result(res(ratio))
        if event is not None:
            out.append(event)
    return out


def load_png(name: str) -> np.ndarray:
    data = np.fromfile(IMAGES / name, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    assert img is not None, f"fixture {name} failed to decode"
    return img


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_starts_normal_on_first_split(self) -> None:
        machine = TransitionDetector(make_config())

        assert machine.state is State.NORMAL
        assert machine.current_split_index == 0
        assert machine.current_split is not None
        assert machine.current_split.id == 1
        assert not machine.finished

    def test_no_splits_configured_is_immediately_done(self) -> None:
        config = make_config(splits=())
        machine = TransitionDetector(config)

        assert machine.current_split is None
        assert machine.feed_result(res(1.0)) is None


# ---------------------------------------------------------------------------
# Spec section 7: transition table
# ---------------------------------------------------------------------------


class TestTransitionTable:
    def test_replays_spec_example_exactly(self) -> None:
        """Spec section 7 table with required_frames=1."""
        machine = TransitionDetector(make_config(required_frames=1))

        # (ratio, expected state after feed, expected event)
        table = [
            (0.10, State.NORMAL, None),          # Normal - nothing
            (0.10, State.NORMAL, None),          # Normal - nothing
            (1.00, State.TRANSITION, (1, "First")),   # Black - split
            (1.00, State.WAITING_FOR_RESET, None),    # Black - waiting
            (1.00, State.WAITING_FOR_RESET, None),    # Black - waiting
            (0.10, State.NORMAL, None),          # Normal - reset detector
            (0.10, State.NORMAL, None),          # Normal - nothing
            (1.00, State.TRANSITION, (2, "Second")),  # Black - split again
        ]

        seen_events: list[tuple[int, str]] = []
        for ratio, expected_state, expected_event in table:
            event = machine.feed_result(res(ratio))
            assert machine.state is expected_state, f"ratio {ratio}"
            if expected_event is not None:
                assert event is not None
                seen_events.append((event.split_id, event.name))
            else:
                assert event is None

        assert seen_events == [(1, "First"), (2, "Second")]

    def test_continuous_black_emits_only_one_split(self) -> None:
        machine = TransitionDetector(make_config(required_frames=2))
        ratios = [1.0] * 20

        assert len(events_of(machine, ratios)) == 1


# ---------------------------------------------------------------------------
# Spec section 8: debouncing
# ---------------------------------------------------------------------------


class TestDebouncing:
    def test_three_consecutive_matches_fire_on_third(self) -> None:
        machine = TransitionDetector(make_config(required_frames=3))

        assert machine.feed_result(res(0.96)) is None
        assert machine.state is State.NORMAL
        assert machine.feed_result(res(0.97)) is None
        assert machine.state is State.NORMAL

        event = machine.feed_result(res(0.96))
        assert event == SplitEvent(split_id=1, name="First")
        assert machine.state is State.TRANSITION

    def test_interrupted_streak_never_fires(self) -> None:
        machine = TransitionDetector(make_config(required_frames=3))

        events = events_of(machine, [0.96, 0.70, 0.96])

        assert events == []
        assert machine.state is State.NORMAL

    def test_single_frame_mode(self) -> None:
        machine = TransitionDetector(make_config(required_frames=1))

        event = machine.feed_result(res(0.99))
        assert event is not None
        assert machine.state is State.TRANSITION


# ---------------------------------------------------------------------------
# Spec section 9: reset threshold / hysteresis
# ---------------------------------------------------------------------------


class TestResetHysteresis:
    @pytest.fixture()
    def waiting_machine(self) -> TransitionDetector:
        machine = TransitionDetector(make_config(required_frames=1))
        machine.feed_result(res(1.0))
        assert machine.state is State.TRANSITION
        machine.feed_result(res(1.0))  # settles to WAITING_FOR_RESET
        assert machine.state is State.WAITING_FOR_RESET
        return machine

    def test_dead_zone_keeps_waiting(self, waiting_machine: TransitionDetector) -> None:
        for ratio in (0.80, 0.85, 0.90, 0.94):
            waiting_machine.feed_result(res(ratio))
            assert waiting_machine.state is State.WAITING_FOR_RESET, f"ratio {ratio}"

    def test_clearing_returns_to_next_split(self, waiting_machine: TransitionDetector) -> None:
        waiting_machine.feed_result(res(0.79))

        assert waiting_machine.state is State.NORMAL
        assert waiting_machine.current_split_index == 1
        assert waiting_machine.current_split is not None
        assert waiting_machine.current_split.id == 2

    def test_high_ratios_while_waiting_do_not_refire(
        self, waiting_machine: TransitionDetector
    ) -> None:
        events_of(waiting_machine, [1.0] * 5)

        assert waiting_machine.state is State.WAITING_FOR_RESET


# ---------------------------------------------------------------------------
# Completion (DONE) and reset
# ---------------------------------------------------------------------------


class TestDoneState:
    def test_last_clear_enters_done(self) -> None:
        machine = TransitionDetector(make_config(required_frames=1))
        machine.feed_result(res(1.0))       # fire split 1
        machine.feed_result(res(0.50))      # clear -> NORMAL on split 2
        machine.feed_result(res(1.0))       # fire split 2

        assert machine.state is State.TRANSITION
        machine.feed_result(res(0.50))      # clear -> DONE

        assert machine.state is State.DONE
        assert machine.finished
        assert machine.current_split is None

    def test_done_ignores_all_further_feeds(self) -> None:
        machine = TransitionDetector(make_config(required_frames=1))
        machine.feed_result(res(1.0))
        machine.feed_result(res(0.50))
        machine.feed_result(res(1.0))
        machine.feed_result(res(0.50))
        assert machine.finished

        for _ in range(10):
            assert machine.feed_result(res(1.0)) is None
        assert machine.state is State.DONE

    def test_midrun_not_done(self) -> None:
        machine = TransitionDetector(make_config(required_frames=1))
        machine.feed_result(res(1.0))   # fire split 1 only
        machine.feed_result(res(0.50))  # clear -> NORMAL on split 2

        assert not machine.finished
        assert machine.state is State.NORMAL


class TestResetMethod:
    def test_reset_restores_initial_state_and_reruns(self) -> None:
        machine = TransitionDetector(make_config(required_frames=2))
        first_run = events_of(machine, [1.0, 1.0])
        machine.reset()

        assert machine.state is State.NORMAL
        assert machine.current_split_index == 0
        second_run = events_of(machine, [1.0, 1.0])

        assert first_run == second_run == [SplitEvent(split_id=1, name="First")]


# ---------------------------------------------------------------------------
# Integration: real PNG fixtures through feed_frame
# ---------------------------------------------------------------------------


class TestFeedFrameIntegration:
    def test_full_sequence_with_color_switch(self) -> None:
        """3 black frames fire split 1; white clears; 3 more black fire
        split 2; white clears; 3 white frames fire split 3 (white target)."""
        cfg = load(FIXTURES / "valid_config.yaml")
        machine = TransitionDetector(cfg)

        black = load_png("roi_black_exact.png")
        white = load_png("roi_white_exact.png")

        sequence = [black] * 3 + [white] * 2 + [black] * 3 + [white] * 2 + [white] * 4
        fired: list[SplitEvent] = []
        for frame in sequence:
            event = machine.feed_frame(frame)
            if event is not None:
                fired.append(event)

        assert [(e.split_id, e.name) for e in fired] == [
            (1, "Transition 1"),
            (2, "Transition 2"),
            (3, "Transition 3"),
        ]
