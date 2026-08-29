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
    StartTransition,
    load,
)
from autosplitter.detection import (
    DetectionResult,
    SplitEvent,
    StartEvent,
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
    fps: float = 60.0,
    restart_quiet_seconds: float = 3.0,
    start_transition: StartTransition | None = None,
) -> Config:
    if splits is None:
        splits = (
            Split(id=1, name="First", color="black", threshold=0.95),
            Split(id=2, name="Second", color="black", threshold=0.95),
        )
    return Config(
        capture=Capture(fps=fps, stream_url="udp://127.0.0.1:9733", width=1920, height=1080),
        roi=DEFAULT_ROI,
        detection=Detection(
            required_frames=required_frames,
            reset_threshold=reset_threshold,
            restart_quiet_seconds=restart_quiet_seconds,
        ),
        colors={"black": BLACK, "white": WHITE},
        splits=splits,
        livesplit=LiveSplit(host="127.0.0.1", port=16834),
        start_transition=start_transition,
    )


def res(ratio: float) -> DetectionResult:
    return DetectionResult(matched_pixels=int(ratio * 100), total_pixels=100, ratio=ratio)


def events_of(machine: TransitionDetector, ratios: list[float]) -> list[SplitEvent | StartEvent]:
    out: list[SplitEvent | StartEvent] = []
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
                assert isinstance(event, SplitEvent)
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
# Optional start_transition gate
# ---------------------------------------------------------------------------


class TestStartGate:
    def _machine(
        self, splits: tuple[Split, ...] | None = None
    ) -> TransitionDetector:
        return TransitionDetector(
            make_config(
                splits=splits,
                required_frames=1,
                start_transition=StartTransition(
                    name="Run Start", color="black", threshold=0.85
                ),
            )
        )

    def test_initial_state_waits_for_start(self) -> None:
        machine = self._machine()

        assert machine.state is State.WAITING_FOR_START
        assert machine.waiting_for_start

    def test_start_settles_to_first_split_without_skipping(self) -> None:
        """Regression: clearing the start gate must NOT advance the index.

        The gate precedes the first split, so after its color clears the
        machine watches splits[0] — previously it jumped to splits[1].
        """
        machine = self._machine()
        assert machine.feed_result(res(1.0)) == StartEvent(name="Run Start")
        assert machine.state is State.TRANSITION

        machine.feed_result(res(1.0))       # settle into the reset wait
        assert machine.state is State.WAITING_FOR_RESET

        machine.feed_result(res(0.10))      # gate color cleared
        assert machine.state is State.NORMAL
        current = machine.current_split
        assert current is not None and current.id == 1

    def test_single_split_game_not_done_right_after_start(self) -> None:
        """One split + a start gate: after the gate clears, that split must
        still be watched (never straight to DONE), and firing it ends the run."""
        only = (Split(id=1, name="Only", color="black", threshold=0.95),)
        machine = self._machine(only)

        assert machine.feed_result(res(1.0)) == StartEvent(name="Run Start")
        machine.feed_result(res(1.0))
        machine.feed_result(res(0.10))
        assert machine.state is State.NORMAL
        current = machine.current_split
        assert current is not None and current.id == 1

        assert machine.feed_result(res(1.0)) == SplitEvent(split_id=1, name="Only")
        machine.feed_result(res(1.0))
        assert machine.finished

    def test_same_color_start_and_first_split_full_flow(self) -> None:
        """User scenario: gate and Level 1 share one color (dark loading
        screen). The gate must clear before Level 1 arms, then Level 1
        fires from the very same color."""
        machine = self._machine()

        assert machine.feed_result(res(1.0)) == StartEvent(name="Run Start")
        machine.feed_result(res(1.0))
        machine.feed_result(res(1.0))       # still lit: hysteresis dead zone
        assert machine.state is State.WAITING_FOR_RESET

        machine.feed_result(res(0.20))      # level 1 gameplay appears
        assert machine.state is State.NORMAL

        assert machine.feed_result(res(1.0)) == SplitEvent(split_id=1, name="First")
        machine.feed_result(res(1.0))
        assert machine.state is State.WAITING_FOR_RESET
        machine.feed_result(res(0.20))
        current = machine.current_split
        assert current is not None and current.id == 2

    def test_reset_returns_to_waiting_for_start(self) -> None:
        machine = self._machine()
        machine.feed_result(res(1.0))

        machine.reset()

        assert machine.state is State.WAITING_FOR_START


# ---------------------------------------------------------------------------
# Completion (DONE) and reset
# ---------------------------------------------------------------------------


class TestDoneState:
    def test_last_split_enters_done_even_while_color_persists(self) -> None:
        """Regression: a persistent victory screen must not stall the run.

        Hysteresis exists to shield a LATER split from re-triggering; the
        final split has none, so it skips WAITING_FOR_RESET entirely —
        even when the ratio is still far above reset_threshold.
        """
        machine = TransitionDetector(make_config(required_frames=1))
        machine.feed_result(res(1.0))       # fire split 1
        machine.feed_result(res(0.50))      # clear -> NORMAL on split 2
        machine.feed_result(res(1.0))       # fire split 2 (last)
        assert machine.state is State.TRANSITION

        # Color stays fully lit; the run must still end immediately.
        machine.feed_result(res(1.0))
        assert machine.state is State.DONE
        assert machine.finished

    def test_last_clear_enters_done(self) -> None:
        machine = TransitionDetector(make_config(required_frames=1))
        machine.feed_result(res(1.0))       # fire split 1
        machine.feed_result(res(0.50))      # clear -> NORMAL on split 2
        machine.feed_result(res(1.0))       # fire split 2 (last)

        assert machine.state is State.TRANSITION
        machine.feed_result(res(0.50))      # next frame -> DONE either way

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


class TestFirstDetector:
    def test_available_while_waiting(self) -> None:
        machine = TransitionDetector(make_config(required_frames=1))

        detector = machine.first_detector
        assert detector is not None
        assert detector is machine.current_detector

    def test_survives_done_when_current_detector_is_none(self) -> None:
        """--loop relies on this: after DONE the rearm watch still needs the
        next run's first split color."""
        machine = TransitionDetector(make_config(required_frames=1))
        machine.feed_result(res(1.0))
        machine.feed_result(res(0.50))
        machine.feed_result(res(1.0))
        machine.feed_result(res(0.50))
        assert machine.finished
        assert machine.current_detector is None

        detector = machine.first_detector
        assert detector is not None

        # Machine stays DONE until reset(); quiet feeds change nothing.
        assert machine.feed_result(res(0.50)) is None
        assert machine.finished


class TestQuietFramesNeeded:
    """Frame-counted --loop quiet period (debounce is frame-based everywhere)."""

    def test_seconds_times_fps(self) -> None:
        machine = TransitionDetector(make_config(fps=60.0, restart_quiet_seconds=3.0))
        assert machine.quiet_frames_needed == 180

    def test_reference_fixture_values(self) -> None:
        cfg = load(FIXTURES / "valid_config.yaml")
        # 2.5 s at 120 fps = exactly 300 frames.
        assert TransitionDetector(cfg).quiet_frames_needed == 300

    def test_zero_seconds_still_requires_one_frame(self) -> None:
        machine = TransitionDetector(make_config(restart_quiet_seconds=0.0))
        assert machine.quiet_frames_needed == 1

    def test_fractional_seconds_round_to_nearest_frame(self) -> None:
        # 0.04 s at 120 fps = 4.8 frames -> 5.
        machine = TransitionDetector(make_config(fps=120.0, restart_quiet_seconds=0.04))
        assert machine.quiet_frames_needed == 5


# ---------------------------------------------------------------------------
# Integration: real PNG fixtures through feed_frame
# ---------------------------------------------------------------------------


class TestFeedFrameIntegration:
    def test_full_sequence_with_color_switch(self) -> None:
        """6 black frames fire split 1; white clears; 6 more black fire
        split 2; white clears; 6 white frames fire split 3 (white target).
        Runs match required_frames=6 in valid_config.yaml."""
        cfg = load(FIXTURES / "valid_config.yaml")
        machine = TransitionDetector(cfg)

        black = load_png("roi_black_exact.png")
        white = load_png("roi_white_exact.png")

        sequence = [black] * 6 + [white] * 2 + [black] * 6 + [white] * 2 + [white] * 6
        fired: list[SplitEvent] = []
        for frame in sequence:
            event = machine.feed_frame(frame)
            if event is not None:
                assert isinstance(event, SplitEvent)
                fired.append(event)

        assert [(e.split_id, e.name) for e in fired] == [
            (1, "Transition 1"),
            (2, "Transition 2"),
            (3, "Transition 3"),
        ]


# ---------------------------------------------------------------------------
# OCR resync
# ---------------------------------------------------------------------------


def _make_machine_with_ocr(start_index: int = 0) -> TransitionDetector:
    """Build a 5-split machine with OCR config."""
    cfg = Config(
        capture=Capture(fps=60, stream_url="udp://127.0.0.1:9733", width=1920, height=1080),
        roi=DEFAULT_ROI,
        detection=Detection(required_frames=3, reset_threshold=0.5),
        colors={"black": Color(name="black", rgb=(0, 0, 0), tolerance=15)},
        splits=tuple(
            Split(id=i, name=f"Split {i}", color="black", threshold=0.9)
            for i in range(1, 6)
        ),
        livesplit=LiveSplit(host="127.0.0.1", port=16834),
        ocr=None,
    )
    machine = TransitionDetector(cfg)
    machine._index = start_index
    return machine


class TestOcrResync:
    def test_resync_in_normal(self) -> None:
        machine = _make_machine_with_ocr(start_index=0)
        assert machine.current_split_index == 0
        machine._resync_index(3)
        assert machine.current_split_index == 3
        assert machine._streak == 0

    def test_resync_blocked_in_transition(self) -> None:
        machine = _make_machine_with_ocr(start_index=1)
        machine._state = State.TRANSITION
        machine._resync_index(3)
        assert machine.current_split_index == 1

    def test_resync_blocked_in_waiting_for_reset(self) -> None:
        machine = _make_machine_with_ocr(start_index=1)
        machine._state = State.WAITING_FOR_RESET
        machine._resync_index(3)
        assert machine.current_split_index == 1

    def test_resync_blocked_in_done(self) -> None:
        machine = _make_machine_with_ocr(start_index=5)
        machine._state = State.DONE
        machine._resync_index(3)
        assert machine.current_split_index == 5

    def test_resync_ignores_same_index(self) -> None:
        machine = _make_machine_with_ocr(start_index=2)
        machine._streak = 5
        machine._resync_index(2)
        assert machine._streak == 5

    def test_resync_ignores_out_of_range(self) -> None:
        machine = _make_machine_with_ocr(start_index=0)
        machine._resync_index(99)
        assert machine.current_split_index == 0
        machine._resync_index(-1)
        assert machine.current_split_index == 0

    def test_resync_resets_streak(self) -> None:
        machine = _make_machine_with_ocr(start_index=0)
        machine._streak = 10
        machine._resync_index(2)
        assert machine._streak == 0

    def test_check_ocr_resync_without_ocr_config(self) -> None:
        cfg = Config(
            capture=Capture(fps=60, stream_url="udp://127.0.0.1:9733", width=1920, height=1080),
            roi=DEFAULT_ROI,
            detection=Detection(required_frames=3, reset_threshold=0.5),
            colors={"black": Color(name="black", rgb=(0, 0, 0), tolerance=15)},
            splits=(Split(id=1, name="S", color="black", threshold=0.9),),
            livesplit=LiveSplit(host="127.0.0.1", port=16834),
            ocr=None,
        )
        machine = TransitionDetector(cfg)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        machine.check_ocr_resync(frame, 0)
        assert machine.current_split_index == 0

    def test_ocr_text_none_initially(self) -> None:
        machine = _make_machine_with_ocr()
        assert machine.ocr_text is None

    def test_ocr_text_none_without_ocr(self) -> None:
        cfg = Config(
            capture=Capture(fps=60, stream_url="udp://127.0.0.1:9733", width=1920, height=1080),
            roi=DEFAULT_ROI,
            detection=Detection(required_frames=3, reset_threshold=0.5),
            colors={"black": Color(name="black", rgb=(0, 0, 0), tolerance=15)},
            splits=(Split(id=1, name="S", color="black", threshold=0.9),),
            livesplit=LiveSplit(host="127.0.0.1", port=16834),
            ocr=None,
        )
        machine = TransitionDetector(cfg)
        assert machine.ocr_text is None

    def test_ocr_text_updates_from_result(self) -> None:
        from autosplitter.detection.ocr_detector import OcrResult

        machine = _make_machine_with_ocr()
        result = OcrResult(text="7", confidence=92.0, split_index=6, frame_number=0)
        machine._ocr_latest = result
        assert machine.ocr_text == "7"

    def test_ocr_text_empty_text_returns_none(self) -> None:
        from autosplitter.detection.ocr_detector import OcrResult

        machine = _make_machine_with_ocr()
        result = OcrResult(text="", confidence=50.0, split_index=None, frame_number=0)
        machine._ocr_latest = result
        assert machine.ocr_text is None
