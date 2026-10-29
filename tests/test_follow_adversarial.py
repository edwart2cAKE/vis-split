"""Adversarial tests for follow_timer increment 1 (advance_to, TimerFollower,
_FollowPoller/_apply_decision, get_split_count, config). Failing tests expose
real defects; do not xfail."""

from __future__ import annotations

from pathlib import Path
from typing import cast, final, override

import pytest

CFG_TEXT = '''capture:
  fps: 60
  stream_url: "udp://127.0.0.1:9733"
  width: 1920
  height: 1080
detection:
  required_frames: 3
  reset_threshold: 0.80
colors:
  black:
    rgb: [0, 0, 0]
    tolerance: 15
splits:
  - id: 1
    name: "S"
    color: black
    threshold: 0.95
livesplit:
  host: "127.0.0.1"
  port: 16834
'''


def write_config(tmp_path: Path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)

from autosplitter import main as main_mod
from autosplitter.config import (
    DEFAULT_ROI,
    Capture,
    Color,
    Config,
    ConfigError,
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
from autosplitter.livesplit import Action, Decision, LiveSplitClient, TimerFollower

BLACK = Color(name="black", rgb=(0, 0, 0), tolerance=15.0)
WHITE = Color(name="white", rgb=(255, 255, 255), tolerance=15.0)


def make_config(n: int, frames: int = 1, start: bool = False) -> Config:
    return Config(
        capture=Capture(fps=60.0, stream_url="udp://127.0.0.1:9733", width=1920, height=1080),
        roi=DEFAULT_ROI,
        detection=Detection(required_frames=frames, reset_threshold=0.80, restart_quiet_seconds=3.0),
        colors={"black": BLACK, "white": WHITE},
        splits=tuple(
            Split(id=i + 1, name=chr(ord("A") + i), color="black", threshold=0.95)
            for i in range(n)
        ),
        livesplit=LiveSplit(host="127.0.0.1", port=16834),
        start_transition=StartTransition(name="Start", color="white", threshold=0.95)
        if start
        else None,
    )


def machine(n: int, frames: int = 1, start: bool = False) -> TransitionDetector:
    return TransitionDetector(make_config(n, frames, start))


def feed(m: TransitionDetector, *ratios: float) -> list[SplitEvent | StartEvent]:
    out: list[SplitEvent | StartEvent] = []
    for r in ratios:
        e = m.feed_result(DetectionResult(int(r * 100), 100, r))
        if e is not None:
            out.append(e)
    return out


def ids(evs: list[SplitEvent | StartEvent]) -> list[int]:
    return [e.split_id for e in evs if isinstance(e, SplitEvent)]


LIT, DARK = 1.0, 0.0

# ------------------------------------------------------------ advance_to / machine


def test_split_then_advance_then_lit_then_clear_then_next_fires_once() -> None:
    m = machine(4)
    assert ids(feed(m, LIT)) == [1]  # TRANSITION, expected 1
    assert not m.advance_to(2)  # refused in TRANSITION
    feed(m, LIT)  # settle into WAITING_FOR_RESET
    assert m.advance_to(2)  # LS manually split once more
    assert m.state is State.WAITING_FOR_RESET
    assert m.expected_ls_index == 2
    assert feed(m, LIT, LIT, LIT) == []  # still lit: no re-fire
    assert m.current_split_index == 2
    assert feed(m, DARK) == []
    assert m.state is State.NORMAL and m.current_split_index == 2
    assert ids(feed(m, LIT)) == [3]


def test_advance_in_waiting_for_reset_cancels_pending_plus_one() -> None:
    m = machine(4)
    feed(m, LIT, LIT)  # fire A, settle into WAITING_FOR_RESET
    assert m.state is State.WAITING_FOR_RESET and m.expected_ls_index == 1
    assert not m.advance_to(1)  # equal to expected: refused
    assert m.advance_to(2)
    feed(m, DARK)
    assert m.current_split_index == 2  # not 3


def test_advance_refused_in_transition_and_done() -> None:
    m = machine(3)
    feed(m, LIT)
    assert m.state is State.TRANSITION
    assert not m.advance_to(2)
    m2 = machine(1)
    feed(m2, LIT, LIT)
    assert m2.state is State.DONE
    assert not m2.advance_to(1)


def test_advance_to_n_goes_done_then_reset_rearms() -> None:
    m = machine(3)
    assert m.advance_to(3)
    assert m.finished and m.current_split is None and m.current_detector is None
    assert m.first_detector is not None
    assert feed(m, LIT, LIT) == []
    m.reset()
    assert m.state is State.NORMAL and m.current_split_index == 0
    assert ids(feed(m, LIT)) == [1]


def test_advance_beyond_n_refused_and_state_untouched() -> None:
    m = machine(3)
    assert not m.advance_to(4)
    assert m.state is State.NORMAL and m.current_split_index == 0


def test_advance_negative_refused() -> None:
    m = machine(3)
    assert not m.advance_to(-1)


def test_advance_to_same_index_refused() -> None:
    m = machine(3)
    assert not m.advance_to(0)


def test_double_advance() -> None:
    m = machine(5)
    assert m.advance_to(2)
    assert m.expected_ls_index == 2
    assert not m.advance_to(2)
    assert m.advance_to(3)
    feed(m, DARK)
    assert m.current_split_index == 3 and m.state is State.NORMAL


def test_advance_while_target_color_lit_blocks_until_clear_then_final_split() -> None:
    m = machine(3)
    assert m.advance_to(2)  # N-1 (last)
    assert feed(m, LIT, LIT, LIT) == []
    feed(m, DARK)
    assert ids(feed(m, LIT)) == [3]
    feed(m, LIT)
    assert m.finished and m.current_split is None


def test_advance_hysteresis_boundary() -> None:
    m = machine(3)
    assert m.advance_to(1)
    feed(m, 0.80)  # exactly reset_threshold: not cleared
    assert m.state is State.WAITING_FOR_RESET
    feed(m, 0.79)
    assert m.state is State.NORMAL


def test_advance_clears_streak_mid_debounce() -> None:
    m = machine(4, frames=3)
    feed(m, LIT, LIT)  # streak 2 on A
    assert m.advance_to(1)
    feed(m, DARK)
    # fresh debounce needed for B
    assert ids(feed(m, LIT, LIT)) == []
    assert ids(feed(m, LIT)) == [2]


def test_streak_not_inherited_after_advance_from_normal() -> None:
    m = machine(4, frames=2)
    feed(m, LIT)  # streak 1
    m.advance_to(1)
    feed(m, DARK, LIT)
    assert m.state is State.NORMAL
    assert ids(feed(m, LIT)) == [2]


def test_clear_streak() -> None:
    m = machine(2, frames=2)
    feed(m, LIT)
    m.clear_streak()
    assert feed(m, LIT) == []
    assert ids(feed(m, LIT)) == [1]


def test_n1_config_advance() -> None:
    m = machine(1)
    assert not m.advance_to(0)
    assert m.advance_to(1)
    assert m.finished


def test_waiting_for_start_advance_zero_allowed_then_gate_does_not_refire() -> None:
    m = machine(3, start=True)
    assert m.state is State.WAITING_FOR_START
    assert m.advance_to(0)  # floor 0 in WFS
    # Now in WAITING_FOR_RESET watching splits[0] color (black); dark -> NORMAL
    assert m.state is State.WAITING_FOR_RESET and m.expected_ls_index == 0
    feed(m, DARK)
    assert m.state is State.NORMAL and m.current_split_index == 0
    assert not m.waiting_for_start  # start gate permanently skipped


def test_waiting_for_start_advance_to_2() -> None:
    m = machine(4, start=True)
    assert m.advance_to(2)
    feed(m, DARK)
    assert m.current_split_index == 2


def test_waiting_for_start_advance_to_n_done() -> None:
    m = machine(2, start=True)
    assert m.advance_to(2)
    assert m.finished
    m.reset()
    assert m.state is State.WAITING_FOR_START


def test_start_settling_advance_requires_gt_expected() -> None:
    m = machine(3, start=True)
    feed(m, LIT)  # gate fires (white==LIT on ratio only)
    feed(m, LIT)  # TRANSITION -> WAITING_FOR_RESET (start settle)
    assert m.state is State.WAITING_FOR_RESET and m.expected_ls_index == 0
    assert not m.advance_to(0)
    assert m.advance_to(1)
    feed(m, DARK)
    assert m.current_split_index == 1 and m.state is State.NORMAL


def test_rollback_after_advance_roundtrip() -> None:
    m = machine(4)
    assert m.advance_to(3)
    feed(m, DARK)
    assert m.rollback_to(1)
    assert m.current_split_index == 1
    assert m.advance_to(3)


def test_manual_split_race_with_detector_fire() -> None:
    """LS manually split (L=E+1), then detector fires: LS ends at idx+2."""
    m = machine(5)
    assert ids(feed(m, LIT, LIT)) == [1]  # LS idx 1, E=1, settled
    # (user manual split: LS idx 2) -> follower would request advance to 2
    assert m.advance_to(2)
    feed(m, DARK)
    assert ids(feed(m, LIT)) == [3]  # third split fires; LS idx 3 == expected
    assert m.expected_ls_index == 3


def test_manual_split_before_detector_fire_results_in_double_count() -> None:
    """LS at E+1 not yet confirmed (1 poll) and detector fires: E becomes L.
    Next poll sees L==E (in step) -> no skip; no double-advance."""
    m = machine(5)
    f = TimerFollower()
    d = f.observe(
        phase="Running", ls_index=1, expected=m.expected_ls_index, n_splits=5, now=10.0,
        last_split_time=None, suppress=False, can_back=True, can_fwd=True, waiting_for_start=False,
    )
    assert d.action is Action.NONE
    feed(m, LIT)  # detector fires A -> expected 1
    d = f.observe(
        phase="Running", ls_index=1, expected=m.expected_ls_index, n_splits=5, now=12.0,
        last_split_time=11.0, suppress=False, can_back=True, can_fwd=True, waiting_for_start=False,
    )
    assert d.action is Action.NONE


# ------------------------------------------------------------ TimerFollower


def obs(f: TimerFollower, **kw: object) -> Decision:
    args: dict[str, object] = {
        "phase": "Running", "ls_index": 2, "expected": 1, "n_splits": 5, "now": 100.0,
        "last_split_time": 0.0, "suppress": False, "can_back": True, "can_fwd": True,
        "waiting_for_start": False,
    }
    args.update(kw)
    return f.observe(**args)  # pyright: ignore[reportArgumentType]


def test_skip_to_last_index_allowed_and_n_refused() -> None:
    f = TimerFollower()
    obs(f, ls_index=4, expected=0)
    assert obs(f, ls_index=4, expected=0) == Decision(Action.SKIP_FORWARD, 4)
    f = TimerFollower()
    obs(f, ls_index=5, expected=0)
    assert obs(f, ls_index=5, expected=0).action is Action.NONE


def test_flapping_phase_resets_candidate() -> None:
    f = TimerFollower()
    obs(f)
    assert obs(f, phase=None).action is Action.NONE
    assert obs(f).action is Action.NONE  # count restarted
    assert obs(f).action is Action.SKIP_FORWARD


def test_ls_index_none_resets_candidate() -> None:
    f = TimerFollower()
    obs(f)
    obs(f, ls_index=None)
    assert obs(f).action is Action.NONE


@pytest.mark.parametrize("phase", ["Ended", "NotRunning", "running", "", "Weird"])
def test_no_skip_for_other_phases(phase: str) -> None:
    f = TimerFollower()
    for _ in range(5):
        assert obs(f, phase=phase).action is Action.NONE


def test_paused_skips() -> None:
    f = TimerFollower()
    obs(f, phase="Paused")
    assert obs(f, phase="Paused").action is Action.SKIP_FORWARD


def test_pause_flap_between_running_paused_keeps_candidate() -> None:
    f = TimerFollower()
    obs(f, phase="Running")
    assert obs(f, phase="Paused").action is Action.SKIP_FORWARD


def test_notrunning_between_resets_candidate() -> None:
    f = TimerFollower()
    obs(f)
    obs(f, phase="NotRunning")
    assert obs(f).action is Action.NONE


def test_l_jumping_around_never_confirms() -> None:
    f = TimerFollower()
    for ls in (2, 3, 2, 3, 2, 3):
        assert obs(f, ls_index=ls).action is Action.NONE


def test_expected_changes_between_polls_resets() -> None:
    f = TimerFollower()
    obs(f, ls_index=3, expected=1)
    assert obs(f, ls_index=3, expected=2).action is Action.NONE
    assert obs(f, ls_index=3, expected=2).action is Action.SKIP_FORWARD


def test_cannot_fwd_clears_candidate() -> None:
    f = TimerFollower()
    obs(f)
    obs(f, can_fwd=False)
    assert obs(f).action is Action.NONE


def test_waiting_for_start_blocks_skip() -> None:
    f = TimerFollower()
    # Increment 2: Running at the gate is MANUAL_START, never SKIP_FORWARD.
    for _ in range(4):
        assert obs(f, waiting_for_start=True).action in {Action.NONE, Action.MANUAL_START}


def test_negative_ls_index_no_decision() -> None:
    f = TimerFollower()
    for _ in range(3):
        assert obs(f, ls_index=-1, expected=0).action is Action.NONE


def test_undo_then_skip_interplay() -> None:
    f = TimerFollower()
    assert obs(f, ls_index=3, expected=3).action is Action.NONE  # arm
    obs(f, ls_index=1, expected=3, last_split_time=None)
    d = obs(f, ls_index=1, expected=3, last_split_time=None)
    assert d == Decision(Action.ROLLBACK, 1)
    # machine rolled back; expected=1. LS user re-splits to 2,3 quickly.
    obs(f, ls_index=3, expected=1)
    assert obs(f, ls_index=3, expected=1) == Decision(Action.SKIP_FORWARD, 3)


def test_rollback_has_priority_and_clears_skip_candidate() -> None:
    f = TimerFollower()
    obs(f, ls_index=2, expected=2)  # arm
    obs(f, ls_index=4, expected=2)  # skip candidate (key 4,2)
    assert f._skip_count == 1  # pyright: ignore[reportPrivateUsage]
    obs(f, ls_index=1, expected=2, last_split_time=None)
    assert f._skip_key is None  # pyright: ignore[reportPrivateUsage]


def test_disarm_forgets_skip_candidate() -> None:
    f = TimerFollower()
    obs(f)
    f.disarm()
    assert obs(f).action is Action.NONE


def test_time_anomaly_does_not_affect_skip() -> None:
    f = TimerFollower()
    obs(f, now=-5.0, last_split_time=1e12)
    assert obs(f, now=-5.0, last_split_time=1e12).action is Action.SKIP_FORWARD


def test_skip_not_blocked_by_suppress() -> None:
    """suppress is for undelivered splits; LS ahead of detector is unrelated."""
    f = TimerFollower()
    obs(f, suppress=True)
    assert obs(f, suppress=True).action is Action.SKIP_FORWARD


# ------------------------------------------------------------ get_split_count


@final
class FakeQ(LiveSplitClient):
    def __init__(self, reply: str | None) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.reply: str | None = reply

    @override
    def query(self, command: str) -> str | None:
        return self.reply


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("5", 5), (" 5\r", 5), ("+5", 5), ("0", 0), ("007", 7),
        ("-1", None), ("5.0", None), ("", None), ("abc", None), ("1e3", None),
        ("5 6", None), (None, None), ("٣", None),
    ],
)
def test_get_split_count_replies(reply: str | None, expected: int | None) -> None:
    assert FakeQ(reply).get_split_count() == expected


# ------------------------------------------------------------ _FollowPoller


@final
class FakeClient:
    def __init__(
        self,
        phase: str | None = "Running",
        index: int | None = 0,
        name: str | None = "A",
        count: int | None = 3,
    ) -> None:
        self.phase: str | None = phase
        self.index: int | None = index
        self.name: str | None = name
        self.count: int | None = count
        self.calls: list[str] = []

    def get_current_timer_phase(self) -> str | None:
        self.calls.append("phase")
        return self.phase

    def get_split_index(self) -> int | None:
        self.calls.append("index")
        return self.index

    def get_current_split_name(self) -> str | None:
        self.calls.append("name")
        return self.name

    def get_split_count(self) -> int | None:
        self.calls.append("count")
        return self.count


def poll(
    p: main_mod._FollowPoller,  # pyright: ignore[reportPrivateUsage]
    c: FakeClient,
    m: TransitionDetector,
    cached: str | None = "A",
    last: float | None = None,
    suppress: bool = False,
) -> main_mod._FollowPollResult:  # pyright: ignore[reportPrivateUsage]
    return p.poll(cast(LiveSplitClient, cast(object, c)), m, cached, last, suppress)


def new_poller(n: int) -> main_mod._FollowPoller:  # pyright: ignore[reportPrivateUsage]
    return main_mod._FollowPoller(n)  # pyright: ignore[reportPrivateUsage]


def test_poll_phase_none_aborts_with_no_more_queries() -> None:
    c = FakeClient(phase=None)
    r = poll(new_poller(3), c, machine(3), cached="X", suppress=True)
    assert c.calls == ["phase"]
    assert not r.ok and r.name == "X" and r.suppress is True


def test_poll_notrunning_skips_index_and_name() -> None:
    c = FakeClient(phase="NotRunning")
    r = poll(new_poller(3), c, machine(3))
    assert c.calls == ["phase"]
    assert not r.ok and r.phase == "NotRunning"


def test_poll_index_none_aborts_before_name() -> None:
    c = FakeClient(index=None)
    r = poll(new_poller(3), c, machine(3))
    assert c.calls == ["phase", "index"]
    assert not r.ok


def test_poll_in_step_second_time_skips_name_query() -> None:
    c = FakeClient(index=0)
    p = new_poller(3)
    m = machine(3)
    poll(p, c, m)
    c.calls.clear()
    poll(p, c, m, cached="A")
    assert c.calls == ["phase", "index"]


def test_poll_name_none_cached_requeries() -> None:
    c = FakeClient(index=0)
    p = new_poller(3)
    m = machine(3)
    poll(p, c, m, cached="A")
    c.calls.clear()
    poll(p, c, m, cached=None)
    assert "name" in c.calls


def test_poll_count_checked_once_per_connection() -> None:
    c = FakeClient()
    p = new_poller(3)
    m = machine(3)
    poll(p, c, m)
    poll(p, c, m)
    assert c.calls.count("count") == 1
    p.connection_lost()
    poll(p, c, m)
    assert c.calls.count("count") == 2


def test_poll_count_none_does_not_retry_and_leaves_skip_enabled() -> None:
    c = FakeClient(count=None, index=2)
    p = new_poller(3)
    m = machine(3)
    poll(p, c, m)
    poll(p, c, m)
    assert c.calls.count("count") == 1  # documented: never rechecked
    assert m.current_split_index == 2 or m.state is State.WAITING_FOR_RESET


def test_poll_count_not_checked_while_phase_unqueryable_then_checked() -> None:
    c = FakeClient(phase="NotRunning")
    p = new_poller(3)
    poll(p, c, machine(3))
    assert "count" not in c.calls


def test_poll_count_mismatch_disables_skip(capsys: pytest.CaptureFixture[str]) -> None:
    c = FakeClient(index=2, count=5)
    p = new_poller(3)
    m = machine(3)
    for _ in range(4):
        poll(p, c, m)
    assert m.current_split_index == 0 and m.state is State.NORMAL
    assert "WARN" in capsys.readouterr().out


def test_poll_count_mismatch_still_allows_rollback() -> None:
    c = FakeClient(index=1, count=5)
    p = new_poller(4)
    m = machine(4)
    m.advance_to(3)
    feed(m, DARK)  # NORMAL at 3
    # arm: L==expected
    c.index = 3
    poll(p, c, m)
    c.index = 1
    poll(p, c, m)
    r = poll(p, c, m)
    assert r.message is not None and m.current_split_index == 1


def test_poll_skip_after_two_polls_and_message() -> None:
    c = FakeClient(index=2, name="C")
    p = new_poller(3)
    m = machine(3)
    r1 = poll(p, c, m)
    assert r1.message is None and m.current_split_index == 0
    r2 = poll(p, c, m, suppress=True)
    assert r2.message is not None and "(index 2)" in r2.message
    assert r2.suppress is False
    assert m.state is State.WAITING_FOR_RESET and m.current_split_index == 2


def test_poll_after_skip_in_step_no_repeat() -> None:
    c = FakeClient(index=2)
    p = new_poller(3)
    m = machine(3)
    poll(p, c, m)
    poll(p, c, m)
    for _ in range(4):
        assert poll(p, c, m).message is None
    assert m.current_split_index == 2


def test_poll_skip_in_transition_state_not_applied_and_resets() -> None:
    c = FakeClient(index=3, count=4)
    p = new_poller(4)
    m = machine(4)
    feed(m, LIT)  # TRANSITION, E=1
    assert m.state is State.TRANSITION
    for _ in range(4):
        assert poll(p, c, m).message is None
    assert m.state is State.TRANSITION


def test_poll_ended_phase_no_skip() -> None:
    c = FakeClient(phase="Ended", index=2)
    p = new_poller(3)
    m = machine(3)
    for _ in range(4):
        assert poll(p, c, m).message is None


def test_poll_waiting_for_start_never_skips() -> None:
    c = FakeClient(index=2)
    p = new_poller(3)
    m = machine(3, start=True)
    # Increment 2: a started timer at the gate is MANUAL_START (one jump),
    # not a skip; afterwards nothing further happens.
    first = poll(p, c, m).message
    assert first is not None and "left the start gate" in first
    assert not m.waiting_for_start
    for _ in range(4):
        assert poll(p, c, m).message is None


def test_poll_rollback_clears_suppress() -> None:
    c = FakeClient(index=2)
    p = new_poller(4)
    m = machine(4)
    m.advance_to(2)
    feed(m, DARK)
    poll(p, c, m)  # arm
    c.index = 0
    poll(p, c, m, last=None, suppress=False)
    r = poll(p, c, m, last=None, suppress=False)
    assert r.message is not None and "UNSPLIT" in r.message
    assert m.current_split_index == 0


def test_poll_suppress_cleared_when_in_step() -> None:
    c = FakeClient(index=0)
    r = poll(new_poller(3), c, machine(3), suppress=True)
    assert r.suppress is False


def test_poll_suppress_kept_when_behind_and_not_confirmed() -> None:
    c = FakeClient(index=0)
    m = machine(3)
    feed(m, LIT)  # expected 1, LS still 0 (split undelivered)
    r = poll(new_poller(3), c, m, suppress=True)
    assert r.suppress is True and r.message is None


def test_poll_rollback_blocked_while_suppressed() -> None:
    c = FakeClient(index=0)
    p = new_poller(3)
    m = machine(3)
    c.index = 0
    poll(p, c, m)  # arm
    feed(m, LIT)
    for _ in range(4):
        r = poll(p, c, m, last=None, suppress=True)
        assert r.message is None
    assert m.expected_ls_index == 1


def test_poll_disarm_prevents_rollback_until_rearmed() -> None:
    c = FakeClient(index=2)
    p = new_poller(4)
    m = machine(4)
    m.advance_to(2)
    feed(m, DARK)
    poll(p, c, m)  # arm
    p.disarm()
    c.index = 0
    for _ in range(4):
        assert poll(p, c, m).message is None


def test_poll_stale_n_splits_after_machine_shrinks() -> None:
    """Poller was built with n=5; machine config later has 3 splits. A skip
    to index 4 must be rejected by the machine, not crash."""
    c = FakeClient(index=4, count=5)
    p = new_poller(5)
    m = machine(3)
    for _ in range(4):
        poll(p, c, m)
    assert m.current_split_index == 0


def test_apply_decision_variants() -> None:
    m = machine(3)
    apply = main_mod._apply_decision  # pyright: ignore[reportPrivateUsage]
    assert apply(m, Decision(Action.NONE)) is None
    assert apply(m, Decision(Action.NONE, 2)) is None
    assert apply(m, Decision(Action.SKIP_FORWARD)) is None
    assert apply(m, Decision(Action.SKIP_FORWARD, 0)) is None  # refused
    msg = apply(m, Decision(Action.SKIP_FORWARD, 2))
    assert msg is not None and '"C"' in msg
    assert apply(m, Decision(Action.ROLLBACK, 5)) is None


def test_apply_decision_skip_to_n_message_has_empty_name() -> None:
    m = machine(3)
    msg = main_mod._apply_decision(m, Decision(Action.SKIP_FORWARD, 3))  # pyright: ignore[reportPrivateUsage]
    assert msg is not None and '""' in msg and m.finished


# ------------------------------------------------------------ config


def _cfg(tmp_path: Path, extra: str) -> str:
    return write_config(tmp_path, CFG_TEXT + extra)


def test_follow_defaults(tmp_path: Path) -> None:
    ls = load(_cfg(tmp_path, "")).livesplit
    assert ls.follow_timer is False and ls.poll_seconds == 2.0


def test_follow_true_and_poll_int(tmp_path: Path) -> None:
    ls = load(_cfg(tmp_path, "  follow_timer: true\n  poll_seconds: 1\n")).livesplit
    assert ls.follow_timer is True and ls.poll_seconds == 1.0
    assert isinstance(ls.poll_seconds, float)


@pytest.mark.parametrize("bad", ["1", "0", '"yes"', "null2", "[true]"])
def test_follow_non_bool_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ConfigError, match="follow_timer must be a boolean"):
        load(_cfg(tmp_path, f"  follow_timer: {bad}\n"))


@pytest.mark.parametrize("bad", ["0", "-1", "-0.5", "0.0"])
def test_poll_seconds_nonpositive_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ConfigError, match="poll_seconds must be greater than 0"):
        load(_cfg(tmp_path, f"  poll_seconds: {bad}\n"))


@pytest.mark.parametrize("bad", ["true", '"2"', "[2]", "abc"])
def test_poll_seconds_non_number_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ConfigError, match="poll_seconds must be a number"):
        load(_cfg(tmp_path, f"  poll_seconds: {bad}\n"))


@pytest.mark.parametrize("bad", [".nan", ".inf", "-.inf"])
def test_poll_seconds_nan_inf_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ConfigError):
        load(_cfg(tmp_path, f"  poll_seconds: {bad}\n"))


def test_follow_null_means_default(tmp_path: Path) -> None:
    ls = load(_cfg(tmp_path, "  follow_timer: null\n  poll_seconds: null\n")).livesplit
    assert ls.follow_timer is False and ls.poll_seconds == 2.0
