"""Tests for autosplitter.livesplit.follow.TimerFollower (increment 1)."""

from __future__ import annotations

import pytest

from autosplitter.livesplit.follow import Action, Decision, TimerFollower


def obs(
    f: TimerFollower,
    *,
    phase: str | None = "Running",
    ls: int | None = 2,
    expected: int = 1,
    n: int = 5,
    now: float = 100.0,
    last: float | None = 0.0,
    suppress: bool = False,
    back: bool = True,
    fwd: bool = True,
    wfs: bool = False,
) -> Decision:
    return f.observe(
        phase=phase,
        ls_index=ls,
        expected=expected,
        n_splits=n,
        now=now,
        last_split_time=last,
        suppress=suppress,
        can_back=back,
        can_fwd=fwd,
        waiting_for_start=wfs,
    )


def armed() -> TimerFollower:
    f = TimerFollower()
    assert obs(f, ls=1, expected=1).action is Action.NONE
    return f


class TestSkip:
    def test_needs_two_agreeing_polls(self) -> None:
        f = TimerFollower()
        assert obs(f).action is Action.NONE
        assert obs(f) == Decision(Action.SKIP_FORWARD, 2)

    def test_count_restarts_after_confirmation(self) -> None:
        f = TimerFollower()
        obs(f)
        obs(f)
        assert obs(f).action is Action.NONE

    def test_changed_key_restarts_count(self) -> None:
        f = TimerFollower()
        assert obs(f, ls=2).action is Action.NONE
        assert obs(f, ls=3).action is Action.NONE
        assert obs(f, ls=3) == Decision(Action.SKIP_FORWARD, 3)

    def test_changed_expected_restarts_count(self) -> None:
        f = TimerFollower()
        obs(f, ls=3, expected=1)
        assert obs(f, ls=3, expected=2).action is Action.NONE
        assert obs(f, ls=3, expected=2).action is Action.SKIP_FORWARD

    def test_double_skip(self) -> None:
        f = TimerFollower()
        obs(f, ls=3, expected=1)
        assert obs(f, ls=3, expected=1) == Decision(Action.SKIP_FORWARD, 3)

    def test_last_segment_allowed_beyond_ignored(self) -> None:
        f = TimerFollower()
        obs(f, ls=4, expected=1, n=5)
        assert obs(f, ls=4, expected=1, n=5) == Decision(Action.SKIP_FORWARD, 4)
        for _ in range(3):
            assert obs(f, ls=5, expected=1, n=5).action is Action.NONE

    @pytest.mark.parametrize("ls", [1, 0])
    def test_equal_or_behind_no_skip(self, ls: int) -> None:
        f = TimerFollower()
        for _ in range(3):
            assert obs(f, ls=ls, expected=1).action is not Action.SKIP_FORWARD

    def test_in_flight_lead_never_triggers(self) -> None:
        """Caller passes E that already includes the +1 lead, so L == E."""
        f = TimerFollower()
        for _ in range(3):
            assert obs(f, ls=2, expected=2).action is Action.NONE

    def test_paused_allowed_ended_not(self) -> None:
        f = TimerFollower()
        obs(f, phase="Paused")
        assert obs(f, phase="Paused").action is Action.SKIP_FORWARD
        for _ in range(3):
            assert obs(f, phase="Ended").action is Action.NONE

    def test_blocked_by_flags(self) -> None:
        f = TimerFollower()
        for _ in range(3):
            assert obs(f, fwd=False).action is Action.NONE
        f = TimerFollower()
        for _ in range(3):
            assert obs(f, wfs=True).action in {Action.NONE, Action.MANUAL_START}

    def test_no_holdoff_or_arming_needed(self) -> None:
        f = TimerFollower()
        obs(f, now=100.0, last=100.0, suppress=True)
        assert obs(f, now=100.0, last=100.0, suppress=True).action is Action.SKIP_FORWARD

    @pytest.mark.parametrize("phase", [None, "running", "Weird", ""])
    def test_bad_phase_ignored_and_clears(self, phase: str | None) -> None:
        f = TimerFollower()
        obs(f)
        assert obs(f, phase=phase).action is Action.NONE
        assert obs(f).action is Action.NONE  # count restarted

    def test_none_index_clears(self) -> None:
        f = TimerFollower()
        obs(f)
        assert obs(f, ls=None).action is Action.NONE
        assert obs(f).action is Action.NONE

    def test_non_candidate_poll_resets(self) -> None:
        f = TimerFollower()
        obs(f)
        obs(f, ls=1, expected=1)
        assert obs(f).action is Action.NONE


class TestUndoViaFollower:
    def test_rollback_after_arming(self) -> None:
        f = armed()
        assert obs(f, ls=0, expected=1).action is Action.NONE
        assert obs(f, ls=0, expected=1) == Decision(Action.ROLLBACK, 0)

    def test_unarmed_never_rolls_back(self) -> None:
        f = TimerFollower()
        for _ in range(4):
            assert obs(f, ls=0, expected=1).action is Action.NONE

    def test_suppress_and_holdoff_block(self) -> None:
        f = armed()
        for _ in range(3):
            assert obs(f, ls=0, expected=1, suppress=True).action is Action.NONE
        for _ in range(3):
            assert obs(f, ls=0, expected=1, now=101.0, last=100.0).action is Action.NONE

    def test_can_back_false_blocks(self) -> None:
        f = armed()
        for _ in range(3):
            assert obs(f, ls=0, expected=1, back=False).action is Action.NONE

    def test_survives_pause(self) -> None:
        f = armed()
        obs(f, phase="Paused", ls=0, expected=1)
        assert obs(f, phase="Paused", ls=0, expected=1).action is Action.ROLLBACK

    def test_from_ended(self) -> None:
        f = TimerFollower()
        assert obs(f, phase="Ended", ls=5, expected=5).action is Action.NONE  # arms
        obs(f, phase="Running", ls=4, expected=5)
        assert obs(f, phase="Running", ls=4, expected=5) == Decision(Action.ROLLBACK, 4)

    def test_not_running_disarms(self) -> None:
        f = armed()
        obs(f, phase="NotRunning", ls=-1)
        for _ in range(3):
            assert obs(f, ls=0, expected=1).action is Action.NONE

    def test_disarm(self) -> None:
        f = armed()
        f.disarm()
        for _ in range(3):
            assert obs(f, ls=0, expected=1).action is Action.NONE


class TestLegacyEquivalence:
    def test_matches_undowatch_on_undo_tables(self) -> None:
        from autosplitter.livesplit.sync import UndoWatch

        polls = [
            ("Running", 1, 1, 100.0),
            ("Running", 0, 1, 100.0),
            ("Running", 0, 1, 100.0),
            ("Running", 0, 2, 100.0),
            ("Running", 0, 2, 101.0),
            ("Paused", 0, 2, 200.0),
            ("Running", 0, 2, 200.0),
            ("Running", 0, 2, 200.0),
        ]
        f = TimerFollower()
        w = UndoWatch()
        for phase, ls, e, now in polls:
            d = obs(f, phase=phase, ls=ls, expected=e, now=now, last=95.0, fwd=False)
            t = w.observe("Running" if phase == "Paused" else phase, ls, e, now, 95.0, False, True)
            assert (d.index if d.action is Action.ROLLBACK else None) == t


def running_seen() -> TimerFollower:
    f = TimerFollower()
    obs(f, phase="Running", ls=1, expected=3, n=5)
    return f


class TestLifecycle:
    def test_stale_ended_never_run_ended(self) -> None:
        f = TimerFollower()
        for _ in range(5):
            assert obs(f, phase="Ended", ls=5, expected=3, n=5).action is Action.NONE

    def test_disarm_and_reset_require_running_again(self) -> None:
        f = running_seen()
        f.disarm()
        for _ in range(4):
            assert obs(f, phase="Ended", ls=5, expected=3, n=5).action is Action.NONE
        f = running_seen()
        assert obs(f, phase="NotRunning", ls=None).action is Action.RESET
        for _ in range(4):
            assert obs(f, phase="Ended", ls=5, expected=3, n=5).action is Action.NONE

    def test_paused_counts_as_seen_running(self) -> None:
        f = TimerFollower()
        obs(f, phase="Paused", ls=1, expected=3, n=5)
        obs(f, phase="Ended", ls=5, expected=3, n=5)
        assert obs(f, phase="Ended", ls=5, expected=3, n=5).action is Action.RUN_ENDED

    def test_manual_start_requires_index_below_n(self) -> None:
        f = TimerFollower()
        assert obs(f, ls=5, expected=0, n=5, wfs=True).action is Action.NONE
        assert obs(f, ls=4, expected=0, n=5, wfs=True) == Decision(Action.MANUAL_START, 4)

    def test_reset_only_after_activity(self) -> None:
        f = TimerFollower()
        for _ in range(3):
            assert obs(f, phase="NotRunning", ls=-1).action is Action.NONE
        obs(f, ls=1, expected=1)
        assert obs(f, phase="NotRunning", ls=None).action is Action.RESET

    @pytest.mark.parametrize("phase", ["Running", "Paused", "Ended"])
    def test_any_active_phase_counts(self, phase: str) -> None:
        f = TimerFollower()
        obs(f, phase=phase, ls=1, expected=1)
        assert obs(f, phase="NotRunning", ls=None).action is Action.RESET

    def test_reset_fires_once_and_single_poll(self) -> None:
        f = TimerFollower()
        obs(f, ls=1, expected=1)
        assert obs(f, phase="NotRunning", ls=None) == Decision(Action.RESET)
        for _ in range(3):
            assert obs(f, phase="NotRunning", ls=None).action is Action.NONE

    def test_none_and_unknown_phase_do_not_reset_or_deactivate(self) -> None:
        f = TimerFollower()
        obs(f, ls=1, expected=1)
        assert obs(f, phase=None).action is Action.NONE
        assert obs(f, phase="bogus").action is Action.NONE
        assert obs(f, phase="NotRunning", ls=None).action is Action.RESET

    def test_reset_clears_candidates_and_arming(self) -> None:
        f = armed()
        obs(f, ls=0, expected=1)
        assert obs(f, phase="NotRunning", ls=None).action is Action.RESET
        obs(f, ls=1, expected=1)  # re-arms
        assert obs(f, ls=0, expected=1).action is Action.NONE  # count was cleared

    def test_disarm_forgets_activity(self) -> None:
        f = TimerFollower()
        obs(f, ls=1, expected=1)
        f.disarm()
        assert obs(f, phase="NotRunning", ls=None).action is Action.NONE

    @pytest.mark.parametrize("phase", ["Running", "Paused"])
    def test_manual_start_single_poll(self, phase: str) -> None:
        f = TimerFollower()
        assert obs(f, phase=phase, ls=0, expected=0, wfs=True) == Decision(
            Action.MANUAL_START, 0
        )

    def test_manual_start_carries_index(self) -> None:
        f = TimerFollower()
        assert obs(f, ls=3, expected=0, wfs=True) == Decision(Action.MANUAL_START, 3)

    def test_manual_start_requires_waiting_for_start_and_valid_phase(self) -> None:
        f = TimerFollower()
        assert obs(f, ls=0, expected=0, wfs=False).action is Action.NONE
        assert obs(f, phase="Ended", ls=0, expected=0, wfs=True).action is not Action.MANUAL_START
        assert obs(f, ls=-1, expected=0, wfs=True).action is Action.NONE
        assert obs(f, ls=0, expected=0, wfs=True, fwd=False).action is Action.NONE

    def test_run_ended_needs_two_polls(self) -> None:
        f = running_seen()
        assert obs(f, phase="Ended", ls=5, expected=3, n=5).action is Action.NONE
        assert obs(f, phase="Ended", ls=5, expected=3, n=5) == Decision(Action.RUN_ENDED, 5)
        assert obs(f, phase="Ended", ls=5, expected=3, n=5).action is Action.NONE

    def test_run_ended_in_step_is_none(self) -> None:
        f = TimerFollower()
        for _ in range(3):
            assert obs(f, phase="Ended", ls=5, expected=5, n=5).action is Action.NONE

    def test_run_ended_key_change_restarts(self) -> None:
        f = running_seen()
        obs(f, phase="Ended", ls=5, expected=3, n=5)
        assert obs(f, phase="Ended", ls=5, expected=4, n=5).action is Action.NONE
        assert obs(f, phase="Ended", ls=5, expected=4, n=5).action is Action.RUN_ENDED

    def test_run_ended_interrupted_by_other_phase_restarts(self) -> None:
        f = TimerFollower()
        obs(f, phase="Ended", ls=5, expected=3, n=5)
        obs(f, phase="Running", ls=2, expected=3, n=5)
        assert obs(f, phase="Ended", ls=5, expected=3, n=5).action is Action.NONE

    def test_run_ended_disabled_when_forwarding_disabled(self) -> None:
        f = TimerFollower()
        for _ in range(3):
            assert obs(f, phase="Ended", ls=5, expected=3, n=5, fwd=False).action is Action.NONE

    def test_run_ended_wrong_index_ignored(self) -> None:
        f = TimerFollower()
        for _ in range(3):
            assert obs(f, phase="Ended", ls=4, expected=3, n=5).action is Action.NONE

    def test_undo_from_ended_still_arms(self) -> None:
        f = TimerFollower()
        assert obs(f, phase="Ended", ls=5, expected=5, n=5).action is Action.NONE
        obs(f, phase="Running", ls=4, expected=5, n=5)
        assert obs(f, phase="Running", ls=4, expected=5, n=5) == Decision(Action.ROLLBACK, 4)


def pobs(f: TimerFollower, phase: str | None, **kw: object) -> Decision:
    base: dict[str, object] = {"ls": 1, "expected": 1}
    base.update(kw)
    return obs(f, phase=phase, **base)  # pyright: ignore[reportArgumentType]


class TestPause:
    def test_pause_once_then_resume(self) -> None:
        f = TimerFollower(pause=True)
        assert pobs(f, "Running").action is Action.NONE
        assert pobs(f, "Paused") == Decision(Action.PAUSE)
        assert f.paused
        assert pobs(f, "Paused").action is Action.NONE  # not repeated
        assert pobs(f, "Running") == Decision(Action.RESUME)
        assert not f.paused
        assert pobs(f, "Running").action is Action.NONE

    def test_resume_without_pause_is_none(self) -> None:
        f = TimerFollower(pause=True)
        for _ in range(3):
            assert pobs(f, "Running").action is Action.NONE

    def test_disabled_never_pauses(self) -> None:
        f = TimerFollower()
        for phase in ("Running", "Paused", "Paused", "Running"):
            d = pobs(f, phase)
            assert d.action not in (Action.PAUSE, Action.RESUME)
        assert not f.paused

    def test_disarm_clears_paused(self) -> None:
        f = TimerFollower(pause=True)
        pobs(f, "Paused")
        f.disarm()
        assert not f.paused

    def test_reset_clears_paused(self) -> None:
        f = TimerFollower(pause=True)
        pobs(f, "Paused")
        assert f.paused
        assert obs(f, phase="NotRunning", ls=None).action is Action.RESET
        assert not f.paused

    def test_none_and_unknown_phase_keep_state(self) -> None:
        f = TimerFollower(pause=True)
        pobs(f, "Paused")
        assert obs(f, phase=None).action is Action.NONE
        assert obs(f, phase="bogus").action is Action.NONE
        assert f.paused
        assert pobs(f, "Running") == Decision(Action.RESUME)

    def test_skip_still_works_while_paused(self) -> None:
        f = TimerFollower(pause=True)
        assert pobs(f, "Paused", ls=3, expected=1).action is Action.PAUSE
        assert pobs(f, "Paused", ls=3, expected=1).action is Action.NONE  # count 1
        assert pobs(f, "Paused", ls=3, expected=1) == Decision(Action.SKIP_FORWARD, 3)

    def test_undo_armed_across_pause(self) -> None:
        f = TimerFollower(pause=True)
        pobs(f, "Running", ls=1, expected=1)  # arms
        assert pobs(f, "Paused", ls=1, expected=1).action is Action.PAUSE
        assert pobs(f, "Running", ls=0, expected=1).action is Action.RESUME
        assert pobs(f, "Running", ls=0, expected=1).action is Action.NONE
        assert pobs(f, "Running", ls=0, expected=1) == Decision(Action.ROLLBACK, 0)

    def test_one_decision_per_poll_phase_first(self) -> None:
        f = TimerFollower(pause=True)
        # Paused with a skip candidate: PAUSE wins this poll; skip counts start later.
        assert pobs(f, "Paused", ls=3, expected=1).action is Action.PAUSE
        assert pobs(f, "Running", ls=3, expected=1).action is Action.RESUME
        assert pobs(f, "Running", ls=3, expected=1).action is Action.NONE
        assert pobs(f, "Running", ls=3, expected=1).action is Action.SKIP_FORWARD
