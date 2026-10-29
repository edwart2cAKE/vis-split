"""Follow LiveSplit's timer: classify poll observations into actions.

Pure logic (no sockets, no machine reference). Produces ``ROLLBACK`` (user
Unsplit, delegated to :class:`UndoWatch`), ``SKIP_FORWARD`` (LiveSplit is
ahead of the detector: skip split or a manual split) and the lifecycle
actions ``RESET`` (timer went NotRunning after activity), ``MANUAL_START``
(timer started while the detector waits at the start gate) and ``RUN_ENDED``
(LiveSplit Ended with every split done). With ``pause=True`` (config
``follow_pause``) it also produces ``PAUSE``/``RESUME`` on Paused/Running
transitions so the caller can freeze detection while the timer is paused.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .sync import CONFIRM_POLLS, UndoWatch

_PHASES = frozenset({"NotRunning", "Running", "Paused", "Ended"})


class Action(Enum):
    NONE = auto()
    ROLLBACK = auto()
    SKIP_FORWARD = auto()
    PAUSE = auto()
    RESUME = auto()
    RESET = auto()
    MANUAL_START = auto()
    RUN_ENDED = auto()


@dataclass(frozen=True)
class Decision:
    action: Action
    index: int | None = None


_NONE = Decision(Action.NONE)


class TimerFollower:
    """Debounced rollback / skip-forward decisions from LiveSplit polls."""

    def __init__(self, pause: bool = False) -> None:
        self._pause_enabled: bool = pause
        self._paused: bool = False
        self._undo: UndoWatch = UndoWatch()
        self._skip_key: tuple[int, int] | None = None
        self._skip_count: int = 0
        self._end_key: tuple[int, int] | None = None
        self._end_count: int = 0
        # True once Running/Paused/Ended was seen; makes NotRunning a RESET
        # rather than "timer never started".
        self._seen_active: bool = False
        # True once Running/Paused was seen: an Ended poll alone never counts,
        # so a stale Ended timer (startup, --loop rearm) cannot fire RUN_ENDED.
        self._seen_running: bool = False

    def _clear_skip(self) -> None:
        self._skip_key = None
        self._skip_count = 0
        self._end_key = None
        self._end_count = 0

    @property
    def paused(self) -> bool:
        """True between a PAUSE and the matching RESUME / disarm."""
        return self._paused

    def unpause(self) -> None:
        """Thaw without touching arming (stuck-frozen safety valve)."""
        self._paused = False

    def disarm(self) -> None:
        """Forget arming, activity and pending candidates (reset / rearm)."""
        self._undo.disarm()
        self._clear_skip()
        self._seen_active = False
        self._seen_running = False
        self._paused = False

    def observe(
        self,
        *,
        phase: str | None,
        ls_index: int | None,
        expected: int,
        n_splits: int,
        now: float,
        last_split_time: float | None,
        suppress: bool,
        can_back: bool,
        can_fwd: bool,
        waiting_for_start: bool,
    ) -> Decision:
        """Feed one poll; return the confirmed action (usually NONE)."""
        if phase is None or phase not in _PHASES:
            self._undo.reset()
            self._clear_skip()
            return _NONE
        if phase == "NotRunning":
            was_active = self._seen_active
            self.disarm()
            return Decision(Action.RESET) if was_active else _NONE
        self._seen_active = True
        if phase in ("Running", "Paused"):
            self._seen_running = True
        elif phase == "Ended":
            # Ended is a stopped timer, not a pause: never stay frozen.
            self._paused = False
        # Pause/resume: single-poll phase actions. A None/unknown phase (handled
        # above) leaves ``_paused`` untouched. Undo stays armed across a pause.
        if self._pause_enabled:
            if phase == "Paused" and not self._paused:
                self._paused = True
                self._undo.reset()
                self._clear_skip()
                return Decision(Action.PAUSE)
            if phase == "Running" and self._paused:
                self._paused = False
                self._undo.reset()
                self._clear_skip()
                return Decision(Action.RESUME)
        if ls_index is None:
            self._undo.reset()
            self._clear_skip()
            return _NONE

        # Phase actions come before index actions.
        if (
            waiting_for_start
            and can_fwd
            and phase in ("Running", "Paused")
            and 0 <= ls_index < n_splits
        ):
            self._undo.reset()
            self._clear_skip()
            return Decision(Action.MANUAL_START, ls_index)
        if (
            phase == "Ended"
            and self._seen_running
            and can_fwd
            and ls_index == n_splits
            and expected < n_splits
        ):
            self._undo.reset()
            key = (ls_index, expected)
            if key == self._end_key:
                self._end_count += 1
            else:
                self._end_key = key
                self._end_count = 1
            self._skip_key = None
            self._skip_count = 0
            if self._end_count >= CONFIRM_POLLS:
                self._clear_skip()
                return Decision(Action.RUN_ENDED, n_splits)
            return _NONE

        # Undo must survive a pause and work from Ended, so the inner watch
        # sees every non-NotRunning phase as Running.
        target = self._undo.observe(
            "Running", ls_index, expected, now, last_split_time, suppress, can_back
        )
        if target is not None:
            self._clear_skip()
            return Decision(Action.ROLLBACK, target)

        is_skip = (
            phase in ("Running", "Paused")
            and expected < ls_index <= n_splits - 1
            and can_fwd
            and not waiting_for_start
        )
        if not is_skip:
            self._clear_skip()
            return _NONE
        key = (ls_index, expected)
        if key == self._skip_key:
            self._skip_count += 1
        else:
            self._skip_key = key
            self._skip_count = 1
        if self._skip_count >= CONFIRM_POLLS:
            self._clear_skip()
            return Decision(Action.SKIP_FORWARD, ls_index)
        return _NONE
