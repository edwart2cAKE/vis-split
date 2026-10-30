"""OCR "go then undo" executor (``ocr.mode: undo``): the ONLY code that sends
LiveSplit ``unsplit`` (a deliberate, opt-in exception to "the detector never
touches timer state"; the late-HUD recovery may send ``split`` again).

Pure orchestration over a client-like object and the state machine, so it is
testable with fakes and a fake TCP server. Protocol bytes stay in
:mod:`autosplitter.livesplit.client`.

Safety rules:

- Every step that needs LiveSplit is verified by an index query. If the
  server is unavailable the detector is NEVER rolled back alone.
- One command per verdict: the confirmed target is consumed even when the
  attempt is skipped, and a second ``unsplit`` is never sent while a first
  one is being reconciled.
- AMBIGUITY IS RECONCILED, NOT GUESSED. A command may have been applied even
  though the write path concluded otherwise (EOF after the write, the verify
  query got no answer, LiveSplit applied it just after answering). In those
  cases the executor re-reads ``getsplitindex``: twice synchronously (~75 ms,
  then ~275 ms after the send; at most ~0.3 s of extra blocking plus two
  queries, only in this rare case) and, if the server still does not answer,
  from the frame loop (:meth:`UndoExecutor.pending_step`, one query at most
  every 0.5 s for up to 5 s). The detector is moved only to match what
  LiveSplit demonstrably reports; at the deadline a loud WARN recommends
  ``livesplit.follow_timer`` (REQUIRED for ocr.mode undo; its rollback/skip rules repair any residual disagreement).
- LiveSplit's real command ordering (is an ``unsplit`` applied before the next
  ``getsplitindex`` reply?) is UNVERIFIED against a real server; the re-checks
  exist precisely so the executor is correct either way.

Frame-loop stall (once per death, never per frame), each query/send bounded by
``livesplit.timeout_seconds`` T:

- normal attempt: phase + index + ``unsplit`` send + index = at most 4 x T
  (0.25 s default => ~1 s with a connected-but-silent server);
- ambiguous attempt: + two re-checks = 5 queries + 1 send + 0.275 s of sleeps,
  at most ~6 x T + 0.275 s (~1.8 s at T = 0.25 s); the recovery ``split`` path
  has one more query (pre-send re-read), 6 queries;
- deferred ``pending_step``: one query at most every 0.5 s, backing off
  (0.5, 1, 2, ... s) after every unanswered one, up to 5 s in total.

On a 120 fps SRT/UDP feed that is 100-200 frames during which the capture pipe
backs up (possible packet loss or stale latency afterwards).

NOT handled: an ``unsplit`` LiveSplit applies later than the ~0.3 s
synchronous window (the detector then stays one ahead until
``livesplit.follow_timer``'s rollback rule repairs it; the executor never rolls back alone),
and a genuine colour fire while a reconcile is pending (the reconcile is
dropped without moving the detector; LiveSplit may then be one behind). Late
replies to a timed-out query may be read by a later query; that can only make
the executor see an OLDER index, i.e. "no effect", never a wrong rollback.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, cast

from autosplitter.detection.state_machine import State

# Synchronous re-check schedule (seconds slept BEFORE each re-read) and the
# deferred schedule used when the server still does not answer.
SYNC_RECHECK_DELAYS = (0.075, 0.2)
DEFERRED_RETRY_SECONDS = 0.5
RECONCILE_DEADLINE_SECONDS = 5.0


class _Client(Protocol):
    def get_current_timer_phase(self) -> str | None: ...
    def get_split_index(self) -> int | None: ...
    def split(self) -> bool: ...
    def unsplit(self) -> bool: ...


class _Machine(Protocol):
    @property
    def state(self) -> State: ...
    @property
    def epoch(self) -> int: ...
    @property
    def current_split_index(self) -> int: ...
    @property
    def expected_ls_index(self) -> int: ...
    @property
    def undo_target(self) -> int | None: ...
    @property
    def recovery_target(self) -> int | None: ...
    @property
    def recovery_lag(self) -> float: ...
    def apply_undo(self, index: int, *, arm_recovery: bool = True) -> bool: ...
    def advance_to(self, index: int) -> bool: ...
    def dismiss_undo_target(self) -> None: ...
    def dismiss_recovery(self) -> None: ...


class OutcomeKind(Enum):
    APPLIED = auto()              # unsplit verified, detector rolled back
    ALREADY_THERE = auto()        # LiveSplit already at target: rolled back only
    USER_UNSPLIT = auto()         # LiveSplit went further back: rolled back to it
    SKIPPED_UNAVAILABLE = auto()  # server unreachable: nothing changed
    SKIPPED = auto()              # precondition failed / reconcile abandoned
    NOT_DELIVERED = auto()        # unsplit definitely never left this process
    IGNORED = auto()              # LiveSplit did not move: detector untouched
    UNVERIFIED = auto()           # sent, not yet confirmed: reconcile pending
    UNCONFIRMED = auto()          # reconcile deadline reached, still unknown
    RECOVERED = auto()            # late HUD: split re-sent, detector advanced
    RECOVERY_SKIPPED = auto()     # late-HUD recovery not performed


_CHANGED = frozenset(
    {
        OutcomeKind.APPLIED,
        OutcomeKind.ALREADY_THERE,
        OutcomeKind.USER_UNSPLIT,
        OutcomeKind.RECOVERED,
    }
)


@dataclass(frozen=True)
class UndoOutcome:
    kind: OutcomeKind
    lines: tuple[str, ...]
    time: float  # the caller's monotonic ``now``

    @property
    def changed(self) -> bool:
        """True when the detector moved (caller resets undo bookkeeping)."""
        return self.kind in _CHANGED


@dataclass
class _Pending:
    """A command that may have been applied but is not confirmed yet."""

    resplit: bool      # False: unsplit (target s, detector d); True: split
    s: int             # undo: S (== d-1); resplit: the detector index (S)
    d: int             # undo: detector index at send time; resplit: S+1
    lag: float
    deadline: float
    next_at: float
    epoch: int         # machine epoch at send time: any later change ends the reconcile
    last: bool = False  # last-split undo: the detector is DONE, LiveSplit was Ended
    retry: float = DEFERRED_RETRY_SECONDS  # doubles after every unanswered re-check


def _wrote(client: _Client) -> bool:
    """Could the last command have left the process? Unknown clients: yes."""
    return cast(object, getattr(client, "last_send_wrote", True)) is not False


class UndoExecutor:
    """One instance per run; holds at most one pending reconcile."""

    def __init__(
        self,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sleep: Callable[[float], None] = sleep
        self._clock: Callable[[], float] = clock
        self._pending: _Pending | None = None
        # True once a LAST-split `unsplit` may have been sent (any outcome); the
        # non-loop DONE exit then waits for a fresh LiveSplit phase (main.py).
        self.last_unsplit_sent: bool = False
        self.last_unsplit_time: float = 0.0

    @property
    def pending(self) -> bool:
        """A command awaits confirmation (no new attempt/recovery starts)."""
        return self._pending is not None

    # ------------------------------------------------------------- undo

    def attempt(
        self, client: _Client, machine: _Machine, target: int, now: float
    ) -> UndoOutcome:
        """Undo split ``target`` (0-based screen index S) in LiveSplit + detector."""
        try:
            return self._attempt(client, machine, target, now)
        finally:
            machine.dismiss_undo_target()

    def _attempt(
        self, client: _Client, machine: _Machine, target: int, now: float
    ) -> UndoOutcome:
        d = machine.current_split_index
        # A death on the LAST level: the detector is DONE (index == N) and
        # LiveSplit is Ended. Only then is phase Ended acceptable.
        last = machine.state is State.DONE
        if self._pending is not None:
            return UndoOutcome(
                OutcomeKind.SKIPPED,
                ("WARN: undo skipped (a previous command is still being confirmed)",),
                now,
            )
        if (
            machine.state not in (State.NORMAL, State.DONE)
            or machine.undo_target != target
            or target != d - 1
        ):
            return UndoOutcome(
                OutcomeKind.SKIPPED,
                ("WARN: undo skipped (detector state changed since the verdict)",),
                now,
            )
        phase = client.get_current_timer_phase()
        ls = client.get_split_index() if phase is not None else None
        if phase is None or ls is None:
            return UndoOutcome(
                OutcomeKind.SKIPPED_UNAVAILABLE,
                ("LiveSplit: undo skipped (server unavailable)",),
                now,
            )
        if phase != "Running" and not (last and phase == "Ended"):
            return UndoOutcome(
                OutcomeKind.SKIPPED, (f"LiveSplit: undo skipped (timer is {phase})",), now
            )
        if ls == target:
            # Already there (user unsplit at the same moment, or the earlier
            # split never reached LiveSplit): nothing to send.
            if machine.apply_undo(target, arm_recovery=False):
                return UndoOutcome(
                    OutcomeKind.ALREADY_THERE,
                    (f"LiveSplit: already at index {ls}; detector rolled back, nothing sent",),
                    now,
                )
            return UndoOutcome(OutcomeKind.SKIPPED, ("WARN: undo skipped (detector refused)",), now)
        if ls != d or (last and phase != "Ended"):
            return UndoOutcome(
                OutcomeKind.SKIPPED,
                (
                    f"WARN: undo skipped: LiveSplit at index {ls}, detector at {d}"
                    + " (not in step)",
                ),
                now,
            )
        sent = client.unsplit()
        if last and (sent or _wrote(client)):
            self.last_unsplit_sent = True
            self.last_unsplit_time = self._clock()
        if not sent and not _wrote(client):
            return UndoOutcome(
                OutcomeKind.NOT_DELIVERED,
                ("LiveSplit: unsplit NOT DELIVERED (server unreachable)",),
                now,
            )
        # From here the unsplit MAY have been applied even if sent is False.
        after = client.get_split_index()
        late = not sent  # reported undelivered yet answered: confirmed after the fact
        if after is None or after == d:
            after = self._recheck(client, d)
            late = True
        if after is None:
            self._start_pending(machine, False, target, d, 0.0, last)
            return UndoOutcome(
                OutcomeKind.UNVERIFIED,
                (
                    f"LiveSplit: unsplit sent but unverified; detector left at {d}"
                    + f" (re-checking for {RECONCILE_DEADLINE_SECONDS:g}s;"
                    + " livesplit.follow_timer (required for ocr.mode undo) repairs it)",
                ),
                now,
            )
        return self._resolve_undo(machine, after, target, d, now, late, last)

    def _resolve_undo(
        self, machine: _Machine, ls: int, s: int, _d: int, now: float, late: bool,
        last: bool = False,
    ) -> UndoOutcome:
        """LiveSplit reports ``ls`` after our unsplit: move the detector to match."""
        if ls == s:
            if machine.apply_undo(s, arm_recovery=not last):  # no recovery after the last split
                text = (
                    "LiveSplit: UNSPLIT applied (confirmed late)"
                    if late
                    else f"LiveSplit: UNSPLIT sent -> index {ls} (verified)"
                )
                return UndoOutcome(OutcomeKind.APPLIED, (text,), now)
            return UndoOutcome(
                OutcomeKind.SKIPPED,
                ("WARN: unsplit applied but the detector refused to roll back",),
                now,
            )
        if 0 <= ls < s and machine.apply_undo(ls, arm_recovery=False):
            return UndoOutcome(
                OutcomeKind.USER_UNSPLIT,
                (
                    f"WARN: LiveSplit moved back to index {ls} (simultaneous"
                    + " unsplit); detector rolled back to match",
                ),
                now,
            )
        reason = (
            f"timer not running: index {ls}" if ls < 0 else f"LiveSplit index {ls}, expected {s}"
        )
        return UndoOutcome(
            OutcomeKind.IGNORED,
            (f"WARN: unsplit had no effect ({reason}); detector left untouched",),
            now,
        )

    # --------------------------------------------------------- recovery

    def recover(self, client: _Client, machine: _Machine, now: float) -> UndoOutcome | None:
        """Late-HUD recovery: re-send ``split`` after a (false) undo.

        Returns None when no recovery is pending (or a reconcile is still
        running: the recovery watch is then consumed, never retried). At most
        one attempt per undo.
        """
        target = machine.recovery_target
        if target is None:
            return None
        lag = machine.recovery_lag
        s = machine.current_split_index
        try:
            if self._pending is not None:
                return None
            return self._recover(client, machine, target, s, lag, now)
        finally:
            machine.dismiss_recovery()

    def _recover(
        self, client: _Client, machine: _Machine, target: int, s: int, lag: float, now: float
    ) -> UndoOutcome:
        head = f"OCR: undo regret — HUD advanced late (+{lag:.1f}s after undo)"

        def skipped(text: str) -> UndoOutcome:
            return UndoOutcome(
                OutcomeKind.RECOVERY_SKIPPED, (f"{head}; recovery not performed", text), now
            )

        if machine.state is not State.NORMAL or target != s + 1:
            return skipped("WARN: recovery skipped (detector state changed)")
        phase = client.get_current_timer_phase()
        ls = client.get_split_index() if phase is not None else None
        if phase is None or ls is None:
            return skipped("WARN: recovery skipped (LiveSplit unavailable); left one split behind")
        if phase != "Running" or ls != s:
            return skipped(
                f"WARN: recovery skipped (LiveSplit {phase} at index {ls}, detector at {s});"
                + " left as is"
            )
        # A user split between the query above and our command would make our
        # split skip a segment (no compare-and-swap in the protocol): re-read
        # right before sending to shrink that window to a few milliseconds.
        again = client.get_split_index()
        if again == target:
            if machine.advance_to(target):
                return UndoOutcome(
                    OutcomeKind.RECOVERED,
                    (
                        f"{head}; LiveSplit already at index {again} (user split?);"
                        + " detector advanced, nothing sent",
                    ),
                    now,
                )
            return skipped("WARN: recovery skipped (detector refused to advance)")
        if again != s:
            return skipped(
                f"WARN: recovery skipped (LiveSplit moved to index {again}); left as is"
            )
        sent = client.split()
        if not sent and not _wrote(client):
            return skipped("LiveSplit: re-split NOT DELIVERED (server unreachable)")
        after = client.get_split_index()
        late = not sent  # reported undelivered yet answered: confirmed after the fact
        if after is None or after == s:
            after = self._recheck(client, s)
            late = True
        if after is None:
            self._start_pending(machine, True, s, target, lag)
            return skipped(
                "WARN: re-split sent but unverified; detector left at"
                + f" {s} (re-checking for {RECONCILE_DEADLINE_SECONDS:g}s;"
                + " livesplit.follow_timer (required for ocr.mode undo) repairs it)"
            )
        return self._resolve_resplit(machine, after, s, lag, now, late)

    def _resolve_resplit(
        self, machine: _Machine, ls: int, s: int, lag: float, now: float, late: bool
    ) -> UndoOutcome:
        base = f"OCR: undo regret — HUD advanced late (+{lag:.1f}s after undo)"
        head = f"{base}; re-splitting"
        if ls == s + 1 and machine.advance_to(ls):
            text = (
                "LiveSplit: SPLIT applied (confirmed late)"
                if late
                else f"LiveSplit: SPLIT re-sent -> index {ls} (verified)"
            )
            return UndoOutcome(OutcomeKind.RECOVERED, (head, text), now)
        if ls > s + 1 and machine.advance_to(ls):
            return UndoOutcome(
                OutcomeKind.RECOVERED,
                (
                    head,
                    "WARN: LiveSplit moved ahead during recovery (user split?);"
                    + f" detector advanced to {ls}",
                ),
                now,
            )
        if ls == s:
            text = f"WARN: re-split had no effect (LiveSplit index {ls}); detector left at {s}"
        else:
            text = (
                f"WARN: re-split result unexpected (LiveSplit index {ls}, detector at {s});"
                + " detector left untouched"
            )
        return UndoOutcome(
            OutcomeKind.RECOVERY_SKIPPED,
            (f"{base}; recovery not performed", text),
            now,
        )

    # -------------------------------------------------------- reconcile

    def _recheck(self, client: _Client, unchanged: int) -> int | None:
        """Re-read the index after short waits until it differs from ``unchanged``.

        Returns the last reading (``unchanged`` => nothing happened, None =>
        the server never answered).
        """
        last: int | None = None
        for delay in SYNC_RECHECK_DELAYS:
            self._sleep(delay)
            last = client.get_split_index()
            if last is not None and last != unchanged:
                return last
        return last

    def _start_pending(
        self, machine: _Machine, resplit: bool, s: int, d: int, lag: float,
        last: bool = False,
    ) -> None:
        t = self._clock()
        self._pending = _Pending(
            resplit, s, d, lag, t + RECONCILE_DEADLINE_SECONDS,
            t + DEFERRED_RETRY_SECONDS, machine.epoch, last,
        )

    def pending_step(self, client: _Client, machine: _Machine) -> UndoOutcome | None:
        """Frame-loop hook: at most ONE index query, at most every 0.5 s.

        Returns an outcome when the reconcile finished (confirmed, dropped,
        abandoned or deadline), else None. No-op without a pending command.
        """
        p = self._pending
        if p is None:
            return None
        t = self._clock()
        if t < p.next_at:
            return None
        ls = client.get_split_index()
        if ls is None:
            if t >= p.deadline:
                self._pending = None
                what = "re-split" if p.resplit else "unsplit"
                return UndoOutcome(
                    OutcomeKind.UNCONFIRMED,
                    (
                        f"LiveSplit: could not confirm {what}; detector at {machine.current_split_index},"
                        + " LiveSplit unknown — livesplit.follow_timer (required for ocr.mode undo) repairs it",
                    ),
                    t,
                )
            # A hung server is unlikely to recover within 0.5 s: back off
            # (0.5, 1, 2, ... s) so the loop is not stalled for ~50% of 5 s.
            p.next_at = min(t + p.retry, p.deadline)
            p.retry *= 2
            return None
        self._pending = None
        expected_here = p.s if p.resplit else p.d
        if (
            machine.epoch != p.epoch
            or machine.state is not (State.DONE if p.last else State.NORMAL)
            or machine.expected_ls_index != expected_here
        ):
            return UndoOutcome(
                OutcomeKind.SKIPPED,
                (
                    f"LiveSplit: {'re-split' if p.resplit else 'unsplit'} reconcile ended"
                    + f" (LiveSplit index {ls}; detector state changed meanwhile)",
                ),
                t,
            )
        if p.resplit:
            return self._resolve_resplit(machine, ls, p.s, p.lag, t, True)
        return self._resolve_undo(machine, ls, p.s, p.d, t, True, p.last)
