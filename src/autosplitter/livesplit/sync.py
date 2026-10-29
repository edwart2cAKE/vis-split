"""Unsplit detection from LiveSplit's split index (pure logic, no sockets).

The detector fires splits; if one fires wrongly the user presses Unsplit in
LiveSplit and the server's split index drops below what the detector expects.
:class:`UndoWatch` debounces that observation across low-rate polls and
returns the index the detector should roll back to.
"""

from __future__ import annotations

HOLDOFF_SECONDS = 3.0
CONFIRM_POLLS = 2


class UndoWatch:
    """Confirm an unsplit after CONFIRM_POLLS agreeing polls."""

    def __init__(self) -> None:
        self._candidate: tuple[int, int] | None = None
        self._count: int = 0
        # Armed once a Running poll shows LiveSplit in step with the detector
        # (L == expected). Without it, splits fired while the timer was not
        # running followed by a late start at index 0 would look like an undo.
        self._armed: bool = False

    def reset(self) -> None:
        self._candidate = None
        self._count = 0

    def disarm(self) -> None:
        """Forget the in-step observation (machine reset / rearm)."""
        self._armed = False
        self.reset()

    def observe(
        self,
        phase: str | None,
        ls_index: int | None,
        expected: int,
        now: float,
        last_split_time: float | None,
        suppress: bool,
        rollback_ok: bool = True,
    ) -> int | None:
        """Feed one poll; return the confirmed rollback target or None.

        ``rollback_ok`` is False when the machine is in a state that cannot
        roll back (TRANSITION / WAITING_FOR_START). ``suppress`` is True
        while a split was not delivered, so LiveSplit is legitimately behind.
        """
        if phase != "Running":
            self.disarm()
            return None
        if ls_index is not None and ls_index == expected:
            # Arming is not subject to the hold-off or rollback_ok.
            self._armed = True
        is_candidate = (
            self._armed
            and ls_index is not None
            and 0 <= ls_index < expected
            and rollback_ok
            and not suppress
            and (last_split_time is None or now - last_split_time >= HOLDOFF_SECONDS)
        )
        if not is_candidate or ls_index is None:
            self.reset()
            return None
        key = (ls_index, expected)
        if key == self._candidate:
            self._count += 1
        else:
            self._candidate = key
            self._count = 1
        if self._count >= CONFIRM_POLLS:
            self.reset()
            return ls_index
        return None
