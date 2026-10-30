"""LiveSplit Server integration (Phase 6).

All protocol logic lives in :mod:`autosplitter.livesplit.client`; the rest
of the application only ever calls methods like :meth:`LiveSplitClient.split`.
"""

from .client import LiveSplitClient
from .follow import Action, Decision, TimerFollower
from .sync import UndoWatch
from .undo_exec import OutcomeKind, UndoExecutor, UndoOutcome

__all__ = [
    "Action",
    "Decision",
    "LiveSplitClient",
    "OutcomeKind",
    "TimerFollower",
    "UndoExecutor",
    "UndoOutcome",
    "UndoWatch",
]
