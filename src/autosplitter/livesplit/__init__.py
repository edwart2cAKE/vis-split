"""LiveSplit Server integration (Phase 6).

All protocol logic lives in :mod:`autosplitter.livesplit.client`; the rest
of the application only ever calls methods like :meth:`LiveSplitClient.split`.
"""

from .client import LiveSplitClient

__all__ = ["LiveSplitClient"]
