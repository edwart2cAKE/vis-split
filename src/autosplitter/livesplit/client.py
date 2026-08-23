"""LiveSplit Server TCP client (Phase 6).

Talks to LiveSplit's built-in command server (Control -> Start TCP Server,
default port 16834). Protocol facts that shape this module:

- Commands are case-sensitive and newline-terminated (``split\\n``).
- ``split`` produces no response; ``ping`` answers ``pong``.
- The server must be started manually each LiveSplit launch.

Hot-loop contract: :meth:`LiveSplitClient.split` NEVER raises and never
blocks longer than ``timeout_seconds``. A dead or hung server degrades to
``split() is False`` while detection continues untouched (spec section 17);
reconnect attempts are rate-limited to one per ``reconnect_seconds``.
"""

from __future__ import annotations

import select
import socket
import time


class LiveSplitClient:
    """Fire-and-forget split delivery over the LiveSplit Server protocol."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_seconds: float,
        reconnect_seconds: float,
    ) -> None:
        self._host: str = host
        self._port: int = port
        self._timeout: float = timeout_seconds
        self._reconnect_interval: float = reconnect_seconds
        self._sock: socket.socket | None = None
        # None = never attempted; monotonic timestamp otherwise.
        self._last_attempt: float | None = None

    @property
    def connected(self) -> bool:
        """True when a socket is currently believed healthy."""
        return self._sock is not None

    def poll_connection(self) -> bool:
        """Rate-limited connection attempt, safe to call every frame."""
        return self._ensure_connected()

    def split(self) -> bool:
        """Send one ``split`` command; False if the server is unreachable."""
        return self._send(b"split")

    def ping(self) -> bool:
        """Round-trip liveness probe — used by tooling, not the hot loop."""
        if not self._send(b"ping"):
            return False
        sock = self._sock
        if sock is None:
            return False
        try:
            deadline = time.monotonic() + self._timeout
            data = b""
            while time.monotonic() < deadline:
                readable, _, _ = select.select([sock], [], [], deadline - time.monotonic())
                if not readable:
                    break
                chunk = sock.recv(4096)
                if not chunk:
                    self.teardown()
                    return False
                data += chunk
                if b"pong" in data:
                    return True
            return False
        except OSError:
            self.teardown()
            return False

    def close(self) -> None:
        """Release the connection (idempotent)."""
        self.teardown()

    def teardown(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _ensure_connected(self) -> bool:
        if self._sock is not None:
            return True
        now = time.monotonic()
        if self._last_attempt is not None and now - self._last_attempt < self._reconnect_interval:
            return False
        self._last_attempt = now
        try:
            # Bounded by timeout_seconds so a hung server cannot stall the
            # caller beyond its budget.
            sock = socket.create_connection((self._host, self._port), self._timeout)
        except OSError:
            return False
        sock.settimeout(self._timeout)
        self._sock = sock
        return True

    def _send(self, command: bytes) -> bool:
        if not self._ensure_connected():
            return False
        sock = self._sock
        if sock is None:
            return False
        try:
            sock.sendall(command + b"\n")
            self._drain(sock)
            return True
        except OSError:
            self.teardown()
            return False

    def _drain(self, sock: socket.socket) -> bool:
        """Read any pending server output without blocking.

        Returns False on EOF (server closed), True otherwise.
        """
        while True:
            readable, _, _ = select.select([sock], [], [], 0)
            if not readable:
                return True
            if sock.recv(4096) == b"":
                raise ConnectionError("LiveSplit Server closed the connection")
