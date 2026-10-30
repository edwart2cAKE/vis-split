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

Query commands (``getcurrenttimerphase``, ``getcurrentsplitname``,
``getsplitindex``, ``getsplitcount``) follow
the same contract and return ``None`` on any transport failure. They are
meant for low-rate polling (a couple of seconds apart), not per-frame use:
each call can stall up to ``timeout_seconds`` waiting for the response line.
"""

from __future__ import annotations

import re
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
        # True when the last send() / split() / unsplit() / start() got its
        # bytes through ``sendall`` (even if a later drain hit EOF and the call
        # returned False): the command MAY have been applied. False means it
        # definitely never left this process.
        self.last_send_wrote: bool = False

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

    def unsplit(self) -> bool:
        """Send one ``unsplit`` command; False if the server is unreachable.

        Fire-and-forget like :meth:`split`. Only the opt-in ``ocr.mode: undo``
        executor (``livesplit/undo_exec.py``) calls it.
        """
        return self._send(b"unsplit")

    def start(self) -> bool:
        """Send one ``start`` command; False if the server is unreachable."""
        return self._send(b"start")

    def query(self, command: str) -> str | None:
        """Send ``command`` and return its first response line.

        The trailing newline (and CR) is stripped; an intentionally empty
        response arrives as ``""``. Returns None when the server is
        unreachable, silent past ``timeout_seconds``, or drops the socket.
        """
        if not self._ensure_connected():
            return None
        sock = self._sock
        if sock is None:
            return None
        try:
            # A previous query that timed out may have its reply arrive late;
            # discard any unread bytes so this query cannot read a stale line
            # (which would leave every later reply off by one).
            self._drain(sock)
            sock.sendall(command.encode("ascii") + b"\n")
            deadline = time.monotonic() + self._timeout
            data = b""
            while b"\n" not in data:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                readable, _, _ = select.select([sock], [], [], remaining)
                if not readable:
                    return None
                chunk = sock.recv(4096)
                if not chunk:
                    self.teardown()
                    return None
                data += chunk
            line = data.split(b"\n", 1)[0]
            return line.decode("utf-8", errors="replace").rstrip("\r")
        except OSError:
            self.teardown()
            return None

    def get_current_timer_phase(self) -> str | None:
        """LiveSplit's timer phase: NotRunning / Running / Paused / Ended."""
        return self.query("getcurrenttimerphase")

    def get_current_split_name(self) -> str | None:
        """Name of the segment LiveSplit is currently timing ('' when done)."""
        return self.query("getcurrentsplitname")

    def get_split_index(self) -> int | None:
        """LiveSplit's current split index (-1 before the run starts).

        None on a failed query, a ``-`` / ``[Error]`` reply, or any
        non-integer response.
        """
        line = self.query("getsplitindex")
        if line is None:
            return None
        text = line.strip()
        # Optional sign plus ASCII digits only (int() alone would accept
        # underscores and non-ASCII digits).
        if re.fullmatch(r"[+-]?[0-9]+", text) is None:
            return None
        return int(text)

    def get_split_count(self) -> int | None:
        """Number of segments in LiveSplit's run (>= 0), None on any failure."""
        line = self.query("getsplitcount")
        if line is None:
            return None
        text = line.strip()
        if re.fullmatch(r"[+]?[0-9]+", text) is None:
            return None
        return int(text)

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
        self.last_send_wrote = False
        if not self._ensure_connected():
            return False
        sock = self._sock
        if sock is None:
            return False
        try:
            sock.sendall(command + b"\n")
            self.last_send_wrote = True
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
