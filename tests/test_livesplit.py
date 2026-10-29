"""Tests for autosplitter.livesplit.client.

Exercises the real TCP path against throwaway loopback servers so wire
format, resilience, and reconnect cadence are all pinned without Wine or a
running LiveSplit. The hung-server timeout path is covered via a faked
socket to keep the suite fast and deterministic.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, cast, override

import pytest

from autosplitter.livesplit import LiveSplitClient

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer


class RestartableServer:
    """Minimal accept-loop server recording everything it receives.

    start()/stop() may be called repeatedly; each start binds a fresh
    listener on the SAME port (SO_REUSEADDR) so clients can reconnect.
    """

    def __init__(self) -> None:
        self.port: int = -1  # set by start()
        self.received: list[bytes] = []
        self.auto_reply: bytes | None = None
        self._lock: threading.Lock = threading.Lock()
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._listener: socket.socket | None = None

    def start(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0 if self.port == -1 else self.port))
        addr = cast("tuple[str, int]", listener.getsockname())
        self.port = addr[1]
        listener.listen(4)
        self._listener = listener
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        listener = self._listener
        if listener is None:
            return
        listener.settimeout(0.2)
        while not self._stop.is_set():
            try:
                accepted = cast("tuple[socket.socket, object]", listener.accept())
                conn = accepted[0]
            except TimeoutError:
                continue
            except OSError:
                break
            with conn:
                conn.settimeout(0.2)
                try:
                    while True:
                        data = conn.recv(4096)
                        if not data:
                            break
                        with self._lock:
                            self.received.append(data)
                        if self.auto_reply is not None:
                            conn.sendall(self.auto_reply)
                except OSError:
                    break

    def received_bytes(self) -> bytes:
        with self._lock:
            return b"".join(self.received)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread = None
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()


@pytest.fixture()
def server() -> Iterator[RestartableServer]:
    srv = RestartableServer()
    srv.start()
    yield srv
    srv.stop()


def make_client(port: int, *, reconnect_seconds: float = 60.0) -> LiveSplitClient:
    return LiveSplitClient(
        "127.0.0.1",
        port,
        timeout_seconds=0.25,
        reconnect_seconds=reconnect_seconds,
    )


class TestWireFormat:
    def test_split_sends_newline_terminated_command(
        self, server: RestartableServer
    ) -> None:
        client = make_client(server.port)

        assert client.split() is True
        assert client.split() is True

        expected = b"split\nsplit\n"
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and server.received_bytes() != expected:
            time.sleep(0.01)
        assert server.received_bytes() == expected
        client.close()

    def test_ping_roundtrip_with_pong(self, server: RestartableServer) -> None:
        server.auto_reply = b"pong\n"
        client = make_client(server.port)

        assert client.ping() is True
        assert client.connected is True
        client.close()

    def test_connected_reflects_connection_lifecycle(
        self, server: RestartableServer
    ) -> None:
        client = make_client(server.port)

        assert client.connected is False
        assert client.poll_connection() is True
        assert client.connected is True
        client.close()
        assert client.connected is False


class TestQueries:
    def test_phase_roundtrip_strips_crlf(self, server: RestartableServer) -> None:
        server.auto_reply = b"Running\r\n"
        client = make_client(server.port)

        assert client.get_current_timer_phase() == "Running"
        client.close()

    def test_split_name_roundtrip(self, server: RestartableServer) -> None:
        server.auto_reply = b"Level 5\n"
        client = make_client(server.port)

        assert client.get_current_split_name() == "Level 5"
        client.close()

    def test_empty_response_is_empty_string(self, server: RestartableServer) -> None:
        """LiveSplit answers with a bare CRLF when the run is finished."""
        server.auto_reply = b"\r\n"
        client = make_client(server.port)

        assert client.get_current_split_name() == ""
        client.close()

    def test_query_returns_none_when_unreachable(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        addr = cast("tuple[str, int]", listener.getsockname())
        free_port = addr[1]
        listener.close()

        client = make_client(free_port)
        started = time.monotonic()

        assert client.query("getcurrenttimerphase") is None
        assert time.monotonic() - started < 1.0
        assert client.connected is False

    def test_query_returns_none_when_server_silent(
        self, server: RestartableServer
    ) -> None:
        """Server accepts but never answers: bail out within timeout budget."""
        server.auto_reply = None
        client = make_client(server.port)

        started = time.monotonic()
        assert client.query("getcurrenttimerphase") is None
        assert time.monotonic() - started < 1.5
        client.close()


class TestSplitIndex:
    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            (b"3\r\n", 3),
            (b"0\n", 0),
            (b"-1\n", -1),
            (b"-\n", None),
            (b"[Error] Unknown command\n", None),
            (b"abc\n", None),
            (b"\r\n", None),
        ],
    )
    def test_reply_parsing(
        self, server: RestartableServer, reply: bytes, expected: int | None
    ) -> None:
        server.auto_reply = reply
        client = make_client(server.port)
        assert client.get_split_index() == expected
        client.close()

    def test_wire_bytes(self, server: RestartableServer) -> None:
        server.auto_reply = b"1\n"
        client = make_client(server.port)
        assert client.get_split_index() == 1
        assert server.received_bytes() == b"getsplitindex\n"
        client.close()

    def test_silent_server_returns_none(self, server: RestartableServer) -> None:
        server.auto_reply = None
        client = make_client(server.port)
        assert client.get_split_index() is None
        client.close()

    def test_dropped_socket_returns_none(self, server: RestartableServer) -> None:
        server.auto_reply = b"2\n"
        client = make_client(server.port, reconnect_seconds=60.0)
        assert client.get_split_index() == 2
        server.stop()
        assert client.get_split_index() is None
        assert client.connected is False

    def test_unreachable_returns_none(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        addr = cast("tuple[str, int]", listener.getsockname())
        listener.close()
        client = make_client(addr[1])
        assert client.get_split_index() is None
        assert client.connected is False

    def test_reconnects_after_server_returns(self, server: RestartableServer) -> None:
        server.auto_reply = b"4\n"
        client = make_client(server.port, reconnect_seconds=0.05)
        assert client.get_split_index() == 4
        server.stop()
        client.teardown()
        assert client.get_split_index() is None
        server.auto_reply = b"5\n"
        server.start()
        deadline = time.monotonic() + 2.0
        got: int | None = None
        while time.monotonic() < deadline and got is None:
            got = client.get_split_index()
            time.sleep(0.02)
        assert got == 5
        client.close()


class TestSplitCount:
    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            (b"5\r\n", 5),
            (b"0\n", 0),
            (b"-1\n", None),
            (b"-\n", None),
            (b"[Error] x\n", None),
            (b"1_0\n", None),
            (b"\xd9\xa3\n", None),
            (b"abc\n", None),
            (b"\r\n", None),
        ],
    )
    def test_reply_parsing(
        self, server: RestartableServer, reply: bytes, expected: int | None
    ) -> None:
        server.auto_reply = reply
        client = make_client(server.port)
        assert client.get_split_count() == expected
        client.close()

    def test_wire_bytes(self, server: RestartableServer) -> None:
        server.auto_reply = b"3\n"
        client = make_client(server.port)
        assert client.get_split_count() == 3
        assert server.received_bytes() == b"getsplitcount\n"
        client.close()

    def test_unreachable_returns_none(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        addr = cast("tuple[str, int]", listener.getsockname())
        listener.close()
        client = make_client(addr[1])
        assert client.get_split_count() is None

    def test_dropped_socket_returns_none(self, server: RestartableServer) -> None:
        server.auto_reply = b"3\n"
        client = make_client(server.port, reconnect_seconds=60.0)
        assert client.get_split_count() == 3
        server.stop()
        assert client.get_split_count() is None


class TestResilience:
    def test_unreachable_server_returns_false_quickly(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        addr = cast("tuple[str, int]", listener.getsockname())
        free_port = addr[1]
        listener.close()

        client = make_client(free_port)
        started = time.monotonic()

        assert client.split() is False
        assert time.monotonic() - started < 1.0
        assert client.connected is False

    def test_reconnect_attempts_are_rate_limited(
        self, server: RestartableServer
    ) -> None:
        port = server.port
        client = make_client(port, reconnect_seconds=30.0)

        # First attempt fails (server down) and latches the rate limiter.
        server.stop()
        assert client.split() is False

        server.start()
        time.sleep(0.05)

        # Within the reconnect window: no second attempt, stays down.
        assert client.split() is False
        assert client.connected is False

    def test_recovers_when_server_comes_back(
        self, server: RestartableServer
    ) -> None:
        client = make_client(server.port, reconnect_seconds=0.05)
        assert client.split() is True

        server.stop()
        client.teardown()
        assert client.split() is False

        server.start()
        deadline = time.monotonic() + 2.0
        delivered = False
        while time.monotonic() < deadline:
            if client.split() is True:
                delivered = True
                break
            time.sleep(0.02)
        assert delivered is True
        client.close()

    def test_hung_send_times_out_and_marks_disconnected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A server that accepts but never reads must not stall the caller."""

        class HungSocket(socket.socket):
            @override
            def sendall(self, data: ReadableBuffer, flags: int = 0, /) -> None:
                raise TimeoutError("simulated send stall")

        created: list[HungSocket] = []

        def fake_create_connection(
            _address: tuple[str, int], _timeout: float | None = None
        ) -> socket.socket:
            sock = HungSocket()
            created.append(sock)
            return sock

        monkeypatch.setattr(socket, "create_connection", fake_create_connection)
        client = make_client(16834, reconnect_seconds=60.0)

        started = time.monotonic()
        assert client.split() is False
        assert time.monotonic() - started < 1.0
        # The dead socket was torn down, so status reports disconnected.
        assert client.connected is False
        # And the next attempt respects the rate limiter (no new socket).
        assert client.split() is False
        assert len(created) == 1
