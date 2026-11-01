"""Increment 4 end to end: ``_run_live`` + real LiveSplitClient + a stateful fake
LiveSplit server (real unsplit/split rules) + a scripted fake OCR worker."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast, final

import numpy as np
import pytest

from autosplitter import main as mainmod
from autosplitter.detection import state_machine as sm
from autosplitter.detection.ocr_detector import OcrResult


@final
class LsServer:
    """Stateful LiveSplit Server: ordered lines, replies end in newline."""

    def __init__(self, phase: str = "Running", idx: int = 0, count: int = 3) -> None:
        self.phase: str = phase
        self.idx: int = idx
        self.count: int = count
        self.commands: list[str] = []
        self.unsplit_delay: float = 0.0  # apply `unsplit` this many seconds late
        self.sock: socket.socket = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port: int = cast("tuple[str, int]", self.sock.getsockname())[1]
        self.stop: bool = False
        self.thread: threading.Thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def seen(self, name: str) -> bool:
        return name in list(self.commands)

    def moves(self) -> list[str]:
        return [c for c in list(self.commands) if c in ("split", "unsplit", "start")]

    def _handle(self, line: str) -> str | None:
        self.commands.append(line)
        if line == "getcurrenttimerphase":
            return self.phase
        if line == "getsplitindex":
            return str(self.idx)
        if line == "getsplitcount":
            return str(self.count)
        if line == "getcurrentsplitname":
            return f"Split {self.idx + 1}"
        if line == "ping":
            return "pong"
        if line == "split" and self.phase == "Running":
            self.idx += 1
            if self.idx >= self.count:
                self.phase = "Ended"  # real LiveSplit: the last split ends the run
        elif line in ("unsplit", "undosplit"):
            if self.unsplit_delay > 0:
                threading.Timer(self.unsplit_delay, self._apply_unsplit).start()
            else:
                self._apply_unsplit()
        return None

    def _apply_unsplit(self) -> None:
        if self.phase in ("Running", "Ended") and self.idx > 0:
            self.idx -= 1
            self.phase = "Running"  # Ended -> Running at N-1

    def _run(self) -> None:
        self.sock.settimeout(0.05)
        conns: list[tuple[socket.socket, bytearray]] = []
        while not self.stop:
            try:
                c = self.sock.accept()[0]
                c.settimeout(0.005)
                conns.append((c, bytearray()))
            except OSError:
                pass
            for c, buf in conns:
                try:
                    d = c.recv(4096)
                except OSError:
                    continue
                if not d:
                    continue
                buf.extend(d)
                while b"\n" in buf:
                    raw, _, rest = bytes(buf).partition(b"\n")
                    buf[:] = rest
                    reply = self._handle(raw.decode())
                    if reply is not None:
                        try:
                            c.sendall(reply.encode() + b"\n")
                        except OSError:
                            pass  # client already gone

    def close(self) -> None:
        self.stop = True
        self.thread.join(1.0)
        self.sock.close()


@final
class Source:
    """Fake MpegTsFrameSource: 4 black frames (the transition) then white."""

    exhausted: bool = False

    def __init__(self, *_a: object, **_k: object) -> None:
        self.n: int = 0

    def read(self) -> np.ndarray:
        time.sleep(0.004)
        self.n += 1
        v = 0 if self.n <= 4 else 255
        return np.full((36, 64, 3), v, np.uint8)

    def close(self) -> None:
        pass


class Hud:
    """Shared script: text shown by the fake worker."""

    def __init__(self, server: LsServer | None, mode: str) -> None:
        self.server: LsServer | None = server
        self.mode: str = mode

    def text(self, frame_idx: int = 0) -> str:
        s = self.server
        if self.mode == "death":
            return "1"
        if self.mode == "genuine":
            # Frame-keyed (no race with the server thread): shown right after
            # the 4-frame transition, well inside settle_seconds.
            return "2" if frame_idx > 8 else "1"
        # "late": the number stays behind until the unsplit went out, then shows up
        return "2" if s is not None and s.seen("unsplit") else "1"


@final
class Worker:
    def __init__(self, hud: Hud) -> None:
        self.hud: Hud = hud
        self.version: int = 0
        self.taken_version: int = 0
        self.status: str = "ok"
        self.latest: OcrResult | None = None

    def start(self) -> None:
        pass

    def submit_frame(self, _frame: np.ndarray, frame_idx: int, epoch: int = 0) -> None:
        self.latest = OcrResult(self.hud.text(frame_idx), 95.0, None, frame_idx, epoch, 0.0)
        self.version += 1

    def take_result(self) -> OcrResult | None:
        self.taken_version = self.version
        return self.latest

    def close(self) -> None:
        pass


def write_cfg(
    tmp_path: Path, port: int, mode: str, extra_undo: str = "", ls_extra: str = "",
    poll: float = 30,
) -> Path:
    if mode == "undo" and "follow_timer" not in ls_extra:
        ls_extra += ", follow_timer: true"  # required by ocr.mode: undo
    text = (
        "capture: {fps: 100, stream_url: 'udp://127.0.0.1:9733', width: 64, height: 36}\n"
        + "roi: {x: 0, y: 0, width: 1, height: 1}\n"
        + "detection: {required_frames: 2, reset_threshold: 0.5}\n"
        + "colors: {black: {rgb: [0, 0, 0], tolerance: 25}}\n"
        + "splits:\n"
        + "  - {id: 1, name: One, color: black, threshold: 0.9}\n"
        + "  - {id: 2, name: Two, color: black, threshold: 0.9}\n"
        + "  - {id: 3, name: Three, color: black, threshold: 0.9}\n"
        + f"livesplit: {{host: '127.0.0.1', port: {port}, timeout_seconds: 0.5,"
        + f" reconnect_seconds: 0.05, poll_seconds: {poll:g}{ls_extra}}}\n"
        + "ocr:\n  roi: {x: 0, y: 0, width: 0.5, height: 0.5}\n"
        + f"  mode: {mode}\n"
        + "  patterns: {'(\\d+)': null}\n"
        + "  filter: {max_age_seconds: 100, consensus_frames: 1, jump_consensus_frames: 1}\n"
        + "  undo: {settle_seconds: 0.05, confirm_reads: 2, confirm_seconds: 0.02,"
        + " window_seconds: 30, require_baseline: false, post_clear_seconds: 0,"
        + f" recover_seconds: 30{extra_undo}}}\n"
    )
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hud_mode: str,
    mode: str = "undo",
    livesplit: bool = True,
    server: LsServer | None = None,
    port: int | None = None,
    frames: int = 220,
    extra_undo: str = "",
    ls_extra: str = "",
    poll: float = 30,
) -> str:
    hud = Hud(server, hud_mode)

    def fake_build(_c: object) -> tuple[Worker, str]:
        return Worker(hud), "fake OK"

    monkeypatch.setattr(sm, "build_ocr_worker", fake_build)
    monkeypatch.setattr(mainmod, "MpegTsFrameSource", Source)
    p = write_cfg(tmp_path, port if port is not None else (server.port if server else 1), mode, extra_undo, ls_extra, poll)
    args = ["--config", str(p), "--live", "x.ts", "--max-frames", str(frames)]
    if livesplit:
        args.append("--livesplit")
    assert mainmod.main(args) == 0
    return capsys.readouterr().out


@pytest.fixture
def server() -> Iterator[LsServer]:
    s = LsServer()
    yield s
    s.close()


def test_death_is_unsplit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                          capsys: pytest.CaptureFixture[str], server: LsServer) -> None:
    out = run(tmp_path, monkeypatch, capsys, "death", server=server)
    assert "OCR undo: ACTIVE" in out
    assert 'OCR: UNDO "One" confirmed (screen 1, detector 2,' in out
    assert "\nLiveSplit: UNSPLIT sent -> index 0 (verified)" in out  # not glued to the status line
    assert "would UNDO" not in out
    assert server.moves() == ["split", "unsplit"], out
    assert server.idx == 0


def test_genuine_split_is_never_unsplit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str], server: LsServer) -> None:
    out = run(tmp_path, monkeypatch, capsys, "genuine", server=server)
    assert "OCR: HUD advanced" in out
    assert server.moves() == ["split"] and server.idx == 1


def test_false_undo_is_recovered_when_the_hud_shows_up_late(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], server: LsServer,
) -> None:
    out = run(tmp_path, monkeypatch, capsys, "late", server=server)
    assert "LiveSplit: UNSPLIT sent -> index 0 (verified)" in out
    assert "OCR: undo regret — HUD advanced late" in out
    assert "LiveSplit: SPLIT re-sent -> index 1 (verified)" in out
    assert server.moves() == ["split", "unsplit", "split"], out
    assert server.idx == 1


def test_recovery_can_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                  capsys: pytest.CaptureFixture[str], server: LsServer) -> None:
    out = run(tmp_path, monkeypatch, capsys, "late", server=server,
              extra_undo=", recover_late_hud: false")
    assert server.moves() == ["split", "unsplit"] and "undo regret" not in out


def test_without_livesplit_flag_degrades_to_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    server: LsServer,
) -> None:
    out = run(tmp_path, monkeypatch, capsys, "death", livesplit=False, server=server)
    assert "WARN: ocr.mode undo needs --livesplit; running as verify_only (dry run)" in out
    assert "would UNDO" in out and "ACTIVE" not in out
    assert "(dry run: ocr.mode undo needs --livesplit)" in out
    assert server.commands == []


def test_verify_only_with_livesplit_stays_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    server: LsServer,
) -> None:
    out = run(tmp_path, monkeypatch, capsys, "death", mode="verify_only", server=server)
    assert "would UNDO" in out and "ACTIVE" not in out
    assert "(ocr.mode is verify_only)" in out
    assert server.moves() == ["split"]  # never unsplit


def test_disconnected_at_fire_time_never_rolls_back_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    dead = socket.socket()
    dead.bind(("127.0.0.1", 0))
    port = cast("tuple[str, int]", dead.getsockname())[1]
    dead.close()
    out = run(tmp_path, monkeypatch, capsys, "death", port=port)
    assert "LiveSplit: undo skipped (server unavailable)" in out
    assert "verified" not in out
    assert out.count("undo skipped") == 1  # one attempt per verdict


def test_check_summary_active_vs_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for mode, word in (("undo", "ACTIVE (will send unsplit)"), ("verify_only", "DRY RUN")):
        p = write_cfg(tmp_path, 1, mode)
        assert mainmod.main(["--config", str(p), "--check"]) == 0
        out = capsys.readouterr().out
        assert f"ocr undo : {word}" in out
        assert f"mode={mode}" in out


@pytest.mark.parametrize(
    ("hud", "moves"),
    [("death", ["split", "unsplit"]), ("late", ["split", "unsplit", "split"]), ("genuine", ["split"])],
)
def test_undo_with_follow_timer_polling_sends_only_expected_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    server: LsServer, hud: str, moves: list[str],
) -> None:
    out = run(tmp_path, monkeypatch, capsys, hud, server=server,
              poll=0.2)
    assert server.moves() == moves, out
    queries = {"getcurrenttimerphase", "getsplitindex", "getsplitcount", "getcurrentsplitname"}
    assert set(server.commands) <= queries | {"split", "unsplit"}
    assert server.commands.count("unsplit") <= 1
    assert "UNSPLIT detected" not in out and "SKIP/manual split" not in out, out
