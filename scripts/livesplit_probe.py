#!/usr/bin/env python3
"""Probe a REAL LiveSplit Server and check the assumptions vis-split relies on.

stdlib only. Run it with a LiveSplit that has the TCP server started
(Control -> Start TCP Server), e.g.:

    uv run python scripts/livesplit_probe.py                # read-only queries
    uv run python scripts/livesplit_probe.py --active       # drives the timer!

The default mode only sends queries (and one deliberately bogus command that
must get no reply). ``--active`` additionally sends ``start``, ``split``,
``unsplit`` (and with ``--test-pause`` ``pause``/``resume``) to the timer. Use
it ONLY with a throwaway splits file / layout, never your PB splits: it starts
and splits the timer, and a reset afterwards will offer to update your times.
The probe never sends ``reset`` unless you pass ``--allow-reset``.

Each check is tied to an assumption the detector makes (see
docs/LIVESPLIT_VALIDATION.md). Exit code 0 = no FAIL, 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import asdict, dataclass

PHASES = {"NotRunning", "Running", "Paused", "Ended"}


@dataclass
class Result:
    ident: str
    what: str
    verdict: str  # PASS | FAIL | WARN | INFO | SKIP
    detail: str


class Conn:
    """A tiny line-oriented client that keeps the raw reply terminators."""

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.buf = b""

    def close(self) -> None:
        self.sock.close()

    def send(self, cmd: str) -> None:
        self.sock.sendall(cmd.encode() + b"\n")

    def recv_line(self, timeout: float) -> str | None:
        """One reply line INCLUDING its terminator, or None on timeout."""
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            self.sock.settimeout(left)
            try:
                chunk = self.sock.recv(4096)
            except TimeoutError:
                return None
            if not chunk:
                raise ConnectionError("server closed the connection")
            self.buf += chunk
        line, _, self.buf = self.buf.partition(b"\n")
        return (line + b"\n").decode("utf-8", errors="replace")

    def drain(self, quiet: float = 0.05) -> list[str]:
        out: list[str] = []
        while (line := self.recv_line(quiet)) is not None:
            out.append(line)
        return out

    def query(self, cmd: str, timeout: float = 1.0) -> str | None:
        """Send one command, return its reply text (terminator stripped)."""
        self.drain(0.02)
        self.send(cmd)
        line = self.recv_line(timeout)
        return None if line is None else line.rstrip("\r\n")


def to_int(text: str | None) -> int | None:
    try:
        return None if text is None else int(text)
    except ValueError:
        return None


def state(c: Conn) -> tuple[str | None, int | None]:
    return c.query("getcurrenttimerphase"), to_int(c.query("getsplitindex"))


def wait_for(c: Conn, want_phase: str, timeout: float = 2.0) -> tuple[str | None, int | None]:
    end = time.monotonic() + timeout
    ph, idx = state(c)
    while ph != want_phase and time.monotonic() < end:
        time.sleep(0.05)
        ph, idx = state(c)
    return ph, idx


def read_only(c: Conn, out: list[Result]) -> tuple[str | None, int | None, int | None]:
    pong = c.query("ping")
    out.append(Result("A0", "server answers `ping` with pong", "PASS" if pong == "pong" else "WARN",
                      f"reply {pong!r} (older servers may not implement ping)"))

    # Raw terminator style (the client strips both \r\n and \n; this is informational).
    c.drain(0.02)
    c.send("getcurrenttimerphase")
    raw = c.recv_line(1.0)
    out.append(Result("A1", "reply line terminator", "INFO", f"raw reply {raw!r}"))

    phase = c.query("getcurrenttimerphase")
    out.append(Result("A2", "getcurrenttimerphase is one of NotRunning/Running/Paused/Ended",
                      "PASS" if phase in PHASES else "FAIL", f"reply {phase!r}"))
    idx = to_int(c.query("getsplitindex"))
    expected = {"NotRunning": "-1"}.get(phase or "", "0..N-1 (Running/Paused) or N (Ended)")
    out.append(Result("A3", "getsplitindex is an integer (-1 when NotRunning)",
                      "PASS" if idx is not None else "FAIL", f"phase {phase}, index {idx} (expected {expected})"))
    if phase == "NotRunning" and idx is not None and idx != -1:
        out.append(Result("A3b", "index is -1 while NotRunning", "FAIL", f"got {idx}"))
    count = to_int(c.query("getsplitcount"))
    out.append(Result("A4", "getsplitcount exists (else vis-split skips the count validation)",
                      "PASS" if count is not None else "WARN",
                      f"reply {count!r}" if count is not None else "no/invalid reply: skip-forward runs unvalidated"))
    name = c.query("getcurrentsplitname")
    out.append(Result("A4b", "getcurrentsplitname answers", "PASS" if name is not None else "WARN", f"reply {name!r}"))
    bogus = c.query("vissplit_probe_bogus_command", timeout=0.6)
    out.append(Result("A5", "unknown command gets NO reply", "PASS" if bogus is None else "WARN",
                      f"reply {bogus!r}"))
    return phase, idx, count


def ordering_test(c: Conn, n: int, repeats: int, out: list[Result]) -> None:
    """The key assumption: is `unsplit` applied before the next getsplitindex reply?"""
    immediate = delayed = 0
    latencies: list[float] = []
    wrong_end: list[str] = []
    for i in range(repeats):
        c.send("split")  # fire-and-forget: LiveSplit sends no reply to `split`
        time.sleep(0.35)
        ph, idx = state(c)
        if (ph, idx) != ("Ended", n):
            out.append(Result("A6", "reach Ended before unsplit", "FAIL", f"round {i}: phase {ph}, index {idx}"))
            return
        c.drain(0.02)
        c.send("unsplit")
        c.send("getsplitindex")
        c.send("getcurrenttimerphase")
        t0 = time.monotonic()
        first = to_int((c.recv_line(1.0) or "").strip() or None)
        phase_reply = (c.recv_line(1.0) or "").strip()
        if first == n - 1:
            immediate += 1
            latencies.append(0.0)
        else:
            delayed += 1
            end = time.monotonic() + 2.0
            while time.monotonic() < end:
                if to_int(c.query("getsplitindex")) == n - 1:
                    latencies.append((time.monotonic() - t0) * 1000.0)
                    break
                time.sleep(0.005)
        ph, idx = state(c)
        if (ph, idx) != ("Running", n - 1):
            wrong_end.append(f"round {i}: {ph}/{idx} (phase reply right after unsplit: {phase_reply!r})")
    out.append(Result("A6", "`unsplit` from Ended -> Running at N-1 (last-split undo depends on it)",
                      "PASS" if not wrong_end else "FAIL", "; ".join(wrong_end) or f"{repeats}/{repeats} rounds"))
    worst = max(latencies) if latencies else float("nan")
    verdict = "PASS" if delayed == 0 else ("WARN" if worst < 275 else "FAIL")
    out.append(Result("A7", "`unsplit` is applied BEFORE the next getsplitindex reply (executor verify)", verdict,
                      f"immediate {immediate}/{repeats}, delayed {delayed}/{repeats}, worst apply latency "
                      f"{worst:.0f} ms (the executor's synchronous re-check window is ~275 ms; later => "
                      "needs follow_timer to repair)"))


def active(c: Conn, n: int, repeats: int, test_pause: bool, out: list[Result]) -> None:
    ph, idx = state(c)
    if (ph, idx) != ("NotRunning", -1):
        out.append(Result("S0", "timer must be NotRunning/-1 before --active", "FAIL",
                          f"phase {ph}, index {idx}: press Reset in LiveSplit (do not save) and re-run"))
        return
    c.send("start")
    ph, idx = wait_for(c, "Running")
    out.append(Result("A8", "`start` from NotRunning -> Running at index 0 (auto_start_timer)",
                      "PASS" if (ph, idx) == ("Running", 0) else "FAIL", f"phase {ph}, index {idx}"))
    c.send("start")  # the detector can send a second start while already running: must be harmless
    time.sleep(0.2)
    ph, idx = state(c)
    out.append(Result("A9", "a second `start` while Running is ignored", "PASS" if (ph, idx) == ("Running", 0) else "FAIL",
                      f"phase {ph}, index {idx}"))

    for k in range(n - 1):  # reach Running at N-1
        time.sleep(0.35)  # LiveSplit ignores a split while RealTime is 0
        c.send("split")
        time.sleep(0.25)
        ph, idx = state(c)
        if idx != k + 1:
            out.append(Result("A10", "`split` advances the index", "FAIL", f"after split {k + 1}: {ph}/{idx}"))
            return
    out.append(Result("A10", "`split` advances the index one at a time", "PASS", f"Running at {n - 1}"))

    if test_pause:
        c.send("pause")
        time.sleep(0.2)
        pph, pidx = state(c)
        c.send("split")
        time.sleep(0.3)
        qph, qidx = state(c)
        out.append(Result("A11", "`split` is IGNORED while Paused (follow_pause assumption)",
                          "PASS" if (qph, qidx) == (pph, pidx) and pph == "Paused" else "FAIL",
                          f"before {pph}/{pidx}, after split {qph}/{qidx}"))
        c.send("resume")
        wait_for(c, "Running")
    else:
        out.append(Result("A11", "`split` ignored while Paused", "SKIP", "pass --test-pause to check"))

    # normal (non-Ended) unsplit, then back; then the Ended ordering test.
    c.drain(0.02)
    c.send("unsplit")
    c.send("getsplitindex")
    first = to_int((c.recv_line(1.0) or "").strip() or None)
    time.sleep(0.3)
    ph, idx = state(c)
    out.append(Result("A12", "`unsplit` while Running -> index N-2", "PASS" if idx == n - 2 else "FAIL",
                      f"reply right after the command {first!r}, settled {ph}/{idx}"))
    time.sleep(0.35)
    c.send("split")
    time.sleep(0.3)
    ordering_test(c, n, repeats, out)

    # `split` into Ended (the detector may fire one after the run finished) is ignored.
    # The ordering rounds end at Running N-1: one split reaches Ended, the next is the probe.
    time.sleep(0.35)
    c.send("split")
    time.sleep(0.3)
    ph, idx = state(c)
    c.send("split")
    time.sleep(0.3)
    ph2, idx2 = state(c)
    out.append(Result("A13", "`split` while Ended is ignored",
                      "PASS" if (ph, idx) == ("Ended", n) and (ph2, idx2) == ("Ended", n) else "WARN",
                      f"{ph}/{idx} -> {ph2}/{idx2}"))

    # unsplit down to 0, then a no-op at 0
    c.send("unsplit")
    time.sleep(0.3)
    for _ in range(n + 1):
        c.send("unsplit")
        time.sleep(0.25)
    ph, idx = state(c)
    out.append(Result("A14", "`unsplit` at index 0 is a no-op (stays Running at 0)",
                      "PASS" if (ph, idx) == ("Running", 0) else "FAIL", f"phase {ph}, index {idx}"))
    c.send("undosplit")
    time.sleep(0.2)
    ph, idx = state(c)
    out.append(Result("A15", "`undosplit` alias accepted (index stays 0 here, so this only checks no error)",
                      "INFO", f"phase {ph}, index {idx}"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=16834)
    ap.add_argument("--timeout", type=float, default=2.0, help="socket connect timeout (s)")
    ap.add_argument("--active", action="store_true", help="ALSO drive the timer (start/split/unsplit); throwaway splits only")
    ap.add_argument("--yes", action="store_true", help="skip the interactive confirmation for --active")
    ap.add_argument("--splits", type=int, default=0, help="number of segments if getsplitcount is unsupported")
    ap.add_argument("--repeats", type=int, default=5, help="rounds of the unsplit-from-Ended ordering test")
    ap.add_argument("--test-pause", action="store_true", help="with --active: also pause/resume (checks follow_pause)")
    ap.add_argument("--allow-reset", action="store_true", help="with --active: send `reset` at the end")
    ap.add_argument("--report", default="", help="write the results as JSON to this path")
    args = ap.parse_args()

    try:
        c = Conn(args.host, args.port, args.timeout)
    except OSError as exc:
        print(f"cannot connect to {args.host}:{args.port}: {exc}\n"
              "Start LiveSplit and use Control -> Start TCP Server.", file=sys.stderr)
        return 2

    results: list[Result] = []
    try:
        count = read_only(c, results)[2]
        n = count if count else args.splits
        if args.active:
            if not n:
                results.append(Result("S0", "number of segments known", "FAIL",
                                      "getsplitcount unsupported: pass --splits N"))
            else:
                print("\n*** --active will START and SPLIT the timer on the connected LiveSplit. ***\n"
                      "Use a throwaway splits file/layout with at least 3 segments. After the run, press\n"
                      "Reset in LiveSplit and choose NOT to update your splits.\n")
                if not args.yes and input('Type "START" to continue: ').strip() != "START":
                    print("aborted")
                    return 2
                active(c, n, args.repeats, args.test_pause, results)
                if args.allow_reset:
                    c.send("reset")
                    time.sleep(0.3)
                    ph, idx = state(c)
                    results.append(Result("S9", "`reset` returns to NotRunning/-1", "PASS" if (ph, idx) == ("NotRunning", -1) else "WARN",
                                          f"phase {ph}, index {idx}"))
                else:
                    print("Press Reset in LiveSplit now (choose not to update splits).")
    finally:
        c.close()

    width = max(len(r.ident) for r in results)
    for r in results:
        print(f"[{r.verdict:4s}] {r.ident:<{width}}  {r.what}\n        {r.detail}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
    fails = [r for r in results if r.verdict == "FAIL"]
    print(f"\n{len(fails)} FAIL, {sum(r.verdict == 'WARN' for r in results)} WARN, "
          f"{sum(r.verdict == 'PASS' for r in results)} PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
