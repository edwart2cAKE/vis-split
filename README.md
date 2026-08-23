# Roblox Visual Autosplitter

A visual autosplitter for Roblox speedruns on Linux: it watches a live capture of the game,
detects screen transitions by color signature, and fires splits to LiveSplit over TCP.
Runs natively on Linux/Wayland, independent of Wine/LiveSplit.

```
OBS Studio ──(sidecar mpegts over UDP)──> ffmpeg subprocess ──> raw BGR frames
                                                                    │
                                              color detector + state machine
                                                                    │
                                              LiveSplit Server <── split()
                                                    127.0.0.1:16834
```

Status: Phases 1–4 implemented — config validation, color detection, transition state
machine, MPEG-TS frame capture with `--live` demo. LiveSplit client comes next.
Design details: see `Project Overview — Roblox Visual.md`; dev rules: see `AGENTS.md`.

## Requirements

- Linux (developed on Arch/Wayland) with **ffmpeg** on PATH (`sudo pacman -S ffmpeg`)
- [uv](https://docs.astral.sh/uv/)
- OBS Studio (for live capture)
- No other Python dependencies beyond what `uv sync` installs (OpenCV, NumPy, PyYAML)

## Setup

```bash
uv sync          # creates .venv from lockfile
uv run pytest -v # sanity check: everything green
```

## Testing — three levels, fastest first

### 1. Automated test suite (~0.3 s, no OBS needed)

```bash
uv run pytest -v
```

Covers config validation (exact error messages), color detection (BGR/RGB order,
tolerance boundaries, ROI math), state machine semantics (debounce, hysteresis,
one split per transition), and frame capture (partial-read assembly plus real
ffmpeg decodes of a committed fixture).

### 2. Offline smoke test (~1 s, no OBS needed)

Plays a committed 64×64 test-pattern `.ts` file through the entire pipeline:
ffmpeg subprocess → raw pipe → frame assembly → state machine.

```bash
uv run python -m autosplitter.main \
    --config config.yaml \
    --live tests/fixtures/testsrc_64x64.ts \
    --width 64 --height 64 \
    --max-frames 40
```

(`--width/--height` are needed here because config declares 1920×1080 while the
fixture is 64×64.)

Success looks like:

```
Live from tests/fixtures/testsrc_64x64.ts @ 64x64 (required_frames=3, reset_threshold=0.80)
Waiting for: 1. "Transition 1" (threshold 95.0%)
#1        0.0 fps  match   0.00%  NORMAL ... #30  11723.7 fps  match   0.00%  NORMAL
Source exhausted (EOF).
```

- Frames counting up + fps climbing = decode pipeline works
- `match 0.00%` is correct: testsrc2 contains no black/white transition screens
- `Source exhausted (EOF)` = clean shutdown, no hang

### 3. Real end-to-end test (with OBS)

Configure OBS once (next section), hit **Start Recording**, then:

```bash
uv run python -m autosplitter.main --config config.yaml --live
```

Success looks like a rolling line updating ~60×/second:

```
#1420    59.8 fps  match  12.34%  NORMAL
```

Then make something black fullscreen (or load a black scene): `match %` spikes,
state flips `NORMAL → TRANSITION`, and a persistent line appears:

```
SPLIT fired: id=1 "Transition 1" at frame 1487
```

Ctrl-C exits cleanly at any time.

## Connecting OBS

The detector never touches your stream. It taps a second, private output that runs
alongside it.

**One-time OBS configuration**

1. File → Settings → **Output** → top-left **Output Mode:** `Advanced`
2. Stay in Output → open the **Recording** tab:
   | Field | Value |
   |---|---|
   | Type | `Custom Output (FFmpeg)` |
   | Container format | `mpegts` |
   | File path or URL | `udp://127.0.0.1:9733?pkt_size=1316` |
   | Encoder | `NVIDIA NVENC H.264` (or `Software (x264)`) |
   | Rate Control / Bitrate | `CBR` / `4000 Kbps` |
   | Keyframe Interval | `1 s` |
3. Settings → **Video** → **Output (Scaled) Resolution:** `1920×1080`
   — this must exactly equal `capture.width` × `capture.height` in `config.yaml`,
   regardless of canvas size (works for both 1920×1080 and 2560×1440 setups;
   both are 16:9).
4. Apply.

**Running it**

Order does not matter — the detector auto-reconnects with backoff, so start
Recording before or after launching `--live`. Streaming is untouched.

- OBS: click **Start Recording** (in addition to Start Streaming if desired)
- Terminal: `uv run python -m autosplitter.main --config config.yaml --live`

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `capture unavailable, reconnecting...` forever | Recording not started / wrong port | Check **Start Recording** is lit red; URL matches `udp://127.0.0.1:9733` |
| Garbage fps / weird match numbers / instant EOF | Resolution mismatch | Sidecar Output (Scaled) Resolution must equal config `capture.width/height` exactly |
| Works but seconds of latency | B-frames/buffering | x264: add `tune=zerolatency bframes=0` in the encoder settings field |
| First frame torn after mid-run Recording start | Joined before next keyframe | Harmless; resolves within keyframe interval (≤1 s) |

## Configuration

All tunables live in `config.yaml`:

- `capture` — sidecar URL (`udp://`/`tcp://`), pinned 1920×1080, target fps
- `roi` — detection region as normalized floats 0.0–1.0 (resolution-independent);
  accepts origin+size (`x/y/width/height`) or corner endpoints (`x1/y1/x2/y2`)
- `detection` — debounce frames required per split + hysteresis reset threshold
- `colors` — named RGB targets with tolerance
- `splits` — ordered transitions referencing defined colors, each with its own match threshold
- `livesplit` — host/port for the TCP client (used by upcoming phases)

Invalid config fails with precise errors such as
`config.yaml: splits[2].threshold must be between 0.0 and 1.0`.

## Development

```bash
uv run pytest -v                        # tests
uvx ruff check .                        # lint
uv run basedpyright src tests           # typecheck (must be 0 warnings)
uv run python -m autosplitter.main --check   # validate config, print summary
```

Verification order per phase: tests → ruff → basedpyright → phase demo command.
Conventions and architecture rules: see `AGENTS.md`.
