# vis-split: Visual Autosplitter

A generic visual autosplitter for speedruns on Linux: it watches a live capture of any game,
detects screen transitions by color signature, and fires splits to LiveSplit over TCP.
Runs natively on Linux/Wayland, independent of Wine/LiveSplit.

```
OBS Studio ──(sidecar mpegts over UDP or SRT)──> ffmpeg subprocess ──> raw BGR frames
                                                                  │
                                                color detector + state machine
                                                                  │
                                                LiveSplit Server <── split()
                                                      127.0.0.1:16834
```

Status: Phases 1–7 implemented — config validation, color detection, transition state
machine, MPEG-TS frame capture (`udp://`, `tcp://` or `srt://`) with `--live` demo, LiveSplit client, `--loop` rearm,
`--overlay` diagnostics window, `--probe` color picker, optional PySide6 GUI (`--gui`).
Design details: see `Project Overview — Roblox Visual.md`; dev rules: see `AGENTS.md`.

## Requirements

- Linux (developed on Arch/Wayland) with **ffmpeg** on PATH (`sudo pacman -S ffmpeg`)
- [uv](https://docs.astral.sh/uv/)
- OBS Studio (for live capture)
- No other required Python dependencies beyond what `uv sync` installs (OpenCV, NumPy, PyYAML); optional extras: `gui` (PySide6) and `ocr` (tesserocr)

## Setup

```bash
uv sync          # creates .venv from lockfile
uv run pytest -v # sanity check: everything green
```

## Quick start — color tuning with `--probe`

1. Capture a screenshot of your game's transition (e.g., level complete flash).
2. Point the ROI in your config to the flash region.
3. Run `--probe` to measure the average color:

```bash
uv run python -m autosplitter.main \
    --config configs/your-game.yaml \
    --probe screenshot.png
```

Output includes average RGB and a ready-to-paste YAML snippet for `colors:`.
Adjust `tolerance` based on how consistent the flash is across runs.

## Game configs

Per-game configs live in `configs/`:
- `configs/start-powerless.yaml` — 15 level splits (purple portal flash)
- `configs/template.yaml` — commented example for new games
- `configs/mr_rich.yaml` — 12 splits (white/black transitions) with a start gate
- `configs/test-white.yaml` — white-flash test config (4 splits) for checking the capture pipeline

Run with:
```bash
uv run python -m autosplitter.main --config configs/start-powerless.yaml --live --livesplit --loop
```

## Testing — three levels, fastest first

### 1. Automated test suite (~2 s, no OBS needed)

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
    --config configs/start-powerless.yaml \
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
uv run python -m autosplitter.main --config configs/start-powerless.yaml --live
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
   - **Type:** `Custom Output (FFmpeg)`
   - Tick **Show all codecs** near the encoder dropdowns (without it you may
     only see `mpeg2video`; if the list looks stale, re-select the container once)

   | Field | Value |
   |---|---|
   | Container format | `mpegts` |
   | File path or URL | `udp://127.0.0.1:9733?pkt_size=1316` |
   | Video Encoder | `h264_nvenc` (GPU) or `libx264` (CPU) — never MPEG-2 |
   | Video Encoder Settings | `bf=0` for nvenc · `tune=zerolatency bf=0` for libx264 |
   | Common FPS Value | e.g. `120` (Settings → Video) — the stream emits at this rate, *not* your monitor's Hz; must match `capture.fps` in your game config (`start-powerless.yaml` uses 120, the other configs 60) |
   | Rate Control / Bitrate | `CBR` / `4000 Kbps` |
   | Keyframe Interval | `1 s` |

   ⚠️ On h264_nvenc, do **not** add `delay=0 zerolatency=1`: measured testing
   showed those disable NVENC's smoothed rate control and produced 10–40 s
   of added latency on scene switches. Plain `bf=0` is the known-good config.
3. Settings → **Video** → **Output (Scaled) Resolution:** `1920×1080`
   — this must exactly equal `capture.width` × `capture.height` in your game config,
   regardless of canvas size (works for both 1920×1080 and 2560×1440 setups;
   both are 16:9).
4. Apply.

**Running it**

Order does not matter — the detector auto-reconnects with backoff, so start
Recording before or after launching `--live`. Streaming is untouched.

- OBS: click **Start Recording** (in addition to Start Streaming if desired)
- Terminal: `uv run python -m autosplitter.main --config configs/start-powerless.yaml --live`

### Alternative: separate low-res SRT feed (OBS Recording stays free)

The Custom Output (FFmpeg) recipe above takes over OBS's Recording slot. To keep
Recording for a normal full-quality archive and give the detector its own cheap feed
instead, use the **Branch Output** plugin (OBS ≥ 30.1; on Arch it is in the AUR as
`obs-branch-output`). It runs its own encoder in the same OBS instance and does
not touch your stream or recording.

> The AUR build can fail under current GCC with `-Wsfinae-incomplete` promoted to an
> error. Adding `-DCMAKE_COMPILE_WARNING_AS_ERROR=OFF` to the cmake call in the
> PKGBUILD fixes it.

1. Right-click your game **scene** → **Filters** → **+** → **Branch Output**.
2. In the filter's **Streaming** section (not Recording, which only writes files) set
   **Stream Count** to 1 and the server to `srt://127.0.0.1:9733`; leave the key empty.
   Plugin outputs are RTMP/SRT only, which is why this path uses SRT and not UDP.
3. Encoder: H.264 (NVENC is fine), **1500 kbps**, keyframe interval 1 s, resolution
   `640×360` via the filter's custom resolution. Very low bitrates (100 kbps measured)
   add ~230 ms of pacing delay; the small resolution is what saves CPU, not the bitrate.
4. Toggle the filter with its eye icon or the *Enable/Disable '<filter>'* hotkeys
   (Settings → Hotkeys). It runs independently of the Start Streaming/Recording buttons.

Run the splitter as the SRT **listener** and start it *before* enabling the filter:

```bash
uv run python -m autosplitter.main --config configs/your-game.yaml \
  --live "srt://127.0.0.1:9733?mode=listener&rcvlatency=20000" \
  --width 640 --height 360 --livesplit --loop
```

- `--width/--height` must equal the filter's output size exactly (or set
  `capture.width/height` in your config). A mismatch garbles frames silently; a fps
  counter reading ~10–20× too high is the tell.
- `rcvlatency` is in **microseconds**. `20000` (20 ms) instead of libsrt's ~120 ms default
  cut about 110 ms of latency on loopback.
- `capture.fps` must equal the filter's real stream rate; check with
  `ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of default=nw=1 "srt://127.0.0.1:9733?mode=listener"`
  before enabling the filter.
- Measured end to end on this setup: about 250 ms from screen to splitter (see
  *Measuring end-to-end latency*).
- If the splitter is started while the filter is already on, connection can take
  >10 s (OBS retries slowly). Start the splitter first, or re-toggle the filter.

## Connecting LiveSplit

The detector talks to LiveSplit's built-in TCP command server running under
Wine (default port `16834`, matching your game config).

1. In LiveSplit: right-click → **Control** → **Start TCP Server**.
   The server must be started manually each LiveSplit launch (or configure
   **Startup Behavior** in the server settings to do it automatically).
2. Run the detector with the flag:

   ```bash
   uv run python -m autosplitter.main --config configs/start-powerless.yaml --live --livesplit
   ```

Behavior:

- **Splits only.** The detector never starts, pauses, or resets your timer —
  you keep full control. Every debounced transition fires one `split`.
- If LiveSplit is down, detection continues unaffected: the status line shows
  `DISCONNECTED (retrying)` and reconnects at `livesplit.reconnect_seconds`
  cadence once the server returns.
- Wire protocol facts (for debugging): commands are newline-terminated
  (`split\n`); `split` produces no response; `ping` answers `pong`.

## Running a full session

Launch order (each step depends on the previous one being up):

1. **OBS** with the sidecar configured per *Connecting OBS* → **Start
   Recording**. Confirm the sidecar's Output (scaled) Resolution matches
   `capture.width/height`.
2. **LiveSplit** under Wine → right-click → **Control** → **Start TCP
   Server**, then start your timer manually.
3. **Your game** in the scene OBS is capturing.
4. The detector:

   ```bash
   uv run python -m autosplitter.main --config configs/start-powerless.yaml --live --livesplit --loop
   ```

Healthy output looks like: rolling fps steady at `capture.fps` (±15%), no
`HINT:` line, `LiveSplit: CONNECTED`, exactly one `SPLIT fired` line per real
transition, and — with `--loop` — a `RUN COMPLETE — rearmed` boundary between
practice runs.

Notes:

- The final split ends the run immediately: `DONE` is entered on the next
  frame after the last `SPLIT fired` line, even if the victory screen keeps
  the transition color on screen (hysteresis only shields *later* splits
  from re-triggering; there is none after the last one).
- `--loop` only rearms the *detection* machine after `DONE` (after
  `detection.restart_quiet_seconds` of the next run's first color staying
  quiet). **Reset your LiveSplit timer between runs yourself** — the detector
  never starts, pauses, or resets it (exception: an optional
  `start_transition.auto_start_timer` sends `start` once per run).
- Without `--loop`, the detector exits after every configured split has
  fired.
- With `--livesplit`, the detector polls LiveSplit every ~2 s for its timer
  phase and current split name (shown in the status line as
  `ls=Running:"Level 5"`). If the split the detector expects next is not
  contained in LiveSplit's active segment name (substring match, so e.g.
  `Level 5` matches `Level 5 - Cave`), a one-time
  `WARN: LiveSplit desync: ...` prints until the two agree again.
- When something looks off, note: fps dips, a late `SPLIT` relative to the
  on-screen transition, `DISCONNECTED` blips, desync `WARN`s, or missing
  splits during heavy packet loss (silent at the UDP layer). Each maps to a
  row in *Troubleshooting*.

## Diagnostics

Add `--overlay` to the live command to open a window showing the ROI rectangle
and a HUD over the live frame:

```bash
uv run python -m autosplitter.main --config configs/start-powerless.yaml --live --livesplit --loop --overlay
```

The window displays:
- Green ROI rectangle (from `roi` in your game config)
- HUD lines:
  - `fps: XX.X` — rolling capture frame rate
  - `match: YY.YY%` — current split's color match ratio
  - `target: "Transition N (color)"` — active split name and color
  - `threshold: ZZ.Z%` — split's detection threshold
  - `state: NORMAL / TRANSITION / WAITING_FOR_RESET / DONE`
  - `split: M/N` — current split index / total (— when DONE)
  - `last: SS.Ss` — seconds since last split (— if none)
  - `ls: CONNECTED / DISCONNECTED` — LiveSplit Server status

Window controls:
- **ESC** or **q** to close the window and stop the detector
- Window close button works normally
- Overlay refreshes at ~`capture.fps / 10` Hz (e.g. 12 Hz at 120 fps) to avoid
  stalling the 120 fps read loop

If `cv2.imshow` fails (e.g. running headless without XWayland), a one-time
`HINT` is printed and the overlay is disabled for that run — the detector
continues running in the terminal.

## Measuring end-to-end latency

When splits feel late, isolate whether the delay is in OBS or in the detector
pipeline by feeding the identical UDP path from a local sender with a
burned-in millisecond clock:

```bash
# 1. Stop OBS Recording first — two senders cannot share the multicast port.
# 2. Terminal 1 — synthetic sender (testsrc displays a running clock):
ffmpeg -re -f lavfi -i testsrc=size=1920x1080:rate=60 -pix_fmt yuv420p \
    -c:v h264_nvenc -b:v 4000k -bf 0 -g 60 -f mpegts \
    "udp://127.0.0.1:9733?pkt_size=1316"
# 3. Terminal 2 — grab a frame after ~15 s and exit:
uv run python -m autosplitter.main --live --snap 900
```

Open `snap.png`: the difference between the burned-in clock and the wall-clock
time elapsed since the sender started is the true latency of the entire
detector chain (±1 s human-timing error). Healthy result: well under 1 s.
If the chain measures clean but live splits still lag seconds behind, the
delay originates in OBS's sidecar output path.

That test bypasses OBS entirely. To include OBS (e.g. the SRT Branch Output feed),
show a wall clock in the OBS scene and compare it to when the splitter saved the frame:

1. Show a millisecond wall clock in a window OBS captures, for example:
   ```bash
   uv run python -c 'exec("import cv2,time,numpy as np\nimg=np.zeros((300,1100,3),np.uint8)\nwhile True:\n img[:]=0\n cv2.putText(img,format(time.time(),\".3f\"),(20,200),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),8)\n cv2.imshow(\"clock\",img)\n if cv2.waitKey(1) in (27,113):\n  break")'
   ```
2. Run `--live ... --snap 300` against the feed (splitter first, then enable the filter).
3. Latency = `stat -c %.3Y snap.png` (when the frame arrived) minus the clock value
   visible in `snap.png`. The clock window's own redraw and OBS's capture add tens of
   ms, so this is an upper bound. Repeat a few times; consistent runs (within ~30 ms)
   mean a fixed delay somewhere in the chain rather than jitter.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `capture unavailable, reconnecting...` forever | Recording not started / wrong port | Check **Start Recording** is lit red; URL matches `udp://127.0.0.1:9733`. For an `srt://...mode=listener` source the message is normal until OBS connects: enable the Branch Output filter |
| SRT source: fps reads in the hundreds/thousands, or frames look striped | The filter's real output size differs from `--width/--height` | `ffprobe` the stream and pass the real size, or fix the filter's resolution setting |
| SRT source: connects only after >10 s | Filter was already on when the splitter started; OBS is waiting to retry | Start the splitter first, then enable the filter (or re-toggle it) |
| SRT source: latency of several hundred ms | Very low encoder bitrate (CBR pacing) and/or libsrt's default ~120 ms receive latency | Raise bitrate to ~1500 kbps; add `&rcvlatency=20000` (µs) to the URL |
| Garbage fps / weird match numbers / instant EOF | Resolution mismatch | Sidecar Output (Scaled) Resolution must equal config `capture.width/height` exactly |
| fps counter pinned below ~60 (e.g. ~45), latency grows the longer it runs | Consumer slower than producer — backlog accumulates every second | Per-frame Python work must stay allocation-free (see AGENTS.md hot-loop rules); check `--live` fps first whenever splits feel late |
| Splits late by many seconds while fps counter stays steady ~60 | Frames arrive stale from upstream — delay sits in OBS's output path, not the detector | Verify chain via latency measurement; revert risky encoder settings (`delay=0 zerolatency=1`) |
| `LiveSplit: DISCONNECTED` never recovers | TCP server not started, or wrong port | Control → **Start TCP Server** each launch; confirm port matches `livesplit.port` |
| `non-existing PPS 0 referenced / no frame!` storm at startup | Joined the broadcast mid-GOP; decoder waits for the next keyframe | Normal and self-healing (≤ keyframe interval). Start Recording before launching `--live` to avoid it |
| Glitchy decode errors but no `Circular buffer overrun` warnings | Packet loss happens silently in the kernel's small UDP socket buffer, before ffmpeg's ring | Don't rely on overrun messages to detect loss; keep encoder bitrate spikes smoothed (no reactive-CBR tricks) |
| First frame torn after mid-run Recording start | Joined before next keyframe | Harmless; resolves within keyframe interval (≤1 s) |

## Configuration

All tunables live in your game config (e.g., `configs/start-powerless.yaml`):

- `capture` — sidecar URL (`udp://`, `tcp://` or `srt://`), frame size (must equal the sidecar's output resolution), target fps
- `roi` — detection region as normalized floats 0.0–1.0 (resolution-independent);
  accepts origin+size (`x/y/width/height`) or corner endpoints (`x1/y1/x2/y2`)
- `detection` — debounce frames required per split + hysteresis reset threshold
- `colors` — named RGB targets with tolerance
- `splits` — ordered transitions referencing defined colors, each with its own match threshold
- `livesplit` — host/port for the TCP client
- `start_transition` (optional) — color gate that starts the run and, with `auto_start_timer`, the LiveSplit timer
- `ocr` (optional, needs the `ocr` extra) — split-name OCR used to resync the detector
- `record` (optional) — splitter-side recording; planned for removal (see `AGENTS.md`)

Invalid config fails with precise errors such as
`configs/start-powerless.yaml: splits[2].threshold must be between 0.0 and 1.0`.

## Development

```bash
uv run pytest -v                        # tests
uvx ruff check .                        # lint
uv run basedpyright src tests           # typecheck (target 0 warnings; ~199 pre-existing, see AGENTS.md)
uv run python -m autosplitter.main --check   # validate config, print summary
```

Verification order per phase: tests → ruff → basedpyright → phase demo command.

Other `main` flags: `--detect IMAGE.png` (run every split's detector over a still image),
`--record PATH` (splitter-side codec-copy .mkv; planned for removal, see `AGENTS.md`),
`--config-poll-interval SECONDS` (with `--live`, how often the config file is re-read) and
`--gui` (PySide6 GUI; needs `uv sync --extra gui`).
Conventions and architecture rules: see `AGENTS.md`.
