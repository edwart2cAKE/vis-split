# AGENTS.md

## Status

Planning repo, Phases 1-4, 5 (Diagnostics overlay), 6 (LiveSplit), and Phase 7 `--loop` rearm feature implemented (config + color detector + state machine + MPEG-TS frame capture + LiveSplit client + loop). The single source of truth is `Project Overview — Roblox Visual.md` — read it before implementing anything. It defines the full design, phased build order (section 19), reference config (section 21), and dev rules (section 20).

Per-game configs live in `configs/` (e.g., `configs/start-powerless.yaml`, `configs/template.yaml`).

## Commands

uv-managed project (src layout). Always use `uv run`; never system pip.

- `uv sync` — create/update `.venv` from lockfile (run first)
- `uv run pytest -v` — run tests
- `uvx ruff check .` — lint (no project install needed)
- `uv run basedpyright src tests` — typecheck in strict "recommended" mode (matches Zed); `[tool.basedpyright]` in pyproject.toml points at `.venv`, which also fixes import resolution in Zed
- `uv run python -m autosplitter.main --config configs/your-game.yaml --check` — validate config, print summary; exit 1 + precise error if invalid
- `uv run python -m autosplitter.main --config configs/your-game.yaml --detect IMAGE.png` — run every split's color detector over a static image; per-split match % / threshold / HIT
- `uv run python -m autosplitter.main --config configs/your-game.yaml --probe IMAGE.png` — load image, crop to ROI, print average RGB + suggested YAML snippet for color tuning
- `uv run python -m autosplitter.main --live [SOURCE] [--width W] [--height H] [--max-frames N] [--snap N] [--livesplit] [--loop] [--overlay] [--record PATH]` — live capture demo: rolling fps/match/state status; SOURCE may be a `.ts` file for offline smoke tests (then pass its dimensions via --width/--height); `--snap N` saves frame N to snap.png and exits; `--livesplit` forwards split events to LiveSplit Server; `--loop` rearms the machine after DONE once the next run's first color stays quiet for `detection.restart_quiet_seconds` (user resets their own timer between runs); `--overlay` shows ROI rectangle + HUD window (throttled ~10 Hz, ESC/q to close); `--record PATH` writes a codec-copy .mkv to PATH while detecting (single ffmpeg process — no second UDP consumer needed); Ctrl-C or EOF stops. If `--record` is omitted but a `record:` config section exists, the path is resolved from the config's `path_template` (supports `{game}`, `{date}`, `{time}`, `{datetime}`, `{attempt}` variables). On exit, if `record.convert_to` is set, ffmpeg converts the .mkv to the target format (best-effort, .mkv never deleted).

Verification order per phase: tests → ruff → basedpyright → phase demo command.

basedpyright strictness: code must stay clean at 0 warnings. PyYAML, argparse, and bare numpy `ndarray` are untyped — pin the boundary with `cast(object, ...)` / `cast(dict[str, object], ...)` / `cast(tuple[int, int], arr.shape[:2])` immediately after isinstance validation rather than disabling rules. Only `reportUnusedCallResult` is off (argparse boilerplate).

Pipeline convention: frames are BGR uint8 `(H, W, 3)` everywhere (OpenCV convention); config colors are RGB and converted to BGR once in `ColorDetector.__init__`. Crop the ROI before any processing — never full-frame work.


## Platform

- Target: Linux (Arch/Wayland) + OBS Studio. Never introduce Windows-only APIs.
- LiveSplit runs under Wine; the detector runs natively on Linux, independent of LiveSplit.
- LiveSplit Server over TCP: `127.0.0.1:16834`.

## Non-negotiables

- All tunables live in your game config (e.g., `configs/your-game.yaml`). Never hard-code resolution, colors, or thresholds.
- ROI coordinates are normalized floats (0.0–1.0) so resolution changes don't break detection.
- Color matching uses configurable tolerance, never exact RGB equality.
- Process only the cropped ROI with vectorized NumPy/OpenCV — no per-pixel Python loops (target ~60 FPS).
- Detection is a state machine (`NORMAL → TRANSITION → WAITING_FOR_RESET`, terminal `DONE`): require N consecutive matching frames before splitting (debounce), use `reset_threshold` hysteresis after a split, and never emit more than one split per continuous transition. Contract: streak counts frames at/above the active split's threshold; clearing is strictly below `reset_threshold`; a frame that clears one split never counts toward the next one's streak. The LAST split skips `WAITING_FOR_RESET` — DONE is entered on the next frame with the index exhausted (so `current_split`/`current_detector` are None even if the victory screen keeps the color lit; a persistent final color must not stall the run). With `start_transition`, its TRANSITION settles through `WAITING_FOR_RESET` but clears WITHOUT advancing the index (the gate precedes splits[0]); `_transition_is_start` tracks which kind of TRANSITION is settling.

## Boundaries

- Keep frame acquisition (`capture/`) separate from detection logic (`detection/`); input source must be swappable.
- Keep all LiveSplit protocol logic inside `livesplit/client.py` (`LiveSplitClient`); the detector only calls methods like `split()`.
- Detector and state machine must be unit-testable from prerecorded frames with no OBS or LiveSplit present (pytest once tests exist).
- v1 ships the color detector only. Do not implement template matching until color detection works.

## Dependencies

Minimal set: OpenCV, NumPy, PyYAML, plus whatever the chosen OBS capture method requires. Do not add frameworks without a demonstrated problem.

## Gotchas

- Capture method (decided after research): OBS sidecar **Recording → Custom Output (FFmpeg)**, container `mpegts`, URL `udp://127.0.0.1:9733?pkt_size=1316` — runs alongside Streaming, pure userspace, no new pip deps (system `ffmpeg` binary decodes to raw BGR24 on a pipe). obs-websocket screenshot polling was rejected (~15-20 fps); virtual cam needs v4l2loopback kernel module. The autosplitter's `--live` is the sole UDP consumer; when `--record PATH` is passed, the same ffmpeg subprocess also writes a .mkv copy (no extra socket needed).
- Pin the sidecar's **Output (scaled) Resolution to 1920x1080** regardless of canvas (works for both the 1920x1080 and 2560x1440 setups; both are 16:9). `capture.width/height` in config must match it exactly — a mismatch garbles frames silently and the pipe reader hits EOF. Receiver-side URLs (config `stream_url`) should only have `pkt_size`; output-only params (`fifo_size`, `overrun_nonfatal`, `buffer_size`) cause ffmpeg to fail when opening the input.
- Sidecar encoder must be H.264 — tick **Show all codecs** in OBS's Custom FFmpeg output or only mpeg2video lists. Video Encoder Settings: nvenc → `bf=0`, libx264 → `tune=zerolatency bf=0`. Do NOT add nvenc `delay=0 zerolatency=1`: measured 10-40s latency regression (reactive CBR pacing). If splits fire late while `--live` fps stays steady ~60, frames are arriving stale from upstream (OBS), not queued in our pipeline (drain would spike fps).
- Latency measurement: stop Recording, send a synthetic clock (`ffmpeg -re -f lavfi -i testsrc=size=1920x1080:rate=60 -c:v h264_nvenc -bf 0 -g 60 -f mpegts udp://127.0.0.1:9733?pkt_size=1316`), run `--live --snap N`, compare snap.png's burned-in clock vs wall time.
- Packet loss is silent: kernel UDP socket buffer overflows before ffmpeg's fifo ring, so missing `Circular buffer overrun` warnings prove nothing.
- Hot loop must stay allocation-free per frame (measured: naive bytearray+copy reader + per-frame `build_detector` capped the pipeline at ~45 fps while a bare `ffmpeg -f rawvideo -pix_fmt bgr24 - >/dev/null` on the same URL sustains 59; fixed consumer paces at 60.1). Invariants: `RawPipeReader` decodes into one reused buffer — the returned ndarray is invalidated by the next read, copy it to retain; `TransitionDetector` builds its `ColorDetector`s once in `__init__`; `ColorDetector` precomputes `cv2.inRange` bounds at construction. Symptom of regression anywhere in this chain: `--live` fps pinned below ~60 with latency climbing steadily.
- Stream rate = OBS's **Common FPS Value** (Settings → Video), not monitor Hz; currently 120 fps, matching `capture.fps`. Debounce is frame-counted: keep `required_frames ≈ capture.fps / 20` (~50 ms of flicker immunity). `--live` prints a one-time HINT when a udp/tcp stream's rolling fps deviates >15% from `capture.fps`.
- LiveSplit Server wire protocol: commands case-sensitive, newline-terminated (`split\n`); `split` has NO response, `ping` returns `pong`; server must be started manually per launch (Control → Start TCP Server) unless Startup Behavior is configured. `LiveSplitClient.split()` never raises and never blocks longer than `livesplit.timeout_seconds`; reconnect attempts are rate-limited to `livesplit.reconnect_seconds`. Detection must keep running when LiveSplit is down. Query commands (`getcurrenttimerphase`, `getcurrentsplitname` via `LiveSplitClient.query()`) return the first response line or None on any failure; they can each stall up to `timeout_seconds`, so `_run_live` polls both at most every ~2 s (wall-clock gated) — never per frame. Polling also drives the desync WARN: while LiveSplit reports Running, a segment name differing from `machine.current_split.name` prints once per distinct pair until they agree again.
- DONE is terminal unless `--loop`: the rearm watch measures splits[0]'s color (via `TransitionDetector.first_detector`, which unlike `current_detector` survives DONE) and calls `reset()` after `detection.restart_quiet_seconds` of continuous ratio-below-`reset_threshold` frames, frame-counted at `capture.fps`. The user resets their own LiveSplit timer between runs — the detector never touches timer state (sole exception: `start_transition.auto_start_timer: true` sends `start` when the gate fires).
- `MpegTsFrameSource` auto-reconnects with backoff for udp/tcp locations only; file locations play once then report exhausted (no replay loop).
- PLANNED REMOVAL (not yet approved to execute): the splitter's own recording (`--record`, the `record:` config section, the `record/` package, ffmpeg tee in `MpegTsFrameSource`, and the matching GUI controls/tests) is to be deleted — recording is OBS's job. Prerequisite: the decoupled OBS setup (normal OBS Recording slot for the archive; a separate low-res, low-bitrate output as the detector sidecar) is working, so nothing is lost meanwhile. Do it as a dedicated change; removing tests/behavior needs explicit approval per the baseline rules below.
- Handle failures gracefully: lost capture → reconnect instead of crash; LiveSplit down → keep detecting, report `DISCONNECTED`, reconnect when available.
- Invalid config must fail with precise errors (e.g. `configs/your-game.yaml: splits[2].threshold must be between 0.0 and 1.0`), not stack traces.

## Multi-Agent Development Protocol

Phases 1–7 are the established project baseline.

Do not rewrite or refactor completed phases unless the current task explicitly
requires it and the orchestrator has approved the architectural change.

Every non-trivial change should follow:

1. Explorer — understand the existing implementation.
2. Researcher — investigate external dependencies or unknown technical behavior
   when necessary.
3. Architect — determine the smallest safe architectural change.
4. Implementer — implement only the approved design.
5. Test Engineer — attempt to break the implementation.
6. Reviewer — independently review the implementation.
7. Integration Validator — validate the runtime pipeline when applicable.

The implementer must not approve its own implementation.

### Baseline protection

- Preserve existing behavior unless the task explicitly changes it.
- Do not remove or weaken existing tests.
- Do not perform unrelated refactors.
- Do not introduce dependencies without justification.
- Do not change public interfaces without architectural justification.
- Treat phase documentation as authoritative.

### Required verification

For production-code changes, run:

    uv run pytest -v
    uvx ruff check .
    uv run basedpyright src tests
    uv run python -m autosplitter.main --check

Then run the applicable phase/demo or integration command.

### Visual detection requirements

Changes to detection behavior should include positive, negative,
tolerance-boundary, debounce, hysteresis/reset, and duplicate-split cases.

### Capture requirements

Changes to capture must consider partial reads, EOF, reconnect behavior,
resolution mismatch, frame-rate stability, packet loss, and decoder startup.

### LiveSplit requirements

Changes to LiveSplit integration must consider unavailable-server,
reconnect, `split\n`, duplicate-split, and desynchronization behavior.

### Performance requirements

The frame-processing hot path is performance-sensitive.

- Keep ROI cropping before image processing.
- Keep per-frame processing allocation-free where practical.
- Do not introduce per-pixel Python loops.
- Do not trade sustained capture FPS for unnecessary abstraction.
- Performance-sensitive changes should include before/after measurements.

### Agent permissions

Read-only agents must not modify production code.
The reviewer is independent of the implementer and must not silently repair
the implementation while reviewing it.
