# AGENTS.md

## Status

Planning repo, Phases 1-4 implemented (config + color detector + state machine + MPEG-TS frame capture). The single source of truth is `Project Overview — Roblox Visual.md` — read it before implementing anything. It defines the full design, phased build order (section 19), reference config (section 21), and dev rules (section 20).

## Commands

uv-managed project (src layout). Always use `uv run`; never system pip.

- `uv sync` — create/update `.venv` from lockfile (run first)
- `uv run pytest -v` — run tests
- `uvx ruff check .` — lint (no project install needed)
- `uv run basedpyright src tests` — typecheck in strict "recommended" mode (matches Zed); `[tool.basedpyright]` in pyproject.toml points at `.venv`, which also fixes import resolution in Zed
- `uv run python -m autosplitter.main --config config.yaml --check` — validate config, print summary; exit 1 + precise error if invalid
- `uv run python -m autosplitter.main --config config.yaml --detect IMAGE.png` — run every split's color detector over a static image; per-split match % / threshold / HIT
- `uv run python -m autosplitter.main --live [SOURCE] [--width W] [--height H] [--max-frames N]` — live capture demo: rolling fps/match/state status; SOURCE may be a `.ts` file for offline smoke tests (then pass its dimensions via --width/--height); Ctrl-C or EOF stops

Verification order per phase: tests → ruff → basedpyright → phase demo command.

basedpyright strictness: code must stay clean at 0 warnings. PyYAML, argparse, and bare numpy `ndarray` are untyped — pin the boundary with `cast(object, ...)` / `cast(dict[str, object], ...)` / `cast(tuple[int, int], arr.shape[:2])` immediately after isinstance validation rather than disabling rules. Only `reportUnusedCallResult` is off (argparse boilerplate).

Pipeline convention: frames are BGR uint8 `(H, W, 3)` everywhere (OpenCV convention); config colors are RGB and converted to BGR once in `ColorDetector.__init__`. Crop the ROI before any processing — never full-frame work.


## Platform

- Target: Linux (Arch/Wayland) + OBS Studio + Sober (Roblox). Never introduce Windows-only APIs.
- LiveSplit runs under Wine; the detector runs natively on Linux, independent of LiveSplit.
- LiveSplit Server over TCP: `127.0.0.1:16834`.

## Non-negotiables

- All tunables live in `config.yaml`. Never hard-code resolution, colors, or thresholds.
- ROI coordinates are normalized floats (0.0–1.0) so resolution changes don't break detection.
- Color matching uses configurable tolerance, never exact RGB equality.
- Process only the cropped ROI with vectorized NumPy/OpenCV — no per-pixel Python loops (target ~60 FPS).
- Detection is a state machine (`NORMAL → TRANSITION → WAITING_FOR_RESET`, terminal `DONE`): require N consecutive matching frames before splitting (debounce), use `reset_threshold` hysteresis after a split, and never emit more than one split per continuous transition. Contract: streak counts frames at/above the active split's threshold; clearing is strictly below `reset_threshold`; a frame that clears one split never counts toward the next one's streak.

## Boundaries

- Keep frame acquisition (`capture/`) separate from detection logic (`detection/`); input source must be swappable.
- Keep all LiveSplit protocol logic inside `livesplit/client.py` (`LiveSplitClient`); the detector only calls methods like `split()`.
- Detector and state machine must be unit-testable from prerecorded frames with no OBS or LiveSplit present (pytest once tests exist).
- v1 ships the color detector only. Do not implement template matching until color detection works.

## Dependencies

Minimal set: OpenCV, NumPy, PyYAML, plus whatever the chosen OBS capture method requires. Do not add frameworks without a demonstrated problem.

## Gotchas

- Capture method (decided after research): OBS sidecar **Recording → Custom Output (FFmpeg)**, container `mpegts`, URL `udp://127.0.0.1:9733?pkt_size=1316` — runs alongside Streaming, pure userspace, no new pip deps (system `ffmpeg` binary decodes to raw BGR24 on a pipe). obs-websocket screenshot polling was rejected (~15-20 fps); virtual cam needs v4l2loopback kernel module.
- Pin the sidecar's **Output (scaled) Resolution to 1920x1080** regardless of canvas (works for both the 1920x1080 and 2560x1440 setups; both are 16:9). `capture.width/height` in config must match it exactly — a mismatch garbles frames silently and the pipe reader hits EOF.
- `MpegTsFrameSource` auto-reconnects with backoff for udp/tcp locations only; file locations play once then report exhausted (no replay loop).
- Handle failures gracefully: lost capture → reconnect instead of crash; LiveSplit down → keep detecting, report `DISCONNECTED`, reconnect when available.
- Invalid config must fail with precise errors (e.g. `config.yaml: splits[2].threshold must be between 0.0 and 1.0`), not stack traces.
