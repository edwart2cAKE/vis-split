# AGENTS.md

## Status

Planning repo, Phases 1-4, 5 (Diagnostics overlay), 6 (LiveSplit), and Phase 7 `--loop` rearm feature and opt-in unsplit rollback (`livesplit.resync_on_unsplit`) and `livesplit.follow_timer` increments 1-3 (rollback, skip-forward, timer reset, manual start, run end, and opt-in `livesplit.follow_pause` pause freezing) implemented (config + color detector + state machine + MPEG-TS frame capture + LiveSplit client + loop). Frame capture accepts `udp://`, `tcp://` and `srt://` sources (SRT added in commit 2819eef; see the SRT gotcha below). The single source of truth is `Project Overview.md` — read it before implementing anything. It defines the full design, phased build order (section 19), reference config (section 21), and dev rules (section 20).

Per-game configs live in `configs/` (e.g., `configs/start-powerless.yaml`, `configs/mr_rich.yaml`, `configs/template.yaml`; `configs/test-white.yaml` is a capture-pipeline test config).

## Commands

uv-managed project (src layout). Always use `uv run`; never system pip.

- `uv sync` — create/update `.venv` from lockfile (run first)
- `uv run pytest -v` — run tests
- `uvx ruff check .` — lint (no project install needed)
- `uv run basedpyright src tests` — typecheck in strict "recommended" mode (matches Zed); `[tool.basedpyright]` in pyproject.toml points at `.venv`, which also fixes import resolution in Zed
- `uv run python -m autosplitter.main --config configs/your-game.yaml --check` — validate config, print summary; exit 1 + precise error if invalid
- `uv run python -m autosplitter.main --config configs/your-game.yaml --detect IMAGE.png` — run every split's color detector over a static image; per-split match % / threshold / HIT
- `uv run python -m autosplitter.main --gui` — launch the optional PySide6 GUI (`uv sync --extra gui`)
- `uv run python -m autosplitter.main --config configs/your-game.yaml --probe IMAGE.png` — load image, crop to ROI, print average RGB + suggested YAML snippet for color tuning
- `uv run python -m autosplitter.main --live [SOURCE] [--width W] [--height H] [--max-frames N] [--snap N] [--livesplit] [--loop] [--overlay] [--record PATH] [--config-poll-interval S]` — live capture demo: rolling fps/match/state status; SOURCE overrides `capture.stream_url` and may be a `.ts` file for offline smoke tests or a `udp://`/`srt://` URL (then pass its dimensions via --width/--height, e.g. `"srt://127.0.0.1:9733?mode=listener&rcvlatency=20000" --width 640 --height 360`); `--snap N` saves frame N to snap.png and exits; `--livesplit` forwards split events to LiveSplit Server; `--loop` rearms the machine after DONE once the next run's first color stays quiet for `detection.restart_quiet_seconds` (user resets their own timer between runs); `--overlay` shows ROI rectangle + HUD window (throttled ~10 Hz, ESC/q to close); `--record PATH` writes a codec-copy .mkv at full source resolution to PATH while detecting (single ffmpeg process — no second UDP consumer needed); when recording, the detection output is scaled to `capture.width`x`capture.height`, so those may differ from the source; without `--record`, ffmpeg emits source-resolution frames and `capture.width/height` must match the source exactly; Ctrl-C or EOF stops. If `--record` is omitted but a `record:` config section exists, the path is resolved from the config's `path_template` (supports `{game}`, `{date}`, `{time}`, `{datetime}`, `{attempt}` variables). On exit, if `record.convert_to` is set, ffmpeg converts the .mkv to the target format (best-effort, .mkv never deleted).

Verification order per phase: tests → ruff → basedpyright → phase demo command.

basedpyright strictness: code must stay clean at 0 warnings (currently 199 pre-existing warnings in `gui/`, OCR, overlay and tests; clearing them is a separate task). PyYAML, argparse, and bare numpy `ndarray` are untyped — pin the boundary with `cast(object, ...)` / `cast(dict[str, object], ...)` / `cast(tuple[int, int], arr.shape[:2])` immediately after isinstance validation rather than disabling rules. Only `reportUnusedCallResult` is off (argparse boilerplate).

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

- Capture method (decided after research): OBS sidecar **Recording → Custom Output (FFmpeg)**, container `mpegts`, URL `udp://127.0.0.1:9733?pkt_size=1316` — runs alongside Streaming, pure userspace, no new pip deps (system `ffmpeg` binary decodes to raw BGR24 on a pipe). obs-websocket screenshot polling was rejected (~15-20 fps); virtual cam needs v4l2loopback kernel module. The autosplitter's `--live` is the sole UDP consumer; when `--record PATH` is passed, the same ffmpeg subprocess also writes a codec-copy .mkv at source resolution (no extra socket needed) and scales the detection frames to `capture.width/height` (`-vf scale`).
- For the Custom Output (FFmpeg)/UDP sidecar, pin its **Output (scaled) Resolution to 1920x1080** regardless of canvas (works for both the 1920x1080 and 2560x1440 setups; both are 16:9). `capture.width/height` in config must match it exactly when not using `--record` — a mismatch garbles frames silently and the pipe reader hits EOF (with `--record`, ffmpeg scales detection frames to `capture.width/height`, so a mismatch is tolerated). Receiver-side UDP URLs (config `stream_url`) should only have `pkt_size`; output-only params (`fifo_size`, `overrun_nonfatal`, `buffer_size`) cause ffmpeg to fail when opening the input.
- Sidecar encoder must be H.264 — tick **Show all codecs** in OBS's Custom FFmpeg output or only mpeg2video lists. Video Encoder Settings: nvenc → `bf=0`, libx264 → `tune=zerolatency bf=0`. Do NOT add nvenc `delay=0 zerolatency=1`: measured 10-40s latency regression (reactive CBR pacing). If splits fire late while `--live` fps stays steady ~60, frames are arriving stale from upstream (OBS), not queued in our pipeline (drain would spike fps).
- Latency measurement: stop Recording, send a synthetic clock (`ffmpeg -re -f lavfi -i testsrc=size=1920x1080:rate=60 -c:v h264_nvenc -bf 0 -g 60 -f mpegts udp://127.0.0.1:9733?pkt_size=1316`), run `--live --snap N`, compare snap.png's burned-in clock vs wall time.
- Packet loss is silent: kernel UDP socket buffer overflows before ffmpeg's fifo ring, so missing `Circular buffer overrun` warnings prove nothing.
- Hot loop must stay allocation-free per frame (measured: naive bytearray+copy reader + per-frame `build_detector` capped the pipeline at ~45 fps while a bare `ffmpeg -f rawvideo -pix_fmt bgr24 - >/dev/null` on the same URL sustains 59; fixed consumer paces at 60.1). Invariants: `RawPipeReader` decodes into one reused buffer — the returned ndarray is invalidated by the next read, copy it to retain; `TransitionDetector` builds its `ColorDetector`s once in `__init__`; `ColorDetector` precomputes `cv2.inRange` bounds at construction. Symptom of regression anywhere in this chain: `--live` fps pinned below ~60 with latency climbing steadily.
- Stream rate = OBS's **Common FPS Value** (Settings → Video), not monitor Hz; per-config (`start-powerless.yaml` 120, `mr_rich`/`test-white`/`template` 60) and must match that config's `capture.fps`. Debounce is frame-counted: keep `required_frames ≈ capture.fps / 20` (~50 ms of flicker immunity). `--live` prints a one-time HINT when a udp/tcp/srt stream's rolling fps deviates >15% from `capture.fps`.
- LiveSplit Server wire protocol: commands case-sensitive, newline-terminated (`split\n`); `split` has NO response, `ping` returns `pong`; server must be started manually per launch (Control → Start TCP Server) unless Startup Behavior is configured. `LiveSplitClient.split()` never raises and never blocks longer than `livesplit.timeout_seconds`; reconnect attempts are rate-limited to `livesplit.reconnect_seconds`. Detection must keep running when LiveSplit is down. Query commands (`getcurrenttimerphase`, `getcurrentsplitname` via `LiveSplitClient.query()`) return the first response line or None on any failure; they can each stall up to `timeout_seconds`, so `_run_live` polls both at most every ~2 s (wall-clock gated) — never per frame. Polling also drives the desync WARN: while LiveSplit reports Running, the desync check is a substring match — it warns only when the detector's expected split name (`machine.current_split.name`) is NOT contained in LiveSplit's segment name, once per distinct pair until they agree again; it is silent when the detector is DONE (no expected split).
- Unsplit rollback (opt-in `livesplit.resync_on_unsplit: true`, default false): the same ~2 s poll also calls `LiveSplitClient.get_split_index()` (`getsplitindex`; -1 valid; `-`, `[Error]` or anything but optional sign + ASCII digits -> None). `query()` drains unread socket bytes before sending so a late reply to a timed-out query cannot be read as the next answer. `TransitionDetector.expected_ls_index` is `_index + 1` in TRANSITION/WAITING_FOR_RESET for a split (not the start gate), else `_index`. `livesplit/sync.py` `UndoWatch` (pure) is ARMED by a Running poll with L == expected (not subject to hold-off), disarmed by any non-Running phase and on `--loop` rearm; only while armed does it confirm an undo: Running, 0 <= L < expected, >= `HOLDOFF_SECONDS` (3.0) since the last fired split, and 2 consecutive polls (`CONFIRM_POLLS`) agreeing on (L, expected). Arming prevents splits fired while the timer was NotRunning plus a late start at index 0 from looking like an undo. An undelivered `split()` suppresses detection (one-line notice) until a poll shows L == expected. Then `rollback_to(L)` (NORMAL/WAITING_FOR_RESET/DONE only; rejected in TRANSITION/WAITING_FOR_START): NORMAL keeps state; the others re-enter WAITING_FOR_RESET with `_transition_is_start=True` (flag now also means 'settle without advance') so a still-lit color must drop below `reset_threshold` first. The detector never writes LiveSplit state. Limits: rollback from DONE is effectively unreachable (without `--loop` the run exits once LiveSplit reports Ended; with `--loop` the rearm after `restart_quiet_seconds` beats the 3 s hold-off plus 2 polls), so final-split undo is NOT supported; a timer reset+restart that lands at index 0 mid-run while armed is indistinguishable from an undo and is treated as a rollback to 0; skipsplit and the GUI worker are not handled; OCR resync and unsplit rollback can disagree and flip-flop (startup WARN if both enabled). Verify `getsplitindex` replies against your LiveSplit first.
- `livesplit.follow_timer` (opt-in, default false; implies undo; `livesplit.poll_seconds`, default 2.0, sets the poll cadence for both modes): increments 1-3 (pause following is the separate opt-in `follow_pause`). `_run_live` uses `_FollowPoller` + `livesplit/follow.py` `TimerFollower` (pure; owns an `UndoWatch`). Poll order: phase (None aborts the poll; NotRunning skips the index/name queries but `observe` still runs so RESET can fire), `getsplitindex` (None aborts), `getsplitcount` once per connection (mismatch with `len(splits)` prints a WARN and disables skip-forward; undo still works), `getcurrentsplitname` only when L != expected or L changed (else cached). Undo works through Paused and from Ended (Ended at L=N=E arms, then Running at N-1 rolls back). SKIP_FORWARD: phase Running/Paused, expected < L <= N-1, state in WAITING_FOR_START/NORMAL/WAITING_FOR_RESET, 2 consecutive polls with identical (L, expected), no hold-off or arming. `TransitionDetector.advance_to(L)` settles like `rollback_to`: WAITING_FOR_RESET with `_transition_is_start=True`, so the target color must read strictly below `reset_threshold` before NORMAL (no re-fire; pending +1 advance cancelled); `index == N` -> DONE; TRANSITION/DONE refused. A manual LiveSplit split looks identical to a skip, and the detector follows it. `getsplitcount` may be unsupported by older servers: then a one-time-per-connection notice is printed, skip-forward stays enabled without validation, and a reconnect re-runs the check (and re-enables skip after a mismatch). An applied action sets `last_split_time`, so the 3 s undo hold-off applies: an Unsplit right after a skip/follow is ignored for ~3 s. A failed poll (phase or index None) clears pending skip/undo candidates so confirmation polls are truly consecutive. `poll_seconds` must be finite and > 0. Increment 2 lifecycle (all behind follow_timer): RESET = phase NotRunning after Running/Paused/Ended was seen (single poll; NotRunning without prior activity, or right after a reconnect, is ignored) -> `machine.reset(settle=True)` (no gate: WAITING_FOR_RESET at index 0 with settle-without-advance so a lit split-0 color cannot fire into an idle timer; with a gate: WAITING_FOR_START) and `last_split_time=None`; MANUAL_START = machine at the start gate and LiveSplit Running/Paused with L >= 0 (single poll) -> `advance_to(L)`, no `start` is sent; RUN_ENDED = Ended with L == N and expected < N, 2 identical polls -> `advance_to(N)` (DONE). A split-count mismatch disables RUN_ENDED and MANUAL_START as well as skip (RESET still works). A RESET while DONE only rewinds with `--loop` (without it the process exits at DONE; undo of the final split is out of scope there). After a RESET with `start_transition.auto_start_timer`, a still-lit start-gate color auto-starts the timer immediately (configured automation); a manual start at the gate does not double-start (the server ignores `start` unless NotRunning). RUN_ENDED (and thus the Ended follow) only fires after the follower has seen the timer Running or Paused since construction / the last `disarm()` / RESET / reconnect, so a stale Ended timer at startup or after a `--loop` rearm is ignored (undo-from-Ended arming does not need it); MANUAL_START also needs L < N. A reset or restart that happens while the splitter is disconnected from LiveSplit is invisible (the follower disarms on reconnect; only the desync WARN fires). Open question for the user: after a RESET with a start gate, a still-lit gate color fires the start immediately (and sends `start` with `auto_start_timer`). The detector never sends reset/pause/resume. `livesplit.follow_pause` (default false, requires `follow_timer`, else ConfigError): PAUSE on the first Paused poll and RESUME on Running (single polls, only while paused); while paused `_run_live` skips `machine.feed_result` (the ratio is still computed for the HUD, the status line shows `FROZEN`; one attribute read per frame), and PAUSE/RESUME call `machine.clear_streak()` so a pre-pause partial streak cannot complete after resume. They do NOT touch `last_split_time` or `undo_suppress` (no hold-off ping-pong). A TRANSITION/WAITING_FOR_RESET machine just stays put while frozen; undo stays armed across Paused. `disarm()` (reconnect, RESET, `--loop` rearm) clears paused; a None/unknown phase leaves it unchanged. Default off because 'LiveSplit ignores `split` while Paused' is inferred from source and not live-tested: test once (pause, trigger a detector split, observe). PAUSE/RESUME are phase-only, so they apply even when the index query fails on that poll; Paused->Ended clears the freeze; a safety valve (`_THAW_AFTER_FAILURES = 3` consecutive failed polls while paused, connection still up) thaws the detector with the log `LiveSplit: no answers while paused — detector unfrozen` (a later successful Paused poll re-freezes). OCR resync (`check_ocr_resync`) still runs while frozen. RESUME-side lag: a split color flash after the user resumes but before the next poll (up to `poll_seconds`) is skipped by the freeze and NOT repaired by undo/skip (LiveSplit ends up behind; a manual split is followed as a skip), so keep `poll_seconds` small with `follow_pause`. Limitation: up to ~`poll_seconds` of detection can run before the pause is observed; a split sent in that window is ignored by a paused LiveSplit and the detector sits one ahead, which the backward (undo) rule repairs after hold-off + 2 polls. When the machine becomes DONE the next LiveSplit poll is forced immediately so the non-loop exit does not wait out `poll_seconds`. NEVER send `pause`/`resume`: the server's pause starts a NotRunning timer. 
- DONE is terminal unless `--loop`: the rearm watch measures splits[0]'s color (via `TransitionDetector.first_detector`, which unlike `current_detector` survives DONE) and calls `reset()` after `detection.restart_quiet_seconds` of continuous ratio-below-`reset_threshold` frames, frame-counted at `capture.fps`. The user resets their own LiveSplit timer between runs — the detector never touches timer state (sole exception: `start_transition.auto_start_timer: true` sends `start` when the gate fires).
- `MpegTsFrameSource` auto-reconnects with backoff for udp/tcp/srt locations only; file locations play once then report exhausted (no replay loop). A live source that is silent for 2 s is torn down and respawned (stall watchdog) — except an `srt://...mode=listener` source before its first frame, where ffmpeg legitimately blocks until the sender connects (the watchdog re-arms once a frame has arrived).
- SRT sidecar (decoupled setup; verified end-to-end with `configs/test-white.yaml`): OBS keeps its normal Recording slot for the full-quality archive, and the **Branch Output** filter (AUR `obs-branch-output`, OBS ≥ 30.1; the AUR build fails under current GCC with `-Werror=sfinae-incomplete` unless the PKGBUILD's cmake call gets `-DCMAKE_COMPILE_WARNING_AS_ERROR=OFF`) is added to the game *scene* and streams a low-res H.264 feed to `srt://127.0.0.1:9733` from its **Streaming** section (server field; Stream Count ≥ 1 — its Recording section only writes files). UDP/`mpegts` is not available in that plugin, hence SRT. The splitter is the SRT listener: `--live "srt://127.0.0.1:9733?mode=listener&rcvlatency=20000" --width 640 --height 360` (SOURCE overrides `capture.stream_url`; width/height must equal the filter's output resolution or frames garble and fps reads ~16x too high). ffmpeg's SRT options are in **microseconds**; `rcvlatency` defaults to libsrt's ~120 ms and SRT latency is the larger of both ends' values.
  - Measured latency (clock window in OBS → `--snap 300` write time): ~594 ms at defaults → ~481 ms with `rcvlatency=20000` → ~252 ms after raising the NVENC bitrate from 100 to 1500 kbps. 100 kbps CBR at 640x360@60 costs ~230 ms of pacing delay; low-res, not low-bitrate, is what saves CPU. Keyframe interval 1 s; do not add nvenc `delay=0 zerolatency=1`.
  - Start order: start the splitter (listener) first, then enable the filter. Enabling the filter first leaves OBS retrying a dead port and the splitter waited >10 s to connect (suspected OBS retry delay; not confirmed in the OBS log). `mode=caller` on the splitter with OBS listening (`mode=listener` in the Branch Output server field) is the untested alternative that would allow leaving the filter on.
  - `capture.fps` must equal the stream's actual rate (check with `ffprobe` against the listener URL); `required_frames ≈ capture.fps / 20` still applies.
  - Known cosmetic issue: the desync WARN can fire once between a split firing and the reset clearing, because the index only advances on clear while LiveSplit has already advanced. LiveSplit ignores `start`/`split` while its timer is `Ended`, so reset the timer between `--loop` runs.
- PLANNED REMOVAL (not yet approved to execute): the splitter's own recording (`--record`, the `record:` config section, the `record/` package, ffmpeg tee in `MpegTsFrameSource`, and the matching GUI controls/tests) is to be deleted — recording is OBS's job. Also covers `scripts/record.sh` (wraps `--record`). Prerequisite: the decoupled OBS setup (normal OBS Recording slot for the archive; a separate low-res output as the detector sidecar) — the SRT Branch Output feed is verified working (see the SRT bullet); still to confirm that OBS's own archive keeps recording at full quality while the splitter runs and restarts. Do it as a dedicated change; removing tests/behavior needs explicit approval per the baseline rules below.
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
