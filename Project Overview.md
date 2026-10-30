# Project Overview — Roblox Visual Autosplitter

> **Implementation status (see `README.md` and `AGENTS.md` for current detail).**
> This document is the original design. Phases 1–7 are implemented and the pipeline
> has been verified end to end with OBS. Where the code differs from the text below,
> the code and `AGENTS.md` win. Main differences, each noted in its section:
>
> - The tool is game-agnostic; Roblox/Sober was the first target (§1).
> - Frames come from an OBS sidecar stream decoded by an `ffmpeg` subprocess over
>   `udp://`, `tcp://` or `srt://` (§3, §22).
> - Config lives in per-game files under `configs/`, not a single `config.yaml` (§12, §21).
> - The state machine also has `WAITING_FOR_START` and a terminal `DONE` state (§7).
> - `LiveSplitClient` only sends `start`/`split` and queries state; resets stay with the user (§11).
> - Beyond the original spec: `--loop` rearm, an optional OCR split-index resync,
>   an optional PySide6 GUI, and splitter-side recording (planned for removal) (§12, §18, §19).

## 1. Project Goal

Create a Linux-compatible autosplitter that monitors a game window captured by OBS and automatically sends split commands to LiveSplit when a configured visual transition occurs. The first target was a Roblox/Sober game; the implementation is game-agnostic, with one config per game.

Primary use case:

```text
Sober / Roblox
      ↓
OBS Window Capture
      ↓
Captured video frames
      ↓
Autosplitter
      ↓
Middle 50% ROI
      ↓
Color percentage analysis
      ↓
Transition detected
      ↓
LiveSplit Server
      ↓
LiveSplit Split
```

The autosplitter should run independently of LiveSplit so that the detector can run natively on Linux while LiveSplit may run under Wine.

## 2. Target Platform

Primary:

- Linux
- Arch Linux
- Wayland
- OBS Studio
- Sober Roblox
- LiveSplit running through Wine
- Python 3

The program should not require Windows-specific APIs.

## 3. Core Requirements

### Frame acquisition

The program must receive frames from the OBS Roblox capture.

The frame acquisition layer should be isolated from the detection logic so that the input source can be changed later.

Architecture:

```text
FrameSource
    │
    └── get_frame()
             ↓
         numpy array
```

Initially, support the simplest reliable OBS-compatible input method available on Linux.

**As built:** the interface is the `FrameSource` protocol in `capture/frame_source.py`
(`read() -> ndarray | None`, `close()`; frames are BGR uint8 `(H, W, 3)`). The
implementation, `MpegTsFrameSource`, runs one `ffmpeg` child that decodes an MPEG-TS
stream from OBS to raw BGR on a pipe. Accepted sources are `udp://`, `tcp://`, `srt://`
and local files, with automatic reconnect for the network ones.

Do not tightly couple the detector to OBS internals.

## 4. Region of Interest

The detector should analyze the middle 50% of the frame by default.

For a frame:

```text
width  = W
height = H
```

Use:

```text
x1 = W * 0.25
x2 = W * 0.75

y1 = H * 0.25
y2 = H * 0.75
```

Therefore:

- ROI width  = 50% of frame width
- ROI height = 50% of frame height
- ROI area   = 25% of total frame area

However, ROI configuration should be supported so the user can eventually specify:

```yaml
roi:
  x: 0.25
  y: 0.25
  width: 0.50
  height: 0.50
```

Coordinates should be normalized from 0.0 to 1.0.

**As built:** the 50% middle region is only the example; the ROI is whatever the game
config specifies, either as `x/y/width/height` or as corners `x1/y1/x2/y2`.

## 5. Color Detection

The detector should determine what percentage of pixels in the ROI match a target color.

Example black detection:

```text
pixel = (R,G,B)

black if:

R <= 15
G <= 15
B <= 15
```

Then:

```text
matching_pixels / total_roi_pixels
```

produces the match percentage.

Example:

```text
ROI pixels:       500,000
Matching pixels:  485,000

match percentage = 97%
```

If the configured threshold is:

```yaml
threshold: 0.95
```

then:

```text
97% >= 95%
```

and the ROI is considered to have entered the target state.

## 6. Color Configuration

Do not hard-code black.

The configuration should support arbitrary RGB colors.

Example:

```yamlgit rm -r --cached -q src/autosplitter/livesplit/__pycache__ src/autosplitter/capture/__pycache__
colors:
  black:
    rgb: [0, 0, 0]
    tolerance: 15

  white:
    rgb: [255, 255, 255]
    tolerance: 15

  blue:
    rgb: [0, 80, 255]
    tolerance: 20
```

Color matching should use a configurable distance/tolerance rather than exact RGB equality.

Prefer a color space that makes tolerance meaningful, such as HSV or another appropriate representation, if practical.

However, the implementation should remain simple enough to run at high FPS.

**As built:** no HSV. A pixel matches when every BGR channel is within `tolerance` of
the target (an inclusive cube), computed with a precomputed `cv2.inRange`. Config colors
are RGB and converted to BGR once when the detector is built.

## 7. Transition Detection

The most important part of the project is that the autosplitter must detect a transition into the target state, not continuously split while the screen remains that color.

Use a state machine.

States:

- `NORMAL`
- `TRANSITION`
- `WAITING_FOR_RESET`

**As built**, two more states exist: `WAITING_FOR_START` (when a `start_transition` gate
is configured; the first color starts the run and, optionally, the LiveSplit timer) and a
terminal `DONE`. The last split goes straight to `DONE` on the next frame instead of
waiting for reset, so a victory screen that keeps the color lit cannot stall the run.
`--loop` calls `reset()` after the next run's first color stays quiet for
`detection.restart_quiet_seconds`.

Conceptually:

```text
NORMAL
  │
  │ target detected
  ▼
TRANSITION
  │
  │ trigger split
  ▼
WAITING_FOR_RESET
  │
  │ target disappears
  ▼
NORMAL
```

Example:

| Frame  | State             | Action                |
| ------ | ----------------- | --------------------- |
| Normal | NORMAL            | Nothing               |
| Normal | NORMAL            | Nothing               |
| Black  | TRANSITION        | Split                 |
| Black  | WAITING_FOR_RESET | Nothing               |
| Black  | WAITING_FOR_RESET | Nothing               |
| Normal | NORMAL            | Reset detector state  |
| Normal | NORMAL            | Nothing               |
| Black  | TRANSITION        | Split                 |

This prevents one long transition from producing multiple splits.

## 8. Debouncing

Require the target condition to persist for a configurable number of frames before triggering.

Example:

```yaml
detection:
  required_frames: 3
```

Then:

```text
Frame 1: 96% black
Frame 2: 97% black
Frame 3: 96% black
                 ↓
              SPLIT
```

But:

```text
Frame 1: 96%
Frame 2: 70%
Frame 3: 95%
```

does not trigger.

This prevents transient frames and capture artifacts from causing false splits.

## 9. Reset Detection

After triggering a split, the detector must wait until the ROI is no longer in the target state.

Example:

```yaml
detection:
  reset_threshold: 0.80
```

If the target was black:

```text
black >= 95% → target detected
black < 80%  → target cleared
```

This creates hysteresis and prevents oscillation around the detection threshold.

## 10. Split Configuration

Each split should be configurable independently.

Example:

```yaml
splits:
  - id: 1
    name: "First Transition"
    color: black
    threshold: 0.95

  - id: 2
    name: "Second Transition"
    color: white
    threshold: 0.95

  - id: 3
    name: "Third Transition"
    color: black
    threshold: 0.90
```

The detector should maintain the current split index.

Conceptually:

```text
Current split = 1

detect split 1
    ↓
send LiveSplit split
    ↓
Current split = 2
    ↓
detect split 2
    ↓
...
```

The split sequence should be controlled by the LiveSplit timer state where possible rather than blindly incrementing locally.

**As built:** the detector advances its own index (only after the target clears). With
`--livesplit` it polls LiveSplit's timer phase and segment name about every 2 s and
prints a one-time desync warning if the expected split name isn't contained in
LiveSplit's current segment name; it never changes LiveSplit's state to fix it. With the
optional OCR section, the on-screen split number is read as read-only "screen evidence"
(`ocr.mode: verify_only`, the default): it is compared with the detector index and reported, never
used to change state or to send anything to LiveSplit (the opt-in `ocr.mode: undo` below is the exception). Reads are rate-capped by
`ocr.max_reads_per_second` (default 4/s; a verdict takes ~0.75 s and costs about half of a niced core (measured +~50%)).

`preprocess.threshold: color` (`text_color`, `text_tolerance`) masks just the HUD text colour so
black bars, gray backgrounds and Otsu polarity no longer matter (example ROI for a black-barred
layout: `x=0.105 y=0.0 width=0.03 height=0.055`, white text; a loose ROI or one covering the bar
reads nothing under `otsu`). The optional `ocr.undo:` section adds a DRY-RUN "go then undo"
judge (`detection/undo_judge.py`): the colour detector still splits on the transition (a death looks
identical to a level change); afterwards the judge checks whether the HUD number advanced (S == D,
logged as `HUD advanced +X.XXs after split`) or stayed one behind past `settle_seconds` for
`confirm_reads` reads over `confirm_seconds` (logged as `would UNDO ... dry run: nothing sent`).
Because the HUD lags the colour event by a game-dependent time, set `settle_seconds` ~1.5x the
largest observed lag. With `mode: verify_only` nothing is sent and detector state never changes.
`ocr.mode: undo` (opt-in per game, `--live --livesplit`, CLI only) makes `livesplit/undo_exec.py` send
`unsplit` after an UNDO verdict: it checks the timer is Running and LiveSplit's index equals the
detector's, sends `unsplit`, verifies the index dropped by one and only then rolls the detector back
(`TransitionDetector.apply_undo`, which always settles in WAITING_FOR_RESET so a still-lit colour cannot
re-fire). Unavailable server, wrong phase/index or an ignored unsplit => skipped and logged, detector untouched
(never a lone rollback). `undo.recover_late_hud` (default true, `recover_seconds` 10): if the HUD then shows the next
level in time with no new colour fire, `split` is sent again and the detector advances; that split is recorded late
by the HUD lag. This is the second explicit exception (besides `start_transition.auto_start_timer`) to "the detector
never touches timer state". Tuning: dry run first, `settle_seconds` ~1.5x the largest `HUD advanced +X.XXs` lag; a
genuine HUD lag beyond settle+post_clear+confirm is wrongly undone (then recovered). Ambiguous outcomes (EOF after the write, no answer to the
verify query, a late-applied command) are reconciled by re-reading the split index (2 immediate re-checks, then one
query per 0.5 s for up to 5 s) before the detector moves; `ocr.mode: undo` requires `livesplit.follow_timer: true` (config error otherwise; the dry run does not), whose rollback/skip rules repair what is left. The
recovery needs its HUD reads to span `confirm_seconds`. LiveSplit's real ordering of an `unsplit` against the next
index reply was checked once against a real server (probe A7, 2026-10-30: immediate). Not handled: multi-step undo, GUI.
Last split (`ocr.undo.apply_to_last_split: true`, default false; opt-in): enable it when your game's HUD level number DISAPPEARS after the finish and keeps showing the level after a death on the last level. The judge then waits `settle_seconds` + `last_split_extra_settle_seconds` (default 2.0) + `post_clear_seconds` + the confirm reads; if the number is gone for the confirm reads the run is FINISHED (`OCR: HUD gone after the last split — run finished, no undo`), otherwise the split is undone (LiveSplit `Ended` -> `Running` at the last segment; `mode: undo` only, `follow_timer` required). A false undo of a genuine finish has NO automatic recovery (no late-HUD recovery for the last split), and a HUD that lingers after a real finish longer than that guard (~7 s with defaults) is wrongly undone. The process holds its exit at DONE while the window is open (up to `window_seconds`; a genuine finish exits about the guard time after the finish) and `--loop` does not rearm meanwhile. Validated offline; real LiveSplit Ended -> Running via `unsplit` was confirmed once with the probe (A6, 2026-10-30), but not against real game footage. Run the dry run (`verify_only`) first and watch the `would UNDO` / `HUD gone` lines. After ANY last-split `unsplit` attempt (even one reported as having no effect or unconfirmed) the non-loop DONE exit also waits for a LiveSplit phase polled at least `poll_seconds + timeout_seconds` AFTER the send, because LiveSplit may apply the unsplit later than the ~0.3 s re-checks and the cached phase may still read Ended; if that fresh phase is Running the process keeps running and follow_timer's rollback repairs the detector. The wait is bounded (6 x poll_seconds + 4 x timeout_seconds + 3 s) and skipped when LiveSplit is unreachable, so a hung server still lets the process exit; apply_to_last_split off, dry runs and FINISHED (nothing sent) exit as before.
Honest notes: (a) the judge measures settle/post_clear/confirm with the clock at PROCESSING time, not the reading's capture time; if the main loop stalls for seconds after DONE (LiveSplit queries timing out, a backlogged pipe) a HUD that really lingers ~1.5 s of video time can age past the ~6 s guard and be wrongly undone — low in practice (confirm_seconds needs wall-clock time between reads and the worker keeps only the latest frame), but it is a second path from a genuine finish to a false undo besides the documented lingering-HUD limit; (b) ANCHORED `ocr.patterns` (`^...$`, e.g. `"^\\s*(\\d+)\\s*$"`) are required in practice for `apply_to_last_split`: end-card/menu text containing a number would otherwise count as a trusted HUD read (startup and `--check` print a non-fatal `WARN: ocr.patterns "<regex>" is not anchored` for `mode: undo` or `apply_to_last_split`); (c) with `--loop` the quiet-frames rearm is held while a last-split window is open (up to `window_seconds`): if the HUD never disappears and a new run starts within that time, its first split is ignored while the machine is still in DONE.
Validator-measured limits: (a) after any UNRESOLVED offset between detector and game (two deaths closer than the ~4.2 s verdict latency, recovery disabled or a HUD later than `recover_seconds`, a late-applied unsplit follow_timer has not yet repaired, a hung-server UNCONFIRMED) `require_baseline` makes undo inert for the rest of the run (`no baseline` on every later split) until fixed manually or by reset; (b) with `recover_late_hud: false` (or a HUD later than `recover_seconds`) LiveSplit and the detector stay one split BEHIND the game and you must fix it manually; the log only says `OCR: screen says split N ..., detector at M (no action)`; (c) at 640x360/30 fps the undo comes ~4.1-4.2 s after the fire at defaults (settle 3.0 / post_clear 1.0 / 3 reads over 1.0 s) and ~1.7 s with the fast set (1.0 / 0.5 / 0.6 s), whose false-undo ceiling is a HUD lag of ~1.7 s; the recovery split follows ~0.9 s after the HUD first reads the new level and is recorded late by roughly the HUD lag (the unsplit+split pair does NOT restore the original timestamp); (d) the frame loop blocks ~1.0 s at a verdict against a connected-but-silent server (100-140 frames of backlog at 120 fps, no loss at 30 fps); (e) a death on the LAST level is not undone unless `apply_to_last_split: true`; (f) validated only offline/synthetic: no real OBS or real game footage (LiveSplit command semantics were checked once against a real LiveSplit with `scripts/livesplit_probe.py`), no 1080p, and `--loop` with undo, the GUI and `--overlay` were not exercised; (g) the stale-reply hazard was only exercised in its benign shape. Run the dry run first; `mode: undo` requires `livesplit.follow_timer: true`.
OCR quality (offline, one frame/font, synthetic motion; README has the numbers): needs >= 640x360 and a sharp downscale; NVENC >= ~200 kbps (100k workable), x264 >= ~400 kbps; for white-on-gray text tune `threshold: color`, `text_tolerance` ~ half the contrast (70), `blur: 0`, `scale: 4`. Degraded frames can yield confident wrong digits that consensus does not catch, so use `undo.min_confidence` >= 85 with `mode: undo`.
Notes: `settle_seconds` is measured from the FIRE, `post_clear_seconds` (default 1.0) from the advancing CLEAR; a D-1 read counts only when both have elapsed (earlier reads are ignored, they do not break the streak). `window_seconds` (from the fire) should exceed the longest transition lit time + settle + confirm. `confirm_reads: 1` with `confirm_seconds: 0` is rejected. Reads are capped by `ocr.max_reads_per_second` (default 4/s), so the default confirm_reads 3 + confirm_seconds 1.0 needs ~5 reads / ~1.0-1.25 s in practice. The baseline AGREE must come from the same epoch (a reset/rollback/advance/clear bumps it). Non-finite (NaN/inf) confidences are untrusted, and huge ints give a ConfigError ('must be a number'). The OCR ROI must cover ALL digits of the highest level number with ~2 px margin (at 640x360 the strict example ROI clips the '1' of '15'; measured read '5'; with 2 px margin it reads '15'). Unknown keys in `ocr.undo` are silently ignored (a typo keeps the default). `text_color`/`text_tolerance` are ignored unless `threshold: color`. The cancel log line reads `undo window cancelled: state changed`; `--check` shows the effective confidence floor max(undo, engine, filter min_confidence).

With the opt-in `livesplit.follow_timer` (implies Unsplit rollback via `getsplitindex`) the
detector also follows LiveSplit: skip/manual split (jump forward), timer reset (rewind),
manual start at the start gate, and run end. The detector never sends reset/pause/resume;
the separate opt-in `livesplit.follow_pause` (requires `follow_timer`; default off because
"LiveSplit ignores `split` while Paused" was confirmed once live, probe A11; re-check on your setup) freezes detection while the
timer is Paused and restarts the debounce on resume, with up to one poll interval of lag
before a pause is seen (the undo rule repairs a split LiveSplit ignored). The same lag applies after resume, where a missed flash is not repaired (keep `poll_seconds` small); 3 consecutive failed polls while paused unfreeze the detector. A split-count mismatch (`getsplitcount`) disables
index-following. Without `--loop` the process exits at DONE, so undo of the final split is
out of scope there.

## 11. LiveSplit Communication

Use LiveSplit Server as the initial communication mechanism.

The detector should have a dedicated abstraction:

```text
LiveSplitClient
```

with methods such as:

- `start()`
- `split()`
- `reset()`
- `pause()`
- `resume()`
- `get_state()`

**As built:** `start()`, `split()`, `ping()`, `query()` with the helpers
`get_current_timer_phase()` and `get_current_split_name()`, and connection handling
(`connected`, `poll_connection()`, `close()`). `reset`, `pause` and `resume` are
deliberately not implemented: the user resets their own timer between runs. `split()`
never raises and never blocks longer than `livesplit.timeout_seconds`.

The detector should not directly implement LiveSplit protocol logic throughout the application.

Instead:

```text
Detector
   │
   ▼
LiveSplitClient
   │
   ▼
TCP
   │
   ▼
LiveSplit Server
```

This makes the LiveSplit communication replaceable later.

## 12. Project Structure

Recommended structure (as built):

```text
vis-split/
│
├── README.md
├── AGENTS.md
├── pyproject.toml
├── config.yaml              # default for --config; real configs live in configs/
├── configs/                 # per-game configs, template.yaml, test-white.yaml
├── docs/PHASE_BASELINE.md
├── scripts/record.sh        # wraps --record (planned for removal)
│
├── src/
│   └── autosplitter/
│       ├── main.py          # CLI: --check --detect --probe --live --gui
│       ├── config.py
│       ├── capture/frame_source.py        # FrameSource, RawPipeReader, MpegTsFrameSource
│       ├── detection/
│       │   ├── color_detector.py
│       │   ├── state_machine.py           # TransitionDetector
│       │   ├── ocr_detector.py            # optional read-only screen evidence (OCR reader + worker)
│       │   └── undo_judge.py              # go-then-undo judge (pure)
│       ├── livesplit/client.py
│       ├── livesplit/undo_exec.py         # ocr.mode: undo executor (unsplit + late-HUD recovery)
│       ├── diagnostics/overlay.py
│       ├── record/                        # recording path templates (planned for removal)
│       └── gui/                           # optional PySide6 GUI
│
└── tests/                   # config, color, state machine, capture, livesplit, ocr, overlay, record
    └── fixtures/
```

## 13. Dependencies

Initial dependencies should be kept relatively small.

Likely:

- Python 3
- OpenCV
- NumPy
- PyYAML

Potentially add a capture-specific package once the OBS frame acquisition method is selected.

**As built:** no capture package was needed; the system `ffmpeg` binary does the decoding.
Optional extras in `pyproject.toml`: `gui` (PySide6). An `ocr` extra (tesserocr) is declared
but currently UNUSED (reserved for a possible future tesserocr backend); the OCR screen
evidence only needs the system `tesseract` binary.

Do not add large frameworks unless they solve a demonstrated problem.

## 14. Diagnostics Mode

This is important for developing the autosplitter.

The program should have a diagnostic mode that displays:

```text
FPS:              59.8
ROI match:        97.4%
Target:           BLACK
Threshold:        95.0%
State:            TRANSITION
Current split:    3
Last split:       2.134s ago
LiveSplit:        CONNECTED
```

Ideally display the ROI as an overlay:

```text
┌───────────────────────────────┐
│                               │
│      ┌─────────────────┐      │
│      │                 │      │
│      │      ROI        │      │
│      │                 │      │
│      └─────────────────┘      │
│                               │
└───────────────────────────────┘
```

This will make tuning the detector much easier.

## 15. Logging

Provide useful logs without flooding the terminal.

Example:

```text
[INFO] Capture initialized: 1920x1080 @ 60 FPS
[INFO] LiveSplit connected
[INFO] Current split: 1
[DEBUG] ROI match: 42.1%
[DEBUG] ROI match: 96.4%
[DEBUG] Target detected for 3 frames
[INFO] Split triggered: First Transition
[DEBUG] Waiting for target to clear
[INFO] Target cleared
[INFO] Ready for next transition
```

Debug logging should be configurable.

## 16. Performance Requirements

The detector should operate comfortably at approximately 60 FPS.

Avoid unnecessary operations on the entire frame.

Preferred pipeline:

```text
Frame
 ↓
Crop ROI
 ↓
Convert color space if necessary
 ↓
Vectorized NumPy/OpenCV comparison
 ↓
Calculate percentage
 ↓
State machine
```

Do not iterate over individual pixels in Python.

Use NumPy/OpenCV vectorized operations.

## 17. Error Handling

The application should gracefully handle:

### OBS capture lost / Capture unavailable

Attempt to reconnect rather than crashing.

### LiveSplit unavailable

Continue detecting but report:

```text
LiveSplit: DISCONNECTED
```

When LiveSplit becomes available, reconnect.

### Invalid configuration

Fail with a useful error:

```text
configs/your-game.yaml: splits[2].threshold must be between 0.0 and 1.0
```

### Resolution changes

The normalized ROI should automatically adapt to different resolutions.

## 18. Future Template Matching

Do not implement this initially unless needed.

The architecture should allow another detector later:

- `ColorDetector`
- `TemplateDetector`
- `OCRDetector`

with a common interface:

```python
Detector.detect(frame) -> DetectionResult
```

This would allow a split to eventually be:

```yaml
type: color
```

or:

```yaml
type: template
```

For example:

```yaml
splits:
  - name: "Transition"
    detector:
      type: color
      color: black
      threshold: 0.95

  - name: "Results Screen"
    detector:
      type: template
      image: images/results.png
      threshold: 0.90
```

But color detection should be the only detector implemented in v1.

**As built:** the color detector is the only one that fires splits. An `OcrDetector`
(optional, `ocr:` config section) exists only to read an on-screen split number as read-only
screen evidence (filtered by consensus, staleness and state-machine epoch; see AGENTS.md);
template matching is still not implemented.

## 19. Development Phases

OpenCode should implement this incrementally.

### Phase 1 — Configuration

Implement:

- YAML configuration
- validation
- split definitions
- ROI definitions
- color definitions

No OBS or LiveSplit yet.

### Phase 2 — Image Detector

Implement:

- NumPy/OpenCV color detection
- ROI extraction
- percentage calculation
- tolerance
- unit tests using static images

### Phase 3 — State Machine

Implement:

- normal state
- target detection
- consecutive-frame debounce
- reset threshold
- split event generation

Test entirely using prerecorded frames.

### Phase 4 — Frame Capture

Implement the OBS-compatible Linux capture source.

Verify:

```text
OBS → application
```

before integrating anything else.

### Phase 5 — Diagnostics

Add:

- FPS
- ROI visualization
- match percentage
- current state
- current split
- connection status

### Phase 6 — LiveSplit

Implement:

```text
LiveSplitClient
```

and connect split events to LiveSplit Server.

### Phase 7 — End-to-End Testing

*(Delivered as the `--loop` rearm feature plus live runs against OBS and LiveSplit; see
`docs/PHASE_BASELINE.md`. A decoupled SRT sidecar feed was verified end to end afterwards.)*

Test:

```text
Sober
 ↓
OBS
 ↓
Detector
 ↓
LiveSplit
```

with real gameplay.

### Phase 8 — Reliability *(ongoing)*

Tune:

- threshold
- color tolerance
- debounce frames
- reset threshold
- capture FPS
- reconnect behavior

## 20. OpenCode Development Rules

Put these directly into the project's AGENTS.md so OpenCode has persistent instructions.

Project rules:

1. This project targets Linux/Arch Linux/Wayland.
2. Do not introduce Windows-only APIs.
3. Keep frame acquisition separate from detection.
4. Keep LiveSplit communication separate from detection.
5. Use NumPy/OpenCV vectorized operations for image processing.
6. Do not iterate through pixels individually in Python.
7. All configurable values belong in the game config (`configs/<game>.yaml`).
8. Never hard-code screen resolution.
9. ROI coordinates must be normalized.
10. Color matching must support configurable tolerance.
11. Detection must use a state machine.
12. A target must persist for N consecutive frames before triggering.
13. After a split, wait for the target to clear before another split.
14. Unit-test detection and state-machine logic independently of OBS and LiveSplit.
15. Do not implement template matching until the color detector is working.
16. Do not add dependencies without a clear reason.
17. Prefer small, testable modules over a single large script.
18. Handle OBS/capture and LiveSplit connection failures gracefully.
19. Diagnostic mode should expose enough information to tune thresholds.
20. Never trigger multiple splits from a single continuous transition.

## 21. Example Initial Configuration

The example below is the original minimal config. A current, fully commented one is
`configs/template.yaml` (it adds `stream_url`, frame size, `start_transition`,
`restart_quiet_seconds` and optional `ocr` / `record` sections).

```yaml
capture:
  fps: 60

roi:
  x: 0.25
  y: 0.25
  width: 0.50
  height: 0.50

detection:
  required_frames: 3
  reset_threshold: 0.80

colors:
  black:
    rgb: [0, 0, 0]
    tolerance: 15

  white:
    rgb: [255, 255, 255]
    tolerance: 15

splits:
  - id: 1
    name: "Transition 1"
    color: black
    threshold: 0.95

  - id: 2
    name: "Transition 2"
    color: black
    threshold: 0.95

  - id: 3
    name: "Transition 3"
    color: white
    threshold: 0.95

livesplit:
  host: "127.0.0.1"
  port: 16834
```

## 22. Important Implementation Note

Before implementing the capture layer, investigate the best OBS frame-access method on Linux/Wayland rather than assuming a particular Python OBS library will work.

The capture implementation is intentionally isolated because it is the most platform-specific part of the application.

**Outcome of that investigation:** obs-websocket screenshot polling managed only ~15–20 fps
and a virtual camera needs the v4l2loopback kernel module, so capture uses a second OBS
output (a sidecar stream) decoded by `ffmpeg`. Two variants work: Recording → Custom Output
(FFmpeg) as `mpegts` over UDP, or the Branch Output filter sending a low-res SRT feed so
OBS's normal Recording stays free for a full-quality archive. See `README.md` ("Connecting
OBS") and `AGENTS.md` for settings and measured latency.

The rest of the system should be testable without OBS by feeding prerecorded frames into the detector.

The development priority should therefore be:

```text
Configuration
    ↓
Color detector
    ↓
State machine
    ↓
Unit tests
    ↓
OBS frame acquisition
    ↓
Diagnostics
    ↓
LiveSplit connection
    ↓
End-to-end testing
```
