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

```yaml
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
optional OCR section, the on-screen split number can also resync the index while the
machine is in `NORMAL`.

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
│       │   └── ocr_detector.py            # optional split-index resync
│       ├── livesplit/client.py
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
Optional extras in `pyproject.toml`: `gui` (PySide6) and `ocr` (tesserocr, plus the system
`tesseract` package).

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
(optional, `ocr:` config section) exists only to resync the split index from an on-screen
number; template matching is still not implemented.

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
