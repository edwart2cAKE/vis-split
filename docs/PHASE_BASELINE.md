# Phase Baseline

This document defines the compatibility baseline for future development.

## Completed phases

- Phase 1 — configuration validation
- Phase 2 — color detection
- Phase 3 — transition state machine
- Phase 4 — MPEG-TS frame capture
- Phase 5 — diagnostics overlay
- Phase 6 — LiveSplit client
- Phase 7 — `--loop` rearm

The local implementation is authoritative if this document becomes stale.

## Protected behavior

- Configuration is game-specific and tunable.
- ROI coordinates are normalized.
- Frames use BGR uint8.
- Color configuration uses RGB and is converted at the detector boundary.
- Detection uses tolerance rather than exact equality.
- Detection processes only the cropped ROI.
- Detection is vectorized.
- The state machine prevents multiple splits for one continuous transition.
- Debounce requires consecutive matching frames.
- Hysteresis controls reset.
- The final split enters DONE without waiting for reset.
- Loop rearm waits for the next run to become quiet.
- LiveSplit communication remains inside `livesplit/client.py`.
- Capture remains separate from detection.
- Detection remains testable without OBS or LiveSplit.

## Verification baseline

    uv run pytest -v
    uvx ruff check .
    uv run basedpyright src tests
    uv run python -m autosplitter.main --check

## Change policy

Changes that violate protected behavior require an explicit reason, updated
architecture documentation, regression tests, reviewer approval, and
integration validation when applicable.
