---
description: Run the detector workflow for color detection, ROI, threshold, debounce, hysteresis, or state-machine changes.
argument-hint: <detection change>
---

Act as the orchestrator for this detection change: $ARGUMENTS

Use for color detection, ROI, threshold, debounce, hysteresis, state-machine,
or visual-transition changes.

1. `explorer` maps detector/state-machine contracts.
2. `architect` defines the smallest change.
3. `implementer` adds implementation and fixtures.
4. `test-engineer` attacks tolerance boundaries, ROI boundaries, noise,
   debounce, hysteresis, duplicate transitions, final split, and loop rearm.
5. `performance-engineer` evaluates hot-loop impact.
6. `reviewer` performs independent review.
7. `integration-validator` validates the complete pipeline when possible.

No detector feature is complete without negative cases. Preserve the state
machine contract in `AGENTS.md` (streak counting, `reset_threshold` clearing,
last-split DONE, `start_transition` gate).
