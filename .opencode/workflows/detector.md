# Detector Workflow

Use for color detection, ROI, threshold, debounce, hysteresis, state-machine,
or visual-transition changes.

1. Explorer maps detector/state-machine contracts.
2. Architect defines the smallest change.
3. Implementer adds implementation and fixtures.
4. Test Engineer attacks tolerance boundaries, ROI boundaries, noise, debounce,
   hysteresis, duplicate transitions, final split, and loop rearm.
5. Performance Engineer evaluates hot-loop impact.
6. Reviewer performs independent review.
7. Integration Validator validates the complete pipeline when possible.

No detector feature is complete without negative cases.
