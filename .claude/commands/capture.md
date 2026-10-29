---
description: Run the capture workflow for OBS, MPEG-TS/SRT, ffmpeg, frame assembly, resolution, or reconnect changes.
argument-hint: <capture change>
---

Act as the orchestrator for this capture change: $ARGUMENTS

1. `explorer` maps capture implementation.
2. `researcher` investigates ffmpeg/OBS behavior when required.
3. `architect` approves design.
4. `implementer` changes capture code/tests.
5. `test-engineer` tests partial reads, EOF, reconnects, malformed input.
6. `performance-engineer` measures throughput and latency.
7. `reviewer` checks architecture/resource handling.
8. `integration-validator` validates OBS → ffmpeg → detector.

Never introduce a second UDP/SRT consumer without explicitly evaluating the
existing single-consumer architecture.
