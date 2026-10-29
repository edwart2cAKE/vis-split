# Capture Workflow

Use for OBS, MPEG-TS, ffmpeg, UDP/SRT, frame assembly, resolution, or reconnect
changes.

1. Explorer maps capture implementation.
2. Researcher investigates ffmpeg/OBS behavior when required.
3. Architect approves design.
4. Implementer changes capture code/tests.
5. Test Engineer tests partial reads, EOF, reconnects, malformed input.
6. Performance Engineer measures throughput and latency.
7. Reviewer checks architecture/resource handling.
8. Integration Validator validates OBS → ffmpeg → detector.

Never introduce a second UDP/SRT consumer without explicitly evaluating the
existing single-consumer architecture.
