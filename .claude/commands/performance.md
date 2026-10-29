---
description: Run the performance workflow for hot-path throughput or latency work.
argument-hint: <performance concern>
---

Act as the orchestrator for this performance work: $ARGUMENTS

1. `explorer` identifies the hot path.
2. `performance-engineer` establishes a baseline.
3. `researcher` investigates alternatives if necessary.
4. `architect` proposes the smallest safe optimization.
5. `implementer` applies the approved change.
6. `performance-engineer` repeats the benchmark.
7. `test-engineer` verifies behavioral equivalence.
8. `reviewer` checks complexity and regressions.
9. `integration-validator` verifies runtime behavior.

Performance claims require measurements. Keep the hot loop allocation-free,
keep ROI cropping before processing, and never add per-pixel Python loops.
