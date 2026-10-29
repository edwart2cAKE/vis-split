---
description: Run the multi-agent bug workflow for a defect.
argument-hint: <symptom or repro>
---

Act as the orchestrator for this defect: $ARGUMENTS

1. `explorer` reproduces/localizes the defect.
2. `test-engineer` creates a failing regression test.
3. `architect` is consulted if architecture changes.
4. `implementer` fixes the defect.
5. `test-engineer` reruns regression/adversarial tests.
6. `reviewer` checks the fix.
7. `integration-validator` runs when the bug crosses process/runtime boundaries.
8. Accept only when the original regression is demonstrably fixed.

Run `uv run pytest -v`, `uvx ruff check .`, `uv run basedpyright src tests`,
and `uv run python -m autosplitter.main --check` before accepting.
