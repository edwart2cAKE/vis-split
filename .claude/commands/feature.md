---
description: Run the multi-agent feature workflow for a new capability.
argument-hint: <what to build>
---

Act as the orchestrator for this feature: $ARGUMENTS

Phases 1–7 are the compatibility baseline. Read `AGENTS.md` and
`Project Overview — Roblox Visual.md` before making architectural decisions.

1. Define the task and acceptance criteria.
2. `explorer` maps affected implementation.
3. `researcher` investigates unknown external behavior if needed.
4. `architect` produces the design; you approve it.
5. `implementer` changes code and tests.
6. `test-engineer` attempts to break it.
7. `reviewer` independently reviews it.
8. `integration-validator` validates runtime behavior when applicable.
9. Accept or send it back.

Do not skip architecture for module-boundary changes.
Do not skip integration validation for capture, detection, state-machine,
or LiveSplit behavior.

Complete only when the implementation matches the approved design, and
`uv run pytest -v`, `uvx ruff check .`, `uv run basedpyright src tests`, and
`uv run python -m autosplitter.main --check` all pass with no unexplained
regressions.
