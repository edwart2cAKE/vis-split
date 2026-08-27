# Feature Workflow

1. Orchestrator defines task and acceptance criteria.
2. Explorer maps affected implementation.
3. Researcher investigates unknown external behavior if needed.
4. Architect produces design.
5. Implementer changes code and tests.
6. Test Engineer attempts to break it.
7. Reviewer independently reviews it.
8. Integration Validator validates runtime behavior when applicable.
9. Orchestrator accepts or sends it back.

Do not skip architecture for module-boundary changes.
Do not skip integration validation for capture, detection, state-machine,
or LiveSplit behavior.
