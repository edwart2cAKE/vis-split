# Bug Workflow

1. Explorer reproduces/localizes the defect.
2. Test Engineer creates a failing regression test.
3. Architect is consulted if architecture changes.
4. Implementer fixes the defect.
5. Test Engineer reruns regression/adversarial tests.
6. Reviewer checks the fix.
7. Integration Validator runs when the bug crosses process/runtime boundaries.
8. Orchestrator accepts only when the original regression is demonstrably fixed.
