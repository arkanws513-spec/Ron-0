# Ron self-improvement

Ron has a bounded, auditable self-improvement layer.

## Lifecycle

1. **Experience** — record what happened and the lesson extracted from it.
2. **Proposal** — a proposer turns an experience into a candidate skill.
3. **Verification** — an independent evaluator compares the candidate with the current baseline and runs tests.
4. **Promotion** — only a candidate with passing tests and a strictly better score becomes active.
5. **Rollback** — an active skill can be reverted to its previous verified version.
6. **Audit** — every promotion or rejection is recorded.

The engine now exposes one complete `improve(...)` cycle that performs these steps in order.

## Boundary

This mechanism improves Ron's learned skills and behavior. It does **not** silently retrain or replace the base model and does not directly overwrite production source code.

For future code-level self-improvement, Ron should create an isolated candidate change, run repository tests and benchmarks, and only then request promotion. GitHub Actions provides the external verification layer for repository changes.
