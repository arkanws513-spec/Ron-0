# Ron self-improvement

Ron now has a bounded, auditable self-improvement layer.

## Lifecycle

1. **Experience** — record what happened and the lesson extracted from it.
2. **Proposal** — a proposer turns a verified experience into a candidate skill.
3. **Verification** — compare the candidate with the current baseline and run tests.
4. **Promotion** — only a candidate with passing tests and a strictly better score becomes active.
5. **Rollback** — an active skill can be reverted to its previous verified version.
6. **Audit** — every promotion or rejection is recorded.

## What this means

Ron can add **skills and learned behavior** without modifying the base model or blindly rewriting production code. The registry is versioned and the promotion gate is explicit.

The current implementation is deliberately model-agnostic. A future model adapter may generate proposals, while the verifier remains independent of that model.

## Safety boundary

Self-improvement is experimental by default. A candidate is inert until verification succeeds. Production code, credentials, and the base model are not modified by this module.

GitHub Actions can run the project's tests on every change, providing an additional verification layer. 
