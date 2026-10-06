# Ron-0 — رون

Ron is a GitHub-first AI assistant project.

## Principles
- GitHub is the source of truth.
- Provider-agnostic model interfaces.
- Explicit memory, tools, and self-improvement boundaries.
- Tests and GitHub Actions before expansion.
- No secrets committed to the repository.

## Current foundation
- Web shell for GitHub Pages.
- Provider contracts.
- Deterministic in-memory recall.
- Explicit tool registry.
- Ron orchestration core.
- Bounded, auditable self-improvement engine.
- Automated CI and Pages deployment.

## Self-improvement
Ron can learn from explicit experiences, turn lessons into candidate skills, verify them against a baseline, promote only passing improvements, and roll back a previously promoted skill. This changes learned behavior/skills; it does not silently retrain or replace the base model.

See `docs/self-improvement.md` for the lifecycle and safety boundary.

## Architecture
`UI → Conversation → Ron Core → Memory/Retrieval/Tools/Self-Improvement → Model Provider`

The project is intentionally built in small verified layers so a failure is fixed before the next layer is added.

## Development
Python 3.11+ and pytest are used for the core. The public web shell is static and can be published with GitHub Pages.
