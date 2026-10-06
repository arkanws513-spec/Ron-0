# Ron-0 — رون

Ron is a GitHub-first AI assistant project.

## Principles
- GitHub is the source of truth.
- Provider-agnostic model interfaces.
- Explicit memory and tool boundaries.
- Tests and GitHub Actions before expansion.
- No secrets committed to the repository.

## Current foundation
- Web shell for GitHub Pages.
- Provider contracts.
- Deterministic in-memory recall.
- Explicit tool registry.
- Ron orchestration core.
- Automated CI and Pages deployment.

## Architecture
`UI → Conversation → Ron Core → Memory/Retrieval/Tools → Model Provider`

The project is intentionally built in small verified layers so a failure is fixed before the next layer is added.

## Development
Python 3.11+ and pytest are used for the core. The public web shell is static and can be published with GitHub Pages.
