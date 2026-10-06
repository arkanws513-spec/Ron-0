# Ron-0 — رون

Ron is a GitHub-first AI assistant project.

## Principles
- GitHub is the source of truth.
- Ron's identity, memory, tools, learning rules, and orchestration stay separate from the model.
- A ready-made local brain can be plugged in without turning Ron into a cloud-provider product.
- No cloud API is required by the architecture.
- No secrets are committed to the repository.
- Tests and GitHub Actions are required before expansion.

## Current foundation
- Web shell for GitHub Pages.
- Provider contracts and a local HTTP model adapter.
- Deterministic memory and natural-language fact learning.
- Explicit tool registry.
- Ron orchestration core.
- Bounded, auditable self-improvement engine.
- Local-brain knowledge transfer with provenance.
- Automated CI and Pages deployment.

## Ready-made brain strategy

Ron does not need to build a useful language model entirely from zero before it
can become capable. A compatible local model can provide the initial language
capability while Ron owns the surrounding system: identity, memory, tools,
learning, evaluation, and self-improvement.

The brain is replaceable and model weights are not committed to this repository.

See `docs/brain-architecture.md`.

## Architecture

`UI → Conversation → Ron Core → Memory/Retrieval/Tools/Learning → Local Brain`
