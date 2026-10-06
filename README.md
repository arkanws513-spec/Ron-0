# Ron-0

Ron (رون) is an open-source AI assistant project built with GitHub as the source of truth.

## Principles
- GitHub-first: source code, architecture, documentation, tests, and history live here.
- Provider-agnostic: model providers are adapters, not the core.
- Modular: memory, retrieval, planning, tools, and UI are replaceable modules.
- Verifiable: capabilities should have tests and automated checks.
- Secure by default: secrets never belong in source control.

## Initial architecture
UI → Conversation Engine → Ron Core → Memory / Retrieval / Tools → Model Router → Provider Adapters

Ron-0 starts deliberately small. Stable interfaces come before providers and external services.

## Status
Foundation phase.
