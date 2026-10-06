"""Model-provider abstractions for Ron.

Ron can run with a deterministic local fallback. External providers are optional
and must be injected explicitly; credentials never belong in source control.
"""
from __future__ import annotations
from dataclasses import dataclass
from .contracts import ModelRequest, ModelResponse

@dataclass(frozen=True)
class LocalTeachingProvider:
    """A tiny deterministic provider used before a real model is connected."""
    name: str = "ron-local"

    def generate(self, request: ModelRequest) -> ModelResponse:
        user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        metadata = {"mode": "local", "external_api_required": False}
        if not user.strip():
            return ModelResponse(
                content="أنا رون. علّمني ما تريد أن أتعلمه.",
                model=self.name,
                metadata=metadata,
            )
        return ModelResponse(
            content=f"سمعتك: {user}\nأنا في وضع التعلم المحلي. يمكنك تعليمي وتصحيح إجاباتي.",
            model=self.name,
            metadata=metadata,
        )

class ProviderRouter:
    """Select an explicitly registered provider without exposing credentials."""
    def __init__(self, default: str = "local") -> None:
        self._providers: dict[str, object] = {}
        self.default = default

    def register(self, name: str, provider: object) -> None:
        if not name.strip():
            raise ValueError("provider name cannot be empty")
        if not callable(getattr(provider, "generate", None)):
            raise TypeError("provider must implement generate")
        self._providers[name] = provider

    def resolve(self, name: str | None = None):
        selected = name or self.default
        if selected not in self._providers:
            raise KeyError(f"unknown provider: {selected}")
        return self._providers[selected]
