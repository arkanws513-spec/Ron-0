"""Model-provider abstractions for Ron.

Ron defaults to a deterministic local provider. Real model adapters can be
injected later without changing the core, and credentials stay outside source.
"""
from __future__ import annotations

from dataclasses import dataclass
from .contracts import ModelRequest, ModelResponse


@dataclass(frozen=True)
class LocalTeachingProvider:
    """Deterministic offline provider used as a safe fallback."""

    name: str = "ron-local"

    def generate(self, request: ModelRequest) -> ModelResponse:
        user = next(
            (message.content for message in reversed(request.messages) if message.role == "user"),
            "",
        )
        metadata = {
            "mode": "local",
            "external_api_required": False,
            "provider": self.name,
        }
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
    """Resolve only explicitly registered providers."""

    def __init__(self, default: str = "local") -> None:
        if not default.strip():
            raise ValueError("default provider cannot be empty")
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

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))
