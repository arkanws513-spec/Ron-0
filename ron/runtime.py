"""Minimal provider-agnostic Ron runtime."""
from __future__ import annotations
from dataclasses import dataclass
from .contracts import Message, ModelProvider, ModelRequest, ModelResponse

@dataclass
class AgentRuntime:
    provider: ModelProvider

    def respond(self, user_text: str) -> ModelResponse:
        request = ModelRequest(
            messages=(Message(role="user", content=user_text),),
            metadata={"runtime": "ron-0"},
        )
        return self.provider.generate(request)
