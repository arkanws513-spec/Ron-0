"""Model-provider abstractions for Ron.

Ron defaults to a deterministic local provider. Real model adapters can be
injected later without changing the core, and credentials stay outside source.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

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

@dataclass(frozen=True)
class LocalHTTPModelProvider:
    """Call a model running locally through an OpenAI-compatible HTTP API.

    No API key is required by default. The endpoint and model are configurable
    through environment variables so secrets and machine-specific settings
    never enter the repository.
    """

    endpoint: str = "http://127.0.0.1:11434/v1/chat/completions"
    model: str = "local-model"
    timeout: float = 120.0

    @classmethod
    def from_environment(cls) -> "LocalHTTPModelProvider":
        return cls(
            endpoint=os.getenv(
                "RON_MODEL_ENDPOINT",
                "http://127.0.0.1:11434/v1/chat/completions",
            ),
            model=os.getenv("RON_MODEL_NAME", "local-model"),
            timeout=float(os.getenv("RON_MODEL_TIMEOUT", "120")),
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role if message.role in {"system", "user", "assistant"} else "user",
                 "content": message.content}
                for message in request.messages
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"local model endpoint unavailable: {self.endpoint}"
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("local model returned no choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError("local model returned invalid message content")

        return ModelResponse(
            content=content,
            model=str(data.get("model") or self.model),
            metadata={
                "mode": "local-http",
                "external_api_required": False,
                "provider": "local-http",
            },
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
