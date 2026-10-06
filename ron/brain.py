"""Ron brain bridge: keep the intelligence layer local, replaceable, and user-controlled."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from .contracts import ModelRequest, ModelResponse
from .providers import LocalHTTPModelProvider, Qwen3TeacherProvider

@dataclass(frozen=True)
class TeachingExample:
    prompt: str
    answer: str
    source: str = "local-teacher"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class KnowledgeTransfer:
    """Turn teacher responses into Ron-owned, auditable training examples."""
    examples: list[TeachingExample] = field(default_factory=list)
    max_examples: int = 5000

    def capture(self, prompt: str, answer: str, source: str = "local-teacher") -> TeachingExample:
        example = TeachingExample(prompt=prompt.strip(), answer=answer.strip(), source=source)
        if example.prompt and example.answer:
            self.examples.append(example)
            del self.examples[:-self.max_examples]
        return example

    def dataset(self) -> list[dict[str, str]]:
        return [{"user": x.prompt, "assistant": x.answer} for x in self.examples]

@dataclass
class LocalBrain:
    """A replaceable local model used by Ron; no cloud provider is implied."""
    provider: LocalHTTPModelProvider
    transfer: KnowledgeTransfer = field(default_factory=KnowledgeTransfer)

    @classmethod
    def from_environment(cls) -> "LocalBrain":
        return cls(provider=LocalHTTPModelProvider.from_environment())

    def think(self, request: ModelRequest, learn: bool = True) -> ModelResponse:
        response = self.provider.generate(request)
        if learn:
            user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
            self.transfer.capture(user, response.content, source=self.provider.model)
        return response

    def generate(self, request: ModelRequest) -> ModelResponse:
        """ModelProvider-compatible entry point for RonCore/ProviderRouter."""
        return self.think(request, learn=True)


@dataclass
class Qwen3Teacher:
    """Optional local Qwen3 teacher that produces Ron-owned examples.

    Teaching is explicit: this component never replaces Ron's identity or
    provider automatically.
    """
    provider: Qwen3TeacherProvider = field(default_factory=Qwen3TeacherProvider.from_environment)
    transfer: KnowledgeTransfer = field(default_factory=KnowledgeTransfer)

    def teach(self, prompt: str, context: str = "") -> TeachingExample:
        messages = []
        if context.strip():
            messages.append({"role": "system", "content": context.strip()})
        messages.append({"role": "user", "content": prompt.strip()})
        request = ModelRequest(
            messages=tuple(
                __import__("ron.contracts", fromlist=["Message"]).Message(
                    role=item["role"], content=item["content"]
                ) for item in messages
            ),
            metadata={"teacher": "qwen3", "purpose": "knowledge-transfer"},
        )
        response = self.provider.generate(request)
        return self.transfer.capture(prompt, response.content, source=f"qwen3:{self.provider.model}")
