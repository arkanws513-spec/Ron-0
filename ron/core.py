"""Ron orchestration core: context -> decision -> bounded execution -> response."""
from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import (
    MemoryItem,
    MemoryStore,
    Message,
    ModelProvider,
    ModelRequest,
    ModelResponse,
)
from .memory import InMemoryStore
from .self_improvement import Experience, SelfImprovementEngine
from .tools import ToolRegistry


@dataclass
class RonCore:
    provider: ModelProvider
    memory: MemoryStore = field(default_factory=InMemoryStore)
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    self_improvement: SelfImprovementEngine = field(default_factory=SelfImprovementEngine)

    def respond(self, user_text: str) -> ModelResponse:
        memories = self.memory.recall(user_text, limit=5)
        context = tuple(Message(role="memory", content=item.content) for item in memories)
        request = ModelRequest(
            messages=context + (Message(role="user", content=user_text),),
            metadata={
                "runtime": "ron-0",
                "memory_hits": len(memories),
                "tools": self.tools.describe(),
                "active_skills": sorted(self.self_improvement.registry.snapshot()),
            },
        )
        response = self.provider.generate(request)
        self.memory.remember(
            MemoryItem(
                key=f"turn:{len(getattr(self.memory, 'items', {}))}",
                content=user_text,
            )
        )
        return response

    def learn_from_experience(self, experience: Experience) -> None:
        """Expose explicit learning without silently changing active behavior."""
        self.self_improvement.learn(experience)
