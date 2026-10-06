"""Ron orchestration core: context -> decision -> bounded execution -> response."""
from __future__ import annotations
from dataclasses import dataclass, field
from .contracts import MemoryItem, MemoryStore, ModelProvider, ModelRequest, ModelResponse, Message
from .memory import InMemoryStore
from .tools import ToolRegistry

@dataclass
class RonCore:
    provider: ModelProvider
    memory: MemoryStore = field(default_factory=InMemoryStore)
    tools: ToolRegistry = field(default_factory=ToolRegistry)

    def respond(self, user_text: str) -> ModelResponse:
        memories = self.memory.recall(user_text, limit=5)
        context = tuple(Message(role="memory", content=item.content) for item in memories)
        request = ModelRequest(
            messages=context + (Message(role="user", content=user_text),),
            metadata={"runtime": "ron-0", "memory_hits": len(memories), "tools": self.tools.describe()},
        )
        response = self.provider.generate(request)
        self.memory.remember(MemoryItem(key=f"turn:{len(getattr(self.memory, 'items', {}))}", content=user_text))
        return response