"""Ron orchestration core: memory -> context -> model -> verified learning."""
from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import MemoryItem, MemoryStore, Message, ModelProvider, ModelRequest, ModelResponse
from .facts import answer_fact_question, extract_fact
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
        fact = extract_fact(user_text)
        if fact is not None:
            self.memory.remember(fact.memory)
            return ModelResponse(
                content=self._fact_confirmation(fact.kind, fact.value),
                model="ron-memory",
                metadata={"runtime": "ron-0", "memory_write": True, "fact_kind": fact.kind},
            )

        remembered_answer = answer_fact_question(user_text, self.memory)
        if remembered_answer is not None:
            return ModelResponse(
                content=remembered_answer,
                model="ron-memory",
                metadata={"runtime": "ron-0", "memory_read": True},
            )

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
                metadata={"kind": "conversation"},
            )
        )
        return response

    @staticmethod
    def _fact_confirmation(kind: str, value: str) -> str:
        if kind == "name":
            return f"تم. سأحفظ أن اسمك {value}."
        if kind == "preference":
            return f"تم. سأحفظ أنك تحب {value}."
        return "تم حفظ المعلومة في ذاكرتي."

    def learn_from_experience(self, experience: Experience) -> None:
        self.self_improvement.learn(experience)
