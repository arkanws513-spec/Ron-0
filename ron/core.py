"""Ron orchestration core: intent -> memory -> context -> model -> learning."""
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
        text = user_text.strip()
        if not text:
            return ModelResponse(
                content="أنا رون. اكتب لي ما تريد.",
                model="ron-core",
                metadata={"runtime": "ron-0", "intent": "empty"},
            )

        fact = extract_fact(text)
        if fact is not None:
            self.memory.remember(fact.memory)
            return ModelResponse(
                content=self._fact_confirmation(fact.kind, fact.value),
                model="ron-memory",
                metadata={"runtime": "ron-0", "memory_write": True, "fact_kind": fact.kind},
            )

        remembered_answer = answer_fact_question(text, self.memory)
        if remembered_answer is not None:
            return ModelResponse(
                content=remembered_answer,
                model="ron-memory",
                metadata={"runtime": "ron-0", "memory_read": True},
            )

        memories = self.memory.recall(text, limit=5)
        memory_text = "\n".join(f"- {item.content}" for item in memories)
        system_context = (
            "أنت رون. استخدم الذاكرة التالية عند الحاجة، ولا تخترع معلومات غير موجودة.\n"
            + (memory_text if memory_text else "- لا توجد ذكريات مرتبطة.")
        )
        request = ModelRequest(
            messages=(
                Message(role="system", content=system_context),
                Message(role="user", content=text),
            ),
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
                content=text,
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
