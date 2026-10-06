"""Ron orchestration core: intent -> memory -> context -> model -> learning."""
from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import MemoryItem, MemoryStore, Message, ModelProvider, ModelRequest, ModelResponse
from .facts import answer_fact_question, extract_facts
from .intent import detect_intent
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

        intent = detect_intent(text)
        if intent is not None:
            if intent.name == "greeting":
                return ModelResponse(content="أهلًا بك. أنا رون.", model="ron-core", metadata={"runtime": "ron-0", "intent": intent.name})
            if intent.name == "assistant_identity":
                return ModelResponse(content="أنا رون، مساعد مستقل قيد التطوير.", model="ron-core", metadata={"runtime": "ron-0", "intent": intent.name})

        facts = extract_facts(text)
        if facts:
            for fact in facts:
                self.memory.remember(fact.memory)
            return ModelResponse(
                content=self._facts_confirmation(facts),
                model="ron-memory",
                metadata={
                    "runtime": "ron-0",
                    "memory_write": True,
                    "fact_kinds": [fact.kind for fact in facts],
                    "facts_written": len(facts),
                },
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
    def _facts_confirmation(facts) -> str:
        values = {fact.kind: fact.value for fact in facts}
        if "name" in values and "age" in values:
            return f"تم. سأحفظ أن اسمك {values['name']} وأن عمرك {values['age']} سنة."
        if "name" in values:
            return f"تم. سأحفظ أن اسمك {values['name']}."
        if "age" in values:
            return f"تم. سأحفظ أن عمرك {values['age']} سنة."
        if "preference" in values:
            return f"تم. سأحفظ أنك تحب {values['preference']}."
        return "تم حفظ المعلومة في ذاكرتي."

    def learn_from_experience(self, experience: Experience) -> None:
        self.self_improvement.learn(experience)
