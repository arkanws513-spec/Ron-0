"""Ron orchestration core: intent -> memory -> conversation context -> model -> learning."""
from __future__ import annotations
from dataclasses import dataclass, field
from .contracts import MemoryItem, MemoryStore, Message, ModelProvider, ModelRequest, ModelResponse
from .facts import answer_fact_question, extract_facts
from .intent import detect_intent
from .memory import InMemoryStore
from .self_improvement import Experience, SelfImprovementEngine
from .session import Conversation
from .tools import ToolRegistry

@dataclass
class RonCore:
    provider:ModelProvider
    memory:MemoryStore=field(default_factory=InMemoryStore)
    tools:ToolRegistry=field(default_factory=ToolRegistry)
    self_improvement:SelfImprovementEngine=field(default_factory=SelfImprovementEngine)
    conversation:Conversation=field(default_factory=lambda:Conversation(session_id="default"))
    max_context_messages:int=12

    def _finish(self,text:str,content:str,model:str,metadata:dict)->ModelResponse:
        response=ModelResponse(content=content,model=model,metadata=metadata)
        self.conversation.add("assistant",content)
        return response

    def respond(self,user_text:str)->ModelResponse:
        text=user_text.strip()
        if not text:
            return self._finish(text,"أنا رون. اكتب لي ما تريد.","ron-core",{"runtime":"ron-0","intent":"empty"})
        self.conversation.add("user",text)
        intent=detect_intent(text)
        if intent is not None:
            if intent.name=="greeting":
                return self._finish(text,"أهلًا بك. أنا رون.","ron-core",{"runtime":"ron-0","intent":intent.name})
            if intent.name=="assistant_identity":
                return self._finish(text,"أنا رون، مساعد مستقل قيد التطوير.","ron-core",{"runtime":"ron-0","intent":intent.name})
        facts=extract_facts(text)
        if facts:
            for fact in facts:self.memory.remember(fact.memory)
            return self._finish(text,self._facts_confirmation(facts),"ron-memory",{"runtime":"ron-0","memory_write":True,"fact_kinds":[f.kind for f in facts],"facts_written":len(facts)})
        remembered_answer=answer_fact_question(text,self.memory)
        if remembered_answer is not None:
            return self._finish(text,remembered_answer,"ron-memory",{"runtime":"ron-0","memory_read":True})
        memories=self.memory.recall(text,limit=5)
        memory_text="\n".join(f"- {item.content}" for item in memories)
        recent=self.conversation.history()[-self.max_context_messages:]
        history_text="\n".join(f"{m.role}: {m.content}" for m in recent)
        system_context=(
            "أنت رون. استخدم الذاكرة وسياق المحادثة عند الحاجة، ولا تخترع معلومات غير موجودة.\n"
            "الذاكرة ذات الصلة:\n"+(memory_text if memory_text else "- لا توجد ذكريات مرتبطة.")+
            "\n\nسياق المحادثة الأخير:\n"+(history_text if history_text else "- لا يوجد سياق.")
        )
        request=ModelRequest(
            messages=(Message(role="system",content=system_context),Message(role="user",content=text)),
            metadata={"runtime":"ron-0","memory_hits":len(memories),"conversation_turns":len(self.conversation.messages),"tools":self.tools.describe(),"active_skills":sorted(self.self_improvement.registry.snapshot())},
        )
        response=self.provider.generate(request)
        self.memory.remember(MemoryItem(key=f"turn:{len(getattr(self.memory,'items',{}))}",content=text,metadata={"kind":"conversation"}))
        self.conversation.add("assistant",response.content)
        return response

    @staticmethod
    def _facts_confirmation(facts)->str:
        values={fact.kind:fact.value for fact in facts}
        if "name" in values and "age" in values:return f"تم. سأحفظ أن اسمك {values['name']} وأن عمرك {values['age']} سنة."
        if "name" in values:return f"تم. سأحفظ أن اسمك {values['name']}."
        if "age" in values:return f"تم. سأحفظ أن عمرك {values['age']} سنة."
        if "preference" in values:return f"تم. سأحفظ أنك تحب {values['preference']}."
        return "تم حفظ المعلومة في ذاكرتي."

    def learn_from_experience(self,experience:Experience)->None:
        self.self_improvement.learn(experience)
