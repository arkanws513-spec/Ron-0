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
from .understanding import UnderstandingEngine
from .reasoning import ReasoningEngine

@dataclass
class RonCore:
    provider:ModelProvider
    memory:MemoryStore=field(default_factory=InMemoryStore)
    tools:ToolRegistry=field(default_factory=ToolRegistry)
    self_improvement:SelfImprovementEngine=field(default_factory=SelfImprovementEngine)
    conversation:Conversation=field(default_factory=lambda:Conversation(session_id="default"))
    max_context_messages:int=12
    understanding:UnderstandingEngine=field(default_factory=UnderstandingEngine)
    reasoning:ReasoningEngine=field(default_factory=ReasoningEngine)

    def _finish(self,text:str,content:str,model:str,metadata:dict)->ModelResponse:
        response=ModelResponse(content=content,model=model,metadata=metadata)
        self.conversation.add("assistant",content)
        return response

    def respond(self,user_text:str)->ModelResponse:
        text=user_text.strip()
        if not text:
            return self._finish(text,"أنا رون. اكتب لي ما تريد.","ron-core",{"runtime":"ron-0","intent":"empty"})
        previous_topic=None
        for message in reversed(self.conversation.messages):
            if message.role=="user" and message.content.strip():
                previous_topic=message.content
                break
        self.conversation.add("user",text)
        if previous_topic==text:
            previous_topic=None
        intent_state=self.understanding.analyze(text,previous_user=previous_topic)
        topic=intent_state.topic
        intent=intent_state.intent or detect_intent(text,topic=topic)
        facts=extract_facts(text)
        if facts:
            for fact in facts:self.memory.remember(fact.memory)
            return self._finish(text,self._facts_confirmation(facts),"ron-memory",{"runtime":"ron-0","memory_write":True,"fact_kinds":[f.kind for f in facts],"facts_written":len(facts)})
        remembered_answer=answer_fact_question(text,self.memory)
        if remembered_answer is not None:
            return self._finish(text,remembered_answer,"ron-memory",{"runtime":"ron-0","memory_read":True})
        memories=self.memory.recall(text,limit=5)
        related=self.memory.related(text,limit=8) if hasattr(self.memory,"related") else memories
        reasoning_facts=[]
        current_fact=self.reasoning.parse_fact(text,source="user",confidence=intent_state.confidence or 0.5)
        if current_fact is not None:
            reasoning_facts.append(current_fact)
        for item in related:
            parsed=self.reasoning.parse_fact(item.content,source=item.key,confidence=float(item.metadata.get("confidence",0.6)))
            if parsed is not None:
                reasoning_facts.append(parsed)
        reasoning_result=self.reasoning.reason(reasoning_facts)
        reasoning_summary=self.reasoning.summarize(reasoning_result)
        memory_text="\n".join(f"- {item.content}" for item in related)
        recent=self.conversation.short_history()[-self.max_context_messages:]
        history_text="\n".join(f"{m.role}: {m.content}" for m in recent)
        system_context=(
            "أنت رون. استخدم الذاكرة وسياق المحادثة عند الحاجة، ولا تخترع معلومات غير موجودة.\n"
            "الذاكرة ذات الصلة:\n"+(memory_text if memory_text else "- لا توجد ذكريات مرتبطة.")+
            "\n\nسياق المحادثة الأخير:\n"+(history_text if history_text else "- لا يوجد سياق.")+
            "\n\nحالة الاستدلال الداخلية:\n"+reasoning_summary+
            "\nملاحظة: هذه خلاصة استدلال منظمة وليست سلسلة التفكير الداخلية."
        )
        request=ModelRequest(
            messages=(Message(role="system",content=system_context),Message(role="user",content=text)),
            metadata={"runtime":"ron-0","intent":intent.name if intent else "unknown","intent_confidence":intent.confidence if intent else 0.0,"topic":topic,"memory_hits":len(related),"conversation_turns":len(self.conversation.messages),"tools":self.tools.describe(),"active_skills":sorted(self.self_improvement.registry.snapshot()),"reasoning":{"facts":len(reasoning_result.facts),"inferences":len(reasoning_result.inferences),"contradictions":len(reasoning_result.contradictions),"confidence":reasoning_result.confidence}},
        )
        try:
            response=self.provider.generate(request)
        except Exception as exc:
            response=ModelResponse(
                content=self._interactive_fallback(text,intent_state),
                model="ron-core-fallback",
                metadata={**request.metadata,"runtime":"ron-0","fallback":True,"provider_error":type(exc).__name__},
            )
        self.memory.remember(MemoryItem(key=f"turn:{len(getattr(self.memory,'items',{}))}",content=text,metadata={"kind":"conversation","topic":topic or ""}))
        return self._finish(text,response.content,response.model or "ron-core",response.metadata)

    @staticmethod
    def _interactive_fallback(text,understanding)->str:
        topic=understanding.topic or text
        if understanding.is_follow_up:
            return f"وصلتني متابعتك. ما زلت أتعامل مع موضوعنا السابق: {topic}. قد لا يكون فهمي كاملًا بعد، لكنني موجود وسأواصل معك."
        return f"وصلتني رسالتك. أفهم أن موضوعنا الآن هو: {topic}. قد أخطئ في الفهم، لكنني سأستمر في التفاعل بدل التوقف."

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
