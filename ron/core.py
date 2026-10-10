"""Ron orchestration core: intent -> memory -> conversation context -> model -> learning."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from .contracts import MemoryItem, MemoryStore, Message, ModelProvider, ModelRequest, ModelResponse
from .facts import answer_fact_question, extract_facts
from .intent import detect_intent
from .memory import InMemoryStore
from .providers import LocalTeachingProvider
from .self_improvement import Experience, SelfImprovementEngine
from .session import Conversation
from .tools import ToolRegistry
from .understanding import UnderstandingEngine
from .reasoning import ReasoningEngine
from .reasoning_advanced import AdvancedReasoner, WeightedEvidence, CausalLink

@dataclass
class RonCore:
    provider:ModelProvider|None=None
    memory:MemoryStore=field(default_factory=InMemoryStore)
    tools:ToolRegistry=field(default_factory=ToolRegistry)
    self_improvement:SelfImprovementEngine=field(default_factory=SelfImprovementEngine)
    conversation:Conversation=field(default_factory=lambda:Conversation(session_id="default"))
    max_context_messages:int=12
    understanding:UnderstandingEngine=field(default_factory=UnderstandingEngine)
    reasoning:ReasoningEngine=field(default_factory=ReasoningEngine)
    advanced_reasoning:AdvancedReasoner=field(default_factory=AdvancedReasoner)

    def __post_init__(self)->None:
        if self.provider is not None:
            return
        checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
        # Prefer the larger native model when a validated Ron-10M checkpoint exists.
        # Keep the baseline as a safe local fallback and never require a hosted model.
        candidates = (
            checkpoint_dir / "ron_native_10m.pt",
            checkpoint_dir / "ron_native_baseline.pt",
        )
        checkpoint = next((path for path in candidates if path.is_file()), None)
        if checkpoint is not None:
            from .native_provider import NativeCheckpointProvider
            self.provider = NativeCheckpointProvider(checkpoint)
        else:
            # A clear offline fallback for checkouts that do not yet contain weights.
            self.provider = LocalTeachingProvider()

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
        semantic_answer=self.reasoning.answer_query(text,reasoning_result)
        if semantic_answer is not None:
            self.memory.remember(MemoryItem(key=f"turn:{len(getattr(self.memory,'items',{}))}",content=text,metadata={"kind":"conversation","topic":topic or ""}))
            return self._finish(text,semantic_answer,"ron-reasoning",{"runtime":"ron-0","reasoning_answer":True,"confidence":reasoning_result.confidence})
        reasoning_summary=self.reasoning.summarize(reasoning_result)
        evidence=tuple(WeightedEvidence(self.reasoning.describe_fact(f), f.confidence, f.confidence, f.source) for f in reasoning_result.facts)
        causal_links=tuple(CausalLink(f.subject, f.object, f.confidence, (self.reasoning.describe_fact(f),)) for f in reasoning_result.facts if f.relation=="causes")
        # Only combine evidence with distinct explicit sources. Treat inferred,
        # unknown, and conversational sources conservatively to avoid false certainty.
        sourced_support = {}
        for item in evidence:
            if item.source not in {"", "unknown", "user", "model"} and not item.source.startswith("rule:"):
                sourced_support[item.source] = max(sourced_support.get(item.source, 0.0), item.strength)
        advanced_confidence = (
            self.advanced_reasoning.combine_independent_confidences(sourced_support.values())
            if sourced_support else max((item.strength for item in evidence), default=0.0)
        )
        if causal_links and topic:
            causal_notes=[]
            for link in causal_links:
                ok, confidence, path=self.advanced_reasoning.causal_reason(causal_links, link.cause, link.effect)
                if ok:
                    causal_notes.append(f"{link.cause} -> {link.effect} ({confidence:.2f})")
            if causal_notes:
                reasoning_summary += "\nالعلاقات السببية المكتشفة: " + "؛ ".join(causal_notes[:6])
        reasoning_summary += f"\nتجميع ثقة الأدلة المستقلة: {advanced_confidence:.2f}"
        memory_text="\n".join(f"- {item.content}" for item in related)
        recent=self.conversation.short_history()[-self.max_context_messages:]
        # The current user turn is supplied separately below; keep only prior turns here
        # so providers can consume dialogue history without duplicating the current question.
        history_messages = recent[:-1] if recent and recent[-1].role == "user" and recent[-1].content == text else recent
        history_text="\n".join(f"{m.role}: {m.content}" for m in history_messages)
        system_context=(
            "أنت رون. استخدم الذاكرة وسياق المحادثة عند الحاجة، ولا تخترع معلومات غير موجودة.\n"
            "الذاكرة ذات الصلة:\n"+(memory_text if memory_text else "- لا توجد ذكريات مرتبطة.")+
            "\n\nسياق المحادثة الأخير:\n"+(history_text if history_text else "- لا يوجد سياق.")+
            "\n\nحالة الاستدلال الداخلية:\n"+reasoning_summary+
            "\nملاحظة: هذه خلاصة استدلال منظمة وليست سلسلة التفكير الداخلية."
        )
        request=ModelRequest(
            messages=(Message(role="system",content=system_context), *history_messages, Message(role="user",content=text)),
            metadata={"runtime":"ron-0","intent":intent.name if intent else "unknown","intent_confidence":intent.confidence if intent else 0.0,"topic":topic,"memory_hits":len(related),"conversation_turns":len(self.conversation.messages),"tools":self.tools.describe(),"active_skills":sorted(self.self_improvement.registry.snapshot()),"reasoning":{"facts":len(reasoning_result.facts),"inferences":len(reasoning_result.inferences),"contradictions":len(reasoning_result.contradictions),"confidence":reasoning_result.confidence,"advanced_confidence":advanced_confidence,"causal_links":len(causal_links)}},
        )
        try:
            assert self.provider is not None
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
