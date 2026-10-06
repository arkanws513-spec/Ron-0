"""Ron internal reasoning engine: structured, auditable, provider-agnostic inference."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Iterable

class EvidenceKind(str, Enum):
    FACT = "fact"
    MEMORY = "memory"
    USER = "user"
    MODEL = "model"
    INFERENCE = "inference"

@dataclass(frozen=True)
class Evidence:
    content: str
    kind: EvidenceKind = EvidenceKind.USER
    confidence: float = 0.5
    source: str = "unknown"

@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    confidence: float = 0.8
    source: str = "unknown"

@dataclass(frozen=True)
class Rule:
    name: str
    premises: tuple[tuple[str, str, str], ...]
    conclusion: tuple[str, str, str]
    weight: float = 0.8

@dataclass(frozen=True)
class Hypothesis:
    statement: str
    confidence: float
    evidence: tuple[str, ...] = ()

@dataclass(frozen=True)
class Inference:
    rule: str
    premises: tuple[str, ...]
    conclusion: str
    confidence: float

@dataclass(frozen=True)
class Contradiction:
    left: str
    right: str
    reason: str

@dataclass(frozen=True)
class ReasoningResult:
    facts: tuple[Fact, ...] = ()
    inferences: tuple[Inference, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    conclusions: tuple[str, ...] = ()
    confidence: float = 0.0
    unknowns: tuple[str, ...] = ()

class ReasoningEngine:
    """A deterministic first layer for deduction, contradiction detection and confidence.

    It deliberately exposes structured results rather than private chain-of-thought.
    External language models may propose evidence, but they do not own the result.
    """

    _TRIPLE = re.compile(r"^\s*(.+?)\s+(is|has|likes|hates|needs|causes|supports|precedes|follows)\s+(.+?)\s*$", re.I)
    _ARABIC = ((re.compile(r"^(.+?)\s+(?:هي|هو)\s+(.+?)$"), "is"), (re.compile(r"^(.+?)\s+(?:يسبب|تسبب|يؤدي إلى|يؤدي الي)\s+(.+?)$"), "causes"), (re.compile(r"^(.+?)\s+(?:يدعم|تدعم)\s+(.+?)$"), "supports"), (re.compile(r"^(.+?)\s+(?:قبل|يسبق)\s+(.+?)$"), "precedes"), (re.compile(r"^(.+?)\s+(?:بعد|يتبع)\s+(.+?)$"), "follows"), (re.compile(r"^(.+?)\s+(?:يحتاج إلى|يحتاج الي|يحتاج)\s+(.+?)$"), "needs"))

    def __init__(self, rules: Iterable[Rule] | None = None, max_steps: int = 32) -> None:
        self.rules = tuple(rules or self.default_rules())
        self.max_steps = max(1, max_steps)

    @staticmethod
    def default_rules() -> tuple[Rule, ...]:
        return (
            Rule("transitive_precedes", (("?a", "precedes", "?b"), ("?b", "precedes", "?c")), ("?a", "precedes", "?c"), 0.92),
            Rule("transitive_supports", (("?a", "supports", "?b"), ("?b", "supports", "?c")), ("?a", "supports", "?c"), 0.88),
            Rule("causal_chain", (("?a", "causes", "?b"), ("?b", "causes", "?c")), ("?a", "causes", "?c"), 0.84),
        )

    @staticmethod
    def _norm(value: str) -> str:
        value = str(value).strip().lower()
        # Normalize common Arabic diacritics/tanween and terminal accusative
        # forms so "ازدحاما" and "ازدحام" represent the same concept.
        value = re.sub(r"[\u064B-\u065F\u0670]", "", value)
        value = re.sub(r"\s+", " ", value)
        value = re.sub(r"([\u0621-\u063A\u0641-\u064A]+)ا$", r"\1", value)
        return value

    @classmethod
    def fact_key(cls, fact: Fact) -> tuple[str, str, str]:
        return tuple(cls._norm(x) for x in (fact.subject, fact.relation, fact.object))

    @classmethod
    def parse_fact(cls, text: str, source: str = "user", confidence: float = 0.7) -> Fact | None:
        value = str(text).strip().rstrip("؟?.,،؛;")
        match = cls._TRIPLE.match(value)
        if match:
            return Fact(match.group(1).strip(), match.group(2).lower(), match.group(3).strip(), max(0.0, min(1.0, confidence)), source)
        normalized = cls._norm(value).replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        for pattern, relation in cls._ARABIC:
            match = pattern.match(normalized)
            if match:
                return Fact(match.group(1).strip(), relation, match.group(2).strip(), max(0.0, min(1.0, confidence)), source)
        return None

    @classmethod
    def parse_facts(cls, text: str, source: str = "user", confidence: float = 0.7) -> tuple[Fact, ...]:
        return tuple(fact for part in re.split(r"[\n.!؟?؛;]+", str(text)) if (fact := cls.parse_fact(part, source, confidence)) is not None)

    @staticmethod
    def _match(pattern: tuple[str, str, str], fact: Fact, bindings: dict[str, str]) -> dict[str, str] | None:
        candidate = (fact.subject, fact.relation, fact.object)
        out = dict(bindings)
        for expected, actual in zip(pattern, candidate):
            if expected.startswith("?"):
                old = out.get(expected)
                if old is not None and old != actual:
                    return None
                out[expected] = actual
            elif expected.lower() != str(actual).lower():
                return None
        return out

    def _apply_rule(self, rule: Rule, facts: list[Fact]) -> list[Inference]:
        out: list[Inference] = []
        def walk(i: int, bindings: dict[str, str], used: list[Fact]) -> None:
            if len(out) >= self.max_steps:
                return
            if i == len(rule.premises):
                rendered = tuple(bindings.get(x, x) for x in rule.conclusion)
                conclusion = Fact(*rendered, confidence=min(rule.weight, *(f.confidence for f in used)), source=f"rule:{rule.name}")
                out.append(Inference(rule.name, tuple(self.describe_fact(f) for f in used), self.describe_fact(conclusion), conclusion.confidence))
                return
            for fact in facts:
                next_bindings = self._match(rule.premises[i], fact, bindings)
                if next_bindings is not None:
                    walk(i + 1, next_bindings, used + [fact])
        walk(0, {}, [])
        return out

    @classmethod
    def describe_fact(cls, fact: Fact) -> str:
        return f"{fact.subject} {fact.relation} {fact.object}"

    def reason(self, facts: Iterable[Fact] = (), hypotheses: Iterable[Hypothesis] = ()) -> ReasoningResult:
        known = list(facts)
        seen = {self.fact_key(f) for f in known}
        inferences: list[Inference] = []
        for _ in range(self.max_steps):
            produced: list[Fact] = []
            for rule in self.rules:
                for inf in self._apply_rule(rule, known):
                    parts = inf.conclusion.split(" ", 2)
                    if len(parts) != 3:
                        continue
                    fact = Fact(parts[0], parts[1], parts[2], inf.confidence, f"rule:{inf.rule}")
                    if self.fact_key(fact) not in seen:
                        seen.add(self.fact_key(fact)); produced.append(fact); inferences.append(inf)
                        if len(inferences) >= self.max_steps: break
                if len(inferences) >= self.max_steps: break
            if not produced: break
            known.extend(produced)

        contradictions: list[Contradiction] = []
        for a in known:
            for b in known:
                if a.subject == b.subject and a.relation == b.relation and a.object != b.object:
                    if a.relation in {"is", "has", "needs", "causes"}:
                        contradictions.append(Contradiction(self.describe_fact(a), self.describe_fact(b), "same subject/relation with different object"))
        conclusion_text = tuple(self.describe_fact(f) for f in known)
        confs = [f.confidence for f in known]
        result_conf = sum(confs) / len(confs) if confs else 0.0
        return ReasoningResult(
            facts=tuple(known),
            inferences=tuple(inferences),
            hypotheses=tuple(hypotheses),
            contradictions=tuple(contradictions),
            conclusions=conclusion_text,
            confidence=min(1.0, result_conf),
            unknowns=(),
        )

    @classmethod
    def answer_query(cls, text: str, result: ReasoningResult) -> str | None:
        q=cls._norm(str(text).rstrip("؟? .،,"))
        facts=list(result.facts)
        m=re.match(r"^(?:ما هي|ما هو|ماهو|ايه|اي)\s+(.+?)\s+(.+)$", q)
        if m:
            relation_words={"عاصمة":"is","عاصمه":"is","يسبب":"causes","تسبب":"causes","يدعم":"supports","تدعم":"supports","قبل":"precedes","يسبق":"precedes","بعد":"follows","يتبع":"follows","يحتاج":"needs"}
            relation=relation_words.get(m.group(1))
            if relation:
                obj=m.group(2)
                matches=[f for f in facts if f.relation==relation and cls._norm(f.object)==obj]
                if matches:
                    return f"{matches[-1].subject} {m.group(1)} {matches[-1].object}."
        m=re.match(r"^(?:هل|هل صحيح ان)\s+(.+?)\s+(يسبب|تسبب|يدعم|تدعم|قبل|يسبق|بعد|يتبع|يحتاج)\s+(.+)$", q)
        if m:
            rel={"يسبب":"causes","تسبب":"causes","يدعم":"supports","تدعم":"supports","قبل":"precedes","يسبق":"precedes","بعد":"follows","يتبع":"follows","يحتاج":"needs"}[m.group(2)]
            key=(cls._norm(m.group(1)),rel,cls._norm(m.group(3)))
            if any(cls.fact_key(f)==key for f in facts):
                return "نعم، هذا مدعوم بالأدلة المتاحة."
            if any(f.subject==m.group(1) and f.relation==rel and f.object!=m.group(3) for f in facts):
                return "لا، توجد معلومة متعارضة مع ذلك."
        m=re.match(r"^(?:ماذا|ما)\s+(?:الذي\s+)?(?:يسبب|تسبب)\s+(.+)$", q)
        if m:
            matches=[f.object for f in facts if f.relation=="causes" and cls._norm(f.subject)==m.group(1)]
            if matches:
                return "المعروف لدي: " + "، ".join(matches) + "."
        return None

    def summarize(self, result: ReasoningResult) -> str:
        if not result.facts and not result.hypotheses:
            return "لا توجد أدلة كافية للاستدلال."
        if result.contradictions:
            return f"يوجد {len(result.contradictions)} تعارض يحتاج إلى مراجعة."
        if result.inferences:
            return f"تم الوصول إلى {len(result.inferences)} استنتاجات منظمة بدرجة ثقة تقريبية {result.confidence:.2f}."
        return f"تم تحليل الأدلة المتاحة بدرجة ثقة تقريبية {result.confidence:.2f}."
