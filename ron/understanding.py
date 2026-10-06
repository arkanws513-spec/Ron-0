"""Ron-native understanding state: intent, topic, entities, context and confidence."""
from __future__ import annotations
from dataclasses import dataclass
import re
from .intent import Intent, detect_intent
from .memory import normalize_arabic

_FOLLOWUP_RE = re.compile(r"^(طيب|طب|وبعدين|وماذا عنه|وماذا عنها|وهل|وضح|اشرح اكثر|كمل|تابع|ماذا تقصد)\s*[؟?]?$")
_WORD_RE = re.compile(r"[\w\u0600-\u06ff]+")

@dataclass(frozen=True)
class Understanding:
    raw: str
    normalized: str
    intent: Intent | None
    topic: str | None
    entities: tuple[str, ...] = ()
    related_memory_keys: tuple[str, ...] = ()
    confidence: float = 0.0
    is_follow_up: bool = False

@dataclass
class UnderstandingEngine:
    """Deterministic front layer that makes every turn interpretable."""

    max_entities: int = 8

    def _tokens(self, text: str) -> list[str]:
        return [x for x in _WORD_RE.findall(normalize_arabic(text)) if len(x) > 1]

    def _entities(self, text: str) -> tuple[str, ...]:
        raw = str(text).strip()
        candidates = re.findall(r"[A-Z][A-Za-z0-9_-]{2,}|[\u0600-\u06ff]{3,}", raw)
        seen: list[str] = []
        for item in candidates:
            if item not in seen:
                seen.append(item)
        return tuple(seen[:self.max_entities])

    def analyze(
        self,
        text: str,
        previous_user: str | None = None,
        related_memory_keys: list[str] | None = None,
    ) -> Understanding:
        raw = str(text).strip()
        normalized = normalize_arabic(raw)
        previous = str(previous_user or "").strip()
        follow_up = bool(_FOLLOWUP_RE.fullmatch(normalized))
        topic = previous if follow_up and previous else (raw or None)
        intent = detect_intent(raw, topic=topic)
        tokens = self._tokens(topic or raw)
        confidence = intent.confidence if intent else 0.0
        if follow_up and previous:
            confidence = max(confidence, 0.96)
        elif tokens:
            confidence = max(confidence, 0.45)
        return Understanding(
            raw=raw,
            normalized=normalized,
            intent=intent,
            topic=topic,
            entities=self._entities(topic or raw),
            related_memory_keys=tuple(related_memory_keys or ()),
            confidence=min(confidence, 1.0),
            is_follow_up=follow_up,
        )
