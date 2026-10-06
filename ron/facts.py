"""Conservative natural-language fact extraction and recall for Ron."""
from __future__ import annotations
import re
from dataclasses import dataclass
from .contracts import MemoryItem, MemoryStore

_ARABIC_DIACRITICS = re.compile(r"[ًٌٍَُِّْـ]")
_SPACES = re.compile(r"\s+")

_NAME_PATTERNS = (
    re.compile(r"^\s*(?:اسمي|انا اسمي|أنا اسمي)\s+(.+?)\s*$", re.I),
    re.compile(r"^\s*اريدك(?: أن| ان)?\s+تعلم\s+أن\s+اسمي\s+هو\s+(.+?)\s*$", re.I),
)
_PREFERENCE_PATTERNS = (
    re.compile(r"^\s*(?:انا|أنا)\s+(?:احب|أحب)\s+(.+?)\s*$", re.I),
    re.compile(r"^\s*(?:انا|أنا)\s+(?:لا احب|لا أحب)\s+(.+?)\s*$", re.I),
)

def normalize_arabic(text: str) -> str:
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    return _SPACES.sub(" ", text.lower()).strip()

@dataclass(frozen=True)
class ExtractedFact:
    key: str
    value: str
    kind: str
    @property
    def memory(self) -> MemoryItem:
        return MemoryItem(
            key=self.key,
            content=self.value,
            metadata={"kind": self.kind, "source": "natural-language"},
        )

def extract_fact(text: str) -> ExtractedFact | None:
    for pattern in _NAME_PATTERNS:
        match = pattern.match(text)
        if match:
            value = match.group(1).strip(" .،,؛;؟?")
            if value:
                return ExtractedFact("user.name", value, "name")
    for pattern in _PREFERENCE_PATTERNS:
        match = pattern.match(text)
        if match:
            value = match.group(1).strip(" .،,؛;؟?")
            if value:
                return ExtractedFact("user.preference", value, "preference")
    return None

def answer_fact_question(text: str, memory: MemoryStore) -> str | None:
    normalized = normalize_arabic(text)
    if any(phrase in normalized for phrase in ("ما اسمي", "ايه اسمي", "ما هو اسمي", "هل تتذكر اسمي")):
        hits = memory.recall("user.name", limit=1)
        return f"اسمك {hits[0].content}." if hits else "لم تخبرني باسمك بعد."
    if any(phrase in normalized for phrase in ("ماذا احب", "ما الذي احبه", "ايه اللي بحبه", "هل تتذكر ما احب")):
        hits = memory.recall("user.preference", limit=5)
        return "أتذكر أنك تحب: " + "، ".join(item.content for item in hits) + "." if hits else "لم تخبرني بتفضيلاتك بعد."
    return None
