"""Lightweight intent routing for Ron before a language model is available."""
from __future__ import annotations
import re
from dataclasses import dataclass

_ARABIC_DIACRITICS = re.compile(r"[ًٌٍَُِّْـ]")
_SPACES = re.compile(r"\s+")

def normalize_arabic(text: str) -> str:
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    return _SPACES.sub(" ", text.lower()).strip()

@dataclass(frozen=True)
class Intent:
    name: str
    confidence: float

def detect_intent(text: str) -> Intent | None:
    value = normalize_arabic(text)
    if not value:
        return None
    if value in {"مرحبا", "اهلا", "السلام عليكم", "سلام عليكم", "هاي"}:
        return Intent("greeting", 1.0)
    if any(phrase in value for phrase in ("ما اسمك", "ايه اسمك", "من انت", "ما هو اسمك")):
        return Intent("assistant_identity", 0.98)
    if any(phrase in value for phrase in ("ما اسمي", "ايه اسمي", "ما هو اسمي", "هل تتذكر اسمي")):
        return Intent("user_name", 0.99)
    if any(phrase in value for phrase in ("ماذا احب", "ما الذي احبه", "ايه اللي بحبه", "هل تتذكر ما احب")):
        return Intent("user_preferences", 0.98)
    return None
