"""Exact-match retrieval over Ron's reviewed, in-repository dialogue curriculum."""
from __future__ import annotations

import re
from pathlib import Path

from .intent import normalize_arabic

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPORA = (
    ROOT / "training" / "arabic_sft_seed.txt",
    ROOT / "training" / "seed_corpus.txt",
)
_PUNCTUATION = re.compile(r"[؟?!.,،؛;:]+")


def normalize_question(text: str) -> str:
    value = normalize_arabic(str(text or ""))
    value = _PUNCTUATION.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


class CuratedKnowledgeBase:
    """A conservative local response index; only exact normalized prompts are retrieved."""

    def __init__(self, corpora: tuple[Path, ...] = DEFAULT_CORPORA) -> None:
        self.responses: dict[str, str] = {}
        self.sources: dict[str, str] = {}
        for path in corpora:
            if not path.is_file():
                continue
            for block in path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n\n"):
                question = None
                answer = None
                for line in block.splitlines():
                    if ":" not in line:
                        continue
                    role, content = line.split(":", 1)
                    role = role.strip().lower()
                    content = content.strip()
                    if role in {"المستخدم", "user"}:
                        question = content
                    elif role in {"رون", "ron", "assistant"}:
                        answer = content
                if question and answer:
                    key = normalize_question(question)
                    if key:
                        # The Arabic SFT curriculum is listed first and has precedence.
                        self.responses.setdefault(key, answer)
                        self.sources.setdefault(key, str(path.relative_to(ROOT)))

    def answer(self, question: str) -> str | None:
        key = normalize_question(question)
        if not key:
            return None
        return self.responses.get(key)

    def describe(self) -> dict[str, int]:
        return {"curated_prompt_count": len(self.responses)}
