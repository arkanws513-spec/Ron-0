"""Lightweight generation-stability guard for Ron's native character model."""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
_REPEAT_PATTERN_RE = re.compile(r"(.{3,10})\1{2,}", re.DOTALL)


def is_degenerate_text(answer: str) -> bool:
    """Detect symbol-only output, obvious word loops, and repeated character fragments."""
    text = str(answer or "").strip()
    if not text:
        return True

    compact = re.sub(r"\s+", "", text.casefold())
    # Reject punctuation/symbol-only generations while allowing normal punctuation.
    if len(compact) >= 4:
        meaningful = sum(char.isalnum() for char in compact)
        if meaningful / len(compact) < 0.25:
            return True

    # Catch short single-character loops as well as longer repeated fragments.
    if len(compact) >= 6 and len(set(compact)) <= 2:
        return True
    if len(compact) >= 18 and _REPEAT_PATTERN_RE.search(compact):
        return True

    tokens = [token.casefold() for token in _TOKEN_RE.findall(text)]
    if len(tokens) < 6:
        return False
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    dominant_ratio = max(counts.values()) / len(tokens)
    unique_ratio = len(counts) / len(tokens)
    return dominant_ratio >= 0.45 or unique_ratio <= 0.35


def guard_generated_text(answer: str, fallback: str) -> tuple[str, bool]:
    """Return generated text unless empty or obviously repetitive; report rejection."""
    candidate = str(answer or "").strip()
    if not candidate or is_degenerate_text(candidate):
        return fallback, True
    return candidate, False
