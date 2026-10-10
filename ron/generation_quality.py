"""Lightweight generation-stability guard for Ron's native character model."""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
_COMPACT_RE = re.compile(r"[a-z0-9\u0600-\u06ff]+", re.IGNORECASE)


def _has_repeated_character_fragment(answer: str) -> bool:
    """Catch character-level loops that do not repeat as whole words."""
    compact = "".join(_COMPACT_RE.findall(str(answer).casefold()))
    if len(compact) < 18:
        return False
    # Look for a 3–8 character fragment repeated at least four times.
    # Require it to account for a substantial share of the output to avoid
    # rejecting ordinary sentences that reuse common Arabic/English fragments.
    for width in range(3, min(8, len(compact) // 4) + 1):
        for start in range(len(compact) - width * 4 + 1):
            fragment = compact[start:start + width]
            count = compact.count(fragment)
            if count >= 4 and (width * count) / len(compact) >= 0.35:
                return True
    return False


def is_degenerate_text(answer: str) -> bool:
    """Detect obvious word or character-fragment loops, not factual errors."""
    if _has_repeated_character_fragment(answer):
        return True
    tokens = [token.casefold() for token in _TOKEN_RE.findall(str(answer))]
    if len(tokens) < 6:
        return False
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    dominant_ratio = max(counts.values()) / len(tokens)
    unique_ratio = len(counts) / len(tokens)
    return dominant_ratio >= 0.45 or unique_ratio <= 0.35


def guard_generated_text(answer: str, fallback: str) -> tuple[str, bool]:
    """Return generated text unless empty or obviously repetitive; also report rejection."""
    candidate = str(answer or "").strip()
    if not candidate or is_degenerate_text(candidate):
        return fallback, True
    return candidate, False
