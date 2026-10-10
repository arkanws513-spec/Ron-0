"""Conservative deterministic routes for tasks that should not need language generation.

These rules belong to Ron's core, not its learned weights. They handle a small,
auditable set of exact facts and elementary arithmetic; unmatched questions still
go to the configured model provider.
"""
from __future__ import annotations

import re
from fractions import Fraction


def _normalize(text: str) -> str:
    value = str(text or "").strip().lower()
    value = value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    value = value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    value = re.sub(r"[ًٌٍَُِّْـ]", "", value)
    return re.sub(r"\s+", " ", value).strip(" \t\r\n؟?!.،,")


_CAPITALS = (
    (("ما عاصمة مصر", "ما هي عاصمة مصر", "مصر عاصمتها القاهرة", "what is the capital of egypt"),
     "عاصمة مصر هي القاهرة."),
    (("ما عاصمة فرنسا", "ما هي عاصمة فرنسا", "what is the capital of france"),
     "عاصمة فرنسا هي باريس."),
    (("ما عاصمة المملكة المتحدة", "ما عاصمة بريطانيا", "what is the capital of the united kingdom"),
     "عاصمة المملكة المتحدة هي لندن."),
)


def _arithmetic(value: str) -> str | None:
    """Safely evaluate only two-number elementary arithmetic; never use eval()."""
    number = r"(-?\d+(?:\.\d+)?)"
    patterns = (
        (rf"^{number}\s*(?:\+|جمع|زائد|plus)\s*{number}$", "+"),
        (rf"^{number}\s*(?:-|ناقص|minus)\s*{number}$", "-"),
        (rf"^{number}\s*(?:\*|×|x|ضرب|مضروب في|times)\s*{number}$", "*"),
        (rf"^{number}\s*(?:/|÷|مقسوم على|على|divided by)\s*{number}$", "/"),
    )
    for pattern, operation in patterns:
        match = re.fullmatch(pattern, value)
        if not match:
            continue
        left, right = (Fraction(item) for item in match.groups())
        if operation == "+":
            result = left + right
        elif operation == "-":
            result = left - right
        elif operation == "*":
            result = left * right
        else:
            if right == 0:
                return "لا يمكن القسمة على صفر."
            result = left / right
        if result.denominator == 1:
            rendered = str(result.numerator)
        else:
            rendered = f"{float(result):.8g}"
        return f"النتيجة: {rendered}"
    equation = re.fullmatch(r"حل المعادلة\s+(-?\d+)\s*x\s*([+-])\s*(-?\d+)\s*=\s*(-?\d+)", value)
    if equation:
        coefficient, sign, offset, target = equation.groups()
        a, b, c = int(coefficient), int(offset), int(target)
        if sign == "-":
            b = -b
        if a == 0:
            return "المعادلة لا تحدد قيمة وحيدة لـ x." if b == c else "لا يوجد حل لهذه المعادلة."
        answer = Fraction(c - b, a)
        rendered = str(answer.numerator) if answer.denominator == 1 else f"{float(answer):.8g}"
        return f"x = {rendered}"
    return None


def answer_deterministic(text: str) -> tuple[str, str] | None:
    """Return (answer, route) only for conservative, locally verifiable matches."""
    value = _normalize(text)
    if not value:
        return None

    if value in {"من انت", "من انت يا رون", "ما اسمك", "هل اسمك رون", "هل انت رون", "who are you", "what is your name"}:
        if any("a" <= char <= "z" for char in value):
            return "I'm Ron, an independent assistant built around Ron's own core and local model.", "identity"
        return "أنا رون، مساعد مستقل له نواة محلية ونموذج خاص بالمشروع.", "identity"

    arithmetic = _arithmetic(value)
    if arithmetic is not None:
        return arithmetic, "arithmetic"

    for prompts, answer in _CAPITALS:
        if value in {_normalize(prompt) for prompt in prompts}:
            return answer, "verified_fact"

    if value in {
        "ماذا تفعل اذا لم تعرف الاجابة",
        "ماذا تفعل اذا لم تعرف الاجابه",
        "ماذا تفعل اذا لم تعرف",
        "what should you do if you do not know an answer",
        "what should you do if you don't know an answer",
    }:
        return "إذا لم أعرف الإجابة، سأوضح أنني لا أعرف بدلًا من اختلاق معلومة، ويمكنني اقتراح طريقة للتحقق منها.", "uncertainty_policy"

    return None
