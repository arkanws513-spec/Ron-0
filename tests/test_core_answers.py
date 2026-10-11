"""Regression tests for Ron's explicit high-confidence core response layer."""
from __future__ import annotations

import pytest

from ron.contracts import Message, ModelRequest
from ron.native_provider import NativeCheckpointProvider


@pytest.fixture
def provider() -> NativeCheckpointProvider:
    # Core-response tests do not need to load a checkpoint or allocate model weights.
    return object.__new__(NativeCheckpointProvider)


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("من أنت؟", "أنا رون"),
        ("هل اسمك رون؟", "أنا رون"),
        ("ما عاصمة مصر؟", "القاهرة"),
        ("ما هي عاصمة مصر؟", "القاهرة"),
        ("مصر عاصمتها القاهرة، صحيح؟", "القاهرة"),
        ("ما عاصمة فرنسا؟", "باريس"),
        ("ما عاصمة المملكة المتحدة؟", "لندن"),
        ("ما عاصمة دولة زيرافيا الخيالية؟", "لا أعرف"),
        ("كم يساوي 9 ضرب 7؟", "63"),
        ("كم يساوي 125 مقسومًا على 5؟", "25"),
        ("حل المعادلة 2x + 4 = 10.", "x = 3"),
        ("ترجم كلمة reliable إلى العربية.", "موثوق"),
        ("ماذا يعني السياق في المحادثة؟", "السياق"),
        ("ما الفرق بين الحقيقة والرأي؟", "الحقيقة"),
        ("هل انخفاض خسارة التدريب يضمن الذكاء؟", "لا"),
        ("كيف أكتب دالة Python تجمع رقمين؟", "a + b"),
        ("كيف تتحقق من خبر؟", "مصدر"),
        ("Who are you?", "I am Ron"),
        ("What should you do if you do not know an answer?", "don't know"),
    ],
)
def test_core_answers_cover_reliable_basics(
    provider: NativeCheckpointProvider, prompt: str, expected: str
) -> None:
    request = ModelRequest(messages=(Message(role="user", content=prompt),))
    answer = provider._known_core_answer(request, prompt)
    assert answer is not None
    assert expected.casefold() in answer.casefold()


def test_followup_uses_conversation_context(provider: NativeCheckpointProvider) -> None:
    request = ModelRequest(
        messages=(
            Message(role="user", content="تذكر أن عاصمة مصر هي القاهرة."),
            Message(role="assistant", content="فهمت، عاصمة مصر هي القاهرة."),
            Message(role="user", content="ماذا فهمت؟"),
        )
    )
    answer = provider._known_core_answer(request, "ماذا فهمت؟")
    assert answer is not None
    assert "القاهرة" in answer


def test_short_confirmation_resolves_prior_topic(provider: NativeCheckpointProvider) -> None:
    request = ModelRequest(
        messages=(
            Message(role="user", content="هل تستطيع شرح معنى الاستدلال؟"),
            Message(role="assistant", content="نعم، أستطيع شرحه."),
            Message(role="user", content="نعم، افعل ذلك."),
        )
    )
    answer = provider._known_core_answer(request, "نعم، افعل ذلك.")
    assert answer is not None
    assert "الاستدلال" in answer
