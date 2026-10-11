from ron.scalable_server import (
    deterministic_math,
    deterministic_social_reply,
    load_seed_responses,
    normalize_lookup_text,
)


def test_deterministic_math_handles_common_arithmetic():
    assert deterministic_math("كم يساوي 5 × 5؟") == "25"
    assert deterministic_math("12 / 3") == "4"
    assert deterministic_math("(2 + 3) * 4") == "20"


def test_deterministic_math_rejects_non_math_and_unsafe_expressions():
    assert deterministic_math("اشرح كيف تتعلم الحواسيب") is None
    assert deterministic_math("__import__('os').system('whoami')") is None
    assert deterministic_math("1 / 0") is None


def test_generation_quality_guard_rejects_symbol_only_output():
    from ron.generation_quality import guard_generated_text, is_degenerate_text

    fallback = "Ron could not form a clear answer."
    assert is_degenerate_text("********")
    assert guard_generated_text("********", fallback) == (fallback, True)
    assert guard_generated_text("aaaaaaaa", fallback) == (fallback, True)
    assert guard_generated_text("The sky looks blue.", fallback) == ("The sky looks blue.", False)


def test_deterministic_social_replies_cover_common_greetings_without_inference():
    assert "أهلًا" in deterministic_social_reply("اهلا")
    assert "وعليكم السلام" in deterministic_social_reply("السلام عليكم")
    assert "جاهز أساعدك" in deterministic_social_reply("كيف حالك")
    assert "Hello" in deterministic_social_reply("hello")
    assert "ready to help" in deterministic_social_reply("how are you")
    assert deterministic_social_reply("اشرح لي الجاذبية") is None


def test_seed_corpus_lookup_normalizes_arabic_question_marks_and_diacritics(tmp_path):
    corpus = tmp_path / "seed_corpus.txt"
    corpus.write_text(
        "المستخدم: لماذا السماء تبدو زرقاء؟\n"
        "رون: يتشتت الضوء الأزرق في الغلاف الجوي.\n",
        encoding="utf-8",
    )
    responses = load_seed_responses(corpus)
    key = normalize_lookup_text("لماذا السماء تبدو زرقاء؟")
    assert responses[key] == "يتشتت الضوء الأزرق في الغلاف الجوي."
    assert responses.get(normalize_lookup_text("لماذا السماء تبدو زرقاء")) == responses[key]
