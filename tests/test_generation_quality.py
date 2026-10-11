from ron.generation_quality import guard_generated_text, is_degenerate_text


def test_detects_repeated_arabic_character_fragment_without_spaces():
    answer = "المالمال الم المالمالمالم المالمالم المالم"
    assert is_degenerate_text(answer)
    guarded, rejected = guard_generated_text(answer, "fallback")
    assert rejected
    assert guarded == "fallback"


def test_keeps_normal_short_arabic_answer():
    answer = "اسمي رون، وأنا مساعد محلي قيد التطوير."
    assert not is_degenerate_text(answer)
    assert guard_generated_text(answer, "fallback") == (answer, False)


def test_keeps_normal_english_answer():
    answer = "The capital of Egypt is Cairo."
    assert not is_degenerate_text(answer)
