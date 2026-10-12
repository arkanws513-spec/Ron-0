from ron.generation_quality import guard_generated_text, is_degenerate_text
from scripts.ron_eval_200 import lexical_quality, normalize


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


def test_lexical_gate_rejects_gibberish_tokens():
    vocabulary = {"الظل", "خلال", "اليوم", "because", "plants", "need", "light"}
    answer = "الوالالمالالأنر والالحد وقوالت والسية"
    result = lexical_quality(answer, vocabulary)
    assert not result["lexical_quality_passed"]
    assert result["known_corpus_word_count"] < 3


def test_lexical_gate_accepts_corpus_supported_sentence():
    vocabulary = {"plants", "need", "sunlight", "because", "light", "helps", "grow"}
    result = lexical_quality("Plants need sunlight because light helps grow.", vocabulary)
    assert result["lexical_quality_passed"]
    assert result["lexical_coverage"] == 1.0


def test_normalize_arabic_diacritics_and_variants():
    assert normalize("إجابةٌ صحيحة") == normalize("اجابه صحيحه")
