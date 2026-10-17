from scripts.evaluate_qwen_lora import assess, classify_language, contains_text, normalize


def test_ascii_required_words_use_token_boundaries():
    answer = normalize("Knowledge is useful.")
    assert not contains_text(answer, "no")
    assert contains_text(normalize("No, that is incorrect."), "no")


def test_arabic_required_phrase_matches_normalized_text():
    answer = "عاصمة مصر هي القاهرة."
    result = assess(answer, {
        "id": "arabic-capital",
        "language": "ar",
        "contains": ["القاهرة"],
    })
    assert result["passed"] is True
    assert result["missing_required"] == []


def test_evaluator_rejects_missing_required_word_even_if_substring_exists():
    result = assess("Knowledge is useful.", {
        "id": "token-boundary",
        "language": "en",
        "contains": ["no"],
    })
    assert result["passed"] is False
    assert result["missing_required"] == ["no"]


def test_evaluator_detects_forbidden_ascii_word_at_boundary():
    result = assess("This is a safe answer.", {
        "id": "forbidden",
        "language": "en",
        "contains": ["safe"],
        "forbidden": ["unsafe"],
    })
    assert result["passed"] is True
    assert result["forbidden_found"] == []



def test_explicit_language_label_overrides_prompt_heuristic():
    assert classify_language({"language": "ar", "prompt": "What is 2+2?"}) == "ar"
    assert classify_language({"language": "en", "prompt": "ما عاصمة مصر؟"}) == "en"
