from scripts.evaluate_native_checkpoint import assess_case, classify_language


def test_classifies_english_question_inside_arabic_instruction_wrapper():
    case = {
        "prompt": "أجب بدقة وباختصار عن السؤال التالي: What is the capital of Egypt?",
        "contains": ["Cairo"],
    }
    assert classify_language(case) == "en"


def test_classifies_arabic_question_inside_arabic_instruction_wrapper():
    case = {
        "prompt": "اقرأ السؤال جيدًا ثم أعط إجابة مباشرة: ما عاصمة مصر؟",
        "contains": ["القاهرة"],
    }
    assert classify_language(case) == "ar"


def test_explicit_language_label_overrides_prompt_heuristic():
    assert classify_language({"language": "en", "prompt": "ما عاصمة مصر؟"}) == "en"
    assert classify_language({"language": "ar", "prompt": "What is the capital of Egypt?"}) == "ar"


def test_assessment_marks_language_and_rejects_repeated_gibberish():
    case = {
        "id": "arabic-loop",
        "prompt": "ما عاصمة مصر؟",
        "contains": ["القاهرة"],
        "critical": True,
    }
    result = assess_case("الالالالا", case)
    assert result["language"] == "ar"
    assert result["passed"] is False
    assert result["degenerate"] is True
