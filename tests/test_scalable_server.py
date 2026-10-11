from ron.scalable_server import deterministic_math


def test_deterministic_math_handles_common_arithmetic():
    assert deterministic_math("كم يساوي 5 × 5؟") == "25"
    assert deterministic_math("12 / 3") == "4"
    assert deterministic_math("(2 + 3) * 4") == "20"


def test_deterministic_math_rejects_non_math_and_unsafe_expressions():
    assert deterministic_math("اشرح كيف تتعلم الحواسيب") is None
    assert deterministic_math("__import__('os').system('whoami')") is None
    assert deterministic_math("1 / 0") is None
