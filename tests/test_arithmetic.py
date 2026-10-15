import pytest

from ron.arithmetic import (
    ArithmeticError,
    calculate_from_text,
    evaluate_expression,
    extract_expression,
    is_previous_calculation_request,
)
from ron.core import RonCore
from ron.contracts import ModelRequest, ModelResponse


class FakeProvider:
    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="إجابة من النموذج", model="fake")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("5*5", 25),
        ("458*345", 158010),
        ("(12 + 8) / 2", 10),
        ("-2 + 5", 3),
        ("2**3", 8),
        ("125 / 5", 25),
    ],
)
def test_safe_arithmetic_evaluates_expected_values(expression, expected):
    assert evaluate_expression(expression) == expected


def test_extracts_arabic_digits_and_multiplication_symbol():
    calculation = calculate_from_text("احسب ٤٥٨ × ٣٤٥")
    assert calculation is not None
    assert calculation.expression == "458*345"
    assert calculation.answer == "158010"


@pytest.mark.parametrize("expression", ["__import__('os').system('id')", "2**13", "1/0", "open('x')"])
def test_rejects_unsafe_or_out_of_bounds_expressions(expression):
    with pytest.raises(ArithmeticError):
        evaluate_expression(expression)


def test_expression_extraction_does_not_treat_plain_numbers_as_calculations():
    assert extract_expression("في عام 2026") is None


def test_previous_calculation_phrase_is_recognized():
    assert is_previous_calculation_request("احسب العملية السابقة")
    assert is_previous_calculation_request("Calculate previous calculation")


def test_core_routes_math_and_remembers_previous_calculation():
    core = RonCore(provider=FakeProvider())
    first = core.respond("458 * 345")
    assert first.content == "الناتج: 158010"
    assert first.metadata["tool"] == "safe-arithmetic"

    second = core.respond("احسب العملية السابقة")
    assert second.content == "نتيجة العملية السابقة (458*345) هي 158010."
    assert second.metadata["previous_calculation"] is True


def test_core_asks_for_missing_expression_instead_of_guessing():
    core = RonCore(provider=FakeProvider())
    response = core.respond("وكم تساوي؟")
    assert "ما العملية الحسابية" in response.content
    assert response.metadata["needs_clarification"] is True


def test_previous_calculation_without_prior_result_asks_for_context():
    core = RonCore(provider=FakeProvider())
    response = core.respond("احسب العملية السابقة")
    assert "لا توجد عملية حسابية سابقة" in response.content
    assert response.metadata["needs_clarification"] is True


def test_core_rejects_invalid_arithmetic_without_model_guessing():
    core = RonCore(provider=FakeProvider())
    response = core.respond("احسب 1 / 0")
    assert response.metadata["invalid_expression"] is True
    assert "بأمان" in response.content
