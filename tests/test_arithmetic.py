import pytest

from ron.arithmetic import answer_arithmetic
from ron.contracts import ModelRequest, ModelResponse
from ron.core import RonCore


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("5*5", "الناتج: 25"),
        ("كم يساوي ٥ × ٥؟", "الناتج: 25"),
        ("ما ناتج 9 ضرب 7؟", "الناتج: 63"),
        ("احسب 125 في 50", "الناتج: 6250"),
        ("10 ناقص 3", "الناتج: 7"),
        ("20 مقسوم على 4", "الناتج: 5"),
        ("(2 + 3) * 4", "الناتج: 20"),
        ("What is 12 divided by 3?", "الناتج: 4"),
        ("5 + 2 * 3", "الناتج: 11"),
    ],
)
def test_arithmetic_examples(prompt, expected):
    assert answer_arithmetic(prompt) == expected


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "ما عاصمة مصر؟",
        "اشرح لي معنى الاستدلال",
        "5 / 0",
        "__import__('os').system('echo unsafe')",
        "2 ** 100",
        "1" * 130,
    ],
)
def test_non_arithmetic_or_unsupported_inputs_are_not_evaluated(prompt):
    assert answer_arithmetic(prompt) is None


def test_core_routes_arithmetic_without_calling_language_model():
    class MustNotRunProvider:
        def generate(self, request: ModelRequest) -> ModelResponse:
            raise AssertionError("arithmetic should be handled before language generation")

    core = RonCore(provider=MustNotRunProvider())
    response = core.respond("كم يساوي 5*5؟")
    assert response.content == "الناتج: 25"
    assert response.model == "ron-arithmetic"
    assert response.metadata["deterministic_tool"] == "arithmetic"
    assert response.metadata["external_model_used"] is False
