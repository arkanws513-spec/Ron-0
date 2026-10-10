from ron.contracts import ModelRequest, ModelResponse
from ron.core import RonCore


class CountingProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return ModelResponse(content="model answer", model="test-provider", metadata={})


def test_core_routes_arabic_arithmetic_without_calling_model():
    provider = CountingProvider()
    core = RonCore(provider=provider)
    response = core.respond("كم يساوي 9 ضرب 7؟")
    assert response.content == "النتيجة: 63"
    assert response.model == "ron-deterministic-core"
    assert response.metadata["deterministic_route"] == "arithmetic"
    assert response.metadata["external_model_used"] is False
    assert provider.calls == 0


def test_core_routes_division_and_linear_equations():
    provider = CountingProvider()
    core = RonCore(provider=provider)
    assert core.respond("125 مقسومًا على 5").content == "النتيجة: 25"
    assert core.respond("حل المعادلة 2x + 4 = 10.").content == "x = 3"
    assert provider.calls == 0


def test_core_answers_known_capitals_and_identity_deterministically():
    provider = CountingProvider()
    core = RonCore(provider=provider)
    assert "القاهرة" in core.respond("ما هي عاصمة مصر؟").content
    assert "رون" in core.respond("من أنت؟").content
    assert provider.calls == 0


def test_core_handles_division_by_zero_safely():
    provider = CountingProvider()
    response = RonCore(provider=provider).respond("10 مقسوم على 0")
    assert response.content == "لا يمكن القسمة على صفر."
    assert provider.calls == 0


def test_unmatched_question_still_reaches_configured_provider():
    provider = CountingProvider()
    response = RonCore(provider=provider).respond("اشرح مفهوم الاستدلال بالتفصيل")
    assert response.content == "model answer"
    assert provider.calls == 1
