from ron.contracts import ModelRequest, ModelResponse
from ron.core import RonCore

class FakeProvider:
    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content=request.messages[-1].content, model="fake")

def test_core_injects_memory_and_records_turn():
    core = RonCore(provider=FakeProvider())
    first = core.respond("remember Ron")
    assert first.content == "remember Ron"
    second = core.respond("Ron")
    assert second.content == "Ron"
    assert len(core.memory.items) == 2
