from ron.contracts import ModelRequest, ModelResponse
from ron.runtime import AgentRuntime

class FakeProvider:
    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content=request.messages[0].content, model="fake")

def test_runtime_delegates_to_provider():
    runtime = AgentRuntime(provider=FakeProvider())
    response = runtime.respond("Hello Ron")
    assert response.content == "Hello Ron"
    assert response.model == "fake"
