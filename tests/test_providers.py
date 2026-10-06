from ron.contracts import ModelRequest
from ron.providers import LocalTeachingProvider, ProviderRouter

def test_local_provider_needs_no_external_key():
    provider = LocalTeachingProvider()
    response = provider.generate(ModelRequest(messages=()))
    assert response.model == "ron-local"
    assert response.metadata["external_api_required"] is False

def test_router_requires_explicit_provider_registration():
    router = ProviderRouter()
    router.register("local", LocalTeachingProvider())
    assert router.resolve().name == "ron-local"


def test_local_http_provider_builds_openai_compatible_request(monkeypatch):
    import json

    from ron.contracts import Message, ModelRequest
    from ron.providers import LocalHTTPModelProvider

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self):
            return json.dumps({
                "model": "test-model",
                "choices": [{"message": {"content": "مرحبا من النموذج"}}],
            }).encode()

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("ron.providers.urlopen", fake_urlopen)
    provider = LocalHTTPModelProvider(endpoint="http://127.0.0.1:1234/v1/chat/completions", model="test-model")
    response = provider.generate(ModelRequest(messages=(Message(role="user", content="مرحبا"),)))

    assert response.content == "مرحبا من النموذج"
    assert response.metadata["external_api_required"] is False
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["messages"][0]["content"] == "مرحبا"


def test_qwen3_teacher_is_explicit_and_local(monkeypatch):
    from ron.providers import Qwen3TeacherProvider

    monkeypatch.setenv("RON_TEACHER_MODEL", "qwen3:1.7b")
    teacher = Qwen3TeacherProvider.from_environment()
    assert teacher.model == "qwen3:1.7b"
    assert teacher.endpoint.startswith("http://127.0.0.1:")
