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
