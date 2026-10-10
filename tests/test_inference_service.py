from fastapi.testclient import TestClient

import service.app as inference
from ron.contracts import ModelResponse


def test_missing_checkpoint_is_not_ready(monkeypatch):
    baseline = inference.Path(__file__).resolve().parents[1] / "ron" / "checkpoints" / "ron_native_baseline.pt"
    monkeypatch.setattr(inference, "CHECKPOINT_PATH", baseline)
    monkeypatch.setattr(inference, "MIN_PARAMETERS", 1_000_000)

    inference.load_checkpoint()

    assert inference.app.state.provider is None
    assert inference.app.state.checkpoint_status["ready"] is False
    assert "minimum is" in inference.app.state.checkpoint_status["error"]


def test_chat_endpoint_uses_configured_native_provider(monkeypatch):
    class StubNativeProvider:
        def generate(self, request):
            assert request.messages[-1].content == "ما عاصمة مصر؟"
            return ModelResponse(
                content="عاصمة مصر هي القاهرة.",
                model="ron-native-checkpoint",
                metadata={"native_weights_loaded": True, "external_model_used": False},
            )

    def fake_load_checkpoint():
        inference.app.state.provider = StubNativeProvider()
        inference.app.state.checkpoint_status = {
            "ready": True,
            "checkpoint": "reviewed-test-checkpoint.pt",
            "parameter_count": 11_000_000,
            "external_model_used": False,
            "error": None,
        }

    monkeypatch.setattr(inference, "load_checkpoint", fake_load_checkpoint)
    with TestClient(inference.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["parameter_count"] == 11_000_000

        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "ما عاصمة مصر؟"}]},
        )
    assert response.status_code == 200
    assert response.json()["content"] == "عاصمة مصر هي القاهرة."
    assert response.json()["metadata"]["external_model_used"] is False


def test_chat_rejects_empty_or_oversized_messages(monkeypatch):
    def fake_load_checkpoint():
        inference.app.state.provider = object()
        inference.app.state.checkpoint_status = {"ready": True, "parameter_count": 11_000_000}

    monkeypatch.setattr(inference, "load_checkpoint", fake_load_checkpoint)
    with TestClient(inference.app) as client:
        assert client.post("/v1/chat", json={"messages": []}).status_code == 422
        assert client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "x" * 4001}]},
        ).status_code == 422
