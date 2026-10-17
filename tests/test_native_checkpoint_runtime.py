import json
from pathlib import Path

from ron.contracts import Message, ModelRequest
from ron.core import RonCore
from ron.native_provider import NativeCheckpointProvider


def test_default_core_uses_the_new_qwen_provider(monkeypatch):
    import ron.llm_provider as llm_provider

    class FakeQwenProvider:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        model_path = "fake-model.gguf"

        def generate(self, request):
            from ron.contracts import ModelResponse
            return ModelResponse(
                content="مرحبًا، أنا رون.",
                model=self.model_id,
                metadata={"local_weights": True, "external_inference_api": False},
            )

    monkeypatch.setattr(llm_provider, "QwenChatProvider", FakeQwenProvider)
    core = RonCore()
    assert isinstance(core.provider, FakeQwenProvider)
    response = core.provider.generate(
        ModelRequest(messages=(Message(role="user", content="مرحبا"),), metadata={})
    )
    assert response.metadata["local_weights"] is True
    assert response.metadata["external_inference_api"] is False


def test_native_provider_builds_bounded_prompt_from_recent_dialogue():
    checkpoint = Path(__file__).resolve().parents[1] / "ron" / "checkpoints" / "ron_native_baseline.pt"
    provider = NativeCheckpointProvider(checkpoint)
    from ron.contracts import Message, ModelRequest

    request = ModelRequest(messages=(
        Message(role="system", content="large system context that is intentionally not copied into the character prompt"),
        Message(role="user", content="ما الفرق بين التدريب والذاكرة؟"),
        Message(role="assistant", content="التدريب يغير الأوزان."),
        Message(role="user", content="اشرح أكثر"),
    ), metadata={})
    prompt = provider._build_dialogue_prompt(request, "اشرح أكثر")
    assert len(prompt) <= provider.config.max_sequence_length
    assert prompt.endswith("\nرون:")
    assert "ما الفرق بين التدريب والذاكرة؟" in prompt
    assert "اشرح أكثر" in prompt


def test_training_metrics_prove_weights_changed_and_vocab_matches_checkpoint():
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / "ron" / "checkpoints" / "metrics.json").read_text(encoding="utf-8"))
    provider = NativeCheckpointProvider(root / "ron" / "checkpoints" / "ron_native_baseline.pt")
    assert metrics["status"] == "completed"
    assert metrics["weights_persisted_from_final_training_step"] is True
    assert metrics["best_validation_weights_persisted"] is True
    assert metrics["changed_parameter_tensors"] > 0
    assert metrics["parameter_delta_l2"] > 0
    assert metrics["vocab_size"] == provider.config.vocab_size == len(provider.vocab)
    assert metrics["training_steps_total"] == provider.training_steps_total
    assert provider.inference_checkpoint == "best_validation"

def test_native_generation_guard_rejects_obvious_token_loops():
    from ron.generation_quality import guard_generated_text, is_degenerate_text

    bad = "أحدد المعلومات المعلومات المعلومات المعلومات المعلومات المعلومات"
    safe = "أراجع المعطيات ثم أشرح النتيجة باختصار واضح."
    assert is_degenerate_text(bad)
    assert not is_degenerate_text(safe)
    fallback, rejected = guard_generated_text(bad, "تعذر توليد إجابة موثوقة.")
    assert rejected is True
    assert fallback == "تعذر توليد إجابة موثوقة."
    assert guard_generated_text(safe, "fallback") == (safe, False)
