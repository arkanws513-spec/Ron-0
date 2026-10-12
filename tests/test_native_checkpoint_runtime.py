import json
from pathlib import Path

from ron.contracts import Message, ModelRequest
from ron.core import RonCore
from ron.native_provider import NativeCheckpointProvider


def test_saved_native_checkpoint_is_used_by_default_core():
    checkpoint_dir = Path(__file__).resolve().parents[1] / "ron" / "checkpoints"
    baseline = checkpoint_dir / "ron_native_baseline.pt"
    promoted_10m = checkpoint_dir / "ron_native_10m.pt"
    assert baseline.is_file(), "trained Ron baseline checkpoint must be present in the repository"

    # The runtime may select Ron-10M only when that checkpoint is present;
    # an experimental continuation must never be placed at this live path.
    expected_checkpoint = promoted_10m if promoted_10m.is_file() else baseline
    core = RonCore()
    assert isinstance(core.provider, NativeCheckpointProvider)
    assert core.provider.checkpoint_path == expected_checkpoint
    assert core.provider.model.training is False
    assert core.provider.config.vocab_size == len(core.provider.vocab)

    response = core.provider.generate(
        ModelRequest(messages=(Message(role="user", content="مرحبا"),), metadata={})
    )
    assert response.metadata["native_weights_loaded"] is True
    assert response.metadata["external_model_used"] is False
    assert 0 < response.metadata["selected_training_step"] <= response.metadata["training_steps_total"]
    assert response.metadata["inference_checkpoint"] in {"best_validation", "final_training_state"}
    assert response.metadata["final_training_step"] >= response.metadata["selected_training_step"]
    assert response.metadata["training_steps_total"] > 0


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
