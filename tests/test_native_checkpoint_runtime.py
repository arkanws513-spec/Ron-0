import json
from pathlib import Path

from ron.contracts import Message, ModelRequest
from ron.core import RonCore
from ron.native_provider import NativeCheckpointProvider


def test_saved_native_checkpoint_is_used_by_default_core():
    checkpoint = Path(__file__).resolve().parents[1] / "ron" / "checkpoints" / "ron_native_baseline.pt"
    assert checkpoint.is_file(), "trained Ron checkpoint must be present in the repository"

    core = RonCore()
    assert isinstance(core.provider, NativeCheckpointProvider)
    assert core.provider.checkpoint_path == checkpoint
    assert core.provider.model.training is False
    assert core.provider.config.vocab_size == len(core.provider.vocab)

    response = core.provider.generate(
        ModelRequest(messages=(Message(role="user", content="مرحبا"),), metadata={})
    )
    assert response.metadata["native_weights_loaded"] is True
    assert response.metadata["external_model_used"] is False
    assert 0 < response.metadata["selected_training_step"] <= response.metadata["training_steps_total"]
    assert response.metadata["inference_checkpoint"] == "best_validation"
    assert response.metadata["final_training_step"] >= response.metadata["selected_training_step"]
    assert response.metadata["training_steps_this_run"] >= 1800
    assert response.metadata["training_steps_total"] >= 2400
    assert response.metadata["optimizer_state_persisted"] is True
    assert response.metadata["checkpoint_source"] == "continued_existing_ron_checkpoint"


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
