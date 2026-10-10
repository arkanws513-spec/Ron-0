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
