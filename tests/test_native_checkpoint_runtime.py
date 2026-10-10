from pathlib import Path

from ron.contracts import Message, ModelRequest
from ron.core import RonCore
from ron.native_provider import NativeCheckpointProvider


def test_saved_native_checkpoint_is_used_by_default_core():
    checkpoint = Path(__file__).resolve().parents[1] / "ron" / "checkpoints" / "ron_native_baseline.pt"
    assert checkpoint.is_file(), "trained Ron checkpoint must be present in the repository"

    core = RonCore()
    assert isinstance(core.provider, NativeCheckpointProvider)

    response = core.provider.generate(
        ModelRequest(messages=(Message(role="user", content="مرحبا"),), metadata={})
    )
    assert response.metadata["native_weights_loaded"] is True
    assert response.metadata["external_model_used"] is False
    assert response.metadata["selected_training_step"] >= 0
    assert response.metadata["training_steps_this_run"] >= 600
    assert response.metadata["checkpoint_source"] != "unknown"
