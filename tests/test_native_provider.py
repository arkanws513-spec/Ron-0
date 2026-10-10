from __future__ import annotations

import torch

from ron.contracts import Message, ModelRequest
from ron.model import RonCausalLM
from ron.model_config import RonModelConfig
from ron.native_provider import NativeCheckpointProvider


def test_native_provider_loads_checkpoint_weights_and_generates(tmp_path):
    vocab = {char: index for index, char in enumerate(" UserRon:مرحبا\n")}
    config = RonModelConfig(
        vocab_size=len(vocab), hidden_size=8, num_layers=1, num_heads=2,
        max_sequence_length=32, dropout=0.0,
    )
    model = RonCausalLM(config)
    checkpoint = tmp_path / "native.pt"
    torch.save({
        "config": {
            "vocab_size": config.vocab_size, "hidden_size": config.hidden_size,
            "num_layers": config.num_layers, "num_heads": config.num_heads,
            "max_sequence_length": config.max_sequence_length, "dropout": config.dropout,
        },
        "vocab": vocab,
        "state_dict": model.state_dict(),
        "best_state_dict": model.state_dict(),
        "selected_step": 12,
        "best_selected_step": 9,
        "training_steps_total": 12,
        "checkpoint_source": "test-fixture",
    }, checkpoint)

    provider = NativeCheckpointProvider(checkpoint)
    response = provider.generate(ModelRequest(messages=(Message(role="user", content="مرحبا"),), metadata={}))
    assert response.model == "ron-native-checkpoint"
    assert response.metadata["native_weights_loaded"] is True
    assert response.metadata["external_model_used"] is False
    assert response.metadata["selected_training_step"] == 9
    assert response.metadata["inference_checkpoint"] == "best_validation"


def test_native_provider_rejects_corrupt_vocabulary(tmp_path):
    vocab = {"a": 0, "b": 2}
    config = {"vocab_size": 2, "hidden_size": 8, "num_layers": 1, "num_heads": 2, "max_sequence_length": 32, "dropout": 0.0}
    checkpoint = tmp_path / "corrupt.pt"
    torch.save({"config": config, "vocab": vocab, "state_dict": {}}, checkpoint)
    try:
        NativeCheckpointProvider(checkpoint)
        assert False, "corrupt vocabulary unexpectedly accepted"
    except ValueError as exc:
        assert "unique and contiguous" in str(exc)
