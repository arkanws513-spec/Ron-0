import torch

from ron.contracts import Message, ModelRequest
from ron.model import RonCausalLM
from ron.model_config import RonModelConfig
from ron.native_provider import NativeCheckpointProvider


def test_native_generation_is_reproducible_with_filtered_sampling(tmp_path):
    torch.manual_seed(23)
    vocab_chars = sorted(set("User: hi\nRon:"))
    vocab = {char: index for index, char in enumerate(vocab_chars)}
    config = RonModelConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        max_sequence_length=32,
        dropout=0.0,
    )
    model = RonCausalLM(config)
    checkpoint = tmp_path / "tiny_native.pt"
    torch.save(
        {
            "config": {
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "max_sequence_length": config.max_sequence_length,
                "dropout": config.dropout,
            },
            "vocab": vocab,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )

    provider = NativeCheckpointProvider(checkpoint)
    request = ModelRequest(messages=[Message(role="user", content="hi")])
    first = provider.generate(request)
    second = provider.generate(request)

    assert first.content == second.content
    assert first.metadata["generation_strategy"] == "deterministic_top_k_sampling"
    assert first.metadata["generation_temperature"] == 0.8
    assert first.metadata["generation_top_k"] == min(8, config.vocab_size)
    assert first.metadata["external_model_used"] is False
