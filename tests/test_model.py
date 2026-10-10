import torch
import pytest
from ron.model import RonCausalLM
from ron.model_config import RonModelConfig


def config():
    return RonModelConfig(vocab_size=32, hidden_size=16, num_layers=2, num_heads=4, max_sequence_length=16)


def test_forward_shape_and_loss():
    model = RonCausalLM(config())
    ids = torch.randint(0, 32, (2, 8))
    output = model(ids, ids)
    assert output.logits.shape == (2, 8, 32)
    assert output.loss is not None and torch.isfinite(output.loss)


def test_generation_preserves_prefix():
    model = RonCausalLM(config())
    prefix = torch.tensor([[1, 2, 3]])
    generated = model.generate(prefix, max_new_tokens=4)
    assert generated.shape == (1, 7)
    assert torch.equal(generated[:, :3], prefix)


def test_sampled_generation_preserves_prefix_and_length():
    model = RonCausalLM(config()).eval()
    prefix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    torch.manual_seed(7)
    generated = model.generate(prefix, max_new_tokens=5, temperature=0.8, top_k=8, top_p=0.9)
    assert generated.shape == (2, 8)
    assert torch.equal(generated[:, :3], prefix)


def test_generation_stops_when_all_sequences_emit_eos():
    model = RonCausalLM(config()).eval()
    # top_k=1 makes sampled generation select the most likely token, regardless of temperature.
    prefix = torch.tensor([[1, 2]])
    greedy = model.generate(prefix, max_new_tokens=1)
    eos = int(greedy[0, -1])
    generated = model.generate(prefix, max_new_tokens=8, temperature=1.0, top_k=1, eos_token_id=eos)
    assert generated.shape == (1, 3)
    assert int(generated[0, -1]) == eos


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"top_k": 0},
        {"top_k": 33},
        {"top_p": 0},
        {"top_p": 1.1},
        {"eos_token_id": 32},
    ],
)
def test_generation_rejects_invalid_sampling_options(kwargs):
    model = RonCausalLM(config())
    with pytest.raises(ValueError):
        model.generate(torch.tensor([[1, 2]]), **kwargs)


def test_dropout_regularization_is_active_only_during_training():
    cfg = RonModelConfig(vocab_size=32, hidden_size=16, num_layers=1, num_heads=4,
                         max_sequence_length=16, dropout=0.5)
    model = RonCausalLM(cfg)
    ids = torch.randint(0, 32, (1, 8))
    model.train()
    torch.manual_seed(1)
    train_a = model(ids).logits
    torch.manual_seed(2)
    train_b = model(ids).logits
    assert not torch.equal(train_a, train_b)
    model.eval()
    eval_a = model(ids).logits
    eval_b = model(ids).logits
    assert torch.equal(eval_a, eval_b)
