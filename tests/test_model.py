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
    model = RonCausalLM(config()).eval()
    prefix = torch.tensor([[1, 2, 3]])
    generated = model.generate(prefix, max_new_tokens=4)
    assert generated.shape == (1, 7)
    assert torch.equal(generated[:, :3], prefix)


def test_cached_and_uncached_logits_match():
    torch.manual_seed(21)
    model = RonCausalLM(config()).eval()
    ids = torch.randint(0, 32, (2, 9))
    full = model(ids).logits
    first = model(ids[:, :6], use_cache=True)
    next_out = model(ids[:, 6:], past_key_values=first.past_key_values, use_cache=True)
    assert first.past_key_values is not None
    assert torch.allclose(full[:, 6:, :], next_out.logits, atol=1e-5, rtol=1e-5)


def test_cached_and_reference_greedy_generation_match():
    torch.manual_seed(22)
    model = RonCausalLM(config()).eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    cached = model.generate(prefix, max_new_tokens=8, use_cache=True)
    reference = model.generate(prefix, max_new_tokens=8, use_cache=False)
    assert torch.equal(cached, reference)


def test_cache_context_window_rollover_matches_reference():
    torch.manual_seed(23)
    cfg = RonModelConfig(vocab_size=32, hidden_size=16, num_layers=2, num_heads=4, max_sequence_length=8)
    model = RonCausalLM(cfg).eval()
    prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
    cached = model.generate(prefix, max_new_tokens=7, use_cache=True)
    reference = model.generate(prefix, max_new_tokens=7, use_cache=False)
    assert torch.equal(cached, reference)


def test_sampled_generation_preserves_prefix_and_length():
    model = RonCausalLM(config()).eval()
    prefix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    torch.manual_seed(7)
    generated = model.generate(prefix, max_new_tokens=5, temperature=0.8, top_k=8, top_p=0.9)
    assert generated.shape == (2, 8)
    assert torch.equal(generated[:, :3], prefix)


def test_generation_stops_when_all_sequences_emit_eos():
    model = RonCausalLM(config()).eval()
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
