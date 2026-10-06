import torch
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
