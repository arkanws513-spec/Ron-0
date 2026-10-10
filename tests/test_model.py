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
