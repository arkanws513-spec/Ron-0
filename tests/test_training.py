import tempfile
from ron.model import RonCausalLM
from ron.model_config import RonModelConfig
from ron.training import load_checkpoint, save_checkpoint, seed_everything, train_steps

def test_tiny_training_reduces_loss():
    seed_everything(7)
    config = RonModelConfig(vocab_size=16, hidden_size=16, num_layers=1, num_heads=4, max_sequence_length=8)
    model = RonCausalLM(config)
    result = train_steps(model, [1,2,3,4,5,6,7,8] * 8, steps=30, sequence_length=7, batch_size=4, learning_rate=0.01)
    assert result.final_loss < result.initial_loss

def test_checkpoint_roundtrip():
    config = RonModelConfig(vocab_size=12, hidden_size=12, num_layers=1, num_heads=3, max_sequence_length=8)
    model = RonCausalLM(config)
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/ron.pt"
        save_checkpoint(model, path)
        restored = RonCausalLM(config)
        load_checkpoint(restored, path)
        assert all(left.equal(right) for left, right in zip(model.parameters(), restored.parameters()))
