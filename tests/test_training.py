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



def test_native_corpus_split_holds_out_complete_pairs_reproducibly():
    from scripts.train_native import split_corpus

    corpus = (
        "المستخدم: سؤال ألف؟\nرون: إجابة ألف.\n"
        "المستخدم: سؤال باء؟\nرون: إجابة باء.\n"
        "المستخدم: سؤال جيم؟\nرون: إجابة جيم.\n"
        "المستخدم: سؤال دال؟\nرون: إجابة دال.\n"
    )
    train_text, validation_text, train_count, validation_count = split_corpus(corpus, seed=19)
    repeated = split_corpus(corpus, seed=19)
    assert (train_text, validation_text, train_count, validation_count) == repeated
    assert train_count + validation_count == 4
    assert train_count >= 1 and validation_count >= 1
    train_pairs = set(train_text.strip().split("\nالمستخدم: "))
    validation_pairs = set(validation_text.strip().split("\nالمستخدم: "))
    assert train_pairs.isdisjoint(validation_pairs)
    assert all(line.startswith(("المستخدم: ", "رون: ")) for line in train_text.splitlines())
    assert all(line.startswith(("المستخدم: ", "رون: ")) for line in validation_text.splitlines())


def test_native_corpus_split_rejects_malformed_pairs():
    import pytest
    from scripts.train_native import split_corpus

    with pytest.raises(ValueError):
        split_corpus("المستخدم: سؤال بلا إجابة\nرون: إجابة\nسطر غير صالح\n")
    with pytest.raises(ValueError):
        split_corpus("المستخدم: واحد\nرون: واحد\n", train_fraction=1.0)

def test_native_evaluation_is_deterministic_and_restores_training_mode():
    import torch
    from scripts.train_native import evaluate

    config = RonModelConfig(vocab_size=16, hidden_size=16, num_layers=1, num_heads=4,
                            max_sequence_length=96, dropout=0.4)
    model = RonCausalLM(config)
    model.train()
    tokens = torch.arange(0, 512, dtype=torch.long) % config.vocab_size
    first = evaluate(model, tokens, count=8)
    second = evaluate(model, tokens, count=8)
    assert first == second
    assert model.training is True
