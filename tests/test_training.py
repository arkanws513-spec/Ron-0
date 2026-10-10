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
    default_count = evaluate(model, tokens)
    explicit_default_count = evaluate(model, tokens, count=16)
    assert first == second
    assert default_count == explicit_default_count
    assert model.training is True

def test_best_validation_checkpoint_survives_a_regressing_training_run():
    from scripts.train_native import choose_best_checkpoint

    state, loss, step, source = choose_best_checkpoint(
        {"weight": "final-training-state"},
        1.40,
        7800,
        {"weight": "prior-best-inference-state"},
        0.98,
        6000,
    )
    assert state == {"weight": "prior-best-inference-state"}
    assert loss == 0.98
    assert step == 6000
    assert source == "prior_best"


def test_best_validation_checkpoint_uses_current_state_when_it_is_better():
    from scripts.train_native import choose_best_checkpoint

    state, loss, step, source = choose_best_checkpoint(
        {"weight": "current-state"},
        0.80,
        7800,
        {"weight": "prior-state"},
        0.98,
        6000,
    )
    assert state == {"weight": "current-state"}
    assert loss == 0.80
    assert step == 7800
    assert source == "current_final"


def test_validation_improvement_does_not_claim_prior_checkpoint_as_this_run_gain():
    from scripts.train_native import validation_improvement_percent, checkpoint_delta_percent

    assert validation_improvement_percent(1.32, 1.21, "prior_best") == 0.0
    assert round(checkpoint_delta_percent(1.32, 1.21), 2) == 8.33
    assert round(validation_improvement_percent(1.32, 1.21, "this_run"), 2) == 8.33


def test_native_training_early_stopping_patience():
    from scripts.train_native import should_stop_early

    assert not should_stop_early(0, patience=3)
    assert not should_stop_early(2, patience=3)
    assert should_stop_early(3, patience=3)


def test_native_training_resumes_from_best_weights_when_corpus_changes(tmp_path, monkeypatch):
    import hashlib
    import torch
    import scripts.train_native as training
    from ron.model import RonCausalLM
    from ron.model_config import RonModelConfig

    old_corpus = "المستخدم: Old question\nرون: Old answer\n"
    new_corpus = "المستخدم: New English question?\nرون: New English answer.\n"
    vocab = {char: index for index, char in enumerate(sorted(set(old_corpus)))}
    config = RonModelConfig(vocab_size=len(vocab), hidden_size=64, num_layers=2, num_heads=4,
                            max_sequence_length=96, dropout=0.1)
    model = RonCausalLM(config)
    final_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_state = {key: value.detach().clone() + 0.25 for key, value in model.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    torch.save({
        "config": config.__dict__, "vocab": vocab, "state_dict": final_state,
        "best_state_dict": best_state, "training_steps_total": 500,
        "selected_step": 500, "best_selected_step": 250,
        "optimizer_state_dict": {"should": "be reset"},
        "corpus_sha256": hashlib.sha256(old_corpus.encode("utf-8")).hexdigest(),
    }, path)
    monkeypatch.setattr(training, "LIVE_CHECKPOINT", path)
    loaded, expanded_vocab, source, prior_step, expanded, optimizer_state, prior_best, prior_best_step = training.load_or_initialize(new_corpus)

    assert source == "continued_best_checkpoint_after_corpus_change"
    assert prior_step == 500
    assert expanded is True
    assert optimizer_state is None
    assert prior_best is None
    assert prior_best_step == 250
    assert len(expanded_vocab) > len(vocab)
    assert torch.equal(loaded.position_embedding.weight, best_state["position_embedding.weight"])
