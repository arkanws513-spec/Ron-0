from __future__ import annotations

import pytest

from ron.tokenizer import RonTokenizer


def test_bpe_tokenizer_round_trip_for_arabic_and_unicode():
    tokenizer = RonTokenizer().fit(["مرحبا بالعالم\nUser: Hello! 😀", "ما عاصمة مصر؟"], vocab_size=512)
    sample = "أهلاً، رون!\nرموز: 😀 العربية ١٢٣"
    ids = tokenizer.encode(sample, add_special_tokens=False)
    assert tokenizer.decode(ids) == sample


def test_bpe_checkpoint_vocab_mismatch_is_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    from ron.bpe_provider import BpeCheckpointProvider
    from ron.model_config import RonModelConfig
    from ron.tokenizer import RonTokenizer

    tokenizer = RonTokenizer().fit(["مرحبا بالعالم"], vocab_size=512)
    config = RonModelConfig(vocab_size=tokenizer.backend.get_vocab_size(), hidden_size=16,
                            num_layers=1, num_heads=2, max_sequence_length=32)
    # Wrong model vocab should fail before loading any weights.
    payload = {
        "tokenizer_type": "byte_bpe",
        "tokenizer_json": tokenizer.to_json(),
        "config": {**config.__dict__, "vocab_size": config.vocab_size + 1},
        "state_dict": {},
    }
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="vocabulary"):
        BpeCheckpointProvider(path)



def test_bpe_provider_loads_matching_checkpoint_and_answers_deterministic_basics(tmp_path):
    torch = pytest.importorskip("torch")
    from ron.bpe_provider import BpeCheckpointProvider
    from ron.contracts import Message, ModelRequest
    from ron.model import RonCausalLM
    from ron.model_config import RonModelConfig
    from ron.tokenizer import RonTokenizer

    tokenizer = RonTokenizer().fit(["User: hello\nRon: hi\nالمستخدم: ما اسمك\nرون: أنا رون"], vocab_size=512)
    config = RonModelConfig(vocab_size=tokenizer.backend.get_vocab_size(), hidden_size=16,
                            num_layers=1, num_heads=2, max_sequence_length=32)
    model = RonCausalLM(config)
    path = tmp_path / "valid.pt"
    torch.save({
        "tokenizer_type": "byte_bpe",
        "tokenizer_json": tokenizer.to_json(),
        "config": config.__dict__,
        "state_dict": model.state_dict(),
        "selected_step": 0,
    }, path)
    provider = BpeCheckpointProvider(path)
    response = provider.generate(ModelRequest(messages=(Message(role="user", content="ما اسمك"),)))
    assert response.content == "أنا رون، مساعد محلي قيد التطوير."
    assert response.metadata["tokenizer_type"] == "byte_bpe"

    # Exercise the open-ended native decode path (including incremental KV caching).
    open_response = provider.generate(ModelRequest(messages=(
        Message(role="user", content="Explain a new fictional concept called flarble."),
    )))
    assert open_response.content
    assert open_response.metadata["tokenizer_type"] == "byte_bpe"
