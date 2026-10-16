import pytest

from ron.tokenizer import RonTokenizer


def sample_corpus():
    return [
        "المستخدم: ما عاصمة مصر؟\nرون: عاصمة مصر هي القاهرة.",
        "المستخدم: كيف حالك؟\nرون: أنا بخير، شكرًا لك.",
        "User: What is the capital of Egypt?\nRon: Cairo is the capital of Egypt.",
        "المستخدم: أريد حساب ٤٥٨ × ٣٤٥.\nرون: الناتج هو ١٥٨٠١٠.",
    ] * 20


def test_bpe_round_trip_preserves_arabic_whitespace_and_punctuation():
    tokenizer = RonTokenizer().fit(sample_corpus(), vocab_size=500)
    text = "أهلاً، رون!\nما عاصمة مصر؟ 123 × ٤٥٨ 🇪🇬"
    encoded = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(encoded) == text


def test_bpe_uses_subwords_and_compresses_repeated_training_text():
    corpus = sample_corpus()
    tokenizer = RonTokenizer().fit(corpus, vocab_size=500)
    text = "المستخدم: ما عاصمة مصر؟\nرون: عاصمة مصر هي القاهرة."
    assert len(tokenizer.encode(text, add_special_tokens=False)) < len(text)
    assert tokenizer.tokenize("القاهرة") != list("القاهرة")


def test_serialized_tokenizer_round_trip():
    tokenizer = RonTokenizer().fit(sample_corpus(), vocab_size=500)
    restored = RonTokenizer.from_json(tokenizer.to_json())
    text = "أهلاً بالعالم 🌍"
    assert restored.decode(restored.encode(text, add_special_tokens=False)) == text


def test_special_tokens_can_be_added_and_removed():
    tokenizer = RonTokenizer().fit(sample_corpus(), vocab_size=500)
    ids = tokenizer.encode("مرحبا", add_special_tokens=True)
    assert tokenizer.decode(ids) == "مرحبا"
    assert ids[0] == tokenizer.backend.token_to_id("<bos>")
    assert ids[-1] == tokenizer.backend.token_to_id("<eos>")


def test_invalid_vocab_size_is_rejected():
    with pytest.raises(ValueError):
        RonTokenizer().fit(["abc"], vocab_size=20)
