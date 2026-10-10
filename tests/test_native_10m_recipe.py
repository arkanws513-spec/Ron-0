"""Regression tests for the dedicated Ron-10M training recipe."""
from scripts.train_native_10m import split_corpus
from scripts.prepare_corpus import SOURCE_DIR, is_pretraining_source
from ron.model import RonCausalLM
from ron.model_config import RonModelConfig


def test_document_split_is_deterministic_and_keeps_paragraphs_intact():
    corpus = "\n\n".join(f"Document {index}: " + ("unique text " * 12) for index in range(20))
    first = split_corpus(corpus, seed=123)
    second = split_corpus(corpus, seed=123)
    assert first == second
    train_text, validation_text, train_count, validation_count = first
    assert train_count + validation_count == 20
    assert train_count > validation_count
    assert set(train_text.split("\n\n")).isdisjoint(set(validation_text.split("\n\n")))


def test_ron_10m_profile_is_approximately_ten_million_parameters():
    config = RonModelConfig(
        vocab_size=128,
        hidden_size=384,
        num_layers=6,
        num_heads=6,
        max_sequence_length=256,
        dropout=0.1,
    )
    model = RonCausalLM(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 10_000_000 <= parameter_count <= 11_500_000


def test_instruction_dialogues_are_excluded_from_pretraining_corpus():
    assert not is_pretraining_source(SOURCE_DIR / "oasst1" / "english_dialogues.txt")
    assert not is_pretraining_source(SOURCE_DIR / "dialogs" / "conversation_dump.txt")
    assert is_pretraining_source(SOURCE_DIR / "gutenberg" / "pg11.txt")
