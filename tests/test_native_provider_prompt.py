from ron.contracts import Message, ModelRequest
from ron.model_config import RonModelConfig
from ron.native_provider import NativeCheckpointProvider


def make_provider():
    provider = object.__new__(NativeCheckpointProvider)
    provider.config = RonModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        max_sequence_length=128,
        dropout=0.0,
    )
    return provider


def test_native_prompt_uses_english_role_labels_for_english_input():
    provider = make_provider()
    request = ModelRequest(messages=(Message(role="user", content="What is 2 + 2?"),))
    prompt = provider._build_dialogue_prompt(request, "What is 2 + 2?")
    assert prompt == "User: What is 2 + 2?\nRon:"


def test_native_prompt_keeps_arabic_role_labels_for_arabic_input():
    provider = make_provider()
    request = ModelRequest(messages=(Message(role="user", content="كم يساوي ٢ + ٢؟"),))
    prompt = provider._build_dialogue_prompt(request, "كم يساوي ٢ + ٢؟")
    assert prompt == "المستخدم: كم يساوي ٢ + ٢؟\nرون:"


def test_native_prompt_preserves_prior_dialogue_in_same_language():
    provider = make_provider()
    request = ModelRequest(messages=(
        Message(role="user", content="What is 2 + 2?"),
        Message(role="assistant", content="Four."),
        Message(role="user", content="Why?"),
    ))
    prompt = provider._build_dialogue_prompt(request, "Why?")
    assert "User: What is 2 + 2?" in prompt
    assert "Ron: Four." in prompt
    assert prompt.endswith("User: Why?\nRon:")
