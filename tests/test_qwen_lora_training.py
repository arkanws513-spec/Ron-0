from scripts.train_qwen_lora import (
    build_turn_examples,
    encode_completion,
    parse_dialogue_blocks,
    split_by_language,
)


def test_parse_english_and_arabic_dialogue_blocks():
    parsed = parse_dialogue_blocks(
        "User: Hi\nRon: Hello\n\nالمستخدم: مرحبا\nرون: أهلا"
    )
    assert len(parsed) == 2
    assert parsed[0][0] == {"role": "user", "content": "Hi"}
    assert parsed[1][1] == {"role": "assistant", "content": "أهلا"}


def test_build_turn_examples_masks_language_by_user_prompt():
    examples = build_turn_examples([
        "User: What is 2+2?\nRon: 4.\n\n"
        "المستخدم: ما عاصمة مصر؟\nرون: القاهرة."
    ])
    assert len(examples) == 2
    assert {item["language"] for item in examples} == {"ar", "en"}
    assert all(item["messages"][-1]["role"] == "assistant" for item in examples)


def test_language_split_is_reproducible_and_keeps_both_languages():
    examples = [
        {"language": language, "messages": [{"role": "user", "content": str(i)}]}
        for language in ("ar", "en") for i in range(20)
    ]
    train_a, val_a = split_by_language(examples, seed=7)
    train_b, val_b = split_by_language(examples, seed=7)
    assert train_a == train_b
    assert val_a == val_b
    assert {item["language"] for item in val_a} == {"ar", "en"}
    assert not {str(item) for item in train_a}.intersection(str(item) for item in val_a)


def test_completion_encoder_masks_prompt_tokens_only():
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            if add_generation_prompt:
                return [10, 11, 12]
            return [10, 11, 12, 13, 14]

    encoded = encode_completion(
        FakeTokenizer(),
        [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
        max_length=16,
    )
    assert encoded["input_ids"] == [10, 11, 12, 13, 14]
    assert encoded["labels"] == [-100, -100, -100, 13, 14]
