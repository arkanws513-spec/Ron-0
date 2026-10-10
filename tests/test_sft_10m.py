import torch

from scripts.train_sft_10m import candidate_starts, encode_examples, make_batch, split_examples


def test_sft_masks_user_text_and_trains_only_ron_response_characters():
    example = "User: What is a useful habit?\nRon: Check evidence before trusting a claim."
    vocab = {char: index for index, char in enumerate(sorted(set(example + "\n")))}
    tokens, mask, unknown, total = encode_examples([example], vocab)

    user_end = example.index("\nRon:")
    assistant_start = user_end + 1
    response_start = assistant_start + len("Ron: ")
    assert not mask[:response_start].any()
    assert mask[response_start:assistant_start + len("Ron: Check evidence before trusting a claim.")].all()
    assert unknown == 0
    assert len(tokens) >= total


def test_sft_batch_uses_ignore_index_for_non_assistant_targets():
    example = "User: Please explain why evidence matters in daily decisions.\nRon: Evidence helps separate reliable conclusions from guesses."
    vocab = {char: index for index, char in enumerate(sorted(set(example + "\n")))}
    tokens, mask, _, _ = encode_examples([example], vocab)
    starts = candidate_starts(mask, length=16)
    x, y = make_batch(tokens, mask, starts, size=2, length=16)
    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
    assert (y != -100).any()


def test_sft_preserves_line_boundaries_and_masks_arabic_user_turns():
    example = "المستخدم: احسب خمسة زائد ثلاثة\nرون: الناتج ثمانية."
    vocab = {char: index for index, char in enumerate(sorted(set(example + "\n")))}
    tokens, mask, unknown, total = encode_examples([example], vocab)
    user_end = example.index("\nرون:")
    assistant_start = user_end + 1
    response_start = assistant_start + len("رون: ")
    assert not mask[:response_start].any()
    assert mask[response_start:assistant_start + len("رون: الناتج ثمانية.")].all()
    assert unknown == 0
    assert len(tokens) > len(example)

def test_sft_split_is_reproducible_and_keeps_examples_disjoint():
    corpus = "\n\n".join(
        f"User: Explain concept {i} in a short and useful way.\nRon: Concept {i} means a distinct thing worth understanding."
        for i in range(120)
    )
    train_a, val_a = split_examples(corpus, seed=12)
    train_b, val_b = split_examples(corpus, seed=12)
    assert train_a == train_b
    assert val_a == val_b
    assert set(train_a).isdisjoint(set(val_a))
