from scripts.download_oasst1_english import is_acceptable_message, iter_conversation_examples


def test_oasst1_exports_only_english_reviewable_dialogue_prefixes():
    tree = {
        "role": "prompter",
        "lang": "en",
        "text": "Please explain why checking evidence matters before making a conclusion.",
        "replies": [
            {
                "role": "assistant",
                "lang": "en",
                "text": "Checking evidence helps distinguish a supported conclusion from a guess.",
                "replies": [],
            }
        ],
    }
    examples = list(iter_conversation_examples(tree))
    assert len(examples) == 1
    assert examples[0].startswith("User: ")
    assert "\nRon: " in examples[0]


def test_oasst1_filter_rejects_deleted_or_flagged_messages():
    assert not is_acceptable_message({
        "role": "assistant", "lang": "en", "text": "A deleted response", "deleted": True,
    })
    assert not is_acceptable_message({
        "role": "assistant", "lang": "en", "text": "A flagged response",
        "labels": {"sexual_content": {"value": 0.9}},
    })
    assert not is_acceptable_message({
        "role": "assistant", "lang": "fr", "text": "Une réponse française",
    })
