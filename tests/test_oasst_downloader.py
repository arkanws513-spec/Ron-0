from scripts.download_oasst1_english import iter_conversation_examples


def test_oasst_nested_prompt_tree_yields_reviewed_english_dialogue():
    tree = {
        "message_tree_id": "root-1",
        "tree_state": "ready_for_export",
        "prompt": {
            "role": "prompter",
            "lang": "en",
            "text": "Can you explain why the sky looks blue in simple terms for a student?",
            "deleted": False,
            "review_result": True,
            "labels": {},
            "replies": [
                {
                    "role": "assistant",
                    "lang": "en",
                    "text": "Sunlight contains many colors. Air molecules scatter shorter blue wavelengths more strongly, so more blue light reaches our eyes from across the sky.",
                    "deleted": False,
                    "review_result": True,
                    "labels": {},
                    "replies": [],
                }
            ],
        },
    }
    examples = list(iter_conversation_examples(tree))
    assert len(examples) == 1
    assert examples[0].startswith("User: Can you explain")
    assert "\nRon: Sunlight contains many colors." in examples[0]


def test_oasst_rejected_or_non_english_root_is_skipped():
    rejected = {
        "prompt": {
            "role": "prompter",
            "lang": "en",
            "text": "A long prompt that should be excluded because it was rejected by review.",
            "review_result": False,
            "replies": [{"role": "assistant", "lang": "en", "text": "A response that should not be kept.", "replies": []}],
        }
    }
    non_english = {
        "prompt": {
            "role": "prompter",
            "lang": "es",
            "text": "Una pregunta en otro idioma que debe excluirse.",
            "replies": [{"role": "assistant", "lang": "es", "text": "Una respuesta que debe excluirse.", "replies": []}],
        }
    }
    assert list(iter_conversation_examples(rejected)) == []
    assert list(iter_conversation_examples(non_english)) == []
