import json

import pytest

from training.finetune import load_approved_jsonl, split_records, validate_training_records


def record(user, assistant):
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def test_validates_and_deduplicates_approved_records():
    records = validate_training_records([
        record("ما اسمك؟", "أنا رون."),
        record("ما اسمك؟", "أنا رون."),
        record("ما عاصمة مصر؟", "القاهرة."),
    ])
    assert len(records) == 2
    assert records[0]["messages"][0]["content"] == "ما اسمك؟"


def test_rejects_bad_roles_empty_content_and_too_few_distinct_examples():
    with pytest.raises(ValueError, match="start with user and end with assistant"):
        validate_training_records([
            {"messages": [
                {"role": "assistant", "content": "رد"},
                {"role": "user", "content": "سؤال"},
            ]},
            record("سؤال آخر", "إجابة"),
        ])
    with pytest.raises(ValueError, match="at least two distinct"):
        validate_training_records([record("سؤال", "إجابة")])
    with pytest.raises(ValueError, match="non-empty"):
        validate_training_records([
            record("سؤال", "إجابة"),
            {"messages": [
                {"role": "user", "content": "سؤال آخر"},
                {"role": "assistant", "content": "   "},
            ]},
        ])


def test_load_approved_jsonl_rejects_malformed_lines_and_accepts_valid_data(tmp_path):
    path = tmp_path / "approved.jsonl"
    path.write_text(json.dumps(record("سؤال أ", "إجابة أ"), ensure_ascii=False) + "\n" +
                    json.dumps(record("سؤال ب", "إجابة ب"), ensure_ascii=False) + "\n",
                    encoding="utf-8")
    assert len(load_approved_jsonl(path)) == 2
    path.write_text("{broken json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON on line 1"):
        load_approved_jsonl(path)


def test_train_validation_split_is_reproducible_and_disjoint():
    records = [record(f"سؤال {index}", f"إجابة {index}") for index in range(20)]
    train, validation = split_records(records, validation_split=0.2, seed=7)
    assert (train, validation) == split_records(records, validation_split=0.2, seed=7)
    assert len(train) == 16
    assert len(validation) == 4
    train_prompts = {item["messages"][0]["content"] for item in train}
    validation_prompts = {item["messages"][0]["content"] for item in validation}
    assert train_prompts.isdisjoint(validation_prompts)


def test_split_rejects_invalid_validation_fraction():
    records = [record("سؤال أ", "إجابة أ"), record("سؤال ب", "إجابة ب")]
    with pytest.raises(ValueError, match="validation_split"):
        split_records(records, validation_split=0.5)
