from ron.distillation import KnowledgeDistiller, TrainingExample


def test_qwen_examples_are_not_memory_by_default():
    d = KnowledgeDistiller()
    d.add(TrainingExample("ما هي عاصمة مصر؟", "القاهرة عاصمة مصر."))
    assert d.approved() == ()
    assert d.export_jsonl() == ""


def test_only_approved_examples_are_exported():
    d = KnowledgeDistiller()
    ex = TrainingExample("سؤال", "إجابة", context="سياق")
    d.add(ex)
    assert d.approve("سؤال", "إجابة")
    assert len(d.approved()) == 1
    assert '"role": "user"' in d.export_jsonl()
