from ron.brain import KnowledgeTransfer

def test_knowledge_transfer_keeps_auditable_examples():
    transfer = KnowledgeTransfer(max_examples=2)
    transfer.capture("ما اسمك؟", "أنا رون.", source="local-brain")
    transfer.capture("تعلم", "تم.", source="local-brain")
    assert len(transfer.examples) == 2
    assert transfer.dataset()[0] == {"user": "ما اسمك؟", "assistant": "أنا رون."}

def test_knowledge_transfer_is_bounded():
    transfer = KnowledgeTransfer(max_examples=2)
    for i in range(3):
        transfer.capture(str(i), "answer")
    assert [x.prompt for x in transfer.examples] == ["1", "2"]
