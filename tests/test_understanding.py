from ron.understanding import UnderstandingEngine

def test_understanding_resolves_short_follow_up_to_previous_topic():
    result = UnderstandingEngine().analyze("طيب؟", previous_user="ما هي الطاقة الشمسية")
    assert result.is_follow_up is True
    assert result.topic == "ما هي الطاقة الشمسية"
    assert result.intent.name == "follow_up"

def test_understanding_produces_confidence_and_entities():
    result = UnderstandingEngine().analyze("رون يعرف Python والذكاء الاصطناعي")
    assert result.confidence > 0
    assert "Python" in result.entities

def test_why_and_clarification_followups_keep_previous_topic():
    engine = UnderstandingEngine()
    for prompt in ("ليه؟", "لماذا؟", "ازاي؟", "ما السبب؟", "كيف ذلك؟", "ماذا تقصد؟"):
        result = engine.analyze(prompt, previous_user="ما هي الطاقة الشمسية؟")
        assert result.is_follow_up is True, prompt
        assert result.topic == "ما هي الطاقة الشمسية؟", prompt
        assert result.intent.name == "follow_up", prompt


def test_complete_how_question_is_not_misclassified_as_followup():
    result = UnderstandingEngine().analyze("كيف يعمل المحرك؟", previous_user="ما هي الطاقة الشمسية؟")
    assert result.is_follow_up is False
    assert result.topic == "كيف يعمل المحرك؟"
