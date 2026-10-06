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
