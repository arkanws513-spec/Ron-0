from ron.reasoning import Fact, Hypothesis, ReasoningEngine

def test_transitive_reasoning():
    engine = ReasoningEngine()
    result = engine.reason([
        Fact("A", "precedes", "B", .9),
        Fact("B", "precedes", "C", .8),
    ])
    assert any("A precedes C" == x for x in result.conclusions)
    assert result.inferences

def test_contradiction_is_retained():
    result = ReasoningEngine().reason([
        Fact("sky", "is", "blue", .9),
        Fact("sky", "is", "green", .4),
    ])
    assert result.contradictions
    assert result.facts[0].object == "blue"

def test_hypotheses_are_separate_from_facts():
    h = Hypothesis("قد يكون السبب هو المطر", .55, ("e1",))
    result = ReasoningEngine().reason([], [h])
    assert result.hypotheses[0] == h
    assert result.facts == ()


def test_semantic_query_uses_relations_not_token_similarity():
    engine = ReasoningEngine()
    result = engine.reason([
        engine.parse_fact("القاهرة هي عاصمة مصر", confidence=.95),
        engine.parse_fact("طيبة هي عاصمة مصر القديمة", confidence=.95),
    ])
    assert engine.answer_query("ما هي عاصمة مصر", result) == "القاهرة عاصمة مصر."
    assert engine.answer_query("ما هي عاصمة مصر القديمة", result) == "طيبة عاصمة مصر القديمة."
