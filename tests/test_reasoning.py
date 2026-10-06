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
