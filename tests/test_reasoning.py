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

def test_arabic_causal_chain_is_inferred_with_provenance():
    engine = ReasoningEngine()
    rain = engine.parse_fact("المطر يسبب البلل", source="verified-example", confidence=.9)
    wet = engine.parse_fact("البلل يسبب انزلاق الطريق", source="verified-example", confidence=.85)
    assert rain is not None and wet is not None
    result = engine.reason([rain, wet])
    assert any("المطر causes انزلاق الطريق" == item for item in result.conclusions)
    assert any(item.rule == "causal_chain" for item in result.inferences)


def test_reasoning_keeps_conflicting_facts_visible():
    engine = ReasoningEngine()
    result = engine.reason([
        Fact("الماء", "is", "سائل", .9, "reference-a"),
        Fact("الماء", "is", "غاز", .3, "unverified-claim"),
    ])
    assert result.contradictions
    assert {fact.source for fact in result.facts} == {"reference-a", "unverified-claim"}

def test_temporal_rules_infer_reverse_and_transitive_relations():
    engine = ReasoningEngine()
    result = engine.reason([
        Fact("A", "precedes", "B", .95),
        Fact("B", "precedes", "C", .9),
    ])
    assert "A precedes C" in result.conclusions
    assert "B follows A" in result.conclusions
    assert "C follows A" in result.conclusions



def test_contradiction_detection_normalizes_arabic_article_variants():
    engine = ReasoningEngine()
    result = engine.reason([
        Fact("ازدحام", "causes", "تأخير", .8, "source-a"),
        Fact("الازدحام", "causes", "التأخير", .7, "source-b"),
        Fact("الماء", "is", "سائل", .9, "source-c"),
        Fact("الماء", "is", "غاز", .4, "source-d"),
    ])
    assert not any("ازدحام" in c.left and "ازدحام" in c.right for c in result.contradictions)
    assert any("سائل" in c.left and "غاز" in c.right for c in result.contradictions)


def test_hypothesis_scoring_counts_negation_as_opposition_not_support():
    from ron.reasoning_advanced import AdvancedReasoner, WeightedEvidence

    scores = AdvancedReasoner().score_hypotheses(
        ["rain causes flooding"],
        [
            WeightedEvidence("rain causes flooding", .9, 1.0, "source-a"),
            WeightedEvidence("not rain causes flooding", .8, 1.0, "source-b"),
        ],
    )
    assert scores[0].support == .9
    assert scores[0].opposition == .8
    assert abs(scores[0].confidence - (.9 / 1.7)) < 1e-9


def test_hypothesis_without_matching_evidence_has_zero_confidence():
    from ron.reasoning_advanced import AdvancedReasoner, WeightedEvidence

    score = AdvancedReasoner().score_hypotheses(
        ["hypothesis"], [WeightedEvidence("unrelated evidence", .9, 1.0)]
    )[0]
    assert score.confidence == 0.0
