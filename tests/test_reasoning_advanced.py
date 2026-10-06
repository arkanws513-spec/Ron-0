from ron.reasoning_advanced import AdvancedReasoner, CausalLink, WeightedEvidence

def test_independent_confidence_combination():
    value=AdvancedReasoner.combine_independent_confidences([.8,.5])
    assert .89 < value < .91

def test_hypothesis_scoring_prefers_supported_statement():
    result=AdvancedReasoner().score_hypotheses(
        ["rain causes wet ground","sun causes wet ground"],
        [WeightedEvidence("rain causes wet ground",.9,.9),
         WeightedEvidence("rain causes wet ground observed",.8,.8)]
    )
    assert result[0].hypothesis=="rain causes wet ground"

def test_causal_chain():
    ok, confidence, path=AdvancedReasoner().causal_reason(
        [CausalLink("A","B",.9),CausalLink("B","C",.8)],"A","C"
    )
    assert ok
    assert path==("A","B","C")
    assert .7 < confidence < .8

def test_counterfactual_reports_unknown_link():
    result=AdvancedReasoner().counterfactual("X",[CausalLink("A","B")],[])
    assert result.blocked_by
