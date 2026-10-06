from ron.evaluation import EvalCase, evaluate_exact

def test_evaluation_reports_exact_passes():
    report = evaluate_exact([EvalCase("2+2", "4")], lambda _: "4")
    assert report.passed_all
    assert report.score == 1.0

def test_evaluation_rejects_wrong_answer():
    report = evaluate_exact([EvalCase("2+2", "4")], lambda _: "5")
    assert not report.passed_all
    assert report.score == 0.0
