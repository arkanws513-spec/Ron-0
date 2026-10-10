from scripts.surprise_eval import BLIND_PROMPTS, ROOT, evaluate, is_degenerate_answer


def test_surprise_prompts_are_not_training_examples():
    corpus = (ROOT / "training" / "seed_corpus.txt").read_text(encoding="utf-8")
    assert len(BLIND_PROMPTS) >= 8
    assert len(set(BLIND_PROMPTS)) == len(BLIND_PROMPTS)
    assert all(prompt not in corpus for prompt in BLIND_PROMPTS)


def test_surprise_evaluation_loads_native_checkpoint_and_reports_limits():
    report = evaluate()
    assert report["case_count"] == len(BLIND_PROMPTS)
    assert report["training_corpus_exact_prompt_leaks"] == 0
    assert report["native_checkpoint_loaded"] is True
    assert report["external_model_used"] is False
    assert len(report["cases"]) == len(BLIND_PROMPTS)
    assert all(isinstance(case["answer"], str) for case in report["cases"])
    assert "not evidence of correctness" in report["interpretation"]

def test_anti_degeneration_check_catches_repetitive_output_without_claiming_correctness():
    assert is_degenerate_answer("أحدد المعلومات المعلومات المعلومات المعلومات المعلومات المعلومات")
    assert is_degenerate_answer("النتيجة هي النتيجة هي النتيجة هي النتيجة هي")
    assert not is_degenerate_answer("أراجع المعطيات ثم أشرح النتيجة باختصار واضح.")


def test_surprise_report_includes_anti_degeneration_gate():
    report = evaluate()
    assert 0 <= report["degenerate_answers"] <= report["case_count"]
    assert 0 <= report["stable_answers"] <= report["case_count"]
    assert report["anti_degeneration_gate_threshold"] == 0.8
    assert isinstance(report["anti_degeneration_gate_passed"], bool)
    assert "not evidence of factual correctness" in report["interpretation"]
