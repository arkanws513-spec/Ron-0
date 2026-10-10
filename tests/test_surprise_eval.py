from scripts.surprise_eval import BLIND_PROMPTS, ROOT, evaluate


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
