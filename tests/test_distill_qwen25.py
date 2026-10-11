import json
from pathlib import Path

from scripts.distill_qwen25 import normalize, read_prompts, write_outputs


def test_normalize_cleans_whitespace():
    assert normalize("  hello \r\n\r\n\r\n world  ") == "hello\n\n world"


def test_read_prompts_accepts_plain_text_and_jsonl(tmp_path: Path):
    path = tmp_path / "prompts.txt"
    path.write_text('# ignored\nHello\n{"prompt":"مرحبا"}\n', encoding="utf-8")
    assert read_prompts(path) == ["Hello", "مرحبا"]


def test_read_prompts_rejects_bad_json(tmp_path: Path):
    path = tmp_path / "prompts.txt"
    path.write_text('{"prompt":\n', encoding="utf-8")
    try:
        read_prompts(path)
    except ValueError as exc:
        assert "invalid JSON" in str(exc)
    else:
        raise AssertionError("expected invalid JSON to be rejected")


def test_write_outputs_creates_jsonl_and_training_corpus(tmp_path: Path):
    jsonl = tmp_path / "teacher.jsonl"
    corpus = tmp_path / "corpus.txt"
    records = [
        {"prompt": "Explain AI", "response": "AI learns patterns.", "teacher_model": "local-test"},
        {"prompt": "ما الذكاء الاصطناعي؟", "response": "هو مجال في علوم الحاسوب.", "teacher_model": "local-test"},
    ]
    write_outputs(records, jsonl, corpus)
    lines = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    text = corpus.read_text(encoding="utf-8")
    assert "User: Explain AI\nRon: AI learns patterns." in text
    assert "\n\n" in text
    assert "User: ما الذكاء الاصطناعي؟" in text


def test_estimator_scales_and_validates_dimensions():
    from scripts.estimate_model_scale import estimate

    small = estimate(4096, 384, 6, 6, 256)
    large = estimate(32000, 1536, 36, 12, 2048)
    assert small["estimated_parameters"] > 0
    assert large["estimated_parameters"] > small["estimated_parameters"]
    assert large["estimated_parameters"] > 1_000_000_000
    assert large["adamw_fp32_weights_gradients_moments_gib"] > large["fp32_inference_weights_gib"]
