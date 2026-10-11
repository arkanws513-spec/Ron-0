from __future__ import annotations
import json
from pathlib import Path
import pytest
import torch
from ron.scalable_model import ScalableRonConfig, ScalableRonLM, count_parameters

def test_scalable_model_forward_and_response_masking() -> None:
    config = ScalableRonConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4,
        num_kv_heads=2, max_sequence_length=16)
    model = ScalableRonLM(config)
    ids = torch.randint(0, 64, (2, 8))
    labels = ids.clone()
    labels[:, :3] = -100
    output = model(ids, labels)
    assert output["logits"].shape == (2, 8, 64)
    assert output["loss"] is not None and torch.isfinite(output["loss"])
    output["loss"].backward()
    assert count_parameters(model) > 0

def test_invalid_gqa_dimensions_rejected() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        ScalableRonConfig(vocab_size=32, hidden_size=32, num_layers=1, num_heads=4,
            num_kv_heads=3, max_sequence_length=8)

def test_config_round_trip(tmp_path: Path) -> None:
    config = ScalableRonConfig(vocab_size=32, hidden_size=32, num_layers=1, num_heads=4,
        num_kv_heads=2, max_sequence_length=8)
    path = tmp_path / "config.json"
    config.save(path)
    assert ScalableRonConfig.load(path) == config

def test_sft_dataset_requires_approved_records(tmp_path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from scripts.train_scalable_sft import PromptResponseDataset
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nanswer", encoding="utf-8")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    tok.train([str(corpus)], trainers.BpeTrainer(vocab_size=300, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok_path = tmp_path / "tokenizer.json"
    tok.save(str(tok_path))
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"prompt":"hello", "response":"world", "review_status":"needs_human_review"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed records"):
        PromptResponseDataset(data, tok_path, 16)


def test_scalable_generation_stays_within_vocabulary() -> None:
    config = ScalableRonConfig(vocab_size=32, hidden_size=32, num_layers=1, num_heads=4,
        num_kv_heads=2, max_sequence_length=8)
    model = ScalableRonLM(config).eval()
    prompt = torch.tensor([[1, 2, 3]])
    generated = model.generate(prompt, max_new_tokens=3, eos_token_id=31)
    assert generated.shape[0] == 1
    assert generated.shape[1] >= prompt.shape[1]
    assert int(generated.min()) >= 0 and int(generated.max()) < config.vocab_size


def test_scalable_estimator_is_separate_from_legacy() -> None:
    from scripts.estimate_model_scale import estimate
    legacy = estimate(1024, 128, 2, 4, 64)
    scalable = estimate(1024, 128, 2, 4, 64, architecture="scalable", num_kv_heads=2)
    assert scalable["architecture"] == "scalable"
    assert scalable["estimated_parameters"] > 0
    assert legacy["architecture"] == "legacy"


def test_optional_training_and_chat_scripts_import_without_tokenizers() -> None:
    from scripts import chat_scalable_ron, pretrain_scalable_ron
    batch = [{"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([1, 2])},
             {"input_ids": torch.tensor([3, 4, 5]), "labels": torch.tensor([3, 4, 5])}]
    padded = pretrain_scalable_ron.collate_blocks(batch, sequence_length=4, pad_id=0)
    assert padded["input_ids"].shape == (2, 4)
    assert padded["labels"][0, 2:].tolist() == [-100, -100]
    assert chat_scalable_ron.__doc__ is not None


def test_ron_100m_preset_has_expected_real_parameter_count() -> None:
    from ron.scale_presets import build_scale_config
    from scripts.estimate_model_scale import estimate
    config = build_scale_config("ron-100m", vocab_size=8000, sequence_length=256)
    assert (config.hidden_size, config.num_layers, config.num_heads, config.num_kv_heads, config.intermediate_size) == (1024, 8, 16, 4, 2880)
    estimate_result = estimate(8000, 1024, 8, 16, 256, architecture="scalable", num_kv_heads=4, intermediate_size=2880)
    assert estimate_result["estimated_parameters"] == 99_959_808
    with torch.device("meta"):
        model = ScalableRonLM(config)
    assert count_parameters(model) == estimate_result["estimated_parameters"]


def test_ron_100m_preset_rejects_mismatched_tokenizer_vocab() -> None:
    from ron.scale_presets import build_scale_config
    with pytest.raises(ValueError, match="8,000-token"):
        build_scale_config("ron-100m", vocab_size=7999, sequence_length=256)
