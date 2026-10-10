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
