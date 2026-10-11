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


def test_tied_embeddings_are_restored_after_assign_checkpoint_loading() -> None:
    config = ScalableRonConfig(vocab_size=32, hidden_size=32, num_layers=1, num_heads=4,
        num_kv_heads=2, max_sequence_length=8, tie_embeddings=True)
    original = ScalableRonLM(config)
    expected_parameters = count_parameters(original)
    checkpoint_state = original.state_dict()

    with torch.device("meta"):
        restored = ScalableRonLM(config)
    restored.load_state_dict(checkpoint_state, strict=True, assign=True)
    restored.tie_weights()

    assert not restored.embed_tokens.weight.is_meta
    assert restored.lm_head.weight is restored.embed_tokens.weight
    assert count_parameters(restored) == expected_parameters


def test_generation_masks_untrained_vocab_rows_and_structural_tokens() -> None:
    config = ScalableRonConfig(vocab_size=8, hidden_size=16, num_layers=1, num_heads=2,
        num_kv_heads=1, max_sequence_length=8)
    model = ScalableRonLM(config).eval()

    # Force the model to prefer an untrained padded ID and then a structural token.
    def fake_forward(input_ids, labels=None):
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], config.vocab_size))
        logits[..., 7] = 100.0  # beyond learned tokenizer vocabulary
        logits[..., 1] = 90.0   # structural token
        logits[..., 2] = 80.0   # EOS, blocked for the first generated token
        logits[..., 4] = 70.0   # first ordinary token
        return {"logits": logits, "loss": None}

    model.forward = fake_forward
    prompt = torch.tensor([[3]])
    generated = model.generate(
        prompt,
        max_new_tokens=2,
        eos_token_id=2,
        valid_vocab_size=6,
        suppress_token_ids=[0, 1, 3],
        min_new_tokens=1,
    )
    assert generated[0, 1].item() == 4
    assert generated[0, 2].item() < 6


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


def test_ron_100m_preset_pads_small_tokenizer_vocabulary() -> None:
    from ron.scale_presets import build_scale_config
    config = build_scale_config("ron-100m", vocab_size=700, sequence_length=32)
    assert config.vocab_size == 8000



def test_ron_180m_preset_has_expected_real_parameter_count() -> None:
    from ron.scale_presets import build_scale_config
    config = build_scale_config("ron-180m", vocab_size=8000, sequence_length=64)
    assert (config.hidden_size, config.num_layers, config.num_heads, config.num_kv_heads, config.intermediate_size) == (1024, 15, 16, 4, 2880)
    with torch.device("meta"):
        model = ScalableRonLM(config)
    assert count_parameters(model) == 180_255_744


def test_sft_dataset_only_accepts_teacher_data_when_explicitly_enabled(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
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
    data.write_text(json.dumps({"prompt": "hello", "response": "world", "review_status": "teacher_generated", "source": "pythia-160m", "human_reviewed": False}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed records"):
        PromptResponseDataset(data, tok_path, 16)
    accepted = PromptResponseDataset(data, tok_path, 16, allow_teacher_generated=True)
    assert len(accepted) == 1
