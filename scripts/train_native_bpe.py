"""Train a fresh native Ron Transformer using byte-level BPE tokens.

This intentionally creates a separate candidate checkpoint. It never loads or
reinterprets character-level weights and never overwrites the live checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from ron.model import RonCausalLM
from ron.model_config import RonModelConfig
from ron.tokenizer import RonTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "native-bpe"
CORPUS_CANDIDATES = (
    ROOT / "training" / "corpus.txt",
    ROOT / "training" / "seed_corpus.txt",
    ROOT / "training" / "corpus" / "oasst1" / "english_dialogues.txt",
)
SEED = 2701
STEPS = int(os.environ.get("RON_BPE_STEPS", "5000"))
BATCH = int(os.environ.get("RON_BPE_BATCH_SIZE", "2"))
LENGTH = int(os.environ.get("RON_BPE_SEQUENCE_LENGTH", "256"))
VOCAB_SIZE = int(os.environ.get("RON_BPE_VOCAB_SIZE", "4096"))
HIDDEN, LAYERS, HEADS = 384, 6, 6


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    corpus_path = next((path for path in CORPUS_CANDIDATES if path.is_file()), None)
    if corpus_path is None:
        raise FileNotFoundError("No training corpus found. Expected training/corpus.txt or a supported corpus path.")
    corpus = corpus_path.read_text(encoding="utf-8")
    if len(corpus) < 100_000:
        raise ValueError(f"Training corpus is too small ({len(corpus)} characters); require at least 100,000.")
    documents = [part.strip() for part in corpus.replace("\r\n", "\n").split("\n\n") if part.strip()]
    if len(documents) < 10:
        raise ValueError("Corpus must contain at least 10 non-empty paragraphs.")
    random.Random(SEED).shuffle(documents)
    validation_count = max(1, round(len(documents) * 0.1))
    validation_text = "\n\n".join(documents[:validation_count])
    train_text = "\n\n".join(documents[validation_count:])
    tokenizer = RonTokenizer().fit((train_text,), vocab_size=VOCAB_SIZE)
    train_ids = torch.tensor(tokenizer.encode(train_text, add_special_tokens=False), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(validation_text, add_special_tokens=False), dtype=torch.long)
    if min(len(train_ids), len(val_ids)) < LENGTH + 2:
        raise ValueError("Tokenized train/validation corpus is too short for configured sequence length.")
    config = RonModelConfig(vocab_size=tokenizer.backend.get_vocab_size(), hidden_size=HIDDEN,
                            num_layers=LAYERS, num_heads=HEADS, max_sequence_length=LENGTH, dropout=0.1)
    model = RonCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    @torch.no_grad()
    def evaluate(tokens: torch.Tensor, windows: int = 16) -> float:
        old = model.training
        model.eval()
        top = len(tokens) - LENGTH - 1
        starts = torch.linspace(0, top - 1, steps=max(1, min(windows, top))).long().unique().tolist()
        losses = [float(model(tokens[s:s+LENGTH].unsqueeze(0),
                              tokens[s+1:s+LENGTH+1].unsqueeze(0)).loss.item()) for s in starts]
        model.train(old)
        if not losses or not all(math.isfinite(v) for v in losses):
            raise RuntimeError("Non-finite BPE validation loss.")
        return sum(losses) / len(losses)

    start_val = evaluate(val_ids)
    best_val, best_step = start_val, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    started = time.time()
    history = []
    for step in range(1, STEPS + 1):
        starts = torch.randint(0, len(train_ids) - LENGTH - 1, (BATCH,))
        x = torch.stack([train_ids[s:s+LENGTH] for s in starts.tolist()])
        y = torch.stack([train_ids[s+1:s+LENGTH+1] for s in starts.tolist()])
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite BPE training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == STEPS:
            val_loss = evaluate(val_ids)
            history.append({"step": step, "validation_loss": val_loss})
            print(f"bpe_step={step} validation_loss={val_loss:.4f}", flush=True)
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_format_version": 2,
        "tokenizer_type": "byte_bpe",
        "tokenizer_json": tokenizer.to_json(),
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "best_state_dict": best_state,
        "best_selected_step": best_step,
        "best_validation_loss": best_val,
        "config": config.__dict__,
        "seed": SEED,
        "selected_step": STEPS,
        "training_steps_this_run": STEPS,
        "training_steps_total": STEPS,
        "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "checkpoint_source": f"bpe_pretraining_from_{corpus_path.name}",
        "training_stage": "byte_bpe_pretraining",
        "optimizer_state_dict": optimizer.state_dict(),
    }
    candidate = OUT / "ron_native_bpe_candidate.pt"
    temporary = OUT / "ron_native_bpe_candidate.tmp.pt"
    torch.save(checkpoint, temporary)
    os.replace(temporary, candidate)
    metrics = {
        "status": "candidate_created_not_promoted",
        "tokenizer_type": "byte_bpe",
        "source_corpus": str(corpus_path.relative_to(ROOT)),
        "source_corpus_characters": len(corpus),
        "train_tokens": int(train_ids.numel()),
        "validation_tokens": int(val_ids.numel()),
        "vocab_size": config.vocab_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_validation_loss": start_val,
        "best_validation_loss": best_val,
        "best_step": best_step,
        "training_steps": STEPS,
        "elapsed_seconds": round(time.time() - started, 2),
        "candidate_checkpoint": str(candidate.relative_to(ROOT)),
        "live_checkpoint_modified": False,
        "history": history,
    }
    (OUT / "bpe_pretraining_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
