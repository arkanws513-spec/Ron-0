"""Reproducible proof-of-learning run for Ron-0's native character-level LM."""
import json, random, time
from pathlib import Path
import torch
from torch import nn
from ron.model import RonCausalLM
from ron.model_config import RonModelConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "native-baseline"
SEED, STEPS, BATCH, LENGTH = 1701, 400, 16, 96

def batch(tokens, size, length):
    top = len(tokens) - length - 1
    if top < 1: raise ValueError("not enough tokens for context length")
    starts = torch.randint(0, top, (size,))
    return (torch.stack([tokens[s:s+length] for s in starts.tolist()]),
            torch.stack([tokens[s+1:s+length+1] for s in starts.tolist()]))

@torch.no_grad()
def evaluate(model, tokens, count=16):
    model.eval()
    vals = [float(model(*batch(tokens, 8, LENGTH)).loss.item()) for _ in range(count)]
    model.train()
    return sum(vals) / len(vals)

def main():
    random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(min(4, torch.get_num_threads()))
    corpus = (ROOT / "training" / "seed_corpus.txt").read_text(encoding="utf-8")
    chars = sorted(set(corpus)); vocab = {ch: i for i, ch in enumerate(chars)}
    ids = torch.tensor([vocab[ch] for ch in corpus], dtype=torch.long)
    split = int(len(ids) * 0.9); train, val = ids[:split], ids[split:]
    cfg = RonModelConfig(vocab_size=len(vocab), hidden_size=64, num_layers=2, num_heads=4, max_sequence_length=LENGTH)
    model = RonCausalLM(cfg); opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    start_train, start_val = evaluate(model, train), evaluate(model, val)
    best_val, best_step, best_state = float("inf"), 0, None
    start = time.time(); history = []
    for step in range(1, STEPS + 1):
        x, y = batch(train, BATCH, LENGTH); opt.zero_grad(set_to_none=True)
        loss = model(x, y).loss
        if loss is None or not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            tr, va = evaluate(model, train), evaluate(model, val, count=24)
            history.append({"step": step, "train_loss": tr, "validation_loss": va})
            print(f"step={step} train_loss={tr:.4f} validation_loss={va:.4f}", flush=True)
            if va < best_val:
                best_val, best_step = va, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None: raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__, "vocab": vocab,
                "seed": SEED, "selected_step": best_step}, OUT / "ron_native_baseline.pt")
    result = {
      "experiment": "ron0-native-character-lm-baseline", "status": "completed",
      "seed": SEED, "training_steps": STEPS, "selected_checkpoint_step": best_step,
      "parameters": sum(p.numel() for p in model.parameters()), "vocab_size": len(vocab),
      "corpus_characters": len(ids), "train_characters": len(train), "validation_characters": len(val),
      "initial_train_loss": start_train, "initial_validation_loss": start_val,
      "best_validation_loss": best_val, "last_train_loss": history[-1]["train_loss"],
      "last_validation_loss": history[-1]["validation_loss"],
      "train_loss_reduction_percent": 100 * (start_train - history[-1]["train_loss"]) / start_train,
      "validation_loss_reduction_percent": 100 * (start_val - best_val) / start_val,
      "elapsed_seconds": round(time.time() - start, 2), "history": history,
      "limitations": ["Tiny hand-written corpus; loss only measures learning on this corpus.",
        "Character-level model is not yet a capable general-purpose conversational assistant.",
        "Validation split is the final 10% of the corpus and is not an independent benchmark.",
        "No Qwen weights or paid inference used."]
    }
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
