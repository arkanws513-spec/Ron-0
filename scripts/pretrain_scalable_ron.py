"""Pretrain a scalable Ron decoder on a local plain-text corpus using causal language modeling."""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from ron.scalable_model import ScalableRonConfig, ScalableRonLM, count_parameters

class PackedTextDataset(Dataset):
    def __init__(self, corpus: Path, tokenizer_path: Path, sequence_length: int,
                 max_documents: int | None = None) -> None:
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        vocab = self.tokenizer.get_vocab()
        bos_id, eos_id = vocab.get("<bos>", 2), vocab.get("<eos>", 3)
        text = corpus.read_text(encoding="utf-8")
        documents = [part.strip() for part in text.split("\n\n") if part.strip()]
        if max_documents is not None:
            documents = documents[:max_documents]
        self.blocks: list[list[int]] = []
        for document in documents:
            ids = [bos_id, *self.tokenizer.encode(document).ids, eos_id]
            for start in range(0, len(ids) - 1, sequence_length):
                block = ids[start:start + sequence_length]
                if len(block) >= 2:
                    self.blocks.append(block)
        if not self.blocks:
            raise ValueError("corpus produced no usable token blocks; add more reviewed text")

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = self.blocks[index]
        labels = list(ids)
        pad_id = self.tokenizer.token_to_id("<pad>")
        pad_id = 0 if pad_id is None else pad_id
        padding = self.sequence_length - len(ids) if hasattr(self, "sequence_length") else 0
        # Sequence length is inferred from the first block by the owning trainer.
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}

def collate_blocks(batch: list[dict[str, torch.Tensor]], sequence_length: int, pad_id: int) -> dict[str, torch.Tensor]:
    input_rows, label_rows = [], []
    for item in batch:
        ids = item["input_ids"].tolist()
        labels = item["labels"].tolist()
        pad = sequence_length - len(ids)
        input_rows.append(ids + [pad_id] * pad)
        label_rows.append(labels + [-100] * pad)
    return {"input_ids": torch.tensor(input_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long)}

def evaluate(model: ScalableRonLM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            result = model(batch["input_ids"].to(device), batch["labels"].to(device))
            loss = result["loss"]
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite or missing validation loss")
            total += float(loss)
            count += 1
    return total / max(count, 1)

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True, help="Reviewed local text corpus")
    p.add_argument("--validation-corpus", type=Path, help="Separate held-out text corpus")
    p.add_argument("--tokenizer", type=Path, default=Path("artifacts/ron-tokenizer/tokenizer.json"))
    p.add_argument("--output", type=Path, default=Path("artifacts/ron-pretrained"))
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--max-documents", type=int, help="Limit documents for a small smoke test")
    p.add_argument("--seed", type=int, default=1701)
    args = p.parse_args()
    if not args.corpus.is_file() or not args.tokenizer.is_file():
        p.error("--corpus and --tokenizer must exist")
    if args.validation_corpus is not None and not args.validation_corpus.is_file():
        p.error("--validation-corpus must exist")
    if args.sequence_length < 8 or args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        p.error("sequence-length must be >= 8; epochs, batch-size and learning-rate must be positive")
    if args.max_documents is not None and args.max_documents < 1:
        p.error("--max-documents must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")
    pad_id = 0 if pad_id is None else pad_id
    ds = PackedTextDataset(args.corpus, args.tokenizer, args.sequence_length, args.max_documents)
    val_ds = PackedTextDataset(args.validation_corpus, args.tokenizer, args.sequence_length) if args.validation_corpus else None
    config = ScalableRonConfig(vocab_size=vocab_size, hidden_size=args.hidden_size,
        num_layers=args.layers, num_heads=args.heads, num_kv_heads=args.kv_heads,
        max_sequence_length=args.sequence_length)
    collate = lambda batch: collate_blocks(batch, args.sequence_length, pad_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if val_ds else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScalableRonLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    config.save(args.output / "config.json")
    (args.output / "tokenizer.json").write_bytes(args.tokenizer.read_bytes())
    history, started = [], time.time()
    for epoch in range(args.epochs):
        model.train()
        total, steps = 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            result = model(batch["input_ids"].to(device), batch["labels"].to(device))
            loss = result["loss"]
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite or missing training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        row = {"epoch": epoch + 1, "train_loss": total / max(steps, 1)}
        if val_loader is not None:
            row["validation_loss"] = evaluate(model, val_loader, device)
        history.append(row)
        torch.save({"model": model.state_dict(), "config": config.__dict__, "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(), "seed": args.seed}, args.output / "checkpoint.pt")
        print(json.dumps(row))
    metrics = {"status": "pretraining_completed_with_held_out_loss" if val_ds else "pretraining_completed_no_independent_evaluation",
        "parameters": count_parameters(model), "vocabulary_size": vocab_size, "train_blocks": len(ds),
        "validation_blocks": len(val_ds) if val_ds else 0, "device": str(device), "epochs": args.epochs,
        "history": history, "elapsed_seconds": round(time.time() - started, 2),
        "limitations": ["This is causal language-model pretraining on the supplied corpus, not evidence of broad competence.",
            "Validation loss does not replace task, factuality, Arabic/English, safety, and regression evaluations.",
            "Scale only after verifying corpus quality, compute, peak memory, and throughput."]}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
