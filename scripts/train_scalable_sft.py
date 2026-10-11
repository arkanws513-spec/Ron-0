"""Train a new scalable Ron checkpoint from reviewed prompt/response JSONL records."""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from ron.scalable_model import ScalableRonConfig, ScalableRonLM, count_parameters

class PromptResponseDataset(Dataset):
    def __init__(self, path: Path, tokenizer_path: Path, max_length: int, limit: int | None = None) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise SystemExit("Install tokenizer tooling with: pip install -e '.[distill]'") from exc
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.max_length = max_length
        vocab = self.tokenizer.get_vocab()
        self.pad_id = vocab.get("<pad>", 0)
        self.bos_id = vocab.get("<bos>", 2)
        self.eos_id = vocab.get("<eos>", 3)
        self.user_id = vocab.get("<user>", 4)
        self.assistant_id = vocab.get("<assistant>", 5)
        self.rows: list[dict[str, str]] = []
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_no}") from exc
                if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
                    raise ValueError(f"line {line_no} must contain string prompt and response fields")
                if row.get("review_status") != "approved":
                    continue
                if row["prompt"].strip() and row["response"].strip():
                    self.rows.append(row)
                if limit is not None and len(self.rows) >= limit:
                    break
        if not self.rows:
            raise ValueError("no usable records; JSONL must contain reviewed records (review_status=approved)")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prompt = self.tokenizer.encode(row["prompt"]).ids
        answer = self.tokenizer.encode(row["response"]).ids
        # Preserve some prompt and prioritize the answer; the assistant prefix is a dedicated token.
        room = self.max_length - 4
        if len(answer) > room - 1:
            answer = answer[:room - 1]
        prompt = prompt[:max(0, room - len(answer))]
        ids = [self.bos_id, self.user_id, *prompt, self.assistant_id, *answer, self.eos_id]
        assistant_pos = 2 + len(prompt)
        labels = [-100] * (assistant_pos + 1) + ids[assistant_pos + 1:]
        ids, labels = ids[:self.max_length], labels[:self.max_length]
        labels = labels[:len(ids)]
        pad = self.max_length - len(ids)
        ids += [self.pad_id] * pad
        labels += [-100] * pad
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="Reviewed JSONL containing prompt/response and review_status=approved")
    p.add_argument("--validation-data", type=Path, help="Separate held-out reviewed JSONL; never reuse training records")
    p.add_argument("--tokenizer", type=Path, default=Path("artifacts/ron-tokenizer/tokenizer.json"))
    p.add_argument("--output", type=Path, default=Path("artifacts/ron-scalable"))
    p.add_argument("--init-checkpoint", type=Path, help="Optional compatible scalable Ron pretraining checkpoint")
    p.add_argument("--vocab-size", type=int, help="Expected tokenizer vocabulary size; defaults to the tokenizer file's actual size")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--limit", type=int, help="Limit reviewed records for a small pilot")
    p.add_argument("--seed", type=int, default=1701)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        p.error("epochs, batch-size and learning-rate must be positive")
    if not args.data.is_file() or not args.tokenizer.is_file():
        p.error("both --data and --tokenizer must point to existing files")
    if args.validation_data is not None and not args.validation_data.is_file():
        p.error("--validation-data must point to an existing file")
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        p.error("--init-checkpoint must point to an existing checkpoint")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PromptResponseDataset(args.data, args.tokenizer, args.sequence_length, args.limit)
    val_ds = PromptResponseDataset(args.validation_data, args.tokenizer, args.sequence_length) if args.validation_data else None
    # Validate tokenizer/model vocabulary agreement before allocating training batches.
    from tokenizers import Tokenizer
    actual_vocab = Tokenizer.from_file(str(args.tokenizer)).get_vocab_size()
    config_vocab_size = args.vocab_size if args.vocab_size is not None else actual_vocab
    if actual_vocab != config_vocab_size:
        p.error(f"tokenizer vocabulary ({actual_vocab}) must match --vocab-size ({config_vocab_size})")
    config = ScalableRonConfig(vocab_size=config_vocab_size, hidden_size=args.hidden_size,
        num_layers=args.layers, num_heads=args.heads, num_kv_heads=args.kv_heads,
        max_sequence_length=args.sequence_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds is not None else None
    model = ScalableRonLM(config).to(device)
    if args.init_checkpoint is not None:
        try:
            initial_state = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        except TypeError:
            initial_state = torch.load(args.init_checkpoint, map_location=device)
        saved_config = initial_state.get("config", {})
        if saved_config != config.__dict__:
            p.error("initial checkpoint config must exactly match the requested student config")
        model.load_state_dict(initial_state["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    config.save(args.output / "config.json")
    (args.output / "tokenizer.json").write_bytes(args.tokenizer.read_bytes())
    started = time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(ids, labels)
            loss = result["loss"]
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite or missing training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            count += 1
        epoch_loss = total / max(count, 1)
        row = {"epoch": epoch + 1, "train_loss": epoch_loss}
        if val_loader is not None:
            model.eval()
            val_total, val_count = 0.0, 0
            with torch.inference_mode():
                for batch in val_loader:
                    result = model(batch["input_ids"].to(device), batch["labels"].to(device))
                    val_loss = result["loss"]
                    if val_loss is None or not torch.isfinite(val_loss):
                        raise RuntimeError("non-finite or missing validation loss")
                    val_total += float(val_loss)
                    val_count += 1
            row["validation_loss"] = val_total / max(val_count, 1)
        history.append(row)
        torch.save({"model": model.state_dict(), "config": config.__dict__, "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(), "seed": args.seed}, args.output / "checkpoint.pt")
        print(json.dumps(history[-1]))
    metrics = {"status": "training_completed_with_held_out_loss" if val_ds is not None else "training_completed_no_independent_evaluation",
        "parameters": count_parameters(model), "device": str(device), "dataset_records": len(ds),
        "validation_records": len(val_ds) if val_ds is not None else 0, "epochs": args.epochs,
        "history": history, "elapsed_seconds": round(time.time() - started, 2),
        "initialized_from_pretraining_checkpoint": args.init_checkpoint is not None,
        "limitations": ["This is a pilot SFT run, not pretraining from scratch.", "Only assistant response tokens contribute to loss.", "All training and validation records must be reviewed and marked review_status=approved.", "Held-out loss does not replace task-based quality, factuality, Arabic/English coverage, safety, or regression evaluations.", "Do not treat the checkpoint as production-ready without independent evaluation."]}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
