"""Train Ron-10M, a larger native character-level Transformer on a prepared corpus."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import tempfile
from pathlib import Path

import torch
from torch import nn

from ron.model import RonCausalLM
from ron.model_config import RonModelConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "native-10m"
LIVE_CHECKPOINT = ROOT / "ron" / "checkpoints" / "ron_native_10m.pt"
SEED = 1701
STEPS = int(os.environ.get("RON_STEPS") or "1800")
BATCH = int(os.environ.get("RON_BATCH_SIZE") or "2")
LENGTH = int(os.environ.get("RON_SEQUENCE_LENGTH") or "256")
HIDDEN, LAYERS, HEADS = 384, 6, 6
MIN_CORPUS_CHARACTERS = 100_000
PATIENCE_EVALS = 10
if STEPS < 1 or BATCH < 1 or LENGTH < 8:
    raise ValueError("RON_STEPS must be positive, RON_BATCH_SIZE must be positive, and RON_SEQUENCE_LENGTH must be at least 8")


def atomic_torch_save(payload: dict, destination: Path) -> None:
    """Write checkpoints atomically so interrupted writes cannot corrupt the live file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def split_corpus(corpus: str, train_fraction: float = 0.9, seed: int = SEED):
    """Split at document/paragraph boundaries to reduce train-validation leakage."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    documents = [part.strip() for part in corpus.replace("\r\n", "\n").split("\n\n") if part.strip()]
    if len(documents) < 10:
        raise ValueError("prepared corpus must contain at least 10 non-empty documents/paragraphs separated by blank lines")
    validation_count = max(1, min(len(documents) - 1, round(len(documents) * (1 - train_fraction))))
    validation_indices = set(random.Random(seed).sample(range(len(documents)), validation_count))
    train_documents = [doc for index, doc in enumerate(documents) if index not in validation_indices]
    validation_documents = [doc for index, doc in enumerate(documents) if index in validation_indices]
    return (
        "\n\n".join(train_documents) + "\n",
        "\n\n".join(validation_documents) + "\n",
        len(train_documents),
        len(validation_documents),
    )

def should_stop_early(stale_evaluations: int, patience: int = PATIENCE_EVALS) -> bool:
    """Stop wasting steps when held-out validation stops improving."""
    if patience < 1 or stale_evaluations < 0:
        raise ValueError("patience must be positive and stale_evaluations non-negative")
    return stale_evaluations >= patience


def choose_best_checkpoint(
    current_state: dict,
    current_validation_loss: float,
    current_selected_step: int,
    prior_best_state: dict | None,
    prior_best_validation_loss: float | None,
    prior_best_step: int,
):
    """Keep the lower-loss inference snapshot while training may resume from final weights."""
    if (
        prior_best_state is not None
        and prior_best_validation_loss is not None
        and prior_best_validation_loss < current_validation_loss
    ):
        return dict(prior_best_state), prior_best_validation_loss, prior_best_step, "prior_best"
    return dict(current_state), current_validation_loss, current_selected_step, "current_final"


def validation_improvement_percent(start_loss: float, best_loss: float, source: str) -> float:
    """Measure improvement made during this run, excluding a reused prior checkpoint."""
    if source != "this_run":
        return 0.0
    return 100 * (start_loss - best_loss) / max(start_loss, 1e-9)


def checkpoint_delta_percent(start_loss: float, best_loss: float) -> float:
    """Compare the selected inference checkpoint against the continuation starting state."""
    return 100 * (start_loss - best_loss) / max(start_loss, 1e-9)


def batch(tokens: torch.Tensor, size: int, length: int):
    top = len(tokens) - length - 1
    if top < 1:
        raise ValueError(f"corpus split has {len(tokens)} tokens; needs at least {length + 2}")
    starts = torch.randint(0, top, (size,))
    return (
        torch.stack([tokens[s:s + length] for s in starts.tolist()]),
        torch.stack([tokens[s + 1:s + length + 1] for s in starts.tolist()]),
    )


@torch.no_grad()
def evaluate(model, tokens: torch.Tensor, count: int = 16) -> float:
    """Evaluate on fixed, evenly spaced windows so checkpoint selection is reproducible."""
    was_training = model.training
    model.eval()
    top = len(tokens) - LENGTH - 1
    if top < 1:
        raise ValueError(f"evaluation split has {len(tokens)} tokens; needs at least {LENGTH + 2}")
    starts = torch.linspace(0, top - 1, steps=max(1, min(count, top))).long().unique().tolist()
    values = []
    for start in starts:
        x = tokens[start:start + LENGTH].unsqueeze(0)
        y = tokens[start + 1:start + LENGTH + 1].unsqueeze(0)
        loss = model(x, y).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite evaluation loss")
        values.append(float(loss.item()))
    model.train(was_training)
    return sum(values) / len(values)


def load_or_initialize(corpus: str):
    """Continue from Ron's live checkpoint when compatible; otherwise initialize Ron's own model."""
    corpus_sha256 = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    if LIVE_CHECKPOINT.is_file():
        payload = torch.load(LIVE_CHECKPOINT, map_location="cpu", weights_only=True)
        corpus_changed = payload.get("corpus_sha256") != corpus_sha256
        old_vocab = payload["vocab"]
        vocab = dict(old_vocab)  # Preserve existing token IDs so old weights keep their meaning.
        for char in sorted(set(corpus) - set(vocab)):
            vocab[char] = len(vocab)

        old_cfg = payload["config"]
        cfg = RonModelConfig(
            vocab_size=len(vocab),
            hidden_size=int(old_cfg["hidden_size"]),
            num_layers=int(old_cfg["num_layers"]),
            num_heads=int(old_cfg["num_heads"]),
            max_sequence_length=int(old_cfg["max_sequence_length"]),
            dropout=0.1,
        )
        if cfg.hidden_size != HIDDEN or cfg.num_layers != LAYERS or cfg.num_heads != HEADS or cfg.max_sequence_length != LENGTH:
            raise RuntimeError("Existing Ron checkpoint architecture differs from this training recipe; refusing to overwrite it.")
        model = RonCausalLM(cfg)
        target = model.state_dict()
        # When the corpus changes, resume from the best inference snapshot rather than overfit final weights.
        old_state = (payload.get("best_state_dict", payload["state_dict"])
                     if corpus_changed else payload["state_dict"])
        copied, expanded = 0, False
        for name, current in target.items():
            previous = old_state.get(name)
            if previous is None:
                continue
            if previous.shape == current.shape:
                current.copy_(previous)
                copied += 1
            elif name in {"token_embedding.weight", "lm_head.weight"} and previous.ndim == 2 and previous.shape[1] == current.shape[1]:
                rows = min(previous.shape[0], current.shape[0])
                current[:rows].copy_(previous[:rows])
                copied += 1
                expanded = True
        model.load_state_dict(target)
        if copied == 0:
            raise RuntimeError("Existing checkpoint contained no compatible weights; refusing silent reinitialization.")
        prior_best_state = payload.get("best_state_dict") if not expanded and not corpus_changed else None
        prior_best_step = int(payload.get("best_selected_step", payload.get("selected_step", 0)))
        source = "continued_best_checkpoint_after_corpus_change" if corpus_changed else "continued_existing_ron_checkpoint"
        optimizer_state = None if corpus_changed or expanded else payload.get("optimizer_state_dict")
        return model, vocab, source, int(payload.get("training_steps_total", payload.get("selected_step", 0))), expanded, optimizer_state, prior_best_state, prior_best_step

    vocab = {char: index for index, char in enumerate(sorted(set(corpus)))}
    cfg = RonModelConfig(vocab_size=len(vocab), hidden_size=HIDDEN, num_layers=LAYERS,
                         num_heads=HEADS, max_sequence_length=LENGTH, dropout=0.1)
    return RonCausalLM(cfg), vocab, "initialized_ron_native_model_no_checkpoint_found", 0, False, None, None, 0


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    corpus_path = ROOT / "training" / "corpus.txt"
    if not corpus_path.is_file():
        raise FileNotFoundError(
            "Ron-10M requires training/corpus.txt. Add appropriately licensed text files under "
            "training/corpus/, run scripts/prepare_corpus.py, then rerun this script."
        )
    corpus = corpus_path.read_text(encoding="utf-8")
    if len(corpus) < MIN_CORPUS_CHARACTERS:
        raise ValueError(
            f"Corpus has only {len(corpus):,} characters; Ron-10M requires at least "
            f"{MIN_CORPUS_CHARACTERS:,} characters for this training recipe. Add more reviewed data."
        )
    model, vocab, checkpoint_source, prior_step, vocab_expanded, prior_optimizer_state, prior_best_state_dict, prior_best_step = load_or_initialize(corpus)
    train_text, validation_text, train_pairs, validation_pairs = split_corpus(corpus)
    train = torch.tensor([vocab[char] for char in train_text], dtype=torch.long)
    val = torch.tensor([vocab[char] for char in validation_text], dtype=torch.long)
    if len(train) < LENGTH + 2 or len(val) < LENGTH + 2:
        raise ValueError("Training corpus is too small for the configured sequence length; expand training/seed_corpus.txt.")

    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    start_train, start_val = evaluate(model, train), evaluate(model, val)
    best_step = 0
    current_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    prior_best_validation = None
    if prior_best_state_dict is not None:
        model.load_state_dict(prior_best_state_dict)
        prior_best_validation = evaluate(model, val, count=16)
        model.load_state_dict(current_state)

    best_state_dict, best_val, best_selected_step_total, best_source = choose_best_checkpoint(
        current_state,
        start_val,
        prior_step,
        prior_best_state_dict,
        prior_best_validation,
        prior_best_step,
    )
    base_lr = 1.5e-4 if checkpoint_source.startswith("continued") else 3e-4
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    if prior_optimizer_state and not vocab_expanded:
        try:
            optimizer.load_state_dict(prior_optimizer_state)
            print("restored_optimizer_state=true", flush=True)
        except (ValueError, RuntimeError) as exc:
            # A vocabulary expansion changes embedding shapes; reset optimizer moments but retain every compatible model weight.
            print(f"optimizer_state_reset_due_to_shape_change={exc}", flush=True)
    warmup_steps = min(100, max(1, STEPS // 10))
    started = time.time()
    history = []
    stale_evaluations = 0
    early_stopped = False
    steps_completed = 0
    model.train()
    for step in range(1, STEPS + 1):
        steps_completed = step
        warmup_factor = min(1.0, step / warmup_steps)
        progress = (step - 1) / max(1, STEPS - 1)
        cosine_factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        current_lr = base_lr * warmup_factor * cosine_factor
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        x, y = batch(train, BATCH, LENGTH)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0:
            train_loss, validation_loss = evaluate(model, train), evaluate(model, val, count=16)
            history.append({"step": step, "train_loss": train_loss, "validation_loss": validation_loss, "learning_rate": current_lr})
            print(f"step={step} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}", flush=True)
            if validation_loss < best_val:
                best_val, best_step = validation_loss, step
                best_selected_step_total = prior_step + step
                best_source = "this_run"
                best_state_dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if should_stop_early(stale_evaluations):
                early_stopped = True
                print(f"early_stopping=true step={step} stale_evaluations={stale_evaluations}", flush=True)
                break

    # Persist the final state after this run, not the starting/best snapshot.
    # This guarantees the next run continues from weights actually updated by this training pass.
    final_train, final_validation = evaluate(model, train), evaluate(model, val, count=16)
    OUT.mkdir(parents=True, exist_ok=True)
    total_steps = prior_step + steps_completed
    final_state_dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    delta_squared = sum(float((final_state_dict[name].float() - initial_state[name].float()).pow(2).sum().item())
                        for name in final_state_dict)
    changed_tensors = sum(not torch.equal(final_state_dict[name], initial_state[name]) for name in final_state_dict)
    checkpoint = {
        # Training resumes from the final weights and matching optimizer state.
        "state_dict": final_state_dict,
        # Inference uses the best validation snapshot to reduce overfitting.
        "best_state_dict": best_state_dict,
        "best_selected_step": best_selected_step_total,
        "best_validation_loss": best_val,
        "config": model.config.__dict__,
        "vocab": vocab,
        "seed": SEED,
        "selected_step": total_steps,
        "training_steps_this_run": steps_completed,
        "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "training_steps_total": total_steps,
        "checkpoint_source": checkpoint_source,
        "prior_selected_step": prior_step,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    # Always retain the candidate as an artifact, but never replace the live model
    # unless this run actually improves held-out validation loss.
    OUT.mkdir(parents=True, exist_ok=True)
    candidate_path = OUT / "ron_native_10m.pt"
    atomic_torch_save(checkpoint, candidate_path)
    checkpoint_promoted = bool(
        best_source == "this_run"
        and math.isfinite(best_val)
        and best_val < start_val
        and changed_tensors > 0
    )
    promotion_reason = (
        "held_out_validation_improved"
        if checkpoint_promoted
        else "validation_gate_not_passed_live_checkpoint_preserved"
    )
    if checkpoint_promoted:
        atomic_torch_save(checkpoint, LIVE_CHECKPOINT)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "experiment": "ron0-native-10m-character-lm",
        "status": "completed",
        "checkpoint_promoted": checkpoint_promoted,
        "promotion_reason": promotion_reason,
        "candidate_checkpoint": str(candidate_path.relative_to(ROOT)),
        "live_checkpoint": str(LIVE_CHECKPOINT.relative_to(ROOT)) if checkpoint_promoted else None,
        "checkpoint_source": checkpoint_source,
        "prior_selected_step": prior_step,
        "vocabulary_expanded": vocab_expanded,
        "seed": SEED,
        "optimizer": "AdamW",
        "base_learning_rate": base_lr,
        "learning_rate_schedule": "linear_warmup_then_cosine_decay_to_10_percent",
        "warmup_steps": warmup_steps,
        "training_steps_this_run": steps_completed,
        "selected_checkpoint_step_this_run": steps_completed,
        "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "early_stopped": early_stopped,
        "early_stopping_patience_evaluations": PATIENCE_EVALS,
        "stale_evaluations_at_stop": stale_evaluations,
        "best_validation_step_this_run": best_step,
        "training_steps_total": total_steps,
        "final_train_loss": final_train,
        "final_validation_loss": final_validation,
        "weights_persisted_from_final_training_step": True,
        "best_validation_weights_persisted": True,
        "live_checkpoint_promoted_only_after_validation_improvement": True,
        "best_selected_step_total": best_selected_step_total,
        "best_checkpoint_source": best_source,
        "optimizer_state_persisted": True,
        "changed_parameter_tensors": changed_tensors,
        "parameter_delta_l2": delta_squared ** 0.5,
        "parameters": parameters,
        "vocab_size": len(vocab),
        "corpus_characters": len(corpus),
        "training_pairs": train_pairs,
        "validation_pairs": validation_pairs,
        "split_strategy": "deterministic_document_paragraph_holdout",
        "train_characters": len(train),
        "validation_characters": len(val),
        "initial_train_loss": start_train,
        "initial_validation_loss": start_val,
        "best_validation_loss": best_val,
        "last_train_loss": final_train,
        "last_validation_loss": final_validation,
        "train_loss_reduction_percent": 100 * (start_train - final_train) / max(start_train, 1e-9),
        "validation_loss_reduction_percent": validation_improvement_percent(start_val, best_val, best_source),
        "best_checkpoint_validation_loss_delta_vs_continuation_start_percent": checkpoint_delta_percent(start_val, best_val),
        "elapsed_seconds": round(time.time() - started, 2),
        "history": history,
        "limitations": [
            "This is an approximately 10-million-parameter character-level prototype, not a general-purpose large language model.",
            "Validation is a deterministic held-out set of documents/paragraphs from the supplied corpus, not an independent benchmark.",
            "The checkpoint stores both final weights for training continuation and best-validation weights for inference; the two snapshots serve different purposes.",
            "No external inference model or paid API is used.",
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
