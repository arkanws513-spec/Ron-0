# Ron model training

The Qwen fine-tuning pipeline is deliberately separate from Ron's native character-level prototype.

1. Qwen responses are stored as candidates.
2. Ron approves only curated examples.
3. Approved examples are validated and deduplicated before any model weights are loaded.
4. The examples are split reproducibly into training and held-out validation sets.
5. LoRA SFT evaluates each epoch and restores the checkpoint with the best validation loss.
6. The adapter and a manifest are saved under `artifacts/`.

Run:

```bash
python -m training.finetune \
  --data training/data/approved.jsonl \
  --validation-split 0.1 \
  --seed 42
```

The output is a **LoRA adapter**, not a self-contained base model. The base model weights (Qwen3 by default) are still required to run that adapter unless a separate, tested merge/export step creates a standalone model. The manifest records this explicitly.

Training requires the optional dependencies in `training/requirements.txt` and hardware memory appropriate for the selected base model. The repository's CPU-friendly native training run is a separate experiment and is not equivalent to fine-tuning a large base model.
