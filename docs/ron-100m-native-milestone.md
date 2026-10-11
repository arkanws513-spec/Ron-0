# Ron 100M native student milestone

## Scope

The scalable subword decoder is now available on `main` alongside the preserved legacy character-level model. The `ron-100m` preset defines a new architecture with exactly **99,959,808 trainable parameters** when using the required 8,000-token tokenizer. It does not resize or overwrite the legacy checkpoint.

## Trainable entry points

Build a tokenizer from a sufficiently large, legally usable, reviewed corpus:

```bash
python scripts/train_ron_tokenizer.py --corpus data/reviewed_corpus.txt --output artifacts/ron-tokenizer --vocab-size 8000
```

Run native causal pretraining with a reviewed local corpus:

```bash
python scripts/pretrain_scalable_ron.py --corpus data/reviewed_corpus.txt --tokenizer artifacts/ron-tokenizer/tokenizer.json --output artifacts/ron-100m-pretrain --scale-preset ron-100m --sequence-length 256 --batch-size 1 --epochs 1
```

For supervised fine-tuning, supply JSONL records whose `review_status` is explicitly `approved`:

```bash
python scripts/train_scalable_sft.py --data data/reviewed_teacher.jsonl --tokenizer artifacts/ron-tokenizer/tokenizer.json --output artifacts/ron-100m-sft --scale-preset ron-100m --sequence-length 256 --batch-size 1 --epochs 1
```

These commands start training only when run in an environment with the dependencies, data, and memory available. They are not evidence that training has already run. The preset's parameter-count test uses PyTorch's meta device to validate tensor shapes without allocating the full model.

## Gate before scaling up

Do not proceed to 1B, 100B, or 500B merely because the configuration instantiates. First run the 100M model on enough diverse licensed data to establish:
- finite training loss and reproducible checkpoint save/reload;
- held-out loss improvement and task-based evaluations in Arabic and English;
- no material regression in generation, safety, or factuality tests;
- measured peak RAM/VRAM, tokens/second, and estimated end-to-end cost;
- verified checkpoints and data provenance.

A 100B or 500B model requires distributed training, very large datasets, and compute well beyond free Colab's guaranteed capacity. Those stages should be separate go/no-go decisions, not automatically launched. Qwen2.5-7B uses a different tokenizer and architecture; its tensors cannot be copied directly into this Ron architecture. Use Qwen as an optional teacher to generate reviewed examples, then train Ron's own weights, unless a separately engineered and validated architecture-compatible conversion is implemented.
