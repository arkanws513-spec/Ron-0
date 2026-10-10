# Ron-0 — رون

Ron-0 is an independent assistant project built around Ron's own local code, reasoning rules, memory, and native language model. **Ron-0 is separate from Ron-1.**

## Non-negotiable design

- Build and improve Ron's core, reasoning engine, learning system, and generation model inside this repository.
- Do not require Qwen, OpenAI, Google, or any other external model or service at runtime.
- Train Ron's native model from Ron's own reviewed corpus; never treat a completed training run as proof of quality.
- Keep memory, deterministic tools, model weights, and evaluation separate so failures can be located and measured.
- Do not promote a candidate checkpoint unless it passes the repository's quality gate.

## Core architecture

`Browser UI → Ron Core → local memory / rules / tools → Ron native model → quality checks → response`

Clear deterministic tasks such as arithmetic should be handled by Ron's local code before general language generation. Memory stores facts; training updates model weights. They are not the same process.

## Native model and training

- Model: `ron/model.py`
- Configuration: `ron/model_config.py`
- Training recipe: `scripts/train_native.py`
- Reviewed seed corpus: `training/seed_corpus.txt`
- Blind evaluation: `scripts/surprise_eval.py`
- Runtime provider: `ron/native_provider.py`

Run the native training script in an environment with the repository's `model` extra installed:

```bash
python -m pip install -e ".[model]" pytest
python -m scripts.train_native
python -m scripts.surprise_eval
```

Training and evaluation reports are written under `artifacts/native-baseline/`. The workflow keeps candidate weights separate from the live checkpoint until the blind anti-degeneration gate passes.

## Current limitations

Ron-0's native Transformer is a small character-level prototype trained on a curated corpus. It is not yet a general-purpose language model. A lower training loss alone is not evidence that responses improved; held-out validation and blind generation checks must also be reviewed.
