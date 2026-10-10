# Ron-0 — رون

Ron-0 is an independent assistant project built around Ron's own local code, reasoning rules, memory, and native language model. **Ron-0 is separate from Ron-1.**

## Non-negotiable design

- Build and improve Ron's core, reasoning engine, learning system, and generation model inside this repository.
- Do not require Qwen, OpenAI, Google, or any other external model or service at runtime.
- Train Ron's native model from Ron's own reviewed corpus; never treat a completed training run as proof of quality.
- Keep memory, deterministic tools, model weights, and evaluation separate so failures can be located and measured.
- Do not promote a candidate checkpoint unless it passes the repository's quality gate.

## Core architecture

`Browser UI → Python RonCore → local memory + reasoning/tools → Ron's native Transformer → response`

The browser is a user interface only. In normal browsers, every chat turn is sent to `/api/chat`; the UI does not silently fall back to a second JavaScript assistant if the Python service is unavailable. The Python service loads one shared native checkpoint and gives each session its own conversation/memory state. The same process exposes `/api/health`, which reports whether native weights really loaded, the checkpoint selected, and the training step metadata.

Run the app and API together from the repository root:

```bash
python -m pip install -e ".[model]" pytest
python -m ron.server
```

Open `http://127.0.0.1:8000`. The server requires a usable native checkpoint by default and refuses to start with only the teaching fallback. For isolated API tests only, set `RON_REQUIRE_NATIVE=0`; this mode is not evidence that the Transformer is active. Static GitHub Pages cannot run Python inference, so it cannot serve as the full single-core deployment by itself; deploy the Python service and UI together on a Python-capable host.

Clear deterministic tasks such as arithmetic should be handled by Ron's local code before general language generation. Memory stores facts; training updates model weights. They are not the same process.

## Native model and training

- Model: `ron/model.py`
- Configuration: `ron/model_config.py`
- Training recipe: `scripts/train_native.py`
- Reviewed seed corpus: `training/seed_corpus.txt`
- Anti-degeneration gate: `scripts/surprise_eval.py` (checks basic generation behavior, not semantic correctness)
- Runtime provider: `ron/native_provider.py`

Run the native training script in an environment with the repository's `model` extra installed:

```bash
python -m pip install -e ".[model]" pytest
python -m scripts.train_native
python -m scripts.surprise_eval
python -m scripts.eval_conversation --strict
python -m scripts.eval_conversation --benchmark training/generalization_benchmark.json --output artifacts/generalization-eval-report.json
```

Training and evaluation reports are written under `artifacts/native-baseline/`. The workflow keeps candidate weights separate from the live checkpoint until the anti-degeneration gate passes. That gate is not a semantic benchmark; use `scripts/eval_conversation.py` for the 25-case curriculum regression set and `training/generalization_benchmark.json` for unseen phrasing.

## Current limitations

The checkpoint currently tracked in the repository is the baseline model with **110,272 parameters**, not the intended 10M model. Its saved metrics show training loss falling from 1.128 to 0.893 while validation loss rises from 0.988 to 1.402; the run therefore shows an overfitting/generalization warning, not a verified quality improvement. The `ron_native_10m.pt` checkpoint and `training/corpus.txt` are ignored external artifacts and are not present in the tracked repository snapshot, so the 10M training run cannot be claimed complete from source code alone.

Ron-0's native Transformer remains a small character-level prototype, not a general-purpose language model. Set `RON_REQUIRE_10M=1` when launching the server to refuse startup unless the active checkpoint has at least 8 million parameters. The default health endpoint reports the actual parameter count, checkpoint hash, selected inference step, and whether the 10M target was reached. A lower training loss alone is not evidence that responses improved; held-out validation and blind generation checks must also be reviewed.


## Ron-10M experimental training track

The feat/native-core-audit branch contains a separate experimental training path; it does not change main by itself.

1. scripts/download_english_corpus.py obtains public-domain starter texts from Project Gutenberg and records source hashes.
2. scripts/prepare_corpus.py normalizes and deduplicates the pretraining texts. OASST dialogue files are deliberately excluded from this split to avoid leakage into the later instruction-tuning validation set.
3. scripts/train_native_10m.py trains Ron's own approximately 10M-parameter character-level Transformer using AdamW, a warmup/cosine learning-rate schedule, held-out validation, early stopping, and atomic checkpoint writes.
4. scripts/download_oasst1_english.py downloads the English OASST1 conversation trees and records a source manifest. Check the dataset card and applicable license/terms before reuse.
5. scripts/train_sft_10m.py performs assistant-response-only supervised fine-tuning on the reviewed English conversations plus training/arabic_sft_seed.txt. The seed examples are curated project data; the script expands the character vocabulary while preserving existing token IDs.

The GitHub Actions workflow archives the pretraining candidate, SFT candidate, metrics, and source manifests. A lower validation loss is only a necessary signal, not proof of factual accuracy or general assistant quality. Review both reports and run blind conversation evaluations before using any candidate as the live checkpoint.

### Reproducibility and limitations

- Run python -m pytest before accepting changes.
- The current tokenizer is still character-level. A trained subword tokenizer (BPE/SentencePiece) is a priority before expecting strong multilingual generation.
- The 10M-parameter model is a small experiment, not comparable in capability to frontier-scale assistants.
- No OpenAI API, external inference model, or proprietary OpenAI internal tooling is required or introduced by this workflow. It uses common, reproducible techniques: a decoder-only Transformer, AdamW, learning-rate scheduling, held-out evaluation, supervised fine-tuning, and checkpoint gating.
