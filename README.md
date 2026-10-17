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

## Live checkpoint rollout

The deployed scalable HTTP runtime keeps its current model available as a fallback while checking the public `ron-live-checkpoint` release for a promoted `ron_native_10m.pt` asset. It checks on startup and every five minutes, verifies the published SHA-256 digest when available, load-tests the candidate, and atomically swaps the native provider only after validation succeeds. Once the 10M checkpoint is promoted, open-ended inference uses those weights; failed downloads or invalid candidates leave the active provider untouched.

Use `GET /api/health` to inspect `active_inference_provider`, `active_checkpoint`, `active_checkpoint_sha256`, and `active_parameter_count`. The browser UI calls the configured `window.RON_API_BASE_URL` when set, otherwise the deployed Ron API. Static hosting alone cannot run PyTorch; it must reach the Ron HTTP service for model-generated answers.


## Optional durable Python-core memory

The Python orchestration core uses isolated in-memory storage by default. To persist its structured facts across process restarts, set `RON_MEMORY_DB_PATH` to a local SQLite file before starting Ron:

```bash
# Linux/macOS example; choose a private, writable directory.
export RON_MEMORY_DB_PATH="$HOME/.ron-0/memory.sqlite3"
python -m your_ron_entrypoint
```

The parent directory is created automatically. Existing facts are loaded on startup and updates are committed to SQLite. Leave the variable unset for isolated/in-memory sessions and tests. The browser UI has its own localStorage-based data and is not redirected to this SQLite file by this setting.

**Privacy note:** the database stores memory content and metadata as local plaintext SQLite data. Keep it in a user-controlled directory with appropriate file permissions, and include it in backups only if you intend to preserve those memories.
