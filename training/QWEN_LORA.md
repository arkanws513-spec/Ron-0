# Ron-0: Qwen LoRA training track

## Target and boundary

The only supported training target in this repository is the same **Qwen2.5-0.5B-Instruct** model used by Ron's live HTTP service through `llama-cpp-python`. This track fine-tunes that matching Hugging Face base model with LoRA; it does not train a separate Ron-native model or touch live weights.

The output is an adapter candidate only. Training never changes Railway settings or replaces the production model automatically.

## Data

1. `training/arabic_sft_seed.txt` contains curated Arabic examples.
2. `python scripts/download_oasst1_english.py` fetches OASST1 conversation trees and exports separately filtered English and Arabic dialogue transcripts. The script writes source hashes and corpus counts to `training/oasst1_manifest.json`.
3. The trainer turns each assistant turn into a supervised example, preserving recent context and applying loss only to the assistant completion.
4. Exact prompts in `training/qwen_eval_cases.json` are excluded from training. This reduces direct leakage but is not a guarantee against semantic overlap; review the corpus and evaluation set before training.

OASST1 is declared Apache-2.0 by its dataset card; retain source/license records and review data quality. Automated filters are imperfect, so inspect samples before using generated corpora.

## GitHub Actions GPU workflow

`.github/workflows/qwen-lora-training.yml` defines a manual GPU training workflow and uploads the adapter/report as a workflow artifact. It deliberately targets a runner labeled `self-hosted, linux, x64, gpu`; it will not run on the standard `ubuntu-latest` runner because that runner has no CUDA GPU. To use this path, the workflow must first be merged into the repository's default branch and a compatible GPU runner must be registered and online. GitHub-hosted GPU larger runners require an eligible organization/enterprise plan and explicit runner configuration; they are not enabled automatically for this personal repository. No paid runner is provisioned by this repository change.

## Train without a local GPU

If you do not have a local CUDA GPU, open the [Google Colab training notebook](https://colab.research.google.com/github/arkanws513-spec/Ron-0/blob/training/qwen-lora-finetune/notebooks/ron0_qwen_lora_colab.ipynb). Select a GPU runtime if one is available and save outputs to your Google Drive. Free GPU availability and session limits vary; this notebook does not deploy or replace production weights.

## Train on a CUDA GPU

Use Python 3.11 or 3.12 with enough GPU memory for Qwen2.5-0.5B plus activations. A 12 GB GPU is a comfortable starting point; reduce sequence length/batch if memory is tight.

```bash
python -m pip install -r requirements-training.txt
python scripts/download_oasst1_english.py
python scripts/train_qwen_lora.py --steps 500 --batch-size 1 --gradient-accumulation 8 --max-sequence-length 512
```

The trainer refuses to start on CPU unless `--allow-cpu` is explicitly supplied. It requires at least 100 assistant-turn examples in each language, uses a deterministic per-language split grouped by the root user prompt (to prevent overlapping dialogue prefixes from leaking across train/validation), masks prompt tokens from the SFT loss, and saves the best validation-loss adapter under `artifacts/qwen-lora-candidate/adapter`.

## Blind evaluation

```bash
python scripts/evaluate_qwen_lora.py \
  --adapter artifacts/qwen-lora-candidate/adapter \
  --cases training/qwen_eval_cases.json \
  --report artifacts/qwen-lora-candidate/evaluation.json \
  --min-pass-rate 0.80 --min-language-pass-rate 0.80
```

The set contains 300 prompt variants grouped into 100 question families. The gate requires unique case IDs, at least 100 prompts per language, at least 40 critical cases, and at least 80% pass rates both per prompt and per question family; each language and the critical question families must independently meet the same minimum. This prevents three paraphrases of one question from counting as three independent successes. This is a necessary filter, not a proof of general intelligence. Review failures and compare the base model and candidate on the same cases before promotion.

## Merge and export

Only after training and blind evaluation pass:

```bash
python scripts/merge_qwen_lora.py \
  --adapter artifacts/qwen-lora-candidate/adapter \
  --output artifacts/qwen-merged
```

Then use a current `llama.cpp` checkout to convert the merged Hugging Face model and quantize it to the runtime's Q4_K_M format:

```bash
python /path/to/llama.cpp/convert_hf_to_gguf.py artifacts/qwen-merged \
  --outfile artifacts/qwen-merged-f16.gguf --outtype f16
/path/to/llama.cpp/build/bin/llama-quantize \
  artifacts/qwen-merged-f16.gguf \
  artifacts/qwen-ron-candidate-q4_k_m.gguf Q4_K_M
```

Run the same blind test on the quantized GGUF through Ron's real `llama.cpp` provider. Quantization can change outputs, so passing the Hugging Face adapter test alone is not enough.

## Deployment gate

Do **not** overwrite the existing Hugging Face model or Railway environment variables before the GGUF candidate has passed evaluation. Publish it as a versioned model artifact first, then test it with a staging service. Change `RON_MODEL_ID` and `RON_MODEL_FILE` only after the staged service is healthy and the generated answers have been checked. Keep the current production model available for rollback.

## Current status

This repository change prepares the data pipeline, LoRA trainer, blind evaluator, and merge step. It does not claim a training run has occurred: a CUDA GPU run and its artifacts are still required.
