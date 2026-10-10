# Qwen2.5-7B teacher distillation and Ron scaling

## Decision

Ron-0 will use Qwen2.5-7B-Instruct as an optional local teacher for generating candidate training examples. Qwen is not called during Ron inference after a distilled Ron checkpoint is trained. Teacher responses are generated data, not verified facts; review and evaluation are required before promotion.

This is not a direct weight conversion. The existing RonCausalLM is a character-level model with learned positional embeddings, LayerNorm, GELU MLPs, and ordinary multi-head attention. Qwen2.5 uses a different tokenizer and architecture. Tensor copying would not produce a compatible Ron checkpoint.

## Generate a teacher corpus locally

Install the optional dependencies:

    pip install -e '.[distill]'

Run against a locally downloaded teacher (recommended when network access is unavailable or weights should remain local):

    python scripts/distill_qwen25.py \
      --model /path/to/Qwen2.5-7B-Instruct \
      --local-files-only \
      --prompts training/distillation/prompts.txt \
      --output-jsonl training/distillation/qwen25_7b_teacher.jsonl \
      --output-corpus training/distillation/ron_distilled_corpus.txt

The model must fit the host's RAM/VRAM. device_map="auto" requires a compatible Accelerate setup; CPU-only inference is possible but may be very slow. No hosted inference API is used. The script never updates Ron's weights. It marks every generated record needs_human_review, normalizes whitespace, removes exact prompt/response duplicates, and writes both JSONL records and blank-line-separated training documents.

The default prompt set is a small smoke-test seed, not a sufficient training corpus. For a meaningful run, prepare a much larger, diverse, legally usable prompt file. Include source metadata and keep held-out evaluation prompts out of the teacher-training corpus. Review for incorrect answers, duplicated templates, privacy-sensitive data, unsafe content, and license restrictions.

## Train Ron's current native student

training/distillation/ron_distilled_corpus.txt is formatted for the current character-level trainer. The trainer's corpus path is configurable with RON_CORPUS_PATH; after reviewing the generated corpus, run:

    RON_CORPUS_PATH=training/distillation/ron_distilled_corpus.txt \
      python scripts/train_native_10m.py

This trains Ron's own weights; it does not preserve all 7B teacher capabilities. The corpus must satisfy the trainer's minimum size and be sufficiently diverse. Keep the existing checkpoint backed up and compare held-out validation and conversation evaluations before promotion.

## Scaling toward billions

Do not change the current 10M dimensions to billion-scale blindly. Increasing dimensions changes tensor shapes and requires a new checkpoint; it does not expand learned knowledge. Training an actually useful billion-parameter model from scratch requires a large licensed token corpus, a subword tokenizer, a scalable architecture, accelerator memory/compute, distributed training, checkpoints, and independent evaluation.

Before a billion-scale training run:
1. Replace character tokenization with a tested subword tokenizer (for example, a BPE tokenizer) and version the tokenizer with each checkpoint.
2. Add a scalable decoder design (RoPE, RMSNorm, SwiGLU, and grouped-query attention where appropriate), preserving the current model/provider as a compatibility baseline.
3. Add model config profiles and exact parameter-count tests; use scripts/estimate_model_scale.py only for planning, not as a promise that a profile can train on the current machine.
4. Run 10M/100M pilot experiments and benchmark loss, generation quality, memory, throughput, and data quality.
5. Scale only when corpus size and hardware are verified. Full AdamW training commonly requires substantially more memory than weights alone; the estimator's optimizer figure excludes activations and temporary buffers.

A parameter-count target is not a quality target. Promotion requires held-out evaluation and reproducible metrics, and no training job should be described as completed unless it actually ran and produced a validated checkpoint.
