# Scalable Ron student training (experimental)

This path is intentionally separate from the legacy character-level `RonCausalLM` and its checkpoints. It introduces a subword BPE tokenizer and a decoder with RoPE, RMSNorm, SwiGLU, and grouped-query attention. It does not copy Qwen weights and does not make Ron a 7B model automatically.

## 1. Build a tokenizer from reviewed local text

Install optional tooling:

    pip install -e '.[distill]'

Use a UTF-8 corpus you have the right to use. The starter teacher corpus is small and unreviewed; do not treat it as sufficient pretraining data.

    python scripts/train_ron_tokenizer.py --corpus training/corpus.txt --output artifacts/ron-tokenizer --vocab-size 16000

The tokenizer trainer may produce fewer entries than requested if the corpus is too small. The SFT trainer therefore infers the actual vocabulary size from `tokenizer.json` unless `--vocab-size` is explicitly passed.

## 2. Generate and review teacher examples

Generate candidate records using the existing local-only-by-default Qwen script:

    python scripts/distill_qwen25.py --model /local/path/Qwen2.5-7B-Instruct --local-files-only

The output JSONL is **not approved training data**. Check each answer for correctness, privacy, duplication, license, and safety. For records that pass review, set `review_status` to `approved`; the trainer skips all other statuses. Keep held-out evaluation prompts and answers separate from training.

## 3. Run a small supervised fine-tuning pilot

    python scripts/train_scalable_sft.py \
      --data training/distillation/qwen25_7b_teacher.jsonl \
      --tokenizer artifacts/ron-tokenizer/tokenizer.json \
      --output artifacts/ron-scalable \
      --validation-data training/distillation/heldout_reviewed.jsonl \
      --epochs 1 --batch-size 2 --limit 64

The loss is applied to assistant response tokens only, not user prompt tokens. A checkpoint, config, tokenizer copy, and training metrics are saved locally. If you provide a genuinely separate `--validation-data` file, response-only validation loss is measured each epoch. Never reuse training examples in that file. CPU execution is supported but can be slow. Start with a small limit and short sequence length; measure memory before scaling.

This is **SFT of a newly initialized student**, not Qwen-to-Ron weight conversion and not general pretraining. It will not inherit the teacher's full capabilities. The generated metrics explicitly say that independent evaluation is still required.

## 4. Scaling gates

Do not jump directly from a small pilot to billions of parameters. Require:
- a large, diverse, legally usable corpus and held-out tests;
- tokenizer coverage checks for Arabic and English;
- repeatable validation loss and task/conversation evaluations;
- measured peak RAM/VRAM, throughput, and checkpoint recovery;
- a staged 10M → 100M → larger pilot plan with cost and compute verified.

Keep the old character-level checkpoint intact. Do not promote a checkpoint to production based only on training loss or parameter count.


## 5. Run the trained student locally

After a successful training run, start interactive local chat:

    python scripts/chat_scalable_ron.py

Or send one prompt:

    python scripts/chat_scalable_ron.py --prompt "Explain how a neural network learns."

This loads only Ron's checkpoint and its tokenizer; Qwen is not needed for inference. An untrained or tiny pilot checkpoint may produce weak or repetitive answers. That is expected and must not be described as production-grade intelligence.


Estimate the new architecture independently from the legacy character model:

    python scripts/estimate_model_scale.py --architecture scalable --vocab-size 16000 --hidden-size 256 --layers 4 --heads 4 --kv-heads 2 --sequence-length 512

The memory result is a planning estimate, not a guarantee that a given machine can train that configuration. Training peak memory also depends on activations, optimizer implementation, precision, sequence length, and runtime buffers.
