# Ron + Qwen base-model architecture

Ron owns identity, memory, tools, learning policy, evaluation, and source code. Qwen3 is the initial **base language/reasoning model**, not Ron's owner.

User → Ron Core → context/memory/reasoning → Qwen3 base model → Ron validation → response

## What changed

The browser bridge now treats Qwen as the base model for substantive turns. If Qwen is unavailable, Ron falls back to its local reasoning layer rather than exposing provider errors.

Qwen answers are **not automatically written into Ron's long-term memory**. They are recorded separately as training candidates. Explicit user teaching is also recorded with provenance, but promotion into a future model dataset is a separate approval step.

## Knowledge transfer

`ron/distillation.py` implements the controlled transfer boundary:

1. collect a Qwen-generated example;
2. keep it marked as `candidate`;
3. approve only examples that pass Ron's evaluation/curation rules;
4. export approved examples as chat-training JSONL;
5. later fine-tune a Qwen base checkpoint into a Ron-specific model.

This is real training-data transfer, not copying Qwen's internal weights.

## Independence path

Qwen3 base → Ron Core + curated examples → Ron-specific fine-tuned model → optional Qwen removal

Removing Qwen later will not delete Ron's source, memory, identity, tools, or approved training data. General capabilities encoded only in Qwen's weights do not automatically migrate; they must be learned through actual training/fine-tuning.

## Model weights

Model weights are deliberately not committed to Git history. The adapter is provider-agnostic and can point at a local OpenAI-compatible runtime.

Qwen3's official project documents SFT, LoRA, and Q-LoRA workflows, so the approved dataset produced here is designed to become training input rather than being treated as permanent memory.