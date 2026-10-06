# Ron native model roadmap

Ron is now being developed toward a model that belongs to the project rather than a permanent dependency on a commercial provider.

## Milestone 1 — interfaces
- tokenizer
- training examples and provenance
- model configuration
- evaluation cases and reports

## Milestone 2 — tiny trainable model
- causal Transformer forward pass
- next-token loss
- CPU smoke-training on a tiny corpus
- checkpoint save/load

## Milestone 3 — real corpus pipeline
- clean and deduplicate licensed/public-domain data
- train a subword tokenizer
- reproducible train/validation splits
- dataset quality reports

## Milestone 4 — useful small model
- larger architecture
- instruction tuning
- Arabic-first evaluation
- memory/tool context integration

## Milestone 5 — Ron model family
- versioned checkpoints
- benchmark gates
- verified promotion and rollback
- optional distributed training

The goal is not to pretend that a tiny prototype is equivalent to GPT or Gemini. The goal is to make every layer replaceable and testable until Ron has a genuinely trainable language model of its own.
