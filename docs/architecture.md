# Ron-0 Architecture

## Goal

Build Ron-0 as an independent local assistant whose original reasoning, memory, learning, and generation components are owned by this repository. Ron-0 is distinct from Ron-1.

## Layers

1. Conversation UI — receives user messages and displays replies.
2. Ron Core — routes intent, checks context, and coordinates execution.
3. Local reasoning and deterministic tools — handles explicit rules and verifiable operations such as arithmetic.
4. Memory — stores user-approved facts separately from model weights.
5. Native model — Ron's own character-level Transformer and local checkpoint.
6. Evaluation and learning — trains on reviewed examples, validates checkpoints, and blocks low-quality candidates.

## Execution lifecycle

1. Receive the message.
2. Detect deterministic requests before generic language handling.
3. Build the relevant local context.
4. Execute local rules, tools, or native generation.
5. Check the result and avoid promoting unsupported output to memory.
6. Record useful experience with provenance.
7. Evaluate training changes on held-out and blind examples before promoting weights.

## Independence rule

Ron-0 must not require an external model or cloud inference service to run. Any improvement to its core language ability must be implemented and evaluated through Ron-0's own code and training process.


## Ron-10M training stages

The experimental 10M path separates two data roles:

- **Pretraining:** deduplicated, appropriately licensed prose. The preparation script excludes OASST1 conversation files and dialogue/conversation dumps, so the SFT validation set is not accidentally seen in pretraining.
- **Supervised fine-tuning:** English OASST1 conversation examples plus a curated Arabic seed set. Loss is masked on user messages and role labels so gradients target Ron's answer text only. The tokenizer vocabulary can be extended for Arabic characters while preserving existing IDs and weights.
- **Evaluation:** held-out response loss, regression tests, and archived manifests/metrics. Candidate weights must be reviewed separately; a successful process exit or reduced loss is not a claim of factual correctness.

The current model remains character-tokenized. A BPE/SentencePiece tokenizer is a planned architectural improvement; adopting one requires a new tokenizer-aware training run and compatible inference code, not merely swapping a vocabulary file.
