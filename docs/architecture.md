# Ron-0 Architecture

## Goal

Build Ron-0 as an independent local assistant whose original reasoning, memory, learning, and generation components are owned by this repository. Ron-0 is distinct from Ron-1.

## Layers

1. Conversation UI — renders messages only; it is not an independent assistant.
2. Python RonCore — the sole production orchestrator for intent, context, memory, reasoning, tools, and generation.
3. Local reasoning and deterministic tools — handles verifiable operations before general generation.
4. Session memory — isolated per conversation and separate from model weights.
5. Native provider — loads Ron's own Transformer checkpoint once and shares its weights across sessions.
6. Evaluation and learning — validates checkpoint loading, held-out loss, conversation quality, and regressions before promotion.

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


## Browser-to-core contract

- `python -m ron.server` serves the UI and `POST /api/chat` from the same origin.
- `GET /api/health` reports `native_weights_loaded`, checkpoint path, selected inference step, and whether an external model was used.
- The server defaults to `RON_REQUIRE_NATIVE=1` and fails closed when no usable native checkpoint is present; it must not silently switch to `LocalTeachingProvider` in production.
- The UI displays an explicit connection error if the Python service is missing. It does not route a live user turn to a JavaScript reasoning agent.
- A static-only host such as GitHub Pages cannot run the Python Transformer. Deploy the UI and Python service together on a Python-capable host.

## Checkpoint promotion and training provenance

Training scripts must write a compatible native checkpoint to `ron/checkpoints/ron_native_10m.pt` (preferred) or `ron/checkpoints/ron_native_baseline.pt`. The native provider validates that the checkpoint vocabulary has unique contiguous IDs and matches the model's configured vocabulary size. `/api/health` exposes the selected training step so the live runtime can be checked rather than inferred from the existence of a training script.
