# Ron native 180M and Pythia-160M distillation pilot

This milestone adds a native Ron configuration with **180,255,744 trainable parameters** (8,000 model vocabulary, hidden size 1,024, 15 decoder blocks, 16 attention heads, 4 KV heads, intermediate size 2,880). The vocabulary/tokenizer remains Ron's own; Pythia weights are not copied into an incompatible architecture.

## Knowledge-transfer path

1. Build Ron's byte-level BPE tokenizer from the repository's starter corpus.
2. Download `EleutherAI/pythia-160m` from Hugging Face on a manually started GitHub Actions run.
3. Generate a small teacher dataset from user/question lines in the starter corpus.
4. Train Ron's native 180M checkpoint on those teacher-generated responses using Ron's tokenizer and model architecture.
5. Verify the actual parameter count, nonzero optimizer-step count, finite loss, and checkpoint files; upload them as a workflow artifact.

## Important limitations

- `Pythia-160M` is a **base language model**, not an instruction-tuned assistant. Its continuations can be incoherent or factually wrong.
- Teacher outputs are marked `review_status=teacher_generated`, `source=pythia-160m`, and `human_reviewed=false`. They are not mislabeled as human-approved facts. The training command must explicitly opt into these synthetic examples.
- This is a small CPU pilot, not comprehensive distillation and not evidence that Ron has acquired all of Pythia's knowledge. The bounded defaults are 16 prompts and 5 optimizer updates.
- The workflow is manual-only and uses GitHub-hosted CPU resources; no paid GPU is required. Runtime and memory depend on current GitHub runner limits and the model download.
- The resulting checkpoint is an artifact, not automatically wired into Ron's live chat service. Deployment requires a separate compatibility and quality review.
