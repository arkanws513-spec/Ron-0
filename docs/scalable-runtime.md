# Running the trained Ron-180M checkpoint

The GitHub Pages site is static and cannot execute PyTorch. The browser chat must reach this Python service to use the trained weights.

## 1. Automatic inference bundle

The service automatically downloads a compact, inference-only FP16 bundle from the [Ron-180M runtime release](https://github.com/arkanws513-spec/Ron-0/releases/tag/ron-180m-runtime-2026-10-11) on first startup. It contains only model weights, configuration, tokenizer, and metrics; optimizer state and teacher data are excluded. If the release is not available yet, use the original [training artifact](https://github.com/arkanws513-spec/Ron-0/actions/runs/38109801662) and extract these files manually:

- `artifacts/ron-180m-pythia-distilled/checkpoint.pt`
- `artifacts/ron-180m-pythia-distilled/config.json`
- `artifacts/ron-180m-pythia-distilled/tokenizer.json`
- `artifacts/ron-180m-pythia-distilled/metrics.json`

The artifact also includes the teacher-data JSONL; it is not needed for inference.

## 2. Install and run

```bash
python -m pip install -e ".[scalable]"
python -m ron.scalable_server
```

Open `http://localhost:8000`. The server loads the checkpoint once at startup and exposes `/api/health`. That endpoint reports the actual parameter count and checkpoint SHA-256 so the loaded model can be verified.

Optional environment variables:

- `RON_SCALABLE_CHECKPOINT`: path to `checkpoint.pt`
- `RON_SCALABLE_CONFIG`: path to `config.json`
- `RON_SCALABLE_TOKENIZER`: path to `tokenizer.json`
- `RON_MAX_NEW_TOKENS`: generation cap (default 96, maximum 128)
- `RON_ALLOWED_ORIGIN`: browser origin allowed by CORS (default `*`; set this to the exact Pages origin for public deployment)

## Important limitations

The current workflow trained a bounded pilot using only 16 teacher prompts and 5 optimizer updates. It is not a full transfer of Pythia's knowledge and should not be described as a capable general assistant yet. This server deliberately fails to start when the artifact is missing or incompatible; it does not silently fall back to the old JavaScript assistant.

The inference bundle stores weights in FP16 (roughly 360 MB for parameters), plus runtime memory and the tokenizer. A Python-capable host with sufficient RAM is still required. The server reports actual precision and checkpoint SHA-256 at /api/health.


## Ron self-improvement roadmap

Ron should improve through a measured, reproducible candidate-evaluation loop rather than editing its live runtime or replacing its own weights during a conversation.

### Improvement cycle

1. **Observe:** record opt-in, privacy-reviewed failure examples and aggregate evaluation results. Do not automatically turn arbitrary user conversations into training data.
2. **Diagnose:** classify failures (knowledge, Arabic fluency, instruction following, reasoning, tokenizer, serving, or decoding) and link each to a reproducible test.
3. **Propose:** generate a candidate change to code, data, training configuration, or model weights on an isolated branch/workspace. Keep the current release immutable.
4. **Train:** run a bounded experiment with fixed seeds, recorded data/version hashes, resource limits, and saved checkpoints.
5. **Evaluate:** run unit/integration tests plus held-out Arabic, English, multi-turn, knowledge, and reasoning benchmarks. Compare against the current champion using the same test set.
6. **Gate:** reject candidates that fail tests, regress key metrics, produce non-finite losses, or exceed resource limits. A lower training loss alone is not sufficient.
7. **Promote and rollback:** publish a versioned candidate only after all gates pass; keep the previous known-good checkpoint and a one-step rollback path. Initially require human approval before promotion or repository writes.
8. **Learn from results:** preserve metrics and failure categories to guide the next experiment, while preventing evaluation-set leakage into training.

### Safety and control boundaries

- The improvement agent may propose changes and run sandboxed tests; it must not have unrestricted shell, secret, production-deployment, or repository-write access by default.
- Never execute model-generated code on the live service. Run candidate code in a disposable, resource-limited environment.
- Never let Ron rewrite its evaluation tests to make itself pass. Keep benchmark definitions and held-out examples outside the candidate's editable workspace.
- Keep credentials out of prompts, logs, datasets, and checkpoints. Require review for changes to permissions, networking, deployment, or data collection.
- Training data must have documented provenance and licensing, with deduplication and privacy filtering.
- Track model version, tokenizer/config version, dataset hash, code commit, evaluation results, and checkpoint hash for every candidate.

### Initial implementation milestones

- **M0 — Observability:** structured inference logs without sensitive prompt retention by default, request IDs, model/checkpoint identity, error categories, and health checks.
- **M1 — Evaluation harness:** a versioned fixed test suite for Arabic/English, multi-turn context, instruction following, basic reasoning, and serving failures.
- **M2 — Experiment runner:** reproducible fine-tuning/distillation runs with bounded steps, validation, early stopping, and checkpoint metadata.
- **M3 — Candidate gate:** compare a candidate to the champion, reject regressions, and publish a report/artifact without replacing production.
- **M4 — Reviewed promotion:** create a proposed release or pull request for review; automate promotion only after the gates have proved reliable.

This roadmap describes intended behavior, not a claim that these automation features are already implemented. The current Pythia-distillation pilot remains too small to establish broad assistant capability.
