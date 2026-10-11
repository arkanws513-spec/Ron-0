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
