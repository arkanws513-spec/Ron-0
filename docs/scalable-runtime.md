# Running the trained Ron-180M checkpoint

The GitHub Pages site is static and cannot execute PyTorch. The browser chat must reach this Python service to use the trained weights.

## 1. Obtain the training artifact

Open [the successful Ron-180M training run](https://github.com/arkanws513-spec/Ron-0/actions/runs/38109801662), download the artifact named `ron-180m-pythia-distillation-pilot`, and extract it so these files exist:

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

The 180M checkpoint uses roughly 720 MB for float32 model weights alone, plus runtime memory and the tokenizer. Verify available RAM before deployment. The server can run locally; a live public deployment additionally requires a Python-capable host with enough memory and the three artifact files.
