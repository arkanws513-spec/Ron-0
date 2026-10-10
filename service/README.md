# Ron native inference service

This API serves Ron's own PyTorch checkpoint. It does not call OpenAI, Gemini, or another hosted model provider.

## Required model mount

Mount a **reviewed and approved 10M+ checkpoint** at `/models/ron_native_10m.pt`, or set `RON_CHECKPOINT_PATH` to another mounted file. By default, the service refuses to become ready if the checkpoint is missing, malformed, or below `RON_MIN_PARAMETERS` (default: 1,000,000). It never silently falls back to the 110k baseline.

## Endpoints

- `GET /health`: readiness and checkpoint metadata; returns HTTP 503 until an acceptable checkpoint is loaded.
- `POST /v1/chat`: accepts `{ "messages": [{"role":"user","content":"Hello"}] }` and returns Ron's native generation plus runtime metadata.

## Environment

- `RON_CHECKPOINT_PATH`: checkpoint file path; default `/models/ron_native_10m.pt`.
- `RON_CHECKPOINT_URL`: optional direct URL used only when the checkpoint file is missing.
- `RON_CHECKPOINT_SHA256`: required when downloading a checkpoint; also verifies an already-mounted file when provided. Downloads are streamed to a temporary file and atomically installed only after the SHA-256 matches.
- `RON_MIN_PARAMETERS`: minimum accepted parameter count; default `1000000`.
- `RON_CORS_ORIGINS`: comma-separated allowed origins. Defaults to the Ron GitHub Pages origin and local development origins.

Do not publish a checkpoint merely because training finished. First run the blind conversational evaluation and promote only a passing candidate.
