# Ron-0 — رون

Ron-0 is an independent assistant project. The active conversational generation engine is now a ready-made, instruction-tuned local language model, not Ron's former tiny character-level prototype.

## Active inference engine

- **Model:** Qwen2.5-0.5B-Instruct, GGUF Q4_K_M quantization (about 491 MB).
- **Inference runtime:** llama.cpp through \`llama-cpp-python\`.
- **Tokenizer:** bundled in the GGUF model; it is loaded with the matching model.
- **Conversation:** instruction-tuned chat format with multi-turn history.
- **External inference API:** not used. Inference runs locally inside the Ron server.
- **License:** Qwen2.5 model card lists Apache-2.0; review the upstream license before redistribution.

The first launch downloads the model file from Hugging Face into its cache. Later launches reuse the cache if the host preserves it. The repository does not commit hundreds of megabytes of model weights.

Model source: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
Runtime source: https://github.com/abetlen/llama-cpp-python

## Run

Python 3.11+ and a C/C++ build toolchain may be required if a compatible prebuilt wheel is not available for the host:

\`\`\`bash
python -m pip install --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu "llama-cpp-python>=0.3.16" "huggingface-hub>=0.25,<2"
python -m pip install --no-deps -e .
python -m ron.native_chat_server
\`\`\`

On first launch, the server downloads the default Q4_K_M model (approximately 491 MB). Set \`RON_MODEL_CACHE\` to a persistent cache directory if the deployment platform supports persistent storage.

## Environment

- \`RON_MODEL_ID\`: defaults to \`Qwen/Qwen2.5-0.5B-Instruct-GGUF\`
- \`RON_MODEL_FILE\`: defaults to \`qwen2.5-0.5b-instruct-q4_k_m.gguf\`
- \`RON_MODEL_REVISION\`: defaults to \`main\`
- \`RON_MODEL_CACHE\`: optional Hugging Face cache directory
- \`RON_CONTEXT_SIZE\`: defaults to 4096 tokens
- \`RON_MAX_NEW_TOKENS\`: defaults to 256
- \`RON_CPU_THREADS\`: defaults to at most 4
- \`RON_ALLOWED_ORIGIN\`: CORS origin; defaults to \`*\`

## API

- \`GET /api/health\` reports the active model and runtime.
- \`POST /api/chat\` accepts JSON containing \`message\`, and optional \`session_id\` and \`request_id\`.
- Static web assets are served from the same process.

## Important limitations

A 0.5B model is a substantial improvement over the old 113K-parameter character model, but it is still small and can make mistakes, especially on complex reasoning or Arabic nuance. The new model has not been fine-tuned on Ron's personal data. Keep deterministic tools (such as arithmetic) separate from language generation when accuracy matters.

The previous native training code and reports remain in the repository for audit and reproducibility, but the live chat server no longer loads or generates from those checkpoints. They are not the active conversational brain.
