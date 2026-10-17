FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RON_HOST=0.0.0.0 \
    RON_CPU_THREADS=2 \
    RON_MODEL_MAX_TOKENS=64 \
    RON_CONTEXT_SIZE=2048 \
    RON_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    RON_MODEL_FILE=qwen2.5-0.5b-instruct-q4_k_m.gguf

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libgomp1 libstdc++6 build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY index.html app.js ron-core.js ron-agent-core.js ron-knowledge-core.js ron-browser-reasoner.js ron-learning-core.js ron-web-search.js manifest.webmanifest ./
COPY assets ./assets
COPY ron ./ron
COPY docs ./docs
COPY training/seed_corpus.txt ./training/seed_corpus.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install --prefer-binary \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
      "llama-cpp-python>=0.3.16" "huggingface-hub>=0.25,<2" && \
    python -m pip install --no-deps -e .

EXPOSE 8000
CMD ["python", "-m", "ron.native_chat_server"]
