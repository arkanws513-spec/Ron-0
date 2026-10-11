FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RON_HOST=0.0.0.0 \
    RON_PRECISION=fp16

WORKDIR /app
COPY pyproject.toml README.md ./
COPY index.html app.js ron-core.js ron-agent-core.js ron-knowledge-core.js ron-browser-reasoner.js ron-learning-core.js ron-web-search.js manifest.webmanifest ./
COPY assets ./assets
COPY ron ./ron
COPY docs ./docs
COPY training/seed_corpus.txt ./training/seed_corpus.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.6.0+cpu" "tokenizers>=0.20" "numpy<2" && \
    python -m pip install --no-deps -e .

EXPOSE 8000
CMD ["python", "-m", "ron.native_chat_server"]
