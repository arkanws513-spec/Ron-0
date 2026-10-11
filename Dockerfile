FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RON_HOST=0.0.0.0 \
    RON_PRECISION=fp16

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ron ./ron
COPY docs ./docs
RUN python -m pip install --upgrade pip && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.6.0+cpu" "tokenizers>=0.20" && \
    python -m pip install --no-deps -e .

EXPOSE 8000
CMD ["python", "-m", "ron.scalable_server"]
