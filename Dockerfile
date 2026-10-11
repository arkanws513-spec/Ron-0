FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RON_HOST=0.0.0.0 \
    RON_PRECISION=fp16

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ron ./ron
COPY docs ./docs
RUN python -m pip install --upgrade pip && python -m pip install -e ".[scalable]"

EXPOSE 8000
CMD ["python", "-m", "ron.scalable_server"]
