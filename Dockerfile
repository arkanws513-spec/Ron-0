FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ron ./ron
COPY service ./service
RUN python -m pip install --upgrade pip && python -m pip install ".[model,server]"

EXPOSE 8080
CMD uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-8080}
