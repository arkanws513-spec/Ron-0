"""HTTP API for serving a reviewed Ron-native checkpoint.

The service deliberately refuses to serve the 110k baseline in production. Set
RON_CHECKPOINT_PATH to a reviewed 10M+ checkpoint mounted into the container.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Literal
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ron.contracts import Message, ModelRequest
from ron.native_provider import NativeCheckpointProvider

DEFAULT_CHECKPOINT = "/models/ron_native_10m.pt"
MIN_PARAMETERS = int(os.getenv("RON_MIN_PARAMETERS", "1000000"))
CHECKPOINT_PATH = Path(os.getenv("RON_CHECKPOINT_PATH", DEFAULT_CHECKPOINT))


def ensure_checkpoint_file(path: Path) -> None:
    """Download a missing checkpoint only from an explicitly pinned URL and SHA-256."""
    expected = os.getenv("RON_CHECKPOINT_SHA256", "").strip().lower()
    if path.is_file():
        if expected:
            with path.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("existing checkpoint SHA-256 does not match RON_CHECKPOINT_SHA256")
        return

    url = os.getenv("RON_CHECKPOINT_URL", "").strip()
    if not url:
        raise FileNotFoundError(
            f"Set RON_CHECKPOINT_PATH to a reviewed checkpoint; missing {path.name}"
        )
    if not expected or len(expected) != 64:
        raise ValueError("RON_CHECKPOINT_SHA256 must be set when downloading a checkpoint")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".download", delete=False
        ) as destination:
            temporary_path = Path(destination.name)
            with urlopen(url, timeout=600) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("downloaded checkpoint SHA-256 does not match RON_CHECKPOINT_SHA256")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

app = FastAPI(title="Ron Native Inference", version="0.1.0")
origins = [
    item.strip()
    for item in os.getenv(
        "RON_CORS_ORIGINS",
        "https://arkanws513-spec.github.io,http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.state.provider = None
app.state.checkpoint_status = {
    "ready": False,
    "checkpoint": CHECKPOINT_PATH.name,
    "parameter_count": 0,
    "error": "checkpoint_not_loaded",
}


def load_checkpoint() -> None:
    """Load and validate the configured local checkpoint; never silently downgrade."""
    app.state.provider = None
    try:
        ensure_checkpoint_file(CHECKPOINT_PATH)
        candidate = NativeCheckpointProvider(CHECKPOINT_PATH)
        parameter_count = sum(parameter.numel() for parameter in candidate.model.parameters())
        if parameter_count < MIN_PARAMETERS:
            raise ValueError(
                f"checkpoint has {parameter_count:,} parameters; minimum is {MIN_PARAMETERS:,}"
            )
        app.state.provider = candidate
        app.state.checkpoint_status = {
            "ready": True,
            "checkpoint": CHECKPOINT_PATH.name,
            "parameter_count": parameter_count,
            "selected_training_step": candidate.selected_step,
            "training_steps_total": candidate.training_steps_total,
            "inference_checkpoint": candidate.inference_checkpoint,
            "external_model_used": False,
            "error": None,
        }
    except Exception as exc:
        app.state.checkpoint_status = {
            "ready": False,
            "checkpoint": CHECKPOINT_PATH.name,
            "parameter_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.on_event("startup")
def startup() -> None:
    load_checkpoint()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=12)


class ChatResponse(BaseModel):
    content: str
    model: str
    metadata: dict


@app.get("/")
def root() -> dict:
    return {"service": "ron-native-inference", "health": "/health", "chat": "/v1/chat"}


@app.get("/health")
def health() -> dict:
    status = app.state.checkpoint_status
    if not status.get("ready"):
        raise HTTPException(status_code=503, detail=status)
    return status


@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    provider = app.state.provider
    if provider is None:
        raise HTTPException(status_code=503, detail=app.state.checkpoint_status)
    messages = tuple(Message(role=item.role, content=item.content) for item in request.messages)
    response = provider.generate(ModelRequest(messages=messages, metadata={"source": "ron-web"}))
    return ChatResponse(
        content=response.content,
        model=response.model or "ron-native-checkpoint",
        metadata=response.metadata,
    )
