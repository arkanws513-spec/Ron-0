"""HTTP chat service for Ron's native 110k-parameter checkpoint.

This runtime is intentionally separate from the experimental scalable-model pilot.
It serves the checkpoint committed at ron/checkpoints/ron_native_baseline.pt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import OrderedDict, defaultdict, deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .contracts import Message, ModelRequest
from .native_provider import NativeCheckpointProvider

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.getenv(
    "RON_NATIVE_CHECKPOINT",
    ROOT / "ron" / "checkpoints" / "ron_native_baseline.pt",
))
MAX_BODY_BYTES = 16 * 1024
MAX_MESSAGE_CHARS = 4000
ALLOWED_ORIGIN = os.getenv("RON_ALLOWED_ORIGIN", "*")
RESPONSE_CACHE_LIMIT = 256
RESPONSE_CACHE: OrderedDict[tuple[str, str], tuple[str, str]] = OrderedDict()
RESPONSE_CACHE_LOCK = threading.Lock()


class NativeChatRuntime:
    def __init__(self) -> None:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"Ron native checkpoint not found: {CHECKPOINT}")
        self.provider = NativeCheckpointProvider(CHECKPOINT)
        self.sessions: dict[str, deque[Message]] = defaultdict(lambda: deque(maxlen=12))
        self.lock = threading.RLock()
        digest = hashlib.sha256()
        with CHECKPOINT.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()
        self.parameter_count = sum(parameter.numel() for parameter in self.provider.model.parameters())

    def health(self) -> dict:
        return {
            "status": "ok",
            "runtime": "ron-0-native-python",
            "provider": "NativeCheckpointProvider",
            "native_weights_loaded": True,
            "external_model_used": False,
            "parameter_count": self.parameter_count,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": self.checkpoint_sha256,
            "selected_training_step": self.provider.selected_step,
            "final_training_step": self.provider.final_training_step,
            "inference_checkpoint": self.provider.inference_checkpoint,
            "training_steps_total": self.provider.training_steps_total,
            "checkpoint_source": self.provider.checkpoint_source,
            "device": "cpu",
        }

    def answer(self, text: str, session_id: str) -> tuple[str, str]:
        with self.lock:
            history = list(self.sessions[session_id])
            history.append(Message(role="user", content=text))
        response = self.provider.generate(ModelRequest(
            messages=tuple(history),
            metadata={"session_id": session_id, "runtime": "ron-native"},
        ))
        reply = response.content.strip()
        if not reply:
            raise RuntimeError("Native checkpoint returned an empty response")
        with self.lock:
            self.sessions[session_id].append(Message(role="user", content=text))
            self.sessions[session_id].append(Message(role="assistant", content=reply))
        fallback = bool(response.metadata.get("generation_quality_fallback", False))
        return reply, "native_quality_fallback" if fallback else "native_checkpoint"


class NativeChatHandler(SimpleHTTPRequestHandler):
    runtime: NativeChatRuntime

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print("ron-native-http:", fmt % args)

    def _headers(self, status: int, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _json(self, status: int, payload: dict) -> None:
        self._headers(status, "application/json; charset=utf-8")
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        self._headers(204, "text/plain; charset=utf-8")

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/api/health":
            self._json(200, {"health": self.runtime.health()})
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/chat":
            self._json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length < 1 or length > MAX_BODY_BYTES:
            self._json(413, {"error": "invalid_body_size"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"error": "invalid_json"})
            return
        if not isinstance(payload, dict):
            self._json(400, {"error": "invalid_payload"})
            return
        message = payload.get("message")
        session_id = payload.get("session_id", "default")
        request_id = payload.get("request_id", "")
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
            self._json(400, {"error": "invalid_message"})
            return
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
            self._json(400, {"error": "invalid_session_id"})
            return
        if request_id != "" and (not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id)):
            self._json(400, {"error": "invalid_request_id"})
            return
        cache_key = (session_id, request_id) if request_id else None
        if cache_key:
            with RESPONSE_CACHE_LOCK:
                cached = RESPONSE_CACHE.get(cache_key)
                if cached is not None:
                    RESPONSE_CACHE.move_to_end(cache_key)
            if cached is not None:
                self._json(200, {
                    "reply": cached[0], "generation_status": cached[1],
                    "health": self.runtime.health(), "cached_retry": True,
                })
                return
        try:
            reply, status = self.runtime.answer(message.strip(), session_id)
        except Exception:
            self._json(500, {"error": "inference_failed", "message": "Ron native inference failed; inspect server logs."})
            return
        if cache_key:
            with RESPONSE_CACHE_LOCK:
                RESPONSE_CACHE[cache_key] = (reply, status)
                RESPONSE_CACHE.move_to_end(cache_key)
                while len(RESPONSE_CACHE) > RESPONSE_CACHE_LIMIT:
                    RESPONSE_CACHE.popitem(last=False)
        self._json(200, {
            "reply": reply, "generation_status": status, "health": self.runtime.health(),
        })


def main() -> None:
    runtime = NativeChatRuntime()
    NativeChatHandler.runtime = runtime
    host = os.getenv("RON_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("RON_PORT", "8000")))
    server = ThreadingHTTPServer((host, port), NativeChatHandler)
    print(json.dumps({"event": "ron_native_server_ready", "health": runtime.health()}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
