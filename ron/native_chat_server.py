"""HTTP chat service for Ron-0's local instruction-tuned model."""
from __future__ import annotations

import json
import os
import re
import threading
from collections import OrderedDict, defaultdict, deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .contracts import Message, ModelRequest
from .llm_provider import QwenChatProvider

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
MAX_BODY_BYTES = 16 * 1024
MAX_MESSAGE_CHARS = 4000
ALLOWED_ORIGIN = os.getenv("RON_ALLOWED_ORIGIN", "*")
RESPONSE_CACHE_LIMIT = 256
RESPONSE_CACHE: OrderedDict[tuple[str, str], tuple[str, str]] = OrderedDict()
RESPONSE_CACHE_LOCK = threading.Lock()


class NativeChatRuntime:
    """Keeps the existing Ron HTTP contract while replacing the old native LM."""

    def __init__(self, provider=None) -> None:
        self.provider = provider or QwenChatProvider()
        self.model_id = getattr(self.provider, "model_id", "local-chat-model")
        self.model_path = getattr(self.provider, "model_path", "local-model")
        self.sessions: dict[str, deque[Message]] = defaultdict(lambda: deque(maxlen=12))
        self.lock = threading.RLock()

    def health(self) -> dict:
        return {
            "status": "ok",
            "runtime": "ron-0-llama-cpp",
            "provider": type(self.provider).__name__,
            "model": self.model_id,
            "weights_loaded": True,
            "local_weights": True,
            "external_inference_api": False,
            "model_path": self.model_path,
            "context_size": getattr(self.provider, "context_size", None),
            "device": "cpu",
        }

    def answer(self, text: str, session_id: str) -> tuple[str, str]:
        with self.lock:
            history = list(self.sessions[session_id])
            history.append(Message(role="user", content=text))
            provider = self.provider
        response = provider.generate(ModelRequest(
            messages=tuple(history),
            metadata={"session_id": session_id, "runtime": "ron-0-llama-cpp"},
        ))
        reply = response.content.strip()
        if not reply:
            raise RuntimeError("Local chat model returned an empty response")
        with self.lock:
            self.sessions[session_id].append(Message(role="user", content=text))
            self.sessions[session_id].append(Message(role="assistant", content=reply))
        return reply, "local_llm"


class NativeChatHandler(SimpleHTTPRequestHandler):
    runtime: NativeChatRuntime

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print("ron-http:", fmt % args)

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
        if request_id != "" and (
            not isinstance(request_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id)
        ):
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
                    "reply": cached[0],
                    "generation_status": cached[1],
                    "health": self.runtime.health(),
                    "cached_retry": True,
                })
                return
        try:
            reply, status = self.runtime.answer(message.strip(), session_id)
        except Exception as exc:
            print(f"ron-inference-error: {type(exc).__name__}: {exc}", flush=True)
            self._json(500, {
                "error": "inference_failed",
                "message": "Ron local model inference failed; inspect server logs.",
            })
            return
        if cache_key:
            with RESPONSE_CACHE_LOCK:
                RESPONSE_CACHE[cache_key] = (reply, status)
                RESPONSE_CACHE.move_to_end(cache_key)
                while len(RESPONSE_CACHE) > RESPONSE_CACHE_LIMIT:
                    RESPONSE_CACHE.popitem(last=False)
        self._json(200, {
            "reply": reply,
            "generation_status": status,
            "health": self.runtime.health(),
        })


def main() -> None:
    runtime = NativeChatRuntime()
    NativeChatHandler.runtime = runtime
    host = os.getenv("RON_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("RON_PORT", "8000")))
    server = ThreadingHTTPServer((host, port), NativeChatHandler)
    print(json.dumps({
        "event": "ron_local_llm_server_ready",
        "health": runtime.health(),
    }, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
