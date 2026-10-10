"""HTTP bridge that exposes Ron's single Python orchestration core to the browser UI.

Run with: python -m ron.server
Production should set RON_REQUIRE_NATIVE=1 (the default), so a missing or broken
native checkpoint fails closed instead of silently serving the teaching fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .core import RonCore
from .native_provider import NativeCheckpointProvider

ROOT = Path(__file__).resolve().parents[1]
MAX_BODY_BYTES = 32 * 1024
MAX_MESSAGE_CHARS = 4000
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class RonServerState:
    """One shared model/provider; isolated conversation and memory per session."""

    def __init__(self, *, require_native: bool = True, require_10m: bool = False) -> None:
        bootstrap = RonCore()
        self.provider = bootstrap.provider
        self.startup_warnings = list(bootstrap.provider_startup_warnings)
        self.native_loaded = isinstance(self.provider, NativeCheckpointProvider)
        self.parameter_count = sum(parameter.numel() for parameter in self.provider.model.parameters()) if self.native_loaded else 0
        self.checkpoint_sha256 = None
        if self.native_loaded:
            digest = hashlib.sha256()
            with self.provider.checkpoint_path.open("rb") as checkpoint_file:
                for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.checkpoint_sha256 = digest.hexdigest()
        self.require_native = require_native
        self.require_10m = require_10m
        if require_native and not self.native_loaded:
            details = ", ".join(self.startup_warnings) or "no native checkpoint file found"
            raise RuntimeError(
                "Ron native Transformer is required but is not loaded; refusing to start "
                f"the chat service ({details}). Train or install a compatible native checkpoint."
            )
        if require_10m and (not self.native_loaded or self.parameter_count < 8_000_000):
            raise RuntimeError(
                f"Ron-10M is required but active checkpoint has {self.parameter_count:,} parameters. "
                "Complete and promote the Ron-10M training run before starting production."
            )
        self.sessions: OrderedDict[str, RonCore] = OrderedDict({"default": bootstrap})
        self.sessions_lock = threading.RLock()
        self.inference_lock = threading.RLock()
        self.rate_lock = threading.Lock()
        self.rate_events: dict[str, list[float]] = {}
        self.max_requests_per_minute = 30
        self.max_sessions = 128

    def allow_request(self, client_ip: str) -> bool:
        now = time.monotonic()
        with self.rate_lock:
            recent = [stamp for stamp in self.rate_events.get(client_ip, []) if now - stamp < 60.0]
            if len(recent) >= self.max_requests_per_minute:
                self.rate_events[client_ip] = recent
                return False
            recent.append(now)
            self.rate_events[client_ip] = recent
            if len(self.rate_events) > 1024:
                self.rate_events = {ip: stamps for ip, stamps in self.rate_events.items() if stamps and now - stamps[-1] < 60.0}
            return True

    def core_for(self, session_id: str) -> RonCore:
        with self.sessions_lock:
            core = self.sessions.get(session_id)
            if core is not None:
                self.sessions.move_to_end(session_id)
                return core
            # Bound session memory while reusing the exact same loaded model weights.
            if len(self.sessions) >= self.max_sessions:
                oldest = next((key for key in self.sessions if key != "default"), None)
                if oldest is not None:
                    self.sessions.pop(oldest, None)
            core = RonCore(provider=self.provider)
            self.sessions[session_id] = core
            return core

    def health(self) -> dict[str, Any]:
        provider = self.provider
        native = self.native_loaded
        return {
            "status": "ok" if (native or not self.require_native) else "degraded",
            "runtime": "ron-0-python-core",
            "single_core": True,
            "provider": type(provider).__name__,
            "native_weights_loaded": native,
            "external_model_used": False,
            "parameter_count": self.parameter_count,
            "model_config": ({
                "vocab_size": provider.config.vocab_size,
                "hidden_size": provider.config.hidden_size,
                "num_layers": provider.config.num_layers,
                "num_heads": provider.config.num_heads,
                "max_sequence_length": provider.config.max_sequence_length,
            } if native else None),
            "target_parameter_count": 10_000_000,
            "target_size_reached": self.parameter_count >= 8_000_000,
            "model_stage": "native-10m" if self.parameter_count >= 8_000_000 else ("native-baseline-prototype" if native else "teaching-fallback"),
            "checkpoint": str(provider.checkpoint_path) if native else None,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_source": getattr(provider, "checkpoint_source", None),
            "inference_checkpoint": getattr(provider, "inference_checkpoint", None),
            "selected_training_step": getattr(provider, "selected_step", None),
            "training_steps_total": getattr(provider, "training_steps_total", None),
            "startup_warnings": self.startup_warnings,
        }


class RonRequestHandler(SimpleHTTPRequestHandler):
    server_version = "RonCore/1.0"
    state: RonServerState

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        # Keep logs useful without echoing user conversation content.
        print("ron-http:", fmt % args)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._json(200, self.state.health())
            return
        if path.startswith("/api/"):
            self._json(404, {"error": "not_found"})
            return
        if path == "/":
            path = "/index.html"
        # Serve only public UI assets; never expose Python source, checkpoints, or training data.
        relative = unquote(path).lstrip("/")
        target = (ROOT / relative).resolve()
        assets_root = (ROOT / "assets").resolve()
        allowed_root_asset = target.parent == ROOT and target.suffix.lower() in {".html", ".js", ".css", ".svg", ".ico", ".webmanifest"}
        allowed_asset_tree = target == assets_root or assets_root in target.parents
        if not (allowed_root_asset or allowed_asset_tree) or not target.is_file():
            self._json(404, {"error": "not_found"})
            return
        self.path = "/" + relative
        super().do_GET()

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/chat":
            self._json(404, {"error": "not_found"})
            return
        client_ip = self.client_address[0] if self.client_address else "unknown"
        if not self.state.allow_request(client_ip):
            self._json(429, {"error": "rate_limited", "retry_after_seconds": 60})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "invalid_content_length"})
            return
        if length < 1 or length > MAX_BODY_BYTES:
            self._json(413 if length > MAX_BODY_BYTES else 400, {"error": "invalid_body_size"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(400, {"error": "invalid_json"})
            return
        if not isinstance(payload, dict):
            self._json(400, {"error": "invalid_payload"})
            return
        message = payload.get("message")
        session_id = payload.get("session_id", "default")
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
            self._json(400, {"error": "invalid_message", "max_chars": MAX_MESSAGE_CHARS})
            return
        if not isinstance(session_id, str) or not SESSION_PATTERN.fullmatch(session_id):
            self._json(400, {"error": "invalid_session_id"})
            return
        if self.state.require_native and not self.state.native_loaded:
            self._json(503, {"error": "native_model_unavailable", "health": self.state.health()})
            return
        core = self.state.core_for(session_id)
        try:
            # The model is shared, so serialize inference to avoid concurrent weight access.
            with self.state.inference_lock:
                response = core.respond(message)
        except Exception as exc:
            self._json(500, {"error": "inference_failed", "kind": type(exc).__name__})
            return
        metadata = dict(response.metadata or {})
        metadata.setdefault("runtime", "ron-0-python-core")
        metadata["native_weights_loaded"] = self.state.native_loaded
        metadata["external_model_used"] = False
        # A loaded checkpoint is not enough: reject provider exceptions and low-quality fallback text.
        if self.state.require_native and (
            metadata.get("fallback") is True
            or metadata.get("generation_quality_fallback") is True
            or metadata.get("native_generated_answer_rejected") is True
        ):
            self._json(503, {"error": "native_inference_rejected", "metadata": metadata, "health": self.state.health()})
            return
        self._json(200, {
            "reply": response.content,
            "model": response.model,
            "metadata": metadata,
            "health": self.state.health(),
        })


def make_server(
    host: str | None = None,
    port: int | None = None,
    *,
    require_native: bool | None = None,
    require_10m: bool | None = None,
):
    host = host or os.environ.get("RON_HOST", "127.0.0.1")
    if port is None:
        port = int(os.environ.get("PORT") or os.environ.get("RON_PORT") or "8000")
    if require_native is None:
        require_native = os.environ.get("RON_REQUIRE_NATIVE", "1").strip().lower() not in {"0", "false", "no"}
    if require_10m is None:
        require_10m = os.environ.get("RON_REQUIRE_10M", "0").strip().lower() not in {"0", "false", "no"}
    state = RonServerState(require_native=require_native, require_10m=require_10m)

    class BoundRonRequestHandler(RonRequestHandler):
        pass

    BoundRonRequestHandler.state = state
    return ThreadingHTTPServer((host, port), BoundRonRequestHandler)


def main() -> None:
    server = make_server()
    print(f"Ron single-core service listening on http://{server.server_address[0]}:{server.server_address[1]}")
    print(json.dumps(server.RequestHandlerClass.state.health(), ensure_ascii=False))
    server.serve_forever()


if __name__ == "__main__":
    main()
