"""HTTP chat service for Ron's native 110k-parameter checkpoint.

This runtime is intentionally separate from the experimental scalable-model pilot.
It serves the checkpoint committed at ron/checkpoints/ron_native_baseline.pt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from collections import OrderedDict, defaultdict, deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .contracts import Message, ModelRequest
from .native_provider import NativeCheckpointProvider

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CHECKPOINT = Path(os.getenv(
    "RON_NATIVE_BASELINE_CHECKPOINT",
    ROOT / "ron" / "checkpoints" / "ron_native_baseline.pt",
))
LIVE_CHECKPOINT = Path(os.getenv(
    "RON_NATIVE_LIVE_CHECKPOINT",
    ROOT / "ron" / "checkpoints" / "ron_native_10m.pt",
))
RELEASE_API_URL = os.getenv(
    "RON_LIVE_CHECKPOINT_RELEASE_API",
    "https://api.github.com/repos/arkanws513-spec/Ron-0/releases/tags/ron-live-checkpoint",
)
POLL_INTERVAL_SECONDS = max(60, int(os.getenv("RON_LIVE_CHECKPOINT_POLL_SECONDS", "300")))
EXPLICIT_CHECKPOINT = os.getenv("RON_NATIVE_CHECKPOINT")
MAX_BODY_BYTES = 16 * 1024
MAX_MESSAGE_CHARS = 4000
ALLOWED_ORIGIN = os.getenv("RON_ALLOWED_ORIGIN", "*")
RESPONSE_CACHE_LIMIT = 256
RESPONSE_CACHE: OrderedDict[tuple[str, str], tuple[str, str]] = OrderedDict()
RESPONSE_CACHE_LOCK = threading.Lock()


class NativeChatRuntime:
    def __init__(self) -> None:
        # Prefer the last promoted 10M checkpoint if it is already on disk.
        # Otherwise keep serving the known baseline while checking the durable release.
        if EXPLICIT_CHECKPOINT:
            self.checkpoint_path = Path(EXPLICIT_CHECKPOINT).expanduser()
        else:
            self.checkpoint_path = LIVE_CHECKPOINT if LIVE_CHECKPOINT.is_file() else BASELINE_CHECKPOINT
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Ron native checkpoint not found: {self.checkpoint_path}")
        self.provider = NativeCheckpointProvider(self.checkpoint_path)
        self.sessions: dict[str, deque[Message]] = defaultdict(lambda: deque(maxlen=12))
        self.lock = threading.RLock()
        self.checkpoint_sha256 = self._sha256(self.checkpoint_path)
        self.parameter_count = sum(parameter.numel() for parameter in self.provider.model.parameters())
        self.refresh_live_checkpoint()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def refresh_live_checkpoint(self) -> bool:
        """Download and atomically activate the latest quality-gated native checkpoint."""
        temporary_path: Path | None = None
        try:
            request = Request(RELEASE_API_URL, headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "Ron-0-live-checkpoint-loader",
            })
            with urlopen(request, timeout=20) as response:
                release = json.loads(response.read().decode("utf-8"))
            assets = [
                asset for asset in release.get("assets", [])
                if asset.get("name") == "ron_native_10m.pt" and asset.get("browser_download_url")
            ]
            if not assets:
                print("ron-native: no promoted checkpoint asset found; keeping current weights", flush=True)
                return False
            asset = assets[0]
            expected_digest = str(asset.get("digest") or "")
            with self.lock:
                current_digest = self.checkpoint_sha256
            if expected_digest.startswith("sha256:") and current_digest == expected_digest.split(":", 1)[1]:
                return False

            LIVE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=LIVE_CHECKPOINT.parent, prefix="ron-checkpoint-", suffix=".pt", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
            download_request = Request(
                asset["browser_download_url"],
                headers={"User-Agent": "Ron-0-live-checkpoint-loader"},
            )
            with urlopen(download_request, timeout=300) as response, temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)

            downloaded_digest = self._sha256(temporary_path)
            if expected_digest.startswith("sha256:") and downloaded_digest != expected_digest.split(":", 1)[1]:
                raise RuntimeError("Downloaded checkpoint SHA-256 does not match the GitHub release asset")
            candidate_provider = NativeCheckpointProvider(temporary_path)
            candidate_parameters = sum(
                parameter.numel() for parameter in candidate_provider.model.parameters()
            )
            if candidate_parameters < 1_000_000:
                raise RuntimeError(f"Refusing to activate a checkpoint with only {candidate_parameters} parameters")

            # The candidate is fully downloaded, checksum-checked, and load-tested before activation.
            os.replace(temporary_path, LIVE_CHECKPOINT)
            temporary_path = None
            candidate_provider.checkpoint_path = LIVE_CHECKPOINT
            with self.lock:
                self.provider = candidate_provider
                self.checkpoint_path = LIVE_CHECKPOINT
                self.checkpoint_sha256 = downloaded_digest
                self.parameter_count = candidate_parameters
            print(json.dumps({
                "event": "ron_live_checkpoint_activated",
                "checkpoint": str(LIVE_CHECKPOINT),
                "sha256": downloaded_digest,
                "parameter_count": candidate_parameters,
                "release": release.get("html_url"),
            }, ensure_ascii=False), flush=True)
            return True
        except HTTPError as error:
            if error.code == 404:
                print("ron-native: live checkpoint release is not published yet; keeping current weights", flush=True)
                return False
            print(f"ron-native: checkpoint refresh failed ({error}); keeping current weights", flush=True)
            return False
        except Exception as error:
            print(f"ron-native: checkpoint refresh failed ({error}); keeping current weights", flush=True)
            return False
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def start_checkpoint_refresh(self) -> None:
        def refresh_loop() -> None:
            while True:
                time.sleep(POLL_INTERVAL_SECONDS)
                self.refresh_live_checkpoint()
        threading.Thread(target=refresh_loop, name="ron-checkpoint-refresh", daemon=True).start()

    def health(self) -> dict:
        with self.lock:
            provider = self.provider
            checkpoint_path = self.checkpoint_path
            checkpoint_sha256 = self.checkpoint_sha256
            parameter_count = self.parameter_count
        return {
            "status": "ok",
            "runtime": "ron-0-native-python",
            "provider": "NativeCheckpointProvider",
            "native_weights_loaded": True,
            "external_model_used": False,
            "parameter_count": parameter_count,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "selected_training_step": provider.selected_step,
            "final_training_step": provider.final_training_step,
            "inference_checkpoint": provider.inference_checkpoint,
            "training_steps_total": provider.training_steps_total,
            "checkpoint_source": provider.checkpoint_source,
            "device": "cpu",
        }

    def answer(self, text: str, session_id: str) -> tuple[str, str]:
        with self.lock:
            history = list(self.sessions[session_id])
            history.append(Message(role="user", content=text))
            provider = self.provider
        response = provider.generate(ModelRequest(
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
    runtime.start_checkpoint_refresh()
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
