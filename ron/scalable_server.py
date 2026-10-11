"""HTTP service for Ron's scalable native checkpoint.

Run from the repository root with:
  python -m pip install -e ".[scalable]"
  python -m ron.scalable_server

By default the service expects the checkpoint, config and tokenizer produced by
the Ron-180M training workflow under artifacts/ron-180m-pythia-distilled/.
Set RON_SCALABLE_CHECKPOINT, RON_SCALABLE_CONFIG and RON_SCALABLE_TOKENIZER
to override those paths. The server refuses to start if any file is missing or
if the tokenizer/checkpoint dimensions disagree.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from operator import add, sub, mul, truediv, mod, pow
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch

from .scalable_model import ScalableRonConfig, ScalableRonLM, count_parameters

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.getenv("RON_SCALABLE_CHECKPOINT", ROOT / "artifacts/ron-180m-pythia-distilled/checkpoint.pt"))
CONFIG = Path(os.getenv("RON_SCALABLE_CONFIG", ROOT / "artifacts/ron-180m-pythia-distilled/config.json"))
TOKENIZER = Path(os.getenv("RON_SCALABLE_TOKENIZER", ROOT / "artifacts/ron-180m-pythia-distilled/tokenizer.json"))
MAX_BODY_BYTES = 16 * 1024
MAX_MESSAGE_CHARS = 4000
MAX_NEW_TOKENS = min(128, max(1, int(os.getenv("RON_MAX_NEW_TOKENS", "96"))))
ALLOWED_ORIGIN = os.getenv("RON_ALLOWED_ORIGIN", "*")


def _safe_math_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_math_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        if abs(node.value) > 1e9:
            raise ValueError("number too large")
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_math_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _safe_math_node(node.left), _safe_math_node(node.right)
        if isinstance(node.op, ast.Add): value = add(left, right)
        elif isinstance(node.op, ast.Sub): value = sub(left, right)
        elif isinstance(node.op, ast.Mult): value = mul(left, right)
        elif isinstance(node.op, ast.Div): value = truediv(left, right)
        elif isinstance(node.op, ast.Mod): value = mod(left, right)
        elif isinstance(node.op, ast.Pow):
            if abs(right) > 8: raise ValueError("exponent too large")
            value = pow(left, right)
        else:
            raise ValueError("unsupported operator")
        if abs(value) > 1e12: raise ValueError("result too large")
        return float(value)
    raise ValueError("unsupported expression")


def deterministic_math(text: str) -> str | None:
    """Handle only an explicitly arithmetic expression, without eval/exec."""
    expression = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    expression = expression.replace("×", "*").replace("÷", "/").replace("−", "-")
    expression = re.sub(r"^(?:كم\s+(?:يساوي|تساوي)\s+|ما\s+الناتج(?:\s+يساوي)?\s*)", "", expression.strip(), flags=re.I)
    expression = expression.strip().rstrip("؟?!.،,; ")
    expression = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expression)
    if not expression or not re.fullmatch(r"[0-9\s.+*/()%\-]+", expression):
        return None
    try:
        value = _safe_math_node(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    if not torch.isfinite(torch.tensor(value)).item():
        return None
    return str(int(value)) if value.is_integer() else f"{value:.8g}"


class ScalableRuntime:
    def __init__(self) -> None:
        for path in (CHECKPOINT, CONFIG, TOKENIZER):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required Ron scalable artifact is missing: {path}. "
                    "Download and extract the ron-180m-pythia-distillation-pilot workflow artifact first."
                )
        from tokenizers import Tokenizer
        self.config = ScalableRonConfig.load(CONFIG)
        self.tokenizer = Tokenizer.from_file(str(TOKENIZER))
        actual_vocab = self.tokenizer.get_vocab_size()
        if actual_vocab != self.config.vocab_size:
            raise RuntimeError(f"Tokenizer vocab size {actual_vocab} does not match model config {self.config.vocab_size}")
        self.vocab = self.tokenizer.get_vocab()
        self.special = {name: self.vocab.get(name) for name in ("<bos>", "<user>", "<assistant>", "<eos>")}
        if any(value is None for value in self.special.values()):
            raise RuntimeError("Tokenizer must define <bos>, <user>, <assistant>, and <eos> tokens")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        state = payload.get("model")
        if not isinstance(state, dict):
            raise RuntimeError("Checkpoint does not contain the expected 'model' state dictionary")
        saved_config = payload.get("config")
        if isinstance(saved_config, dict):
            for key in ("vocab_size", "hidden_size", "num_layers", "num_heads", "num_kv_heads", "max_sequence_length"):
                if key in saved_config and int(saved_config[key]) != int(getattr(self.config, key)):
                    raise RuntimeError(f"Checkpoint/config mismatch for {key}")
        self.model = ScalableRonLM(self.config).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.parameter_count = count_parameters(self.model)
        self.checkpoint_sha256 = self._sha256(CHECKPOINT)
        metrics_path = CHECKPOINT.parent / "metrics.json"
        self.metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        self.sessions: dict[str, deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=8))
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.rate_lock = threading.Lock()
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": "ron-0-scalable-python",
            "provider": "ScalableRonLM",
            "native_weights_loaded": True,
            "external_model_used": False,
            "parameter_count": self.parameter_count,
            "target_parameter_count": 180_255_744,
            "target_size_reached": self.parameter_count == 180_255_744,
            "model_config": {
                "vocab_size": self.config.vocab_size,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "num_heads": self.config.num_heads,
                "num_kv_heads": self.config.num_kv_heads,
                "max_sequence_length": self.config.max_sequence_length,
            },
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_status": self.metrics.get("status"),
            "optimizer_steps": self.metrics.get("optimizer_steps"),
            "training_metrics": self.metrics.get("history", [])[-3:],
            "device": str(self.device),
        }

    def allow_request(self, client_ip: str) -> bool:
        now = time.monotonic()
        with self.rate_lock:
            events = self.requests[client_ip]
            while events and now - events[0] >= 60:
                events.popleft()
            if len(events) >= 20:
                return False
            events.append(now)
            if len(self.requests) > 1024:
                self.requests = defaultdict(deque, {key: val for key, val in self.requests.items() if val and now - val[-1] < 60})
            return True

    def answer(self, text: str, session_id: str) -> str:
        math_answer = deterministic_math(text)
        if math_answer is not None:
            return math_answer
        with self.lock:
            history = list(self.sessions[session_id])
            history.append({"role": "user", "text": text})
        ids = [int(self.special["<bos>"])]
        for message in history:
            ids.append(int(self.special["<user>" if message["role"] == "user" else "<assistant>"]))
            ids.extend(self.tokenizer.encode(message["text"]).ids)
            if message["role"] == "assistant":
                ids.append(int(self.special["<eos>"]))
        ids.append(int(self.special["<assistant>"]))
        ids = ids[-self.config.max_sequence_length:]
        if len(ids) >= self.config.max_sequence_length:
            ids = ids[-(self.config.max_sequence_length - 1):]
            if ids[0] != int(self.special["<bos>"]):
                ids[0] = int(self.special["<bos>"])
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self.inference_lock, torch.inference_mode():
            output = self.model.generate(input_ids, MAX_NEW_TOKENS, eos_token_id=int(self.special["<eos>"]))[0, len(ids):].tolist()
        reply = self.tokenizer.decode(output, skip_special_tokens=True).strip()
        if not reply:
            reply = "لم ينتج النموذج إجابة نصية. نحتاج إلى مراجعة التدريب والتقييم."
        with self.lock:
            self.sessions[session_id].append({"role": "user", "text": text})
            self.sessions[session_id].append({"role": "assistant", "text": reply})
        return reply


class RonHandler(SimpleHTTPRequestHandler):
    runtime: ScalableRuntime

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print("ron-scalable-http:", fmt % args)

    def _headers(self, status: int, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8")
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self._headers(204, "text/plain; charset=utf-8")

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._json(200, {"health": self.runtime.health()})
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/chat":
            self._json(404, {"error": "not_found"})
            return
        client_ip = self.client_address[0]
        if not self.runtime.allow_request(client_ip):
            self._json(429, {"error": "rate_limited", "message": "Too many requests; retry in one minute."})
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
        message = payload.get("message") if isinstance(payload, dict) else None
        session_id = payload.get("session_id", "default") if isinstance(payload, dict) else "default"
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
            self._json(400, {"error": "invalid_message", "message": "Message must be 1–4000 characters."})
            return
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
            self._json(400, {"error": "invalid_session_id"})
            return
        try:
            reply = self.runtime.answer(message.strip(), session_id)
        except Exception:
            self._json(500, {"error": "inference_failed", "message": "Ron inference failed; inspect server logs."})
            return
        self._json(200, {"reply": reply, "health": self.runtime.health()})

    def end_headers(self) -> None:
        # Static assets may be cached by the browser; app.js is versioned on deploy.
        super().end_headers()


def main() -> None:
    runtime = ScalableRuntime()
    RonHandler.runtime = runtime
    host = os.getenv("RON_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("RON_PORT", "8000")))
    server = ThreadingHTTPServer((host, port), RonHandler)
    print(json.dumps({"event": "ron_server_ready", "health": runtime.health()}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
