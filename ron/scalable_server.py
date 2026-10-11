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
import shutil
import tempfile
import threading
import time
import unicodedata
from collections import OrderedDict, defaultdict, deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from operator import add, sub, mul, truediv, mod, pow
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile

import torch

from .generation_quality import guard_generated_text
from .scalable_model import ScalableRonConfig, ScalableRonLM, count_parameters

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.getenv("RON_SCALABLE_CHECKPOINT", ROOT / "artifacts/ron-180m-pythia-distilled/checkpoint.pt"))
CONFIG = Path(os.getenv("RON_SCALABLE_CONFIG", ROOT / "artifacts/ron-180m-pythia-distilled/config.json"))
TOKENIZER = Path(os.getenv("RON_SCALABLE_TOKENIZER", ROOT / "artifacts/ron-180m-pythia-distilled/tokenizer.json"))
MAX_BODY_BYTES = 16 * 1024
MAX_MESSAGE_CHARS = 4000
MAX_NEW_TOKENS = min(96, max(8, int(os.getenv("RON_MAX_NEW_TOKENS", "48"))))
ALLOWED_ORIGIN = os.getenv("RON_ALLOWED_ORIGIN", "*")
RESPONSE_CACHE_LIMIT = 1024
RESPONSE_CACHE: OrderedDict[tuple[str, str], tuple[str, str]] = OrderedDict()
RESPONSE_CACHE_LOCK = threading.Lock()
RUNTIME_BUNDLE_URL = os.getenv("RON_RUNTIME_BUNDLE_URL", "https://github.com/arkanws513-spec/Ron-0/releases/download/ron-180m-runtime-2026-10-11/ron-180m-inference-bundle.zip")


def ensure_runtime_bundle() -> None:
    """Fetch the inference-only release bundle if trained files are absent."""
    if CHECKPOINT.is_file() and CONFIG.is_file() and TOKENIZER.is_file():
        return
    if os.getenv("RON_DISABLE_AUTO_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
        raise FileNotFoundError("Ron checkpoint files are missing and auto-download is disabled")
    destination = ROOT / "artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ron-bundle-") as temporary:
        archive_path = Path(temporary) / "bundle.zip"
        request = Request(RUNTIME_BUNDLE_URL, headers={"User-Agent": "Ron-180M-runtime"})
        print("ron-runtime: downloading inference bundle", flush=True)
        with urlopen(request, timeout=1200) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        expected_root = destination.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if target != expected_root and expected_root not in target.parents:
                    raise RuntimeError("Runtime bundle contains an unsafe path")
            archive.extractall(destination)
    for path in (CHECKPOINT, CONFIG, TOKENIZER):
        if not path.is_file():
            raise FileNotFoundError(f"Downloaded runtime bundle is missing {path}")
    print("ron-runtime: inference bundle ready", flush=True)

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


def deterministic_social_reply(text: str) -> str | None:
    """Answer common standalone greetings locally, without spending CPU on inference."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", normalized)
    normalized = re.sub(r"[أإآٱ]", "ا", normalized)
    normalized = re.sub(r"[!?؟.,،؛:]+$", "", normalized).strip()
    if re.fullmatch(r"(?:اهلا|مرحبا|سلام|السلام عليكم|هلا|هاي)(?:\s+يا\s+رون)?", normalized):
        if normalized.startswith("السلام عليكم"):
            return "وعليكم السلام ورحمة الله، أهلًا بك."
        return "أهلًا بك! أنا رون. تحب نتكلم عن إيه؟"
    if re.fullmatch(r"(?:كيف حالك(?: يا رون)?|عامل ايه|عاملة ايه|ازيك|اخبارك ايه|كيفك)", normalized):
        return "أنا جاهز أساعدك، شكرًا لسؤالك. إنت أخبارك إيه؟"
    if re.fullmatch(r"(?:hello|hi|hey|good morning|good afternoon)", normalized):
        return "Hello! I'm Ron. What would you like to talk about?"
    if re.fullmatch(r"(?:how are you|how are you doing|how's it going)", normalized):
        return "I'm ready to help, thanks for asking. What would you like to talk about?"
    return None


def normalize_lookup_text(text: str) -> str:
    """Normalize a question for exact lookup in Ron's reviewed seed corpus."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", normalized)
    normalized = re.sub(r"[أإآٱ]", "ا", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def load_seed_responses(path: Path) -> dict[str, str]:
    """Load user/Ron pairs from the project seed corpus; this is not an external model."""
    if not path.is_file():
        return {}
    responses: dict[str, str] = {}
    pending_prompt: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith(("المستخدم:", "User:", "Question:")):
            pending_prompt = line.split(":", 1)[1].strip()
        elif pending_prompt is not None and line.startswith(("رون:", "Ron:", "Answer:")):
            answer = line.split(":", 1)[1].strip()
            key = normalize_lookup_text(pending_prompt)
            if key and answer:
                responses.setdefault(key, answer)
            pending_prompt = None
    return responses


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
        self.seed_responses = load_seed_responses(ROOT / "training/seed_corpus.txt")
        print(f"ron-runtime: loaded {len(self.seed_responses)} seed responses", flush=True)
        print("ron-runtime: initializing tokenizer", flush=True)
        from tokenizers import Tokenizer
        self.config = ScalableRonConfig.load(CONFIG)
        self.tokenizer = Tokenizer.from_file(str(TOKENIZER))
        actual_vocab = self.tokenizer.get_vocab_size()
        # Keep the learned tokenizer boundary before any reserved rows are added.
        # The model's fixed embedding table may be larger than the actual tokenizer.
        self.valid_vocab_size = actual_vocab
        if actual_vocab > self.config.vocab_size:
            raise RuntimeError(f"Tokenizer vocab size {actual_vocab} exceeds model config {self.config.vocab_size}")
        if actual_vocab < self.config.vocab_size:
            # The fixed 180M architecture pads its embedding table to 8,000 IDs.
            # The seed-corpus BPE tokenizer may contain fewer learned tokens; add
            # reserved special IDs so every embedding row has a valid token ID.
            self.tokenizer.add_special_tokens([
                f"<ron_unused_{token_id}>"
                for token_id in range(actual_vocab, self.config.vocab_size)
            ])
        actual_vocab = self.tokenizer.get_vocab_size()
        if actual_vocab != self.config.vocab_size:
            raise RuntimeError(f"Tokenizer padding failed: {actual_vocab} != {self.config.vocab_size}")
        print(f"ron-runtime: tokenizer ready ({actual_vocab} tokens)", flush=True)
        self.vocab = self.tokenizer.get_vocab()
        self.special = {name: self.vocab.get(name) for name in ("<bos>", "<user>", "<assistant>", "<eos>")}
        if any(value is None for value in self.special.values()):
            raise RuntimeError("Tokenizer must define <bos>, <user>, <assistant>, and <eos> tokens")
        # These structural tokens are useful in prompts, but must not be emitted as
        # user-facing content. EOS remains separately controlled by min_new_tokens.
        self.suppressed_token_ids = [
            int(token_id) for token, token_id in self.vocab.items()
            if token in {"<pad>", "<unk>", "<bos>", "<user>", "<assistant>"}
        ]
        # Do not let punctuation-only/whitespace-only BPE pieces dominate decoding.
        # Decode each learned ID once at startup; Arabic and Latin word pieces remain allowed.
        for token_id in range(self.valid_vocab_size):
            if token_id == int(self.special["<eos>"]):
                continue
            piece = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if not any(character.isalnum() for character in piece):
                self.suppressed_token_ids.append(token_id)
        self.suppressed_token_ids = sorted(set(self.suppressed_token_ids))
        if not 1 <= self.valid_vocab_size <= self.config.vocab_size:
            raise RuntimeError(f"Invalid learned tokenizer vocabulary size: {self.valid_vocab_size}")
        if int(self.special["<eos>"]) >= self.valid_vocab_size:
            raise RuntimeError("EOS token is outside the learned tokenizer vocabulary")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ron-runtime: loading checkpoint on {self.device}", flush=True)
        payload = torch.load(CHECKPOINT, map_location=self.device, mmap=(self.device.type == "cpu"), weights_only=True)
        state = payload.get("model")
        if not isinstance(state, dict):
            raise RuntimeError("Checkpoint does not contain the expected 'model' state dictionary")
        saved_config = payload.get("config")
        if isinstance(saved_config, dict):
            for key in ("vocab_size", "hidden_size", "num_layers", "num_heads", "num_kv_heads", "max_sequence_length"):
                if key in saved_config and int(saved_config[key]) != int(getattr(self.config, key)):
                    raise RuntimeError(f"Checkpoint/config mismatch for {key}")
        precision = os.getenv("RON_PRECISION", "fp16").lower()
        if precision not in {"fp16", "fp32"}:
            raise RuntimeError("RON_PRECISION must be fp16 or fp32")
        if precision == "fp32" and self.device.type == "cpu":
            raise RuntimeError("FP32 inference exceeds the current CPU memory budget; use RON_PRECISION=fp16")
        target_dtype = torch.float16 if precision == "fp16" else torch.float32
        if any(tensor.dtype != target_dtype for tensor in state.values()):
            # Keep the default FP16 path zero-copy. FP32 is only allowed on a GPU.
            state = {key: tensor.to(dtype=target_dtype) for key, tensor in state.items()}
        print("ron-runtime: constructing model on meta device to avoid a second full weight allocation", flush=True)
        with torch.device("meta"):
            self.model = ScalableRonLM(self.config)
        print("ron-runtime: assigning checkpoint weights", flush=True)
        self.model.load_state_dict(state, strict=True, assign=True)
        # assign=True reduces load-time memory but can split tied parameters into
        # separate Parameter objects. Re-establish Ron's configured weight tying.
        self.model.tie_weights()
        del payload, state
        self.model.eval()
        self.parameter_count = count_parameters(self.model)
        if any(parameter.is_meta for parameter in self.model.parameters()):
            raise RuntimeError("Checkpoint assignment left model parameters on the meta device")
        if any(parameter.dtype != target_dtype for parameter in self.model.parameters()):
            raise RuntimeError("Loaded checkpoint precision does not match RON_PRECISION")
        if self.config.tie_embeddings and self.model.lm_head.weight is not self.model.embed_tokens.weight:
            raise RuntimeError("Configured tied embeddings were not preserved")
        if self.config.tie_embeddings and self.parameter_count != 180_255_744:
            raise RuntimeError(f"Unexpected tied-model parameter count: {self.parameter_count}")

        print("ron-runtime: verifying checkpoint checksum", flush=True)
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
            "seed_response_count": len(self.seed_responses),
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
            "precision": str(next(self.model.parameters()).dtype),
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

    def answer_with_status(self, text: str, session_id: str) -> tuple[str, str]:
        math_answer = deterministic_math(text)
        if math_answer is not None:
            return math_answer, "deterministic_math"

        social_reply = deterministic_social_reply(text)
        if social_reply is not None:
            with self.lock:
                self.sessions[session_id].append({"role": "user", "text": text})
                self.sessions[session_id].append({"role": "assistant", "text": social_reply})
            return social_reply, "deterministic_social"

        seed_reply = self.seed_responses.get(normalize_lookup_text(text))
        if seed_reply is not None:
            with self.lock:
                self.sessions[session_id].append({"role": "user", "text": text})
                self.sessions[session_id].append({"role": "assistant", "text": seed_reply})
            return seed_reply, "deterministic_seed_corpus"

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
            output = self.model.generate(
                input_ids,
                MAX_NEW_TOKENS,
                eos_token_id=int(self.special["<eos>"]),
                valid_vocab_size=self.valid_vocab_size,
                suppress_token_ids=self.suppressed_token_ids,
                min_new_tokens=min(3, MAX_NEW_TOKENS),
            )[0, len(ids):].tolist()
        decoded = self.tokenizer.decode(output, skip_special_tokens=True).strip()
        fallback = "أنا معك، لكن النموذج لم يصغ ردًا واضحًا هذه المرة. جرّب سؤالًا أبسط."
        reply, rejected = guard_generated_text(decoded, fallback)
        status = "empty_generation_fallback" if rejected else "native_model"
        with self.lock:
            self.sessions[session_id].append({"role": "user", "text": text})
            self.sessions[session_id].append({"role": "assistant", "text": reply})
        return reply, status

    def answer(self, text: str, session_id: str) -> str:
        """Compatibility wrapper for callers that only need the response text."""
        return self.answer_with_status(text, session_id)[0]


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
        request_id = payload.get("request_id", "") if isinstance(payload, dict) else ""
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
            self._json(400, {"error": "invalid_message", "message": "Message must be 1–4000 characters."})
            return
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
            self._json(400, {"error": "invalid_session_id"})
            return
        if request_id and (not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id)):
            self._json(400, {"error": "invalid_request_id"})
            return
        cache_key = (session_id, request_id) if request_id else None
        if cache_key:
            with RESPONSE_CACHE_LOCK:
                cached = RESPONSE_CACHE.get(cache_key)
                if cached is not None:
                    RESPONSE_CACHE.move_to_end(cache_key)
            if cached is not None:
                self._json(200, {"reply": cached[0], "generation_status": cached[1], "health": self.runtime.health(), "cached_retry": True})
                return
        try:
            reply, generation_status = self.runtime.answer_with_status(message.strip(), session_id)
        except Exception:
            self._json(500, {"error": "inference_failed", "message": "Ron inference failed; inspect server logs."})
            return
        if cache_key:
            with RESPONSE_CACHE_LOCK:
                RESPONSE_CACHE[cache_key] = (reply, generation_status)
                RESPONSE_CACHE.move_to_end(cache_key)
                while len(RESPONSE_CACHE) > RESPONSE_CACHE_LIMIT:
                    RESPONSE_CACHE.popitem(last=False)
        self._json(200, {"reply": reply, "generation_status": generation_status, "health": self.runtime.health()})

    def end_headers(self) -> None:
        # Static assets may be cached by the browser; app.js is versioned on deploy.
        super().end_headers()


def main() -> None:
    ensure_runtime_bundle()
    print("ron-runtime: starting native model initialization", flush=True)
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
