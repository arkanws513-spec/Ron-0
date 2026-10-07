import json
import os
import re
import threading
from typing import Any

import httpx
import torch
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Ron Core — Qwen + DeepSeek + Gemini")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://arkanws513-spec.github.io",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Qwen is Ron's local open-weight base model. No Qwen API key is required.
MODEL_ID = os.getenv("RON_QWEN_MODEL", "Qwen/Qwen3-0.6B")
MAX_NEW_TOKENS = min(max(int(os.getenv("RON_MAX_NEW_TOKENS", "192")), 64), 512)
MAX_INPUT_CHARS = min(max(int(os.getenv("RON_MAX_INPUT_CHARS", "12000")), 4000), 40000)
MODEL_THREADS = min(max(int(os.getenv("RON_MODEL_THREADS", str(min(os.cpu_count() or 2, 4)))), 1), 16)
try:
    torch.set_num_threads(MODEL_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
MODEL_REVISION = os.getenv("RON_QWEN_REVISION", "")

# Optional external reasoning engines. Keys stay server-side in Railway.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY2", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

PROVIDER_ORDER = [
    x.strip().lower()
    for x in os.getenv("RON_PROVIDER_ORDER", "qwen,deepseek,gemini").split(",")
    if x.strip()
]

_tokenizer = None
_model = None
_model_lock = threading.Lock()
_generation_lock = threading.Lock()


def error_response(message: str, status: int = 503):
    return Response(
        content=json.dumps({"error": {"message": message}}, ensure_ascii=False),
        media_type="application/json",
        status_code=status,
    )


def load_model():
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return
    with _model_lock:
        if _model is not None and _tokenizer is not None:
            return
        kwargs = {}
        if MODEL_REVISION:
            kwargs["revision"] = MODEL_REVISION
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
        dtype = torch.bfloat16 if getattr(torch, "cpu", None) is not None and torch.backends.cpu.is_bf16_supported() else torch.float32
        try:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **kwargs,
            )
        except Exception:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                **kwargs,
            )
        _model.eval()


def clean_generated(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^assistant\s*:\s*", "", text, flags=re.I).strip()
    return text


def build_messages(body: dict[str, Any]):
    messages = body.get("messages") or []
    normalized = []
    for item in messages[-18:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        content = str(content or "").strip()
        if len(content) > MAX_INPUT_CHARS:
            content = content[-MAX_INPUT_CHARS:]
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def provider_messages(messages: list[dict[str, str]], provider: str):
    if provider == "gemini":
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        contents = []
        if system_parts:
            contents.append({"role": "user", "parts": [{"text": "\n\n".join(system_parts)}]})
            contents.append({"role": "model", "parts": [{"text": "فهمت سياق رون وتعليماته."}]})
        for m in messages:
            if m["role"] == "system":
                continue
            contents.append({
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            })
        return contents
    return messages


async def ask_deepseek(messages: list[dict[str, str]], temperature: float, max_tokens: int):
    if not DEEPSEEK_API_KEY:
        return None
    async with httpx.AsyncClient(timeout=httpx.Timeout(28.0, connect=8.0)) as client:
        response = await client.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        data = response.json()
        if response.status_code >= 400:
            raise RuntimeError(f"DeepSeek HTTP {response.status_code}: {data.get('error', {}).get('message', 'request failed')}")
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(text, list):
            text = "".join(
                str(x.get("text", "")) if isinstance(x, dict) else str(x)
                for x in text
            )
        text = clean_generated(str(text or ""))
        return {
            "text": text,
            "provider": "deepseek",
            "model": DEEPSEEK_MODEL,
        } if text else None


async def ask_gemini(messages: list[dict[str, str]], temperature: float, max_tokens: int):
    if not GEMINI_API_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    async with httpx.AsyncClient(timeout=httpx.Timeout(28.0, connect=8.0)) as client:
        response = await client.post(
            url,
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json={
                "contents": provider_messages(messages, "gemini"),
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
        )
        data = response.json()
        if response.status_code >= 400:
            message = data.get("error", {}).get("message", "request failed")
            raise RuntimeError(f"Gemini HTTP {response.status_code}: {message}")
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
        text = clean_generated(text)
        return {
            "text": text,
            "provider": "gemini",
            "model": GEMINI_MODEL,
        } if text else None


def ask_qwen_sync(messages: list[dict[str, str]], temperature: float, max_tokens: int):
    load_model()
    prompt_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
    inputs = _tokenizer.apply_chat_template(messages, **prompt_kwargs)
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with _generation_lock, torch.inference_mode():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.05,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = clean_generated(_tokenizer.decode(generated, skip_special_tokens=True))
    if not text:
        return None
    return {
        "text": text,
        "provider": "qwen",
        "model": MODEL_ID,
    }


async def run_provider_chain(messages: list[dict[str, str]], temperature: float, max_tokens: int):
    errors = []
    for provider in PROVIDER_ORDER:
        try:
            if provider == "qwen":
                # Keep the CPU model isolated from the async event loop and do not let
                # a slow first-load/generation block the external fallback engines.
                import asyncio
                result = await asyncio.wait_for(
                    asyncio.to_thread(ask_qwen_sync, messages, temperature, max_tokens),
                    timeout=float(os.getenv("RON_QWEN_TIMEOUT_SECONDS", "24")),
                )
            elif provider == "deepseek":
                result = await ask_deepseek(messages, temperature, max_tokens)
            elif provider == "gemini":
                result = await ask_gemini(messages, temperature, max_tokens)
            else:
                continue
            if result and result.get("text"):
                result["chain"] = PROVIDER_ORDER
                result["fallback_used"] = provider != PROVIDER_ORDER[0]
                return result, errors
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
    return None, errors


@app.get("/")
async def root():
    return {
        "service": "ron-core",
        "core": "Qwen open-weight",
        "model": MODEL_ID,
        "qwen_api_required": False,
        "provider_order": PROVIDER_ORDER,
        "external_engines": {
            "gemini_configured": bool(GEMINI_API_KEY),
            "deepseek_configured": bool(DEEPSEEK_API_KEY),
        },
        "status": "ok",
    }


@app.get("/health")
async def health():
    return {
        "ok": True,
        "core": "qwen-open-weight-local",
        "model": MODEL_ID,
        "model_loaded": _model is not None,
        "provider_order": PROVIDER_ORDER,
        "gemini_configured": bool(GEMINI_API_KEY),
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
    }


@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        return error_response("invalid JSON request", 400)

    messages = build_messages(body)
    if not messages:
        return error_response("messages is required", 400)

    temperature = min(max(float(body.get("temperature", 0.45) or 0.45), 0.1), 1.2)
    max_tokens = min(
        max(int(body.get("max_tokens", MAX_NEW_TOKENS) or MAX_NEW_TOKENS), 64),
        MAX_NEW_TOKENS,
    )

    result, errors = await run_provider_chain(messages, temperature, max_tokens)
    if not result:
        return error_response(
            "كل محركات رون غير متاحة حاليًا: " + " | ".join(errors[-3:]),
            503,
        )

    return {
        "id": "ron-core-response",
        "object": "chat.completion",
        "model": result["model"],
        "provider": result["provider"],
        "fallback_used": result["fallback_used"],
        "provider_chain": result["chain"],
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result["text"],
            },
            "finish_reason": "stop",
        }],
    }


@app.post("/v1/teacher/status")
async def teacher_status():
    return {
        "qwen": True,
        "gemini": bool(GEMINI_API_KEY),
        "deepseek": bool(DEEPSEEK_API_KEY),
        "provider_order": PROVIDER_ORDER,
        "note": "Ron owns orchestration, memory and learning; external engines provide optional reasoning support.",
    }
