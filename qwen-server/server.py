import json
import os
import re
import threading
from typing import Any

import torch
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Ron Core — local Qwen open-weight runtime")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://arkanws513-spec.github.io"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Ron's actual base model. No Qwen API key is used.
MODEL_ID = os.getenv("RON_QWEN_MODEL", "Qwen/Qwen3-0.6B")
MAX_NEW_TOKENS = min(max(int(os.getenv("RON_MAX_NEW_TOKENS", "384")), 64), 768)
MODEL_REVISION = os.getenv("RON_QWEN_REVISION", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEY2 = os.getenv("GEMINI_API_KEY2", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

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
        # Qwen publishes the model in bfloat16; keep that dtype on CPU to avoid
        # unnecessarily doubling the weight memory.
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
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
        if content:
            normalized.append({"role": role, "content": content})
    return normalized

@app.get("/")
async def root():
    return {
        "service": "ron-core",
        "core": "Qwen open-weight",
        "model": MODEL_ID,
        "qwen_api_required": False,
        "external_teachers": {
            "gemini_configured": bool(GEMINI_API_KEY or GEMINI_API_KEY2),
            "deepseek_configured": bool(DEEPSEEK_API_KEY),
        },
        "status": "ok",
    }

@app.get("/health")
async def health():
    return {
        "ok": _model is not None,
        "core": "qwen-open-weight-local",
        "model": MODEL_ID,
        "model_loaded": _model is not None,
        "gemini_teacher_configured": bool(GEMINI_API_KEY or GEMINI_API_KEY2),
        "deepseek_teacher_configured": bool(DEEPSEEK_API_KEY),
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

    try:
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

        temperature = float(body.get("temperature", 0.6) or 0.6)
        temperature = min(max(temperature, 0.1), 1.2)
        max_tokens = min(max(int(body.get("max_tokens", MAX_NEW_TOKENS) or MAX_NEW_TOKENS), 64), MAX_NEW_TOKENS)

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
            return error_response("local Qwen returned an empty response", 502)

        return {
            "id": "ron-local-qwen",
            "object": "chat.completion",
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        }
    except Exception as exc:
        return error_response("Ron local Qwen runtime error: " + str(exc), 503)

@app.post("/v1/teacher/status")
async def teacher_status():
    return {
        "gemini": bool(GEMINI_API_KEY or GEMINI_API_KEY2),
        "deepseek": bool(DEEPSEEK_API_KEY),
        "note": "External teachers are optional and are not used as Ron's core.",
    }
