import json
import os
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Ron Model Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://arkanws513-spec.github.io"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_UPSTREAM = os.getenv("QWEN_UPSTREAM", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-4b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_UPSTREAM = os.getenv("GEMINI_UPSTREAM", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_UPSTREAM = os.getenv("DEEPSEEK_UPSTREAM", "https://api.deepseek.com/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_UPSTREAM = os.getenv("OPENROUTER_UPSTREAM", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "")

def error_response(message: str, status: int = 503):
    return Response(content=json.dumps({"error": {"message": message}}, ensure_ascii=False), media_type="application/json", status_code=status)

async def call_provider(client, url, key, model, body, extra_headers=None, thinking=False):
    if not key or not model:
        return None, "provider is not configured"
    payload = dict(body)
    payload["model"] = model
    payload["stream"] = False
    payload.pop("enable_thinking", None)
    if thinking:
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = "high"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    try:
        response = await client.post(url, json=payload, headers=headers)
        try:
            data = response.json()
        except Exception:
            data = {"error": {"message": response.text[:2000]}}
        if response.is_success:
            return data, None
        msg = data.get("error", {}).get("message") if isinstance(data, dict) else None
        return None, msg or f"HTTP {response.status_code}"
    except httpx.HTTPError as exc:
        return None, f"network error: {exc}"

@app.get("/")
async def root():
    providers = []
    if DEEPSEEK_API_KEY: providers.append("deepseek")
    if GEMINI_API_KEY: providers.append("gemini")
    if QWEN_API_KEY: providers.append("qwen3")
    if OPENROUTER_API_KEY and OPENROUTER_MODEL: providers.append("openrouter")
    return {"service":"ron-model-gateway","providers":providers,"primary":"deepseek","fallback_chain":["gemini","qwen3","openrouter"],"status":"ok"}

@app.get("/health")
async def health():
    return {
        "ok": bool(DEEPSEEK_API_KEY or GEMINI_API_KEY or QWEN_API_KEY or (OPENROUTER_API_KEY and OPENROUTER_MODEL)),
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "qwen_configured": bool(QWEN_API_KEY),
        "openrouter_configured": bool(OPENROUTER_API_KEY and OPENROUTER_MODEL),
        "deepseek_model": DEEPSEEK_MODEL,
        "gemini_model": GEMINI_MODEL,
        "qwen_model": QWEN_MODEL,
        "openrouter_model": OPENROUTER_MODEL or None,
    }

@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        return error_response("invalid JSON request", 400)
    errors = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        data, err = await call_provider(client, DEEPSEEK_UPSTREAM, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, body, thinking=True)
        if data is not None:
            return Response(content=json.dumps(data, ensure_ascii=False), media_type="application/json", status_code=200)
        errors.append("DeepSeek: " + str(err))
        data, err = await call_provider(client, GEMINI_UPSTREAM, GEMINI_API_KEY, GEMINI_MODEL, body)
        if data is not None:
            return Response(content=json.dumps(data, ensure_ascii=False), media_type="application/json", status_code=200)
        errors.append("Gemini: " + str(err))
        data, err = await call_provider(client, QWEN_UPSTREAM, QWEN_API_KEY, QWEN_MODEL, body, {"X-Provider":"qwen3"})
        if data is not None:
            return Response(content=json.dumps(data, ensure_ascii=False), media_type="application/json", status_code=200)
        errors.append("Qwen3: " + str(err))
        data, err = await call_provider(client, OPENROUTER_UPSTREAM, OPENROUTER_API_KEY, OPENROUTER_MODEL, body, {"HTTP-Referer":"https://arkanws513-spec.github.io/Ron-0/","X-Title":"Ron"})
        if data is not None:
            return Response(content=json.dumps(data, ensure_ascii=False), media_type="application/json", status_code=200)
        errors.append("OpenRouter: " + str(err))
    return error_response("No language model is available. " + " | ".join(errors))
