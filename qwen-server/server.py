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

# Secondary provider: OpenRouter. The key stays on Railway, never in the browser.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_UPSTREAM = os.getenv("OPENROUTER_UPSTREAM", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

def error_response(message: str, status: int = 503):
    return Response(
        content=json.dumps({"error": {"message": message}}, ensure_ascii=False),
        media_type="application/json",
        status_code=status,
    )

async def call_provider(client, url, key, model, body, extra_headers=None):
    if not key:
        return None, "provider key is not configured"
    payload = dict(body)
    payload["model"] = model
    payload["stream"] = False
    payload.pop("enable_thinking", None)
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
    if QWEN_API_KEY: providers.append("qwen3")
    if OPENROUTER_API_KEY: providers.append("openrouter")
    return {"service": "ron-model-gateway", "providers": providers, "primary": "qwen3", "fallback": "openrouter" if OPENROUTER_API_KEY else None, "status": "ok"}

@app.get("/health")
async def health():
    return {
        "ok": bool(QWEN_API_KEY or OPENROUTER_API_KEY),
        "qwen_configured": bool(QWEN_API_KEY),
        "fallback_configured": bool(OPENROUTER_API_KEY),
        "qwen_model": QWEN_MODEL,
        "fallback_model": OPENROUTER_MODEL,
    }

@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        return error_response("invalid JSON request", 400)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        data, qwen_error = await call_provider(
            client, QWEN_UPSTREAM, QWEN_API_KEY, QWEN_MODEL, body,
            {"X-Provider": "qwen3"},
        )
        if data is not None:
            return Response(content=json.dumps(data, ensure_ascii=False), media_type="application/json", status_code=200)

        # Qwen failed/unavailable: transparently try the secondary model.
        fallback_body = dict(body)
        data, fallback_error = await call_provider(
            client, OPENROUTER_UPSTREAM, OPENROUTER_API_KEY, OPENROUTER_MODEL, fallback_body,
            {"HTTP-Referer": "https://arkanws513-spec.github.io/Ron-0/", "X-Title": "Ron"},
        )
        if data is not None:
            return Response(content=json.dumps(data, ensure_ascii=False), media_type="application/json", status_code=200)

    return error_response(
        "No language model is available. Qwen3 failed: "
        + str(qwen_error)
        + "; fallback failed: "
        + str(fallback_error)
    )
