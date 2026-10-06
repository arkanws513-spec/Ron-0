import os
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(title="Ron Qwen Teacher Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://arkanws513-spec.github.io",
        "https://github.com",
    ],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

API_KEY = os.getenv("QWEN_API_KEY", "")
UPSTREAM = os.getenv(
    "QWEN_UPSTREAM",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
)
MODEL = os.getenv("QWEN_MODEL", "qwen3-4b")

@app.get("/")
async def root():
    return {"service": "ron-qwen-teacher", "model": MODEL, "status": "ok"}

@app.get("/health")
async def health():
    return {"ok": bool(API_KEY), "model": MODEL}

@app.post("/v1/chat/completions")
async def chat(request: Request):
    if not API_KEY:
        return {"error": {"message": "QWEN_API_KEY is not configured"}}

    body = await request.json()
    body["model"] = MODEL
    body["stream"] = False

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(UPSTREAM, json=body, headers=headers)

    try:
        data = response.json()
    except Exception:
        data = {"error": {"message": response.text[:2000]}}

    return Response(content=__import__("json").dumps(data, ensure_ascii=False), media_type="application/json", status_code=response.status_code)
