import json
import os
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Ron Qwen Teacher Bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://arkanws513-spec.github.io"],
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
    return {
        "ok": bool(API_KEY),
        "model": MODEL,
        "upstream_configured": bool(UPSTREAM),
        "thinking_disabled": True,
    }

@app.post("/v1/chat/completions")
async def chat(request: Request):
    if not API_KEY:
        return Response(
            content=json.dumps(
                {"error": {"message": "QWEN_API_KEY is not configured"}},
                ensure_ascii=False,
            ),
            media_type="application/json",
            status_code=503,
        )

    try:
        body = await request.json()
    except Exception:
        return Response(
            content=json.dumps(
                {"error": {"message": "invalid JSON request"}},
                ensure_ascii=False,
            ),
            media_type="application/json",
            status_code=400,
        )

    # Ron needs the final answer, not an internal reasoning trace.
    # Qwen remains a teacher/advisor; it does not own Ron's memory or identity.
    body["model"] = MODEL
    body["stream"] = False
    body["enable_thinking"] = False

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=15.0)
        ) as client:
            response = await client.post(UPSTREAM, json=body, headers=headers)
    except httpx.HTTPError as exc:
        return Response(
            content=json.dumps(
                {"error": {"message": f"upstream network error: {exc}"}},
                ensure_ascii=False,
            ),
            media_type="application/json",
            status_code=502,
        )

    try:
        data = response.json()
    except Exception:
        data = {"error": {"message": response.text[:2000]}}

    return Response(
        content=json.dumps(data, ensure_ascii=False),
        media_type="application/json",
        status_code=response.status_code,
    )
