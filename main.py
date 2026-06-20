import asyncio
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai

APP_NAME = "CarryAI"

# Render env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_PRIMARY = os.getenv("GEMINI_MODEL_PRIMARY", "gemini-3.1-flash-lite").strip()
GEMINI_MODEL_FALLBACK = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-2.5-flash").strip()
USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() == "true"

# Keep this strict so the backend only touches public share links.
ALLOWED_HOST_SUFFIXES = {
    s.strip().lower()
    for s in os.getenv(
        "ALLOWED_HOST_SUFFIXES",
        "claude.ai,chatgpt.com,gemini.google.com,gemini.ai,kimi.ai,chat.kimi.com"
    ).split(",")
    if s.strip()
}

OUT_DIR = Path(os.getenv("CARRYAI_OUT_DIR", tempfile.gettempdir())) / "carryai_ctx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, Dict] = {}


def short_chat_id_from_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]


def normalize_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are allowed.")

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Invalid URL host.")

    allowed = any(host == s or host.endswith("." + s) for s in ALLOWED_HOST_SUFFIXES)
    if not allowed:
        raise ValueError("URL host is not allowed.")

    return url


async def fetch_html(url: str) -> tuple[str, str, str]:
    """
    Fetch public share page.
    Returns: html, final_url, source_name
    """
    headers = {
        "User-Agent": "CarryAI/1.0 (+public-share-context-import)"
    }

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
        headers=headers,
    ) as client:
        resp = await client.get(url)

    if resp.status_code != 200:
        raise ValueError(f"Link is not publicly accessible. HTTP {resp.status_code}")

    final_url = str(resp.url)
    host = urlparse(final_url).hostname or ""
    source_name = host.split(".")[0].capitalize()
    return resp.text, final_url, source_name


def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    main = soup.find("main")
    if main:
        text = main.get_text("\n", strip=True)
    else:
        text = soup.get_text("\n", strip=True)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


def build_prompt(source_name: str, source_url: str, chat_id: str, transcript: str) -> str:
    return f"""
You are helping convert a public AI conversation into a portable context file for another AI assistant.

Rules:
- Output plain text only.
- Avoid first-person pronouns and second-person pronouns.
- Refer to the person as "the user" or use neutral phrasing.
- Preserve the user's words verbatim where possible, especially instructions, preferences, corrections, decisions, names, project details, and open tasks.
- Remove greetings, filler, repeated lines, and irrelevant small talk.
- If a detail is uncertain, write "unknown" rather than guessing.
- Keep the output compact and highly useful for continuation in another model.

Return the output in exactly this structure:

CTX_VERSION: 1
SOURCE: {source_name}
CHAT_ID: {chat_id}
SOURCE_URL: {source_url}

1. Demographics Information
- ...

2. Interests & Preferences
- ...

3. Relationships
- ...

4. Dated Events, Projects & Plans
- ...

5. Instructions
- ...

6. Important Verbatim Quotes
- ...

7. Current Open Tasks
- ...

Imported from: {source_name}

Transcript:
{transcript}
""".strip()


def make_ctx_text(source_name: str, chat_id: str, source_url: str, body: str) -> str:
    return f"""CTX_VERSION: 1
SOURCE: {source_name}
CHAT_ID: {chat_id}
SOURCE_URL: {source_url}

{body}

END_CTX
"""


def choose_file_name(source_name: str, chat_id: str) -> str:
    safe_source = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_name.lower())[:20] or "source"
    safe_chat = re.sub(r"[^a-zA-Z0-9_-]+", "_", chat_id)[:20] or "chat"
    return f"{safe_source}_{safe_chat}.ctx"


def write_file(file_name: str, content: str) -> Path:
    path = OUT_DIR / file_name
    path.write_text(content, encoding="utf-8")
    return path


def gemini_compress(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    last_error = None
    for model in [GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK]:
        if not model:
            continue
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[prompt],
            )
            text = getattr(resp, "text", None) or ""
            text = text.strip()
            if not text:
                raise RuntimeError(f"{model} returned empty text.")
            return text
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Gemini failed on both models: {last_error}")


async def process_job(job_id: str, url: str) -> None:
    jobs[job_id] = {"status": "starting", "progress": 1, "error": None}

    try:
        jobs[job_id].update(status="validating_url", progress=5)
        clean_url = normalize_url(url)

        jobs[job_id].update(status="fetching_page", progress=15)
        html, final_url, source_name = await fetch_html(clean_url)

        jobs[job_id].update(status="extracting_text", progress=35)
        transcript = extract_visible_text(html)
        if len(transcript) < 200:
            raise ValueError("No usable public chat text found on that link.")

        chat_id = short_chat_id_from_url(final_url)
        file_name = choose_file_name(source_name, chat_id)

        jobs[job_id].update(status="compressing_with_gemini", progress=60)
        prompt = build_prompt(source_name, final_url, chat_id, transcript)
        body = await asyncio.to_thread(gemini_compress, prompt) if USE_GEMINI else transcript

        jobs[job_id].update(status="writing_ctx_file", progress=90)
        ctx_text = make_ctx_text(source_name, chat_id, final_url, body)
        file_path = write_file(file_name, ctx_text)

        jobs[job_id].update(
            status="done",
            progress=100,
            file_name=file_path.name,
            download_url=f"/download/{job_id}",
        )

    except Exception as e:
        jobs[job_id].update(status="error", progress=100, error=str(e))


@app.get("/health")
async def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "gemini_primary": GEMINI_MODEL_PRIMARY,
        "gemini_fallback": GEMINI_MODEL_FALLBACK,
        "use_gemini": USE_GEMINI,
    }


@app.websocket("/ws/context")
async def ws_context(websocket: WebSocket):
    await websocket.accept()
    job_id = hashlib.sha1(os.urandom(24)).hexdigest()[:12]

    try:
        payload = await websocket.receive_json()
        url = str(payload.get("url", "")).strip()
        if not url:
            await websocket.send_json({"status": "error", "progress": 100, "error": "URL is required."})
            return

        await websocket.send_json({"job_id": job_id, "status": "accepted", "progress": 1})

        task = asyncio.create_task(process_job(job_id, url))

        while not task.done():
            state = jobs.get(job_id, {})
            await websocket.send_json({
                "status": state.get("status", "working"),
                "progress": state.get("progress", 0),
            })
            await asyncio.sleep(0.4)

        await task

        await websocket.send_json(jobs.get(job_id, {
            "status": "error",
            "progress": 100,
            "error": "Unknown failure."
        }))

    except WebSocketDisconnect:
        return


@app.get("/download/{job_id}")
async def download(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job.")

    file_name = job.get("file_name")
    if not file_name:
        raise HTTPException(status_code=404, detail="File not ready.")

    path = OUT_DIR / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing.")

    return FileResponse(
        path=path,
        filename=file_name,
        media_type="text/plain",
    )
