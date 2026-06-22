
import asyncio
import hashlib
import json
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
from playwright.async_api import async_playwright

APP_NAME = "CarryAI"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_PRIMARY = os.getenv("GEMINI_MODEL_PRIMARY", "gemini-2.5-flash").strip()
GEMINI_MODEL_FALLBACK = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-1.5-flash").strip()
USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() == "true"

# Proxy support: If your IP is blocked by Cloudflare, set this environment variable
# Example: export CARRYAI_PROXY="http://username:password@proxy.example.com:8080"
PROXY_URL = os.getenv("CARRYAI_PROXY", "").strip()

ALLOWED_HOST_SUFFIXES = {
    s.strip().lower()
    for s in os.getenv(
        "ALLOWED_HOST_SUFFIXES",
        "claude.ai,chatgpt.com,gemini.google.com,g.co,gemini.ai,"
        "kimi.ai,chat.kimi.com,perplexity.ai,pplx.ai"
    ).split(",")
    if s.strip()
}

SOURCE_NAME_MAP = {
    "claude.ai": "Claude",
    "chatgpt.com": "ChatGPT",
    "gemini.google.com": "Gemini",
    "g.co": "Gemini",
    "gemini.ai": "Gemini",
    "kimi.ai": "Kimi",
    "chat.kimi.com": "Kimi",
    "perplexity.ai": "Perplexity",
    "pplx.ai": "Perplexity",
}

NOISE_PATTERNS = re.compile(
    r"^(continue (this )?conversation|sign up|log in|sign in|new chat|"
    r"share|copy link|regenerate|edit|retry|download|try for free|"
    r"get started|cookie|privacy policy|terms of service|\d+\s*/\s*\d+)$",
    re.IGNORECASE,
)

# Hardened Stealth Script to bypass Cloudflare Turnstile
STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
window.chrome = { runtime: {} };
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) => (
  parameters.name === 'notifications' ?
    Promise.resolve({ state: Notification.permission }) :
    originalQuery(parameters)
);
"""

# Standard user agent for modern browsers
HTTPX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
}

# Disguise as Googlebot (Many AI companies whitelist Googlebot to index share URLs)
GOOGLEBOT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
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


# ─── HELPERS ────────────────────────────────────────────────────────────────

def short_chat_id_from_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:20]

def normalize_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed.")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Invalid URL host.")
    if not any(host == s or host.endswith("." + s) for s in ALLOWED_HOST_SUFFIXES):
        raise ValueError(f"Host '{host}' is not an allowed platform.")
    return url

def get_host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()

def resolve_source_name(host: str) -> str:
    for suffix, name in SOURCE_NAME_MAP.items():
        if host == suffix or host.endswith("." + suffix):
            return name
    return host.split(".")[0].capitalize()

def is_bot_wall(html: str) -> bool:
    """Detects if we are stuck on a Cloudflare or standard Bot Wall."""
    text = html.lower()
    if "just a moment..." in text and "cloudflare" in text: return True
    if "cf-browser-verification" in text: return True
    if "verify you are human" in text and "cloudflare" in text: return True
    if "enable javascript and cookies to continue" in text: return True
    soup = BeautifulSoup(html, "html.parser")
    if soup.find(id="challenge-running"): return True
    return False

def has_valid_content(html: str) -> bool:
    """Checks if the page actually contains conversation DOM elements."""
    if "__NEXT_DATA__" in html: return True
    if "font-user-message" in html or "data-is-user" in html: return True
    if "data-message-author-role" in html: return True
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(strip=True)
    return len(text) > 300

def _deep_get(obj, *keys):
    for key in keys:
        if obj is None: return None
        if isinstance(obj, dict): obj = obj.get(key)
        elif isinstance(obj, list) and isinstance(key, int): obj = obj[key] if key < len(obj) else None
        else: return None
    return obj


# ─── FETCHERS ───────────────────────────────────────────────────────────────

async def fetch_html_httpx(url: str, headers: dict) -> tuple[str, str, str]:
    proxies = {"all://": PROXY_URL} if PROXY_URL else None
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=30.0, headers=headers, proxy=proxies
    ) as client:
        resp = await client.get(url)

    if resp.status_code not in (200, 403): # We pass 403 back to check if it's a bot wall
        raise ValueError(f"Not publicly accessible. HTTP {resp.status_code}")

    final_url = str(resp.url)
    return resp.text, final_url, resolve_source_name(get_host(final_url))


async def fetch_html_playwright(url: str) -> tuple[str, str, str]:
    async with async_playwright() as p:
        pw_proxy = {"server": PROXY_URL} if PROXY_URL else None
        browser = await p.chromium.launch(
            proxy=pw_proxy,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
            ]
        )
        try:
            context = await browser.new_context(
                user_agent=HTTPX_HEADERS["User-Agent"],
                viewport={"width": 1366, "height": 768},
                locale="en-US",
                timezone_id="America/New_York",
            )
            await context.add_init_script(STEALTH_SCRIPT)
            page = await context.new_page()

            resp = await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            final_url = page.url
            
            # --- TURNSTILE EVASION LOGIC ---
            # Simulate human cursor movements
            await page.mouse.move(100, 100)
            await page.wait_for_timeout(200)
            await page.mouse.move(300, 400)
            await page.wait_for_timeout(500)
            
            # If there's an iframe (Turnstile challenge), attempt to click its center
            try:
                iframe = await page.query_selector("iframe")
                if iframe:
                    box = await iframe.bounding_box()
                    if box:
                        await page.mouse.click(box["x"] + box["width"]/2, box["y"] + box["height"]/2)
                        await page.wait_for_timeout(3000) # wait for challenge to process
            except Exception:
                pass
            
            # Explicitly wait for main chat elements to load
            try:
                await page.wait_for_selector(
                    "div[data-is-user], .font-user-message, [data-message-author-role], main", 
                    timeout=15000
                )
            except Exception:
                pass

            # Scroll to trigger lazy loading
            for _ in range(5):
                await page.mouse.wheel(0, 3000)
                await page.wait_for_timeout(400)

            html = await page.content()
            
            # Fast fail if still stuck on bot wall
            if is_bot_wall(html):
                raise ValueError("Blocked by Cloudflare Turnstile anti-bot protection. Try setting the CARRYAI_PROXY environment variable with a residential proxy.")

            source_name = resolve_source_name(get_host(final_url))
            return html, final_url, source_name
        finally:
            await browser.close()


async def fetch_html(url: str) -> tuple[str, str, str]:
    """Orchestrator: Tries Googlebot first, then Standard, then Playwright."""
    
    # Attempt 1: Googlebot bypass (Often whitelisted for public share links)
    try:
        html, final_url, source_name = await fetch_html_httpx(url, GOOGLEBOT_HEADERS)
        if not is_bot_wall(html) and has_valid_content(html):
            return html, final_url, source_name
    except Exception:
        pass

    # Attempt 2: Standard HTTPX
    try:
        html, final_url, source_name = await fetch_html_httpx(url, HTTPX_HEADERS)
        if not is_bot_wall(html) and has_valid_content(html):
            return html, final_url, source_name
    except Exception:
        pass
            
    # Attempt 3: Playwright with Turnstile clicker
    return await fetch_html_playwright(url)


# ─── EXTRACTORS ─────────────────────────────────────────────────────────────

def extract_claude_dom(soup: BeautifulSoup) -> Optional[str]:
    messages = soup.find_all(lambda tag: tag.name == "div" and (
        tag.has_attr('data-is-user') or
        tag.has_attr('data-test-render-role') or
        (tag.get('class') and any('font-user-message' in c or 'font-claude-message' in c for c in tag.get('class', [])))
    ))
    
    if messages:
        lines = []
        for msg in messages:
            role = "UNKNOWN"
            if msg.get('data-is-user') == 'true': role = "USER"
            elif msg.get('data-is-user') == 'false': role = "CLAUDE"
            elif msg.get('data-test-render-role') == 'user': role = "USER"
            elif msg.get('data-test-render-role') == 'assistant': role = "CLAUDE"
            elif any('font-user-message' in c for c in msg.get('class', [])): role = "USER"
            elif any('font-claude-message' in c for c in msg.get('class', [])): role = "CLAUDE"
            
            text = msg.get_text(separator=' ', strip=True)
            if text:
                lines.append(f"{role}: {text}")
        return "\n\n".join(lines)
    return None

def extract_claude_next(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string: return ""
    
    try: data = json.loads(tag.string)
    except Exception: return ""
    
    pp = _deep_get(data, "props", "pageProps") or {}
    conversation = pp.get("sharedConversation") or pp.get("conversation") or _deep_get(pp, "initialData", "conversation") or {}
    messages = conversation.get("chat_messages") or conversation.get("messages") or []

    lines = []
    for msg in messages:
        role = (msg.get("sender") or msg.get("role") or "unknown").upper()
        text = msg.get("text") or ""
        if not text:
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
        if text.strip():
            lines.append(f"{role}: {text.strip()}")
    return "\n\n".join(lines)


def extract_chatgpt_dom(soup: BeautifulSoup) -> Optional[str]:
    messages = soup.select("[data-message-author-role]")
    if messages:
        lines = []
        for msg in messages:
            role = msg.get("data-message-author-role", "unknown").upper()
            lines.append(f"{role}: {msg.get_text(separator=' ', strip=True)}")
        return "\n\n".join(lines)
    return None


def extract_generic(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "button", "svg", "img"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = root.get_text("\n", strip=True)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line or NOISE_PATTERNS.match(line): continue
        lines.append(line)
    return "\n".join(lines)


def extract_visible_text(html: str, source_name: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    
    if source_name == "Claude":
        text = extract_claude_dom(soup)
        if text: return text
        text = extract_claude_next(html)
        if text: return text

    if source_name == "ChatGPT":
        text = extract_chatgpt_dom(soup)
        if text: return text

    return extract_generic(html)


# ─── COMPRESSION ─────────────────────────────────────────────────────────────

def build_prompt(source_name: str, source_url: str, chat_id: str, transcript: str) -> str:
    return f"""
You are helping convert a public AI conversation into a portable context file for another AI assistant.

Rules:
- Output plain text only.
- Avoid first-person and second-person pronouns.
- Refer to the person as "the user".
- Preserve the user's words verbatim where possible: instructions, preferences, corrections, decisions, names, project details, open tasks.
- If a detail is uncertain, write "unknown".
- VERY IMPORTANT: You MUST copy the transcript EXACTLY underneath the "Transcript:" heading. Do NOT summarize or skip the Transcript section.

Return output in exactly this structure:

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
        if not model: continue
        try:
            resp = client.models.generate_content(model=model, contents=[prompt])
            text = (getattr(resp, "text", None) or "").strip()
            if not text:
                raise RuntimeError(f"{model} returned empty text.")
            return text
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Gemini compression failed: {last_error}")


# ─── JOB PIPELINE ────────────────────────────────────────────────────────────

async def process_job(job_id: str, url: str) -> None:
    jobs[job_id] = {"status": "starting", "progress": 1, "error": None}
    try:
        jobs[job_id].update(status="validating_url", progress=5)
        clean_url = normalize_url(url)

        jobs[job_id].update(status="fetching_page", progress=15)
        html, final_url, source_name = await fetch_html(clean_url)

        jobs[job_id].update(status="extracting_text", progress=35)
        transcript = extract_visible_text(html, source_name)

        if len(transcript) < 100:
            raise ValueError(
                f"No usable chat content found on this {source_name} link "
                f"({len(transcript)} chars). Ensure the link is public and accessible."
            )

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
            source_name=source_name,
            chars_extracted=len(transcript),
        )

    except Exception as e:
        jobs[job_id].update(status="error", progress=100, error=str(e))


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "gemini_primary": GEMINI_MODEL_PRIMARY,
        "gemini_fallback": GEMINI_MODEL_FALLBACK,
        "use_gemini": USE_GEMINI,
        "proxy_enabled": bool(PROXY_URL)
    }


@app.websocket("/ws/context")
async def ws_context(websocket: WebSocket):
    await websocket.accept()
    job_id = hashlib.sha1(os.urandom(24)).hexdigest()[:12]
    try:
        payload = await websocket.receive_json()
        url = str(payload.get("url", "")).strip()
        if not url:
            await websocket.send_json({
                "status": "error", "progress": 100, "error": "URL is required."
            })
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
            "error": "Unknown failure.",
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
    return FileResponse(path=path, filename=file_name, media_type="text/plain")

