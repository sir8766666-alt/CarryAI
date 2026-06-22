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

# === CONFIGURATION ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_PRIMARY = os.getenv("GEMINI_MODEL_PRIMARY", "gemini-2.5-flash").strip()
GEMINI_MODEL_FALLBACK = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-1.5-flash").strip()
USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() == "true"

ALLOWED_HOST_SUFFIXES = {
    s.strip().lower()
    for s in os.getenv(
        "ALLOWED_HOST_SUFFIXES",
        "claude.ai,chatgpt.com,gemini.google.com,g.co,gemini.ai,kimi.ai,"
        "chat.kimi.com,perplexity.ai,pplx.ai,bing.com"
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
    "bing.com": "Bing",
}

SSR_PLATFORMS = {"claude.ai", "chatgpt.com"}

NOISE_PATTERNS = re.compile(
    r"^(continue (this )?conversation|sign up|log in|sign in|new chat|share|copy link|"
    r"regenerate|edit|retry|download|try for free|get started|cookie|privacy policy|"
    r"terms of service|d+s*/s*d+)$",
    re.IGNORECASE,
)

# === STEALTH SCRIPT FOR PLAYWRIGHT ===
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

HTTPX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
}

OUT_DIR = Path(os.getenv("CARRYAI_OUT_DIR", tempfile.gettempdir())) / "carryai_ctx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === FASTAPI APP ===
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, Dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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


def is_ssr_platform(host: str) -> bool:
    return any(host == s or host.endswith("." + s) for s in SSR_PLATFORMS)


def is_bot_wall(html: str) -> bool:
    text = html.lower()
    if "just a moment..." in text and "cloudflare" in text:
        return True
    if "cf-browser-verification" in text:
        return True
    if "verify you are human" in text and "cloudflare" in text:
        return True
    if "enable javascript and cookies to continue" in text:
        return True
    soup = BeautifulSoup(html, "html.parser")
    if soup.find(id="challenge-running"):
        return True
    return False


def _deep_get(obj, *keys):
    for key in keys:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(key)
        elif isinstance(obj, list) and isinstance(key, int):
            obj = obj[key] if key < len(obj) else None
        else:
            return None
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_html_httpx(url: str) -> tuple[str, str, str]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0, headers=HTTPX_HEADERS) as client:
        resp = await client.get(url)

    if resp.status_code != 200:
        raise ValueError(f"Not publicly accessible. HTTP {resp.status_code}")

    final_url = str(resp.url)
    return resp.text, final_url, resolve_source_name(get_host(final_url))


async def fetch_html_playwright(url: str) -> tuple[str, str, str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
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

            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            final_url = page.url

            # Wait for content to load
            try:
                await page.wait_for_selector(
                    "div[data-is-user], .font-user-message, [data-message-author-role], main, [data-mention], .prose",
                    timeout=15000
                )
            except Exception:
                pass

            # Scroll to trigger lazy loading
            for _ in range(5):
                await page.mouse.wheel(0, 3000)
                await page.wait_for_timeout(400)

            html = await page.content()

            if is_bot_wall(html):
                raise ValueError("Blocked by Cloudflare Turnstile anti-bot protection.")

            source_name = resolve_source_name(get_host(final_url))
            return html, final_url, source_name
        finally:
            await browser.close()


async def fetch_html(url: str) -> tuple[str, str, str]:
    host = get_host(url)

    if is_ssr_platform(host):
        try:
            html, final_url, source_name = await fetch_html_httpx(url)

            is_mojibake = "" in html or len(html) < 250
            has_content = "__NEXT_DATA__" in html or "font-user-message" in html or "data-is-user" in html or "data-message-author-role" in html or "prose" in html

            if is_bot_wall(html) or is_mojibake or not has_content:
                print("HTTPX blocked or missing content. Falling back to Playwright...")
                return await fetch_html_playwright(url)

            return html, final_url, source_name
        except Exception as e:
            print(f"HTTPX failed ({e}), falling back to Playwright...")
            return await fetch_html_playwright(url)

    return await fetch_html_playwright(url)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def extract_claude_dom(soup: BeautifulSoup) -> Optional[str]:
    """Extract messages from Claude - uses multiple methods for reliability"""
    # Method 1: data-is-user attribute (most reliable)
    user_msgs = soup.find_all("div", attrs={"data-is-user": "true"})
    assistant_msgs = soup.find_all("div", attrs={"data-is-user": "false"})

    if user_msgs or assistant_msgs:
        all_msgs = []
        for msg in user_msgs:
            text = msg.get_text(separator=' ', strip=True)
            if text:
                all_msgs.append(("USER", text))
        for msg in assistant_msgs:
            text = msg.get_text(separator=' ', strip=True)
            if text:
                all_msgs.append(("CLAUDE", text))

        if all_msgs:
            lines = [f"{role}: {text}" for role, text in all_msgs]
            return "

".join(lines)

    # Method 2: data-test-render-role attribute
    messages = soup.find_all(lambda tag: tag.name == "div" and tag.has_attr('data-test-render-role'))

    if messages:
        lines = []
        for msg in messages:
            role = msg.get('data-test-render-role', 'unknown').upper()
            if role == 'user':
                role = "USER"
            elif role == 'assistant':
                role = "CLAUDE"
            text = msg.get_text(separator=' ', strip=True)
            if text:
                lines.append(f"{role}: {text}")
        return "

".join(lines)

    # Method 3: CSS classes
    messages = soup.find_all(lambda tag:
        tag.name == "div" and tag.get('class') and any(
            'font-user-message' in c or 'font-claude-message' in c for c in tag.get('class', [])
        ))

    if messages:
        lines = []
        for msg in messages:
            classes = msg.get('class', [])
            role = "UNKNOWN"
            if any('font-user-message' in c for c in classes):
                role = "USER"
            elif any('font-claude-message' in c for c in classes):
                role = "CLAUDE"
            text = msg.get_text(separator=' ', strip=True)
            if text:
                lines.append(f"{role}: {text}")
        return "

".join(lines)

    return None


def extract_claude_next(html: str) -> str:
    """Legacy fallback for Next.js Claude"""
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return ""

    try:
        data = json.loads(tag.string)
    except Exception:
        return ""

    pp = _deep_get(data, "props", "pageProps") or {}
    conversation = pp.get("sharedConversation") or pp.get("conversation") or _deep_get(pp, "initialData", "conversation") or {}
    messages = conversation.get("chat_messages") or conversation.get("messages") or []

    lines = []
    for msg in messages:
        role = (msg.get("sender") or msg.get("role") or "unknown").upper()
        if role == 'user':
            role = "USER"
        elif role in ['assistant', 'claude']:
            role = "CLAUDE"
        text = msg.get("text") or ""
        if not text:
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
        if text.strip():
            lines.append(f"{role}: {text.strip()}")
    return "

".join(lines)


def extract_chatgpt_dom(soup: BeautifulSoup) -> Optional[str]:
    """Extract messages from ChatGPT"""
    messages = soup.select("[data-message-author-role]")
    if messages:
        lines = []
        for msg in messages:
            role = msg.get("data-message-author-role", "unknown").upper()
            if role == 'user':
                role = "USER"
            elif role in ['assistant', 'model']:
                role = "CHATGPT"
            lines.append(f"{role}: {msg.get_text(separator=' ', strip=True)}")
        return "

".join(lines)
    return None


def extract_gemini_dom(soup: BeautifulSoup) -> Optional[str]:
    """Extract messages from Gemini"""
    messages = soup.find_all(lambda tag:
        tag.name == "div" and (
            tag.has_attr('data-user-query') or
            tag.has_attr('data-model-response') or
            (tag.get('class') and any('user-query' in c or 'model-response' in c for c in tag.get('class', [])))
        ))

    if messages:
        lines = []
        for msg in messages:
            classes = msg.get('class', [])
            role = "UNKNOWN"
            if any('user-query' in c or 'data-user-query' in str(msg.attrs) for c in classes):
                role = "USER"
            elif any('model-response' in c or 'data-model-response' in str(msg.attrs) for c in classes):
                role = "GEMINI"
            text = msg.get_text(separator=' ', strip=True)
            if text:
                lines.append(f"{role}: {text}")
        return "

".join(lines)
    return None


def extract_perplexity_dom(soup: BeautifulSoup) -> Optional[str]:
    """Extract messages from Perplexity"""
    messages = soup.find_all(lambda tag:
        tag.name == "div" and (
            (tag.get('class') and any('user' in c.lower() or 'assistant' in c.lower() for c in tag.get('class', []))) or
            tag.has_attr('data-message-role')
        ))

    if messages:
        lines = []
        for msg in messages:
            role = "UNKNOWN"
            if msg.has_attr('data-message-role'):
                role = msg.get('data-message-role', 'unknown').upper()
                if role == 'user':
                    role = "USER"
                elif role == 'assistant':
                    role = "PERPLEXITY"
            elif msg.get('class'):
                classes = msg.get('class', [])
                if any('user' in c.lower() for c in classes):
                    role = "USER"
                elif any('assistant' in c.lower() for c in classes):
                    role = "PERPLEXITY"
            text = msg.get_text(separator=' ', strip=True)
            if text:
                lines.append(f"{role}: {text}")
        return "

".join(lines)
    return None


def extract_bing_dom(soup: BeautifulSoup) -> Optional[str]:
    """Extract messages from Bing Chat"""
    messages = soup.find_all(lambda tag:
        tag.name == "div" and (
            tag.has_attr('data-search-context') or
            (tag.get('class') and any('user' in c.lower() or 'bot' in c.lower() or 'answer' in c.lower() for c in tag.get('class', [])))
        ))

    if messages:
        lines = []
        for msg in messages:
            role = "UNKNOWN"
            if msg.get('class'):
                classes = msg.get('class', [])
                if any('user' in c.lower() for c in classes):
                    role = "USER"
                elif any('bot' in c.lower() or 'answer' in c.lower() for c in classes):
                    role = "BING"
            text = msg.get_text(separator=' ', strip=True)
            if text:
                lines.append(f"{role}: {text}")
        return "

".join(lines)
    return None


def extract_generic(html: str) -> str:
    """Generic text extraction for unknown platforms"""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "button", "svg", "img"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = root.get_text("
", strip=True)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"s+", " ", line).strip()
        if not line or NOISE_PATTERNS.match(line):
            continue
        lines.append(line)
    return "
".join(lines)


def extract_visible_text(html: str, source_name: str) -> str:
    """Main extraction function that routes to platform-specific extractors"""
    soup = BeautifulSoup(html, "html.parser")

    if source_name == "Claude":
        text = extract_claude_dom(soup)
        if text:
            return text
        text = extract_claude_next(html)
        if text:
            return text

    if source_name == "ChatGPT":
        text = extract_chatgpt_dom(soup)
        if text:
            return text

    if source_name == "Gemini":
        text = extract_gemini_dom(soup)
        if text:
            return text

    if source_name == "Perplexity":
        text = extract_perplexity_dom(soup)
        if text:
            return text

    if source_name == "Bing":
        text = extract_bing_dom(soup)
        if text:
            return text

    return extract_generic(html)


# ─────────────────────────────────────────────────────────────────────────────
# COMPRESSION & FILE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(source_name: str, source_url: str, chat_id: str, transcript: str) -> str:
    """Build prompt for Gemini compression"""
    return f"""
You are helping convert a public AI conversation into a portable context file for another AI assistant.

Rules:
- Output plain text only.
- Avoid first-person and second-person pronouns.
- Refer to the person as "the user".
- Preserve the user's words verbatim where possible.
- If a detail is uncertain, write "unknown".
- VERY IMPORTANT: Copy the transcript EXACTLY underneath the "Transcript:" heading. Do NOT summarize.

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
    """Create final .ctx file content"""
    return f"""CTX_VERSION: 1
SOURCE: {source_name}
CHAT_ID: {chat_id}
SOURCE_URL: {source_url}

{body}

END_CTX
"""


def choose_file_name(source_name: str, chat_id: str) -> str:
    """Generate safe filename"""
    safe_source = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_name.lower())[:20] or "source"
    safe_chat = re.sub(r"[^a-zA-Z0-9_-]+", "_", chat_id)[:20] or "chat"
    return f"{safe_source}_{safe_chat}.ctx"


def write_file(file_name: str, content: str) -> Path:
    """Write file to output directory"""
    path = OUT_DIR / file_name
    path.write_text(content, encoding="utf-8")
    return path


def gemini_compress(prompt: str) -> str:
    """Compress using Gemini API"""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing.")
    client = genai.Client(api_key=GEMINI_API_KEY)
    last_error = None
    for model in [GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK]:
        if not model:
            continue
        try:
            resp = client.models.generate_content(model=model, contents=[prompt])
            text = (getattr(resp, "text", None) or "").strip()
            if not text:
                raise RuntimeError(f"{model} returned empty text.")
            return text
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Gemini compression failed: {last_error}")


# ─────────────────────────────────────────────────────────────────────────────
# JOB PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

async def process_job(job_id: str, url: str) -> None:
    """Process a chat URL extraction job"""
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
                f"({len(transcript)} chars). Ensure the link is public."
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


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

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
