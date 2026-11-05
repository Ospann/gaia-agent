"""Custom tools for the GAIA agent: reading attached files, audio transcription, YouTube transcripts."""
import os
import re
import time

from smolagents import tool

MAX_CHARS = 40_000
PAGE_MAX_CHARS = 150_000
NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg"]
NOISE_SELECTORS = [".mw-editsection", ".navbox", ".vertical-navbox", "#toc", ".toc"]
USER_AGENT = "gaia-agent/1.0 (HF Agents Course Unit 4; research bot)"


def _truncate(text: str, limit: int = MAX_CHARS) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n\n[... truncated, total {len(text)} chars]"
    return text


def _main_html(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    main = soup.select_one("#mw-content-text") or soup.find("main") or soup.find("article") or soup.body or soup
    title = soup.title.get_text(strip=True) if soup.title else ""
    return f"<h1>{title}</h1>{main}"


@tool
def read_file(path: str) -> str:
    """Read a local file attached to the question and return its content as text.
    Supports spreadsheets (.xlsx, .xls, .csv - every sheet is returned as a table),
    PDF (.pdf), Word (.docx) and any text-based file (.txt, .py, .json, .md, .html ...).
    For images use your vision ability, for audio use transcribe_audio.

    Args:
        path: Local path to the file.
    """
    if not os.path.exists(path):
        return f"File not found: {path}"
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".xlsx", ".xls"):
            import pandas as pd

            sheets = pd.read_excel(path, sheet_name=None)
            parts = []
            for name, df in sheets.items():
                parts.append(f"### Sheet: {name} (rows={len(df)}, cols={len(df.columns)})\n{df.to_string()}")
            return _truncate("\n\n".join(parts))
        if ext == ".csv":
            import pandas as pd

            df = pd.read_csv(path)
            return _truncate(f"rows={len(df)}, cols={len(df.columns)}\n{df.to_string()}")
        if ext == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(path)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            return _truncate(text)
        if ext == ".docx":
            import docx

            document = docx.Document(path)
            return _truncate("\n".join(p.text for p in document.paragraphs))
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return _truncate(f.read())
    except Exception as e:  # noqa: BLE001
        return f"Error reading {path}: {e}"


_whisper_model = None


@tool
def transcribe_audio(path: str) -> str:
    """Transcribe a local audio file (.mp3, .wav, .m4a, ...) to text using Whisper.

    Args:
        path: Local path to the audio file.
    """
    global _whisper_model
    if not os.path.exists(path):
        return f"File not found: {path}"
    try:
        from faster_whisper import WhisperModel

        if _whisper_model is None:
            size = os.getenv("WHISPER_MODEL", "small")
            _whisper_model = WhisperModel(size, device="cpu", compute_type="int8")
        segments, _ = _whisper_model.transcribe(path, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments)
    except Exception as e:  # noqa: BLE001
        return f"Error transcribing {path}: {e}"


@tool
def youtube_transcript(url: str) -> str:
    """Get the spoken-text transcript (subtitles) of a YouTube video.
    Note: it only contains what is said, not what is shown on screen.

    Args:
        url: Full YouTube URL or video id.
    """
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    video_id = match.group(1) if match else url.strip()
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        try:  # youtube-transcript-api >= 1.0
            fetched = YouTubeTranscriptApi().fetch(video_id)
            parts = [s.text for s in fetched]
        except AttributeError:  # older versions
            data = YouTubeTranscriptApi.get_transcript(video_id)
            parts = [d["text"] for d in data]
        return _truncate(" ".join(parts))
    except Exception as e:  # noqa: BLE001
        return f"Could not get transcript for {video_id}: {e}"


_ddgs = None


@tool
def web_search(query: str) -> str:
    """Search the web (DuckDuckGo) and return the top results with titles, links and snippets.
    If nothing is found, try a shorter or less specific query.

    Args:
        query: The search query.
    """
    global _ddgs
    try:
        from ddgs import DDGS

        if _ddgs is None:
            _ddgs = DDGS()
        results = []
        for attempt in range(2):
            try:
                results = [r for r in _ddgs.text(query, max_results=8) if r.get("href")]
            except Exception:  # noqa: BLE001
                results = []
            if results:
                break
            time.sleep(1 + attempt)
        if not results:
            return f"No results found for '{query}'. Try a shorter or less specific query."
        return "## Search Results\n\n" + "\n\n".join(
            f"[{r.get('title', '')}]({r['href']})\n{r.get('body', '')}" for r in results
        )
    except Exception as e:  # noqa: BLE001
        return f"Search failed for '{query}': {e}"


@tool
def visit_webpage(url: str) -> str:
    """Visit a webpage at the given url and read its main content as markdown (navigation and menus are removed).
    Works for Wikipedia pages too. Long pages are returned almost in full: store the result in a variable
    and search/slice it in code instead of printing everything.

    Args:
        url: The url of the webpage to visit.
    """
    try:
        import requests
        from markdownify import markdownify

        response = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" in content_type or response.content[:5] == b"%PDF-":
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(response.content))
            content = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
        elif "html" in content_type or not content_type:
            content = markdownify(_main_html(response.text)).strip()
        else:
            content = response.text.strip()
        return _truncate(re.sub(r"\n{3,}", "\n\n", content), PAGE_MAX_CHARS)
    except Exception as e:  # noqa: BLE001
        return f"Error fetching the webpage: {e}"
