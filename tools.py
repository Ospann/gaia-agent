"""Custom tools for the GAIA agent: reading attached files, audio transcription, YouTube transcripts."""
import os
import re

from smolagents import tool

MAX_CHARS = 40_000


def _truncate(text: str) -> str:
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS] + f"\n\n[... truncated, total {len(text)} chars]"
    return text


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
