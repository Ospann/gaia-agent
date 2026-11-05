"""GAIA agent built with smolagents. The model is configured via MODEL_ID in .env (any LiteLLM model id)."""
import os
import re

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    VisitWebpageTool,
    WikipediaSearchTool,
)

from tools import read_file, transcribe_audio, youtube_transcript

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

INSTRUCTIONS = """You are a precise research assistant solving a question from the GAIA benchmark.

How to work:
- Think step by step and make a short plan first.
- Use web search and visit the actual pages to verify facts. Prefer primary/authoritative sources (Wikipedia, official sites).
- If a question refers to a specific Wikipedia version or date, take that into account.
- For attached files use read_file (spreadsheets, pdf, text, code) or transcribe_audio (audio). Images are passed to you directly.
- Do arithmetic, counting, sorting and data analysis in Python code, never in your head.
- Read the question very carefully: some questions are tricky (reversed text, logic puzzles, "except", "first name only", etc.).
- Before answering, double-check that the answer matches exactly what was asked.

Answer format (the answer is checked by EXACT MATCH, so this is critical):
- Call final_answer() with ONLY the answer itself. No explanation, no full sentence, no "FINAL ANSWER" prefix.
- A number: write digits only, without thousands separators, and without units or symbols ($, %) unless the question asks for them.
- A string: as few words as possible, no articles, no abbreviations (e.g. write city names in full) unless asked.
- A list: comma-separated, with a single space after each comma, e.g. "apple, banana, cherry". Apply the rules above to each element.
- If the question specifies a format (order, rounding, codes, units, capitalization), follow it exactly.
"""


def build_model():
    model_id = os.getenv("MODEL_ID", "gpt-4.1")
    return LiteLLMModel(model_id=model_id)


def build_agent() -> CodeAgent:
    return CodeAgent(
        tools=[
            DuckDuckGoSearchTool(max_results=8),
            VisitWebpageTool(),
            WikipediaSearchTool(),
            read_file,
            transcribe_audio,
            youtube_transcript,
        ],
        model=build_model(),
        additional_authorized_imports=[
            "pandas", "numpy", "math", "statistics", "datetime", "re", "json",
            "collections", "itertools", "fractions", "decimal", "unicodedata", "csv",
        ],
        max_steps=int(os.getenv("MAX_STEPS", "15")),
        verbosity_level=1,
    )


def clean_answer(answer) -> str:
    """Normalize the model output for exact-match scoring."""
    s = str(answer).strip()
    s = re.sub(r"^\s*(final answer|answer)\s*[:\-]\s*", "", s, flags=re.IGNORECASE)
    s = s.strip().strip('"').strip("'").strip()
    # drop a single trailing period ("Paris." -> "Paris"), but keep things like "U.S."
    if s.endswith(".") and s.count(".") == 1:
        s = s[:-1].strip()
    return s


def answer_question(question: str, file_path: str | None = None) -> str:
    agent = build_agent()  # a fresh agent per question => clean memory
    task = f"{INSTRUCTIONS}\n\nQuestion:\n{question}"
    images = None

    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        task += f"\n\nA file is attached to this question. It is saved locally at: {file_path}"
        if ext in IMAGE_EXTS:
            from PIL import Image

            images = [Image.open(file_path).convert("RGB")]
            task += "\nThe image is also attached to this message, look at it carefully."
        elif ext in AUDIO_EXTS:
            task += "\nIt is an audio file: use transcribe_audio on it."
        else:
            task += "\nUse read_file on it (or open it with pandas in code)."

    result = agent.run(task, images=images) if images else agent.run(task)
    return clean_answer(result)
