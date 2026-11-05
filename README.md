# HF Agents Course — Final Assignment (Unit 4)

GAIA agent built with [smolagents](https://github.com/huggingface/smolagents) for the
[Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit4/hands-on).

Tools: web search (DuckDuckGo), web page reader, Wikipedia, attached-file reader
(xlsx/csv/pdf/docx/text), audio transcription (Whisper), YouTube transcripts, Python execution.
Images are passed to the model directly (needs a vision-capable model).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your key, username, repo link
```

## Usage

```bash
python run.py run --limit 1   # test on one question
python run.py run             # answer all 20 (cached in answers.json)
python run.py show            # review answers
python run.py run --only <task_id>   # re-try a specific question
python run.py submit          # submit and get the score
```
