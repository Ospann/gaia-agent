"""Run the agent on the Unit 4 questions and submit answers to the course scoring API.

Usage:
    python run.py run              # answer all questions (answers are cached in answers.json)
    python run.py run --limit 1    # answer just one new question (for debugging)
    python run.py run --force      # re-answer everything, ignoring the cache
    python run.py run --only <task_id> [<task_id> ...]   # re-answer specific questions
    python run.py show             # print cached answers
    python run.py submit           # send cached answers and print the score
"""
import argparse
import json
import os
import shutil

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://agents-course-unit4-scoring.hf.space"
CACHE_PATH = "answers.json"
FILES_DIR = "files"
FATAL_ERRORS = ("credit balance is too low", "invalid x-api-key", "authentication_error")
GAIA_REPO = "gaia-benchmark/GAIA"
GAIA_SPLITS = ("2023/validation", "2023/test")


def fetch_questions() -> list[dict]:
    resp = requests.get(f"{API_URL}/questions", timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_file(task_id: str, file_name: str | None) -> str | None:
    if not file_name:
        return None
    os.makedirs(FILES_DIR, exist_ok=True)
    path = os.path.join(FILES_DIR, file_name)
    if os.path.exists(path):
        return path
    resp = requests.get(f"{API_URL}/files/{task_id}", timeout=60)
    if resp.status_code == 200:
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    print(f"  ! course API has no file for {task_id} (HTTP {resp.status_code}), trying the GAIA dataset")
    return download_from_gaia(file_name)


def download_from_gaia(file_name: str) -> str | None:
    token = os.getenv("HF_TOKEN")
    if not token:
        print("  ! HF_TOKEN is not set, cannot download from the GAIA dataset")
        return None
    from huggingface_hub import hf_hub_download

    for split in GAIA_SPLITS:
        try:
            cached = hf_hub_download(GAIA_REPO, f"{split}/{file_name}", repo_type="dataset", token=token)
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue
        path = os.path.join(FILES_DIR, file_name)
        shutil.copyfile(cached, path)
        return path
    print(f"  ! could not download {file_name} from the GAIA dataset: {type(last_error).__name__}")
    return None


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def cmd_run(args) -> None:
    from agent import answer_question

    questions = fetch_questions()
    cache = load_cache()
    print(f"Got {len(questions)} questions, {len(cache)} already answered.\n")

    processed = 0
    for i, q in enumerate(questions, 1):
        task_id = q["task_id"]
        if args.only and task_id not in args.only:
            continue
        if task_id in cache and not args.force and not args.only:
            continue

        print(f"=== [{i}/{len(questions)}] {task_id}")
        print(q["question"][:300])
        file_path = download_file(task_id, q.get("file_name"))
        try:
            missing = q.get("file_name") if q.get("file_name") and not file_path else None
            answer = answer_question(q["question"], file_path, missing)
        except Exception as e:  # noqa: BLE001
            if any(marker in str(e).lower() for marker in FATAL_ERRORS):
                raise SystemExit(f"  ! stopping: the LLM provider rejected the request ({e})")
            print(f"  ! agent error, not caching (will retry on next run): {e}\n")
            continue
        print(f"  -> ANSWER: {answer!r}\n")

        cache[task_id] = {"question": q["question"], "file": q.get("file_name") or None, "answer": answer}
        save_cache(cache)

        processed += 1
        if args.limit and processed >= args.limit:
            break

    print(f"Done. Answers in cache: {len(cache)}/{len(questions)}. Next: python run.py submit")


def cmd_show(_args) -> None:
    for task_id, item in load_cache().items():
        print(f"{task_id}\n  Q: {item['question'][:150]}\n  A: {item['answer']!r}\n")


def cmd_submit(_args) -> None:
    username = os.getenv("HF_USERNAME")
    agent_code = os.getenv("AGENT_CODE_URL")
    if not username or not agent_code:
        raise SystemExit("Set HF_USERNAME and AGENT_CODE_URL in .env")

    from agent import clean_answer

    cache = load_cache()
    answers = [{"task_id": t, "submitted_answer": clean_answer(v["answer"])} for t, v in cache.items()]
    payload = {"username": username, "agent_code": agent_code, "answers": answers}

    print(f"Submitting {len(answers)} answers as {username} ...")
    resp = requests.post(f"{API_URL}/submit", json=payload, timeout=120)
    try:
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(resp.status_code, resp.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="HF Agents Course - Unit 4 runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="answer questions")
    p_run.add_argument("--limit", type=int, default=0, help="stop after N new questions")
    p_run.add_argument("--force", action="store_true", help="ignore cache, re-answer all")
    p_run.add_argument("--only", nargs="+", help="re-answer only these task ids")
    p_run.set_defaults(func=cmd_run)

    sub.add_parser("show", help="print cached answers").set_defaults(func=cmd_show)
    sub.add_parser("submit", help="submit cached answers").set_defaults(func=cmd_submit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
