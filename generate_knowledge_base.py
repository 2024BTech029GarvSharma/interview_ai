"""
generate_knowledge_base.py
Generates the RAG knowledge base (sample interview questions + strong answers)
using the LLM. Run once from the project root:

    python generate_knowledge_base.py

Afterwards, open each JSON file in data/knowledge_base/ and review/edit by hand.
"""
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
MODEL = os.getenv("LLM_MODEL")

OUT_DIR = Path("data/knowledge_base")
ENTRIES_PER_FILE = 10

# (role, category) -> one JSON file each
COMBOS = [
    ("software engineering", "behavioral"),
    ("software engineering", "technical"),
    ("business consulting", "behavioral"),
    ("business consulting", "technical"),
    ("general", "hr"),
]

CATEGORY_HINTS = {
    "behavioral": (
        "Questions ask about past experiences (teamwork, conflict, failure, "
        "leadership, tight deadlines). Answers must follow the STAR method "
        "(Situation, Task, Action, Result) and end with a concrete result."
    ),
    "technical": (
        "Questions test domain knowledge for the role. For software engineering, "
        "cover core concepts (data structures, databases, OOP, APIs, debugging). "
        "For business consulting, use case-style and problem-solving questions "
        "(market sizing, profitability, prioritisation). Answers must be clear, "
        "structured, and explain the reasoning step by step."
    ),
    "hr": (
        "General questions asked in almost every interview (tell me about yourself, "
        "strengths and weaknesses, why this company, career goals, handling "
        "pressure). Answers must be honest, concise and well structured."
    ),
}


def build_prompt(role: str, category: str) -> str:
    return f"""You are an experienced interview coach. Create {ENTRIES_PER_FILE} different interview questions for the role "{role}", category "{category}", each with a strong model answer.

{CATEGORY_HINTS[category]}

Rules:
- Every question must be different and cover a different topic.
- strong_answer: 90 to 150 words, first person, natural spoken English (it will be spoken aloud), no bullet points.
- why_it_works: 1 to 2 sentences explaining what makes the answer strong.
- Use realistic but generic details. Do not name real companies or real people.

Return ONLY a JSON array, with no text before or after it. Each item must have exactly these keys: "question", "strong_answer", "why_it_works"."""


def parse_json(text: str):
    """LLMs often wrap JSON in ```json fences; strip them before parsing."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def validate(entries) -> None:
    if not isinstance(entries, list) or not entries:
        raise ValueError("response is not a non-empty JSON list")
    for item in entries:
        for key in ("question", "strong_answer", "why_it_works"):
            if key not in item or not str(item[key]).strip():
                raise ValueError(f"an entry is missing '{key}'")


def generate_entries(role: str, category: str, attempts: int = 4):
    """Call the LLM; retry with increasing waits if the call or parsing fails."""
    for attempt in range(1, attempts + 1):
        try:
            reply = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": build_prompt(role, category)}],
            )
            entries = parse_json(reply.choices[0].message.content)
            validate(entries)
            return entries
        except Exception as error:
            wait = 5 * attempt
            print(f"  attempt {attempt} failed: {error}")
            if attempt < attempts:
                print(f"  retrying in {wait}s...")
                time.sleep(wait)
    raise RuntimeError(f"giving up on {role} / {category} after {attempts} attempts")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for role, category in COMBOS:
        filename = f"{role.replace(' ', '_')}_{category}.json"
        path = OUT_DIR / filename

        # Never overwrite a file: protects entries you have already edited by hand.
        if path.exists():
            print(f"skipped {filename} (already exists)")
            continue

        print(f"generating {role} / {category} ...")
        entries = generate_entries(role, category)

        records = []
        for i, item in enumerate(entries, start=1):
            records.append(
                {
                    "id": f"{role.replace(' ', '_')}_{category}_{i:02d}",
                    "role": role,
                    "category": category,
                    "question": item["question"].strip(),
                    "strong_answer": item["strong_answer"].strip(),
                    "why_it_works": item["why_it_works"].strip(),
                }
            )

        path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved   {filename} ({len(records)} entries)")
        time.sleep(2)  # be gentle with the free-tier rate limit

    print("\nDone. Now review and edit the files in data/knowledge_base/")


if __name__ == "__main__":
    main()
