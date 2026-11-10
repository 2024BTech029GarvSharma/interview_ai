"""
Shared helper for all LLM calls.
Retries on temporary errors (503, 429) and falls back to a second model.
"""
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))

MODELS = [m for m in (os.getenv("LLM_MODEL"), os.getenv("LLM_FALLBACK_MODEL")) if m]


def chat(prompt: str, system: str = "", attempts_per_model: int = 3) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    for model in MODELS:
        for attempt in range(1, attempts_per_model + 1):
            try:
                reply = _client.chat.completions.create(model=model, messages=messages)
                return reply.choices[0].message.content.strip()
            except Exception as error:
                last_error = error
                print(f"[llm] {model} attempt {attempt} failed: {error}")
                if attempt < attempts_per_model:
                    time.sleep(5 * attempt)
    raise RuntimeError(f"All LLM attempts failed. Last error: {last_error}")