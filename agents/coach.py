"""
Coach Agent
Job: Writes feedback using the transcript, metrics and retrieved context.
Writes to state: "feedback", "revision_count"
"""
import json

from llm import chat
from state import InterviewState

SYSTEM_PROMPT = (
    "You are a supportive but honest interview coach. Use only the information "
    "you are given. The reference examples show what a strong answer looks like: "
    "learn from them, but never copy them word for word. The candidate's answer is "
    "an automatic transcript, so do not criticise odd words that look like "
    "transcription mistakes. Be specific, constructive and concise. Never invent "
    "details about the candidate's experience. If the answer is already strong, "
    "say so plainly and suggest only small refinements; never manufacture problems."
)


def _build_prompt(state: InterviewState) -> str:
    context = "\n\n".join(state.get("retrieved_context", []))
    metrics = json.dumps(state["metrics"], indent=2)

    prompt = f"""INTERVIEW QUESTION ({state['role']}, {state['category']}):
{state['question']}

CANDIDATE'S ANSWER (automatic transcript):
{state['transcript']}

DELIVERY METRICS (measured by software, accurate):
{metrics}

REFERENCE EXAMPLES OF STRONG ANSWERS TO SIMILAR QUESTIONS:
{context}

Write feedback with exactly these five sections. Write each section title as a markdown heading on its own line (for example "### Overall impression"), without numbers, followed by the content:
1. Overall impression (2 sentences)
2. What worked well (2-3 points)
3. What to improve (2-3 points on content and structure, based on what the reference examples do well)
4. Delivery (comment on pace, filler words and sentence length, quoting the numbers above)
5. Stronger version (rewrite the answer in 80-120 words using only facts from the transcript. Where a strong answer needs a detail the candidate did not give, write a placeholder in square brackets, such as [your specific result], and never invent a fact)"""

    notes = state.get("critic_notes")
    if notes and state.get("feedback"):
        prompt += f"""

A reviewer rejected your previous draft for this reason:
{notes}

PREVIOUS DRAFT:
{state['feedback']}

Write a corrected version that fixes these problems. The candidate will never see the reviewer or the previous draft, so do not mention them, the corrections, or invented details anywhere in your feedback. Write as if this were your first and only draft."""
    return prompt


def run(state: InterviewState) -> dict:
    feedback = chat(_build_prompt(state), system=SYSTEM_PROMPT)
    return {
        "feedback": feedback,
        "revision_count": state.get("revision_count", 0) + 1,
    }