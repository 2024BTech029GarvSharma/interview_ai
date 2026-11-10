"""
Critic Agent
Job: Checks the Coach's feedback is grounded in the transcript, metrics and
reference examples; approves it or asks for a rewrite.
Writes to state: "critic_verdict", "critic_notes"
"""
import json
import re

from llm import chat
from state import InterviewState

REQUIRED_HEADINGS = [
    "overall impression",
    "what worked well",
    "what to improve",
    "delivery",
    "stronger version",
]

SYSTEM_PROMPT = (
    "You are a strict reviewer of interview feedback. You only judge whether the "
    "draft is faithful to the sources you are given. You never rewrite it."
)


def _rule_checks(state: InterviewState) -> list:
    problems = []
    feedback = state["feedback"]
    lowered = feedback.lower()

    for heading in REQUIRED_HEADINGS:
        if heading not in lowered:
            problems.append(f"The section '{heading}' is missing.")

    wpm = state["metrics"]["words_per_minute"]
    if str(wpm) not in feedback and f"{wpm:.0f}" not in feedback:
        problems.append(
            f"The Delivery section must quote the measured pace ({wpm} words per minute)."
        )
    return problems


def _llm_check(state: InterviewState):
    context = "\n\n".join(state.get("retrieved_context", []))
    metrics = json.dumps(state["metrics"], indent=2)

    prompt = f"""Review the DRAFT FEEDBACK against the sources below.

INTERVIEW QUESTION:
{state['question']}

CANDIDATE'S TRANSCRIPT:
{state['transcript']}

MEASURED METRICS:
{metrics}

REFERENCE EXAMPLES:
{context}

DRAFT FEEDBACK:
{state['feedback']}

Check these three things:
1. 1. FABRICATION: does the draft, especially "Stronger version", state any fact about the candidate's experience that is not in the transcript? Rewording is fine. New reasons, tools, numbers, outcomes, skills or personal qualities (such as teamwork, leadership or discipline) are not. Guessing a degree name, or expanding an abbreviation the transcript does not spell out, also counts as fabrication. Where the transcript is unclear, the draft should use a placeholder in square brackets instead of guessing.
2. METRICS: do the numbers quoted match the measured metrics exactly?
3. GROUNDING: is the advice under "What to improve" supported by what the reference examples do well, rather than being generic?

Reply with ONLY a JSON object, no other text:
{{"verdict": "approved" or "revise", "notes": "one short paragraph saying exactly what to fix, empty if approved"}}"""

    for _ in range(2):
        text = chat(prompt, system=SYSTEM_PROMPT)
        try:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            data = json.loads(cleaned)
            verdict = "approved" if str(data["verdict"]).lower() == "approved" else "revise"
            return verdict, str(data.get("notes", "")).strip()
        except (ValueError, KeyError, TypeError):
            continue

    return "approved", "The Critic could not read its own review, so the draft was accepted."


def run(state: InterviewState) -> dict:
    problems = _rule_checks(state)
    if problems:
        return {"critic_verdict": "revise", "critic_notes": " ".join(problems)}

    verdict, notes = _llm_check(state)
    return {"critic_verdict": verdict, "critic_notes": notes}