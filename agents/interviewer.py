"""
Interviewer Agent
Job: Chooses the interview question. Classifies a custom question, or
generates a fresh one for the chosen role and category.
Writes to state: "question", "category", "asked_questions"
"""
from llm import chat
from state import InterviewState

CATEGORIES = ("behavioral", "technical", "hr")

SYSTEM_PROMPT = "You are an experienced interviewer. Follow the output format exactly."

HINTS = {
    "behavioral": "Ask about a specific past experience (teamwork, conflict, failure, leadership, pressure).",
    "technical": (
        "Ask a domain question the candidate can answer aloud in one to two minutes "
        "(explaining a concept or a short problem). For business consulting, ask a "
        "short case or estimation question."
    ),
    "hr": "Ask a general question that appears in almost every interview (introduction, strengths, motivation, goals).",
}


def _classify(question: str) -> str:
    prompt = f"""Classify this interview question into exactly one category.

behavioral: asks about a past experience or situation (teamwork, conflict, failure, leadership)
technical: tests knowledge or problem-solving skills for a role
hr: general questions asked in every interview (introduction, strengths, goals, motivation)

Examples:
"Tell me about a time you missed a deadline." -> behavioral
"What is the difference between a process and a thread?" -> technical
"Where do you see yourself in five years?" -> hr

Question: "{question}"

Reply with ONLY one word: behavioral, technical or hr."""
    answer = chat(prompt, system=SYSTEM_PROMPT).lower()
    for category in CATEGORIES:
        if category in answer:
            return category
    return "hr"


def _generate(role: str, category: str, asked: list) -> str:
    avoid = "\n".join(f"- {q}" for q in asked) or "(none yet)"
    prompt = f"""Ask ONE interview question for a {role} candidate.
Category: {category}. {HINTS[category]}

Do not repeat or closely resemble these earlier questions:
{avoid}

Reply with ONLY the question text, nothing else."""
    return chat(prompt, system=SYSTEM_PROMPT).strip().strip('"')


def run(state: InterviewState) -> dict:
    asked = list(state.get("asked_questions", []))
    question = state.get("question", "").strip()

    if question:  # the user typed their own question
        category = state.get("category") or _classify(question)
    else:  # generate a fresh one
        category = state["category"]
        question = _generate(state.get("role", "general"), category, asked)

    asked.append(question)
    return {"question": question, "category": category, "asked_questions": asked}