"""
The shared state passed between all agents.
Each agent reads some fields and writes others.
"""
from typing import TypedDict, List, Dict


class InterviewState(TypedDict, total=False):
    # inputs chosen by the user
    role: str                 # e.g. "software engineering"
    category: str             # "behavioral" | "technical" | "hr"

    # written by Interviewer Agent
    question: str
    asked_questions: List[str]   # avoids repeating questions in one session

    # written by Transcription Agent
    audio_path: str
    transcript: str
    audio_duration: float     # seconds, used later for words-per-minute

    # written by Delivery Analyst Agent
    metrics: Dict             # filler count, words per minute, length...

    # written by Retrieval Agent
    retrieved_context: List[str]

    # written by Coach Agent
    feedback: str

    # written by Critic Agent
    critic_verdict: str       # "approved" | "revise"
    critic_notes: str         # what the Critic wants fixed (read by the Coach)
    revision_count: int       # number of drafts the Coach has written
