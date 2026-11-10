"""
The Orchestrator: connects all agents into a LangGraph workflow.

Two ways to run it:
  - no audio_path in the input -> the Interviewer picks a question, then stop
  - audio_path in the input    -> transcribe, analyse, retrieve, coach, critique
"""
from langgraph.graph import StateGraph, START, END

from agents import coach, critic, delivery_analyst, interviewer, retriever, transcription
from state import InterviewState

MAX_DRAFTS = 3


def entry_router(state: InterviewState) -> str:
    return "evaluate" if state.get("audio_path") else "ask"


def after_critic(state: InterviewState) -> str:
    if state["critic_verdict"] == "revise" and state["revision_count"] < MAX_DRAFTS:
        return "rewrite"
    return "finish"


def build_graph():
    g = StateGraph(InterviewState)

    g.add_node("interviewer", interviewer.run)
    g.add_node("transcription", transcription.run)
    g.add_node("delivery_analyst", delivery_analyst.run)
    g.add_node("retriever", retriever.run)
    g.add_node("coach", coach.run)
    g.add_node("critic", critic.run)

    g.add_conditional_edges(
        START, entry_router, {"ask": "interviewer", "evaluate": "transcription"}
    )
    g.add_edge("interviewer", END)

    g.add_edge("transcription", "delivery_analyst")
    g.add_edge("transcription", "retriever")
    g.add_edge(["delivery_analyst", "retriever"], "coach")

    g.add_edge("coach", "critic")
    g.add_conditional_edges(
        "critic", after_critic, {"rewrite": "coach", "finish": END}
    )

    return g.compile()