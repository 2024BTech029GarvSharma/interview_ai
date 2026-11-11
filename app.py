"""
Streamlit UI for InterVue AI.
Run with:  streamlit run app.py
"""
import time
from pathlib import Path

import streamlit as st

from graph import build_graph

st.set_page_config(page_title="InterVue AI", page_icon="🎤", layout="centered")


@st.cache_resource
def get_app():
    return build_graph()


app = get_app()

for key, default in [("asked", []), ("question", ""), ("category", ""), ("result", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("InterVue AI")
st.caption("Practice a mock interview and get coached by a team of AI agents.")

with st.sidebar:
    role = st.selectbox("Role", ["software engineering", "business consulting"])
    category = st.selectbox("Question type", ["behavioral", "technical", "hr"])
    st.divider()
    custom = st.text_input("...or type your own question")

col1, col2 = st.columns(2)

if col1.button("Get a question", use_container_width=True):
    with st.spinner("Interviewer Agent is choosing a question..."):
        out = app.invoke(
            {"role": role, "category": category, "asked_questions": st.session_state.asked}
        )
    st.session_state.question = out["question"]
    st.session_state.category = out["category"]
    st.session_state.asked = out["asked_questions"]
    st.session_state.result = None

if col2.button("Use my question", use_container_width=True, disabled=not custom.strip()):
    with st.spinner("Interviewer Agent is classifying your question..."):
        out = app.invoke(
            {"role": role, "question": custom.strip(), "asked_questions": st.session_state.asked}
        )
    st.session_state.question = out["question"]
    st.session_state.category = out["category"]
    st.session_state.asked = out["asked_questions"]
    st.session_state.result = None

if st.session_state.question:
    st.subheader("Question")
    st.info(st.session_state.question)
    st.caption(f"Category: {st.session_state.category}")

    audio = st.audio_input("Record your answer", key=f"audio_{len(st.session_state.asked)}")

    if st.button("Get feedback", type="primary", disabled=audio is None):
        audio_path = Path("data/audio") / f"answer_{int(time.time())}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(audio.getvalue())

        with st.spinner("Transcribing, analysing, retrieving examples and writing feedback (about a minute)..."):
            try:
                st.session_state.result = app.invoke(
                    {
                        "role": role,
                        "category": st.session_state.category,
                        "question": st.session_state.question,
                        "asked_questions": st.session_state.asked,
                        "audio_path": str(audio_path),
                    }
                )
            except Exception as error:
                st.session_state.result = None
                st.error(f"Something went wrong: {error}")

result = st.session_state.result
if result:
    st.divider()
    st.subheader("Your transcript")
    st.write(result["transcript"])

    m = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pace (wpm)", m["words_per_minute"], m["pace"], delta_color="off")
    c2.metric("Fillers", m["filler_total"], f"{m['filler_per_100_words']} per 100 words", delta_color="off")
    c3.metric("Words", m["word_count"], f"{m['duration_seconds']} s of speech", delta_color="off")
    c4.metric("Avg sentence", m["avg_sentence_length"])
    if m["filler_breakdown"]:
        st.caption("Fillers heard: " + ", ".join(f"{w} x{n}" for w, n in m["filler_breakdown"].items()))

    st.subheader("Coach feedback")
    st.markdown(result["feedback"])

    drafts = result["revision_count"]
    if result["critic_verdict"] == "approved":
        st.success(f"Reviewed and approved by the Critic Agent after {drafts} draft(s).")
    else:
        st.warning(f"The Critic Agent still had concerns after {drafts} drafts: {result['critic_notes']}")

    with st.expander("Examples the Retrieval Agent used"):
        for item in result["retrieved_context"]:
            st.text(item)