from agents import retriever, coach, critic
from tests.sample_data import SAMPLE_STATE

BAD_FEEDBACK = """1. Overall impression
Good answer.
2. What worked well
- Clear story.
3. What to improve
- Add more detail.
4. Delivery
Your pace was 140.0 words per minute, with 1 filler word.
5. Stronger version
In my team project I disagreed with a teammate about the database. I built a benchmark that showed MySQL was 40% faster, so we chose it and the app launched two weeks early."""

# Case A: a real draft written by the Coach
state = dict(SAMPLE_STATE)
state.update(retriever.run(state))
state.update(coach.run(state))
state.update(critic.run(state))
print("CASE A (real Coach draft)")
print("verdict:", state["critic_verdict"])
print("notes:", state["critic_notes"])

# Case B: a deliberately bad draft with invented facts
bad = dict(SAMPLE_STATE)
bad["retrieved_context"] = state["retrieved_context"]
bad["feedback"] = BAD_FEEDBACK
bad.update(critic.run(bad))
print("\nCASE B (draft with fabricated facts)")
print("verdict:", bad["critic_verdict"])
print("notes:", bad["critic_notes"])