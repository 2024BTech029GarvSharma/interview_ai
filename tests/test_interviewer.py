from agents import interviewer

# Mode 1: generate three fresh questions in a row
state = {"role": "software engineering", "category": "behavioral"}
for _ in range(3):
    state.update(interviewer.run(state))
    print("Q:", state["question"])
    state.pop("question")  # the UI clears the question before the next turn

# Mode 2: classify custom questions
custom_questions = [
    "Explain how a hash map handles collisions.",
    "Tell me about a time you led a team under pressure.",
    "Why do you want to work here?",
]
print()
for q in custom_questions:
    print(interviewer.run({"question": q})["category"], "<-", q)