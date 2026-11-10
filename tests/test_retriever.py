from agents import retriever

state = {
    "role": "software engineering",
    "category": "technical",
    "question": "Explain how a hash map works.",
}

for item in retriever.run(state)["retrieved_context"]:
    print(item)
    print("-----")