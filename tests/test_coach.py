from agents import retriever, coach

state = {
    "role": "software engineering",
    "category": "behavioral",
    "question": "Tell me about a time you disagreed with a teammate.",
    "transcript": (
        "So, um, there was this one time in my project where me and my teammate "
        "disagreed about which database to use. I wanted MongoDB and he wanted "
        "MySQL. We argued a lot, but finally we went with MySQL. It worked out fine."
    ),
    "audio_duration": 18.0,
    "metrics": {
        "word_count": 42,
        "duration_seconds": 18.0,
        "words_per_minute": 140.0,
        "pace": "good",
        "filler_total": 1,
        "filler_per_100_words": 2.4,
        "filler_breakdown": {"um": 1},
        "avg_sentence_length": 10.5,
    },
}

state.update(retriever.run(state))
state.update(coach.run(state))
print(state["feedback"])
print("\nDrafts written:", state["revision_count"])