from agents import delivery_analyst

state = {
    "transcript": (
        "So, um, I am a third year student, like, really interested in AI. "
        "I like building projects, you know, and basically I worked on a portal. "
        "Uh, I mean, it was actually a great experience."
    ),
    "audio_duration": 15.0,
}
print(delivery_analyst.run(state))