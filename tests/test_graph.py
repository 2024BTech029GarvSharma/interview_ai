from pathlib import Path

from graph import build_graph

app = build_graph()

# Stage 1: the Interviewer asks a question
state = app.invoke({"role": "software engineering", "category": "behavioral"})
print("QUESTION:", state["question"])

# Stage 2: you answer it
input("\nRecord your answer in Voice Recorder, save it as data/audio/answer.m4a, then press Enter...")
state["audio_path"] = "data/audio/answer.m4a"
result = app.invoke(state)

print("\nVerdict:", result["critic_verdict"])
print("Drafts written:", result["revision_count"])
print("Metrics:", result["metrics"])

Path("data/last_feedback.txt").write_text(
    f"QUESTION:\n{result['question']}\n\n"
    f"TRANSCRIPT:\n{result['transcript']}\n\n"
    f"CRITIC NOTES:\n{result['critic_notes']}\n\n"
    f"FEEDBACK:\n{result['feedback']}\n",
    encoding="utf-8",
)
print("\nFull result saved to data/last_feedback.txt, open it in VS Code to read it.")