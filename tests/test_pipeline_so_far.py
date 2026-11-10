from agents import transcription, delivery_analyst

state = {"audio_path": "data/audio/test.m4a"}

state.update(transcription.run(state))
print("TRANSCRIPT:", state["transcript"])
print("DURATION:", state["audio_duration"])

state.update(delivery_analyst.run(state))
print("METRICS:", state["metrics"])
