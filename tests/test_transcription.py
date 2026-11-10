from agents import transcription

state = {"audio_path": "data/audio/test.m4a"}
print(transcription.run(state))