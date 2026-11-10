"""
Transcription Agent
Job: Converts the recorded audio into text using Whisper.
Writes to state: "transcript", "audio_duration"
"""
import whisper
from state import InterviewState

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("small")
    return _model


def run(state: InterviewState) -> dict:
    result = _get_model().transcribe(
        state["audio_path"],
        language="en",
        fp16=False,
        initial_prompt="Umm, let me think, like, hmm... okay, so, uh, you know, I mean, basically.",
    )
    segments = result["segments"]
    duration = (segments[-1]["end"] - segments[0]["start"]) if segments else 0.0
    return {
        "transcript": result["text"].strip(),
        "audio_duration": duration,
    }