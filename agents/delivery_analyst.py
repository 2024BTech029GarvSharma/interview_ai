"""
Delivery Analyst Agent
Job: Measures filler words, pacing (words per minute) and sentence length.
Writes to state: "metrics"
"""
from collections import Counter

import spacy
from spacy.matcher import PhraseMatcher

from state import InterviewState

nlp = spacy.load("en_core_web_sm")

HESITATIONS = {"um", "umm", "uh", "uhh", "er", "erm", "hmm", "ah"}
WORD_FILLERS = {"basically", "actually", "literally", "honestly"}
PHRASE_FILLERS = ["you know", "i mean", "kind of", "sort of"]

IDEAL_WPM_MIN = 120
IDEAL_WPM_MAX = 160

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("FILLER", [nlp.make_doc(p) for p in PHRASE_FILLERS])


def _count_fillers(doc) -> Counter:
    counts = Counter()
    in_phrase = set()

    for _, start, end in matcher(doc):
        counts[doc[start:end].text.lower()] += 1
        in_phrase.update(range(start, end))

    for token in doc:
        if token.i in in_phrase:
            continue
        word = token.lower_
        if word in HESITATIONS or word in WORD_FILLERS:
            counts[word] += 1
        elif word == "like" and token.pos_ != "VERB":
            counts[word] += 1

    return counts


def _pace_label(wpm: float) -> str:
    if wpm < IDEAL_WPM_MIN:
        return "too slow"
    if wpm > IDEAL_WPM_MAX:
        return "too fast"
    return "good"


def run(state: InterviewState) -> dict:
    doc = nlp(state["transcript"])
    duration = state.get("audio_duration", 0.0)

    words = [t for t in doc if t.is_alpha]
    word_count = len(words)

    wpm = word_count / (duration / 60) if duration > 0 else 0.0

    fillers = _count_fillers(doc)
    filler_total = sum(fillers.values())
    filler_rate = (filler_total / word_count * 100) if word_count else 0.0

    sentences = list(doc.sents)
    avg_sentence_len = word_count / len(sentences) if sentences else 0.0

    return {
        "metrics": {
            "word_count": word_count,
            "duration_seconds": round(duration, 1),
            "words_per_minute": round(wpm, 1),
            "pace": _pace_label(wpm),
            "filler_total": filler_total,
            "filler_per_100_words": round(filler_rate, 1),
            "filler_breakdown": dict(fillers),
            "avg_sentence_length": round(avg_sentence_len, 1),
        }
    }