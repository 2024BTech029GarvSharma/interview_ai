"""
Retrieval Agent
Job: Finds best-practice answers from the knowledge base (RAG).
Writes to state: "retrieved_context"
"""
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from state import InterviewState

KB_DIR = Path("data/knowledge_base")
INDEX_DIR = Path("data/index")
INDEX_FILE = INDEX_DIR / "kb.faiss"
META_FILE = INDEX_DIR / "kb_meta.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

_embedder = None
_index = None
_entries = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _load_entries():
    entries = []
    for path in sorted(KB_DIR.glob("*.json")):
        entries.extend(json.loads(path.read_text(encoding="utf-8")))
    if not entries:
        raise FileNotFoundError("No knowledge base files found in data/knowledge_base/")
    return entries


def _index_is_stale() -> bool:
    if not (INDEX_FILE.exists() and META_FILE.exists()):
        return True
    newest_kb = max((p.stat().st_mtime for p in KB_DIR.glob("*.json")), default=0)
    return newest_kb > INDEX_FILE.stat().st_mtime


def _build_index():
    entries = _load_entries()
    questions = [e["question"] for e in entries]
    vectors = _get_embedder().encode(questions, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return index, entries


def _ensure_ready():
    global _index, _entries
    if _index is not None:
        return
    if _index_is_stale():
        _index, _entries = _build_index()
    else:
        _index = faiss.read_index(str(INDEX_FILE))
        _entries = json.loads(META_FILE.read_text(encoding="utf-8"))


def run(state: InterviewState) -> dict:
    _ensure_ready()
    role = state.get("role", "general").lower()
    category = state.get("category", "").lower()

    query = _get_embedder().encode([state["question"]], normalize_embeddings=True)
    _, ids = _index.search(query, len(_entries))

    results = []
    for i in ids[0]:
        entry = _entries[i]
        if entry["category"] != category or entry["role"] not in (role, "general"):
            continue
        results.append(
            f"Example question: {entry['question']}\n"
            f"Strong answer: {entry['strong_answer']}\n"
            f"Why it works: {entry['why_it_works']}"
        )
        if len(results) == TOP_K:
            break

    return {"retrieved_context": results}