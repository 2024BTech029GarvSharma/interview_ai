# InterVue AI
A multi-agent mock interview coach: speech-to-text, delivery metrics, RAG over best-practice answers, and LLM feedback reviewed by a Critic agent.

## Setup
1. Install Python 3.11+ and ffmpeg (`winget install Gyan.FFmpeg --source winget`), then restart your terminal.
2. Clone the repo and enter the folder.
3. Create and activate a virtual environment:
   `python -m venv venv` then `venv\Scripts\activate`
4. Install dependencies:
   `pip install -r requirements.txt`
   `python -m spacy download en_core_web_sm`
5. Copy `.env.example` to `.env` and put your own Gemini API key in it (free from aistudio.google.com). Never commit `.env`.
6. Run the app: `streamlit run app.py`

The first run downloads the Whisper and embedding models, which takes a few minutes.

## Structure
- `agents/`: one file per agent (interviewer, transcription, delivery_analyst, retriever, coach, critic)
- `graph.py`: LangGraph orchestrator
- `state.py`: shared state passed between agents
- `llm.py`: shared LLM helper with retries and fallback model
- `data/knowledge_base/`: interview Q&A used for retrieval
- `app.py`: Streamlit UI