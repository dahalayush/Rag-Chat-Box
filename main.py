import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import RAGEngine

# -------------------------
# SETUP
# -------------------------
app = FastAPI(title="Sherpa - Nepal RAG Voice Assistant")

# Build the index / load models ONCE at startup.
engine = RAGEngine()

# Simple in-memory, single-session history (fine for a demo / single user).
# For multiple concurrent users, key this dict by a session id instead.
conversation_history: list[dict] = []

# Where the latest TTS reply is written so /audio-response can serve it.
TTS_OUTPUT_PATH = os.path.join(tempfile.gettempdir(), "sherpa_response.wav")

STT_MODEL = "whisper-large-v3"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_TEXT_PATH = BASE_DIR / "sample.txt"
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = BASE_DIR / "index.html"
SAMPLE_TEXT = SAMPLE_TEXT_PATH.read_text(encoding="utf-8").strip()

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# -------------------------
# SCHEMAS
# -------------------------
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


def build_rag_input(user_input: str) -> str:
    if not SAMPLE_TEXT:
        return user_input

    return f"""User question:
{user_input}

Reference text from sample_text.txt:
{SAMPLE_TEXT}"""


# -------------------------
# ROUTES
# -------------------------
@app.get("/", response_class=HTMLResponse)
def serve_frontend() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    answer = engine.chat(build_rag_input(request.message), conversation_history)

    conversation_history.append({"role": "user", "content": request.message})
    conversation_history.append({"role": "assistant", "content": answer})

    return ChatResponse(answer=answer)


@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    # Persist the upload with its original extension (e.g. .webm, .wav)
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(await audio.read())
        input_path = tmp_in.name

    try:
        # 1) Speech -> text (Groq Whisper)
        with open(input_path, "rb") as f:
            transcription = engine.client.audio.transcriptions.create(
                file=f,
                model=STT_MODEL,
            )
        user_text = transcription.text.strip()
    finally:
        os.remove(input_path)

    if not user_text:
        return {"transcript": "", "answer": "I didn't catch any speech, please try again.", "audio_url": None}

    # 2) Text -> RAG answer
    answer = engine.chat(build_rag_input(user_text), conversation_history)
    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": answer})

    # 3) Text -> speech (Groq Orpheus)
    speech = engine.client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=answer,
        response_format="wav",
    )
    speech.write_to_file(TTS_OUTPUT_PATH)

    return {"transcript": user_text, "answer": answer, "audio_url": "/audio-response"}


@app.get("/audio-response")
def audio_response():
    if not os.path.exists(TTS_OUTPUT_PATH):
        return {"error": "No audio response generated yet."}
    return FileResponse(TTS_OUTPUT_PATH, media_type="audio/wav")


@app.post("/reset")
def reset():
    conversation_history.clear()
    return {"status": "conversation reset"}
