
import os
import tempfile

import sounddevice as sd
import soundfile as sf
from openai import BadRequestError

from rag_engine import RAGEngine

# -------------------------
# CONFIG
# -------------------------
RECORD_SECONDS = 5
SAMPLE_RATE = 16000

STT_MODEL = "whisper-large-v3"              # Groq speech-to-text model
TTS_MODEL = "canopylabs/orpheus-v1-english" # Groq / Orpheus text-to-speech model
TTS_VOICE = "troy"                          # Pick any voice from Groq's Orpheus voice list
TTS_TERMS_URL = "https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english"


# -------------------------
# AUDIO HELPERS
# -------------------------
def record_audio(filepath: str, duration: int = RECORD_SECONDS, samplerate: int = SAMPLE_RATE) -> None:
    print(f"\n🎙️  Recording for {duration}s... speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    sf.write(filepath, audio, samplerate)
    print("✅ Recording captured.")


def play_audio(filepath: str) -> None:
    data, samplerate = sf.read(filepath, dtype="int16")
    sd.play(data, samplerate)
    sd.wait()


# -------------------------
# GROQ STT / TTS HELPERS
# -------------------------
def transcribe(client, filepath: str) -> str:
    with open(filepath, "rb") as f:
        result = client.audio.transcriptions.create(
            file=f,
            model=STT_MODEL,
        )
    return result.text.strip()


def synthesize(client, text: str, filepath: str) -> bool:
    text = (text or "").strip()
    if not text:
        print("Assistant returned no text, skipping voice output.\n")
        return False

    try:
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format="wav",
        )
    except BadRequestError as exc:
        if getattr(exc, "code", None) == "model_terms_required":
            print("⚠️  Voice output is disabled until your Groq org admin accepts Orpheus TTS terms.")
            print(f"   Accept here: {TTS_TERMS_URL}\n")
            return False
        raise

    response.write_to_file(filepath)
    return True


# -------------------------
# MAIN LOOP
# -------------------------
def run_voice_agent() -> None:
    engine = RAGEngine()
    history: list[dict] = []

    print("\n🎤 Voice RAG Agent Started")
    print("Press ENTER to record a question, or type 'exit' + ENTER to quit.\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.wav")
        out_path = os.path.join(tmpdir, "output.wav")

        while True:
            cmd = input("Press ENTER to talk (or 'exit'): ")
            if cmd.strip().lower() in ["exit", "quit"]:
                break

            record_audio(in_path)

            user_text = transcribe(engine.client, in_path)
            if not user_text:
                print("⚠️  No speech detected, try again.\n")
                continue

            print(f"You said: {user_text}")

            answer = engine.chat(user_text, history)
            print(f"Assistant: {answer}\n")

            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": answer})

            if synthesize(engine.client, answer, out_path):
                play_audio(out_path)


if __name__ == "__main__":
    run_voice_agent()
