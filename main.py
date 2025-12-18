import os
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# =========================
# Load environment
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("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")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set!")
    st.stop()

client = Groq(api_key="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")

# =========================
# Audio conversion helper
# =========================
def convert_to_wav(input_path: Path) -> Path:
    """
    Convert any audio to 16kHz mono WAV (Whisper-safe)
    """
    output_path = input_path.with_suffix(".wav")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    return output_path

# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="Speech-to-Text (Groq Whisper)",
    layout="centered"
)

st.title("🎙️ Speech-to-Text")
st.write("Record your voice and transcribe using **Groq Whisper API**")

st.markdown("---")

language_choice = st.selectbox(
    "Select Language",
    ["Auto Detect", "Persian (fa)", "English (en)"]
)

lang_map = {
    "Auto Detect": None,
    "Persian (fa)": "fa",
    "English (en)": "en",
}

audio_file = st.audio_input("Click to record")

if audio_file:
    st.audio(audio_file)

    raw_bytes = audio_file.read()

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(raw_bytes)
    tmp.flush()

    raw_audio_path = Path(tmp.name)
    wav_audio_path = convert_to_wav(raw_audio_path)

    st.success("Audio recorded successfully")

    if st.button("Transcribe"):
        with st.spinner("Transcribing with Whisper..."):
            try:
                with open(wav_audio_path, "rb") as f:
                    kwargs = {}
                    if lang_map[language_choice]:
                        kwargs["language"] = lang_map[language_choice]

                    response = client.audio.transcriptions.create(
                        file=f,
                        model="whisper-large-v3",
                        **kwargs
                    )

                transcript = response.text

                st.subheader("📝 Transcription")
                st.write(transcript)
                st.caption(f"Word count: {len(transcript.split())}")

                if st.button("Save Transcript"):
                    Path("transcript.txt").write_text(
                        transcript, encoding="utf-8"
                    )
                    st.success("Saved as transcript.txt")

            except Exception as e:
                st.error("Transcription failed")
                st.exception(e)
