import os
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
from groq import Groq

# API KEY 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set as environment variable")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Audio conversion
def convert_to_wav(input_path: Path) -> Path:
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


# Streamlit UI
st.set_page_config(
    page_title="Speech-to-Text (Groq Whisper)",
    layout="centered"
)

st.title("🎙️ Speech to Text")
st.write("Groq Whisper API — Persian & English Supported")

language_choice = st.selectbox(
    "Language",
    ["Auto Detect", "Persian (fa)", "English (en)"]
)

lang_map = {
    "Auto Detect": None,
    "Persian (fa)": "fa",
    "English (en)": "en",
}

audio_file = st.audio_input("Record your voice")

if audio_file:
    st.audio(audio_file)

    raw_bytes = audio_file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(raw_bytes)
    tmp.flush()

    raw_audio = Path(tmp.name)
    wav_audio = convert_to_wav(raw_audio)

    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            try:
                with open(wav_audio, "rb") as f:
                    kwargs = {}
                    if lang_map[language_choice]:
                        kwargs["language"] = lang_map[language_choice]

                    response = client.audio.transcriptions.create(
                        file=f,
                        model="whisper-large-v3",
                        **kwargs
                    )

                transcript = response.text

                st.subheader("Transcription")
                st.write(transcript)
                st.caption(f"Word count: {len(transcript.split())}")

                if st.button("Save Transcript"):
                    Path("transcript.txt").write_text(
                        transcript, encoding="utf-8"
                    )
                    st.success("Saved transcript.txt")

            except Exception as e:
                st.error("Transcription failed")
                st.exception(e)
