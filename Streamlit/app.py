"""
Streamlit demo for PAER-LLM.

Run from the project root:

    streamlit run Streamlit/app.py

Bugs fixed
----------
1. ``project_root = os.path.abspath("..")`` resolved against the *working
   directory*, not the file. Launching from the project root put
   ``F:\\PhD`` on ``sys.path`` and the imports failed. It now derives from
   ``__file__``.

2. ``audio_path = "temp.wav"`` wrote into whatever directory Streamlit happened
   to start in, and never cleaned up - that is where the stray
   ``Streamlit/temp.wav`` in the repo came from. Concurrent users would also
   overwrite each other's file. Uses a per-session temporary file now.

3. The LLM was loaded eagerly at startup even if the user never uploaded
   anything - several GB of RAM and a long cold start for a page that might
   just be being looked at. It is loaded on first use.

4. No error handling: a non-RAVDESS or corrupt WAV raised a raw traceback into
   the browser.

5. The caption claimed "128-dimensional emotion embedding" as a hard-coded
   fact. It now reports the model's actual embedding width.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.inference import load_emotion_model, load_llm, predict  # noqa: E402

st.set_page_config(page_title="PAER-LLM", page_icon="🎧", layout="wide")


@st.cache_resource(show_spinner="Loading emotion model...")
def get_emotion_model():
    return load_emotion_model()


@st.cache_resource(show_spinner="Loading language model...")
def get_llm():
    return load_llm()


st.title("🎧 PAER-LLM")
st.subheader(
    "Psychoacoustic-Aware Emotion-Sensitive Human-Computer Interaction"
)
st.markdown("---")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.getvalue()

    # Hand Streamlit the bytes directly - no file on disk needed for playback.
    st.audio(audio_bytes, format="audio/wav")

    if st.button("Analyse emotion", type="primary"):
        try:
            model = get_emotion_model()
        except FileNotFoundError as exc:
            st.error(f"{exc}")
            st.stop()

        # Temp file lives only for the duration of feature extraction, and is
        # removed even if extraction raises.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = Path(tmp.name)

            with st.spinner("Analysing vocal features..."):
                emotion, confidence, embedding, probabilities = predict(
                    str(tmp_path), model
                )
        except Exception as exc:  # noqa: BLE001 - surface to the user, not the log
            st.error(f"Could not analyse this file: {exc}")
            st.stop()
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Emotion prediction")
            st.metric("Detected emotion", emotion.capitalize())
            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(min(max(confidence / 100, 0.0), 1.0))

            st.caption("Full class distribution")
            st.bar_chart(probabilities)

            if confidence < 60:
                st.warning(
                    "Low confidence - the tone is ambiguous. Treat this "
                    "reading as a suggestion, not a determination."
                )

        with col2:
            st.subheader("🤖 AI response")
            try:
                builder, llm = get_llm()
                with st.spinner("Generating an emotion-aware response..."):
                    prompt = builder.build_prompt(
                        emotion=emotion,
                        confidence=confidence,
                        probabilities=probabilities,
                    )
                    response = llm.generate(prompt, max_tokens=120)
                st.success(response)
            except Exception as exc:  # noqa: BLE001
                st.info(f"Language model unavailable: {exc}")

        st.markdown("---")
        with st.expander("Fused multimodal embedding"):
            st.write(embedding[:10])
            st.caption(
                f"First 10 of {embedding.shape[0]} dimensions from the fusion layer."
            )
