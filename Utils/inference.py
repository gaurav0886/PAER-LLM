"""
inference.py

End-to-end single-file inference: audio -> emotion -> empathetic response.

Bugs fixed
----------
1. **Paths were relative to the caller's cwd.** ``torch.load("../Models/best_model.pth")``
   only worked when Python was launched from ``Notebook/`` or ``Streamlit/``.
   From the project root it raised FileNotFoundError. All paths now come from
   ``Utils.paths``.

2. **Scalers were re-loaded from disk on every single prediction.** Two
   ``joblib.load`` calls per ``predict()`` - noticeable in the Streamlit app and
   pointless. They are cached now.

3. **Emotion labels were hard-coded.** The module carried its own
   ``emotion_labels`` list. It happens to match ``LabelEncoder``'s alphabetical
   ordering, so it worked - but if the label encoder were ever refit on a
   subset of classes, every prediction would be silently mislabelled with no
   error. The saved ``label_encoder.pkl`` is now authoritative.

4. **Doubly-nested ``with torch.no_grad():``** - harmless but a sign the block
   was pasted twice.

5. **``generate_response`` reloaded the emotion model and the 3.8B-parameter
   LLM on every call**, then deleted the emotion model and ran ``gc.collect()``.
   For a single file that is minutes of overhead per request. Both are cached
   now; call ``unload_llm()`` explicitly if you are memory-constrained.

6. **No feature-dimension check.** If ``extract_psycho_features`` returned a
   vector of the wrong length the scaler raised a confusing sklearn error deep
   in the stack. There is an explicit check with a readable message.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from Features.bark import extract_bark_features
from Features.hubert import extract_hubert_features
from Features.loudness import extract_loudness_features
from Features.pitch import extract_pitch_features
from Models.multimodal_model import MultimodalEmotionModel
from Utils.constants import PSYCHO_DIM
from Utils.paths import BEST_MODEL, HUBERT_SCALER, LABEL_ENCODER, PSYCHO_SCALER


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Cached artefact loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_label_encoder():
    if not LABEL_ENCODER.exists():
        raise FileNotFoundError(
            f"{LABEL_ENCODER} not found - run `python -m Training.train` first."
        )
    return joblib.load(LABEL_ENCODER)


@lru_cache(maxsize=1)
def load_scalers():
    for path in (PSYCHO_SCALER, HUBERT_SCALER):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run `python -m Training.train` first."
            )
    return joblib.load(PSYCHO_SCALER), joblib.load(HUBERT_SCALER)


@lru_cache(maxsize=1)
def load_emotion_model():
    """Load and cache the trained multimodal classifier."""
    if not BEST_MODEL.exists():
        raise FileNotFoundError(
            f"{BEST_MODEL} not found - run `python -m Training.train` first."
        )
    model = MultimodalEmotionModel.load(BEST_MODEL, map_location=get_device())
    model.to(get_device()).eval()
    print(f"[inference] emotion model loaded on {get_device()}")
    return model


@lru_cache(maxsize=1)
def load_llm():
    """Load and cache the prompt builder and the response LLM."""
    from LLM.llm_model import EmotionLLM
    from LLM.prompt_builder import PromptBuilder

    builder = PromptBuilder()
    llm = EmotionLLM()
    return builder, llm


def unload_llm() -> None:
    """Free the LLM's memory. Call only if you are actually constrained."""
    import gc

    load_llm.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def extract_psycho_features(audio_path: str) -> np.ndarray:
    """Build the 34-dim psychoacoustic vector.

    The concatenation order (pitch, loudness, bark) must match the column order
    used when ``psychoacoustic_features.csv`` was built, or the scaler and the
    model receive permuted inputs and degrade silently.
    """
    pitch = extract_pitch_features(audio_path)
    loudness = extract_loudness_features(audio_path)
    bark = extract_bark_features(audio_path)

    features = np.asarray(
        list(pitch) + list(loudness) + list(bark), dtype=np.float32
    )

    if features.shape[0] != PSYCHO_DIM:
        raise ValueError(
            f"expected {PSYCHO_DIM} psychoacoustic features, got "
            f"{features.shape[0]} (pitch={len(pitch)}, loudness={len(loudness)}, "
            f"bark={len(bark)})"
        )

    return features


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(audio_path: str, model=None) -> tuple[str, float, np.ndarray, dict]:
    """Predict the emotion of one recording.

    Returns
    -------
    emotion : str
    confidence : float
        Softmax probability of the top class, in percent.
    embedding : numpy.ndarray
        The fused multimodal representation.
    probabilities : dict[str, float]
        Full class distribution, in percent - useful for the UI and for
        reporting calibration rather than a single opaque number.
    """
    model = model or load_emotion_model()
    device = get_device()

    psycho_scaler, hubert_scaler = load_scalers()
    encoder = load_label_encoder()

    psycho = extract_psycho_features(audio_path)
    hubert = extract_hubert_features(audio_path)

    psycho = psycho_scaler.transform(psycho.reshape(1, -1))
    hubert = hubert_scaler.transform(hubert.reshape(1, -1))

    psycho_t = torch.as_tensor(psycho, dtype=torch.float32).to(device)
    hubert_t = torch.as_tensor(hubert, dtype=torch.float32).to(device)

    logits, embedding, _attention = model(psycho_t, hubert_t)
    probabilities = F.softmax(logits, dim=1)

    confidence, predicted = torch.max(probabilities, dim=1)

    labels = list(encoder.classes_)
    emotion = labels[int(predicted.item())]

    probs = {
        label: round(float(p) * 100, 2)
        for label, p in zip(labels, probabilities.squeeze(0).cpu().numpy())
    }

    return (
        emotion,
        float(confidence.item()) * 100.0,
        embedding.squeeze(0).cpu().numpy(),
        probs,
    )


def generate_response(audio_path: str, max_tokens: int = 120) -> dict:
    """Full pipeline: classify the audio, then generate an empathetic reply."""
    model = load_emotion_model()
    emotion, confidence, embedding, probabilities = predict(audio_path, model)

    builder, llm = load_llm()

    prompt = builder.build_prompt(
        emotion=emotion,
        confidence=confidence,
        probabilities=probabilities,
    )
    response = llm.generate(prompt, max_tokens=max_tokens)

    return {
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": probabilities,
        "embedding": embedding,
        "response": response,
    }


__all__ = [
    "load_emotion_model",
    "load_llm",
    "unload_llm",
    "load_scalers",
    "load_label_encoder",
    "extract_psycho_features",
    "predict",
    "generate_response",
    "get_device",
]
