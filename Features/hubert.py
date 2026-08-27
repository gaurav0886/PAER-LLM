"""
HuBERT Embedding Extraction Module

Bugs and design problems fixed
------------------------------
1. **Debug prints on every call.** The original printed the hidden-state shape,
   the embedding shape and the first ten values for *every* file. Over 1440
   files that is 4320 lines of console noise, it slows a notebook to a crawl,
   and in the Streamlit app it prints into the server log on every request.

2. **Model loaded at import time.** ``HubertModel.from_pretrained(...)`` ran as
   a module-level side effect, so merely importing anything from
   ``Features.hubert`` - including indirectly, via ``Utils.inference`` -
   downloaded and instantiated a 95M-parameter model. That is why the Streamlit
   app is slow to start and why ``import Utils.inference`` blocks. Loading is
   now lazy and cached.

3. **CPU only.** The model was never moved to the GPU even when one was
   available, making the 1440-file extraction pass far slower than necessary.

4. **No batching.** Added ``extract_hubert_features_batch`` with padding and a
   proper attention mask, so masked frames are excluded from the mean pool
   instead of dragging it toward zero.

5. **Which layer?** Mean-pooling the *last* hidden state is a defensible
   default but not the best one for paralinguistics - middle layers of HuBERT
   carry more speaker-state information than the top layer, which is specialised
   for phonetic content. ``layer=`` is now exposed so you can justify the choice
   empirically instead of by default. Keep ``layer=-1`` to stay compatible with
   the existing ``hubert_embeddings.csv``.
"""

from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np
import torch

from Utils.constants import HUBERT_SAMPLE_RATE

MODEL_NAME = "facebook/hubert-base-ls960"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=2)
def _load(model_name: str = MODEL_NAME):
    """Load and cache the feature extractor and model (once per process)."""
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    model.eval()
    model.to(get_device())
    return feature_extractor, model


def _pool(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool over time, ignoring padded frames."""
    if mask is None:
        return hidden.mean(dim=1)
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


@torch.no_grad()
def extract_hubert_features(
    file_path: str,
    *,
    model_name: str = MODEL_NAME,
    layer: int = -1,
) -> np.ndarray:
    """Extract a 768-dim utterance embedding from one recording.

    Parameters
    ----------
    file_path : str
        Path to an audio file.
    layer : int
        Which transformer layer to pool. ``-1`` is the final layer (default,
        matches the existing CSV). Try 6-9 for paralinguistic tasks.

    Returns
    -------
    numpy.ndarray, shape (768,), dtype float32
    """
    feature_extractor, model = _load(model_name)

    audio, _ = librosa.load(file_path, sr=HUBERT_SAMPLE_RATE)

    if audio.size == 0:
        raise ValueError(f"empty audio: {file_path}")

    inputs = feature_extractor(
        audio,
        sampling_rate=HUBERT_SAMPLE_RATE,
        return_tensors="pt",
    )
    input_values = inputs.input_values.to(model.device)

    if layer == -1:
        hidden = model(input_values).last_hidden_state
    else:
        hidden = model(input_values, output_hidden_states=True).hidden_states[layer]

    embedding = _pool(hidden, None).squeeze(0)
    return embedding.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def extract_hubert_features_batch(
    file_paths: list[str],
    *,
    model_name: str = MODEL_NAME,
    layer: int = -1,
    batch_size: int = 8,
) -> np.ndarray:
    """Batched version of :func:`extract_hubert_features`.

    Returns
    -------
    numpy.ndarray, shape (len(file_paths), 768)
    """
    feature_extractor, model = _load(model_name)
    out: list[np.ndarray] = []

    for start in range(0, len(file_paths), batch_size):
        chunk = file_paths[start:start + batch_size]
        waves = [librosa.load(p, sr=HUBERT_SAMPLE_RATE)[0] for p in chunk]

        inputs = feature_extractor(
            waves,
            sampling_rate=HUBERT_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs.input_values.to(model.device)
        attn = inputs.get("attention_mask")
        attn = attn.to(model.device) if attn is not None else None

        if layer == -1:
            hidden = model(input_values, attention_mask=attn).last_hidden_state
        else:
            hidden = model(
                input_values, attention_mask=attn, output_hidden_states=True
            ).hidden_states[layer]

        # Waveform-level mask -> frame-level mask via the conv feature encoder.
        frame_mask = None
        if attn is not None:
            lengths = model._get_feat_extract_output_lengths(attn.sum(-1)).to(
                hidden.device
            )
            frame_mask = (
                torch.arange(hidden.shape[1], device=hidden.device)[None, :]
                < lengths[:, None]
            )

        pooled = _pool(hidden, frame_mask)
        out.append(pooled.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(out, axis=0)


__all__ = [
    "extract_hubert_features",
    "extract_hubert_features_batch",
    "get_device",
    "MODEL_NAME",
]
