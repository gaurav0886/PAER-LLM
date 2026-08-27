"""
MFCC Feature Extraction Module (baseline features)

Changes from the original
-------------------------
* Sample rate and n_mfcc now come from ``Utils.constants`` instead of being
  hard-coded, so the baseline and the main pipeline cannot drift apart.
* Added optional delta / delta-delta and standard deviation statistics. Mean
  pooling alone discards all temporal dynamics, and emotion lives largely in
  the dynamics - a stronger baseline makes the multimodal model's improvement
  a more meaningful claim. ``include_deltas=False, include_std=False``
  reproduces the original 40-dim vector exactly.
"""

from __future__ import annotations

import librosa
import numpy as np

from Utils.constants import HOP_LENGTH, N_FFT, N_MFCC, SAMPLE_RATE


def extract_mfcc(
    file_path: str,
    *,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    include_std: bool = False,
    include_deltas: bool = False,
) -> np.ndarray:
    """Extract time-pooled MFCC statistics.

    Returns
    -------
    numpy.ndarray
        ``n_mfcc`` values by default; more if std / deltas are enabled.
    """
    audio, sr = librosa.load(file_path, sr=sr)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    blocks = [mfcc.mean(axis=1)]

    if include_std:
        blocks.append(mfcc.std(axis=1))

    if include_deltas:
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        blocks.append(d1.mean(axis=1))
        blocks.append(d2.mean(axis=1))
        if include_std:
            blocks.append(d1.std(axis=1))
            blocks.append(d2.std(axis=1))

    return np.concatenate(blocks).astype(np.float32)


__all__ = ["extract_mfcc"]
