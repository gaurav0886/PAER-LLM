"""
Spectral Feature Extraction Module

Two real bugs fixed
-------------------
1. ``np.mean(contrast)`` collapsed a 7-band x T spectral-contrast matrix into a
   single scalar. Averaging across bands destroys the very thing spectral
   contrast measures - the per-band peak-to-valley difference. The seven band
   means are now returned separately, giving 11 features instead of 5.

2. These features were never actually used. ``05_feature_fusion.ipynb`` loads
   ``spectral_features.csv``, checks that its ``file`` column matches, and then
   concatenates only pitch, loudness and bark. ``spectral_features.csv`` is
   dead weight in Outputs/ and none of the 34 psychoacoustic dimensions come
   from it. Either wire it in (see ``build_psychoacoustic_table``) or drop the
   claim from the thesis - right now the write-up and the code disagree.

Note on backwards compatibility
-------------------------------
``extract_spectral_features`` now returns 11 values, not 5. Nothing in the
trained model depends on it (see point 2), so this is safe - but if you
re-generate ``spectral_features.csv`` the column set will change.
"""

from __future__ import annotations

import librosa
import numpy as np

from Utils.constants import HOP_LENGTH, N_FFT, SAMPLE_RATE

N_CONTRAST_BANDS = 7  # librosa default: 6 octave bands + 1 residual

SPECTRAL_COLUMNS = (
    ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]
    + [f"spectral_contrast_{i}" for i in range(N_CONTRAST_BANDS)]
    + ["spectral_flatness"]
)


def extract_spectral_features(
    file_path: str,
    *,
    sr: int = SAMPLE_RATE,
) -> list[float]:
    """Extract time-averaged spectral descriptors from one recording.

    Returns
    -------
    list[float]
        centroid, bandwidth, rolloff, 7 contrast bands, flatness (11 values).
    """
    audio, sr = librosa.load(file_path, sr=sr)

    kwargs = dict(n_fft=N_FFT, hop_length=HOP_LENGTH)

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, **kwargs)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, **kwargs)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, **kwargs)[0]
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, **kwargs)
    flatness = librosa.feature.spectral_flatness(y=audio, **kwargs)[0]

    features = [
        float(np.mean(centroid)),
        float(np.mean(bandwidth)),
        float(np.mean(rolloff)),
    ]
    # Per-band means, not one global mean.
    features.extend(float(v) for v in np.mean(contrast, axis=1))
    features.append(float(np.mean(flatness)))

    return features


def extract_spectral_features_dict(file_path: str) -> dict[str, float]:
    return dict(zip(SPECTRAL_COLUMNS, extract_spectral_features(file_path)))


__all__ = [
    "extract_spectral_features",
    "extract_spectral_features_dict",
    "SPECTRAL_COLUMNS",
]
