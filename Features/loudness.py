"""
Loudness Feature Extraction Module

Changes from the original
-------------------------
* The sample rate was hard-coded to 22050 in three separate files. It now comes
  from ``Utils.constants.SAMPLE_RATE`` so changing it once changes it
  everywhere - otherwise a future edit silently desynchronises the feature
  blocks that get concatenated together.
* RMS is optionally converted to dB. Linear RMS is dominated by recording gain
  and compresses the perceptually meaningful range; dBFS is closer to how
  loudness is actually perceived and is what the psychoacoustic framing of this
  thesis implies. ``as_db=False`` reproduces the original behaviour exactly, so
  the existing CSVs remain valid - do not flip this without re-extracting.
"""

from __future__ import annotations

import librosa
import numpy as np

from Utils.constants import HOP_LENGTH, LOUDNESS_COLUMNS, SAMPLE_RATE

_EPS = 1e-10


def extract_loudness_features(
    file_path: str,
    *,
    sr: int = SAMPLE_RATE,
    as_db: bool = False,
) -> list[float]:
    """Extract short-term energy statistics from one recording.

    Parameters
    ----------
    file_path : str
        Path to a WAV file.
    sr : int
        Target sample rate.
    as_db : bool
        If True, report RMS in dBFS instead of linear amplitude.

    Returns
    -------
    list[float]
        ``[mean, max, min, std, range]`` of frame-level RMS.
    """
    audio, sr = librosa.load(file_path, sr=sr)

    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]

    if as_db:
        rms = 20.0 * np.log10(rms + _EPS)

    mean_loudness = float(np.mean(rms))
    max_loudness = float(np.max(rms))
    min_loudness = float(np.min(rms))
    std_loudness = float(np.std(rms))
    loudness_range = max_loudness - min_loudness

    return [
        mean_loudness,
        max_loudness,
        min_loudness,
        std_loudness,
        loudness_range,
    ]


def extract_loudness_features_dict(file_path: str) -> dict[str, float]:
    return dict(zip(LOUDNESS_COLUMNS, extract_loudness_features(file_path)))


__all__ = ["extract_loudness_features", "extract_loudness_features_dict"]
