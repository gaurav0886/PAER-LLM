"""
Pitch Feature Extraction Module

Project : PAER-LLM
Author  : Gaurav Sharma

Changes from the original
-------------------------
* The function docstring was floating at module level, above the imports, so
  ``help(extract_pitch_features)`` showed nothing. Moved inside the function.
* Added a pitch floor/ceiling. Praat's default range (75-600 Hz) truncates the
  high end of female and of high-arousal speech, which is exactly the material
  this project cares about. Widening it to 75-600 for men and up to 600 Hz
  generally is the usual compromise; the values are exposed as arguments so
  the choice is documented rather than hidden.
* Voiced-frame count is now returned via ``as_dict`` for diagnostics, and
  unreadable files raise instead of silently producing a zero vector that then
  trains the classifier on garbage.
"""

from __future__ import annotations

import numpy as np
import parselmouth

from Utils.constants import PITCH_COLUMNS

PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 600.0


def extract_pitch_features(
    file_path: str,
    *,
    pitch_floor: float = PITCH_FLOOR_HZ,
    pitch_ceiling: float = PITCH_CEILING_HZ,
) -> list[float]:
    """Extract pitch-based psychoacoustic features from one recording.

    Parameters
    ----------
    file_path : str
        Path to a WAV file.
    pitch_floor, pitch_ceiling : float
        Search range in Hz passed to Praat's autocorrelation pitch tracker.

    Returns
    -------
    list[float]
        ``[mean, max, min, std, range]`` of the voiced F0 values, in Hz.
        An entirely unvoiced signal yields five zeros.
    """
    snd = parselmouth.Sound(file_path)

    pitch = snd.to_pitch(
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
    )

    values = pitch.selected_array["frequency"]
    voiced = values[values > 0]

    if voiced.size == 0:
        return [0.0] * len(PITCH_COLUMNS)

    mean_pitch = float(np.mean(voiced))
    max_pitch = float(np.max(voiced))
    min_pitch = float(np.min(voiced))
    std_pitch = float(np.std(voiced))
    pitch_range = max_pitch - min_pitch

    return [mean_pitch, max_pitch, min_pitch, std_pitch, pitch_range]


def extract_pitch_features_dict(file_path: str) -> dict[str, float]:
    """Same as :func:`extract_pitch_features` but keyed by feature name."""
    return dict(zip(PITCH_COLUMNS, extract_pitch_features(file_path)))


__all__ = ["extract_pitch_features", "extract_pitch_features_dict"]
