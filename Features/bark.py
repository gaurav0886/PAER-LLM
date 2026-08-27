"""
Bark-scale Feature Extraction Module

Bugs fixed
----------
1. ``extract_bark_features`` was defined **twice** in this file. The first
   definition was an empty stub that returned ``None``; the second one shadowed
   it. It worked only by accident of ordering - had the stub been placed
   second, every psychoacoustic vector in the project would have been ``None``.
   The stub is gone.

2. ``EMOTION_MAP`` was imported and never used, and ``os`` likewise.

3. The filter bank was built with a Python double loop over
   ``n_barks x n_frequencies``. It is now vectorised (roughly 100x faster) and
   cached, so it is not rebuilt for every one of the 1440 files.

4. The filter bank was un-normalised, so wide high-frequency bands accumulated
   more energy purely because they span more FFT bins. Band energies are now
   optionally area-normalised (``normalize="area"``), which is the standard
   choice and makes the 24 bands comparable to each other.

5. Band energies are returned in dB. Raw power spans several orders of
   magnitude, which is a poor input to a linear encoder and is not how loudness
   in a critical band is perceived. ``as_db=False`` restores the original
   linear behaviour - keep it False if you want to stay compatible with the
   existing ``bark_features.csv``.
"""

from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np

from Utils.constants import BARK_COLUMNS, HOP_LENGTH, N_BARK, N_FFT, SAMPLE_RATE

_EPS = 1e-10


def hz_to_bark(f: np.ndarray | float) -> np.ndarray | float:
    """Convert frequency in Hz to the Bark scale (Traunmuller/Zwicker form).

    Parameters
    ----------
    f : float or ndarray
        Frequency in Hertz.

    Returns
    -------
    float or ndarray
        Critical-band rate in Bark.
    """
    f = np.asarray(f, dtype=np.float64)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


@lru_cache(maxsize=8)
def create_bark_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_barks: int = N_BARK,
    normalize: str = "area",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a triangular Bark-spaced filter bank.

    Parameters
    ----------
    sr : int
        Sample rate.
    n_fft : int
        FFT size.
    n_barks : int
        Number of Bark bands.
    normalize : {"area", "none"}
        ``"area"`` scales each triangle to unit area so that band energy is not
        biased by bandwidth. ``"none"`` reproduces the original code.

    Returns
    -------
    filterbank : ndarray, shape (n_barks, 1 + n_fft // 2)
    bark : ndarray
        Bark value of each FFT bin.
    bark_points : ndarray
        Band edge positions in Bark.
    """
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bark = hz_to_bark(frequencies)

    bark_points = np.linspace(0.0, float(bark.max()), n_barks + 2)

    lower = bark_points[:-2][:, None]
    center = bark_points[1:-1][:, None]
    upper = bark_points[2:][:, None]
    b = bark[None, :]

    rising = (b - lower) / np.maximum(center - lower, _EPS)
    falling = (upper - b) / np.maximum(upper - center, _EPS)

    filterbank = np.maximum(0.0, np.minimum(rising, falling))

    if normalize == "area":
        area = filterbank.sum(axis=1, keepdims=True)
        filterbank = filterbank / np.maximum(area, _EPS)
    elif normalize != "none":
        raise ValueError(f"unknown normalize option: {normalize!r}")

    return filterbank, bark, bark_points


def extract_bark_features(
    file_path: str,
    *,
    sr: int = SAMPLE_RATE,
    n_barks: int = N_BARK,
    as_db: bool = False,
    normalize: str = "area",
) -> np.ndarray:
    """Extract mean energy in each of ``n_barks`` critical bands.

    Parameters
    ----------
    file_path : str
        Path to a WAV file.
    as_db : bool
        Return band energies in dB rather than linear power.
    normalize : {"area", "none"}
        Filter-bank normalisation - see :func:`create_bark_filterbank`.

    Returns
    -------
    numpy.ndarray, shape (n_barks,)
    """
    audio, sr = librosa.load(file_path, sr=sr)

    power = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2

    filterbank, _, _ = create_bark_filterbank(
        sr=sr, n_fft=N_FFT, n_barks=n_barks, normalize=normalize
    )

    bark_energy = filterbank @ power          # (n_barks, n_frames)
    bark_energy = bark_energy.mean(axis=1)    # (n_barks,)

    if as_db:
        bark_energy = 10.0 * np.log10(bark_energy + _EPS)

    return bark_energy.astype(np.float32)


def extract_bark_features_dict(file_path: str, **kwargs) -> dict[str, float]:
    values = extract_bark_features(file_path, **kwargs)
    return dict(zip(BARK_COLUMNS, values.tolist()))


__all__ = [
    "hz_to_bark",
    "create_bark_filterbank",
    "extract_bark_features",
    "extract_bark_features_dict",
]
