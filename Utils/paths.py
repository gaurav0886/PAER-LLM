"""
paths.py

Single source of truth for project paths.

Why this file exists
--------------------
The original code used relative paths like ``"../Models/best_model.pth"`` and
``os.path.abspath("..")``. Those resolve against the *current working
directory*, so the same function worked from ``Notebook/`` and crashed from the
project root, from ``Streamlit/``, or from a scheduled job.

Everything here is derived from the location of this file, so it is correct no
matter where Python is launched from.
"""

from pathlib import Path

# .../PhD_Project/Utils/paths.py -> .../PhD_Project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "Dataset"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
MODELS_DIR = PROJECT_ROOT / "Models"
EVALUATION_DIR = PROJECT_ROOT / "Evaluation"
DOCS_DIR = PROJECT_ROOT / "Docs"
NOTEBOOK_DIR = PROJECT_ROOT / "Notebook"

# Feature tables
PITCH_CSV = OUTPUTS_DIR / "pitch_features.csv"
LOUDNESS_CSV = OUTPUTS_DIR / "loudness_features.csv"
SPECTRAL_CSV = OUTPUTS_DIR / "spectral_features.csv"
BARK_CSV = OUTPUTS_DIR / "bark_features.csv"
PSYCHO_CSV = OUTPUTS_DIR / "psychoacoustic_features.csv"
HUBERT_CSV = OUTPUTS_DIR / "hubert_embeddings.csv"
MULTIMODAL_CSV = OUTPUTS_DIR / "multimodal_features.csv"

# Model artefacts
BEST_MODEL = MODELS_DIR / "best_model.pth"
PSYCHO_SCALER = MODELS_DIR / "psycho_scaler.pkl"
HUBERT_SCALER = MODELS_DIR / "hubert_scaler.pkl"
LABEL_ENCODER = MODELS_DIR / "label_encoder.pkl"
SPLIT_JSON = MODELS_DIR / "split_actors.json"
TRAIN_CONFIG = MODELS_DIR / "train_config.json"


def ensure_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    for d in (OUTPUTS_DIR, MODELS_DIR, EVALUATION_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATASET_DIR",
    "OUTPUTS_DIR",
    "MODELS_DIR",
    "EVALUATION_DIR",
    "DOCS_DIR",
    "NOTEBOOK_DIR",
    "PITCH_CSV",
    "LOUDNESS_CSV",
    "SPECTRAL_CSV",
    "BARK_CSV",
    "PSYCHO_CSV",
    "HUBERT_CSV",
    "MULTIMODAL_CSV",
    "BEST_MODEL",
    "PSYCHO_SCALER",
    "HUBERT_SCALER",
    "LABEL_ENCODER",
    "SPLIT_JSON",
    "TRAIN_CONFIG",
    "ensure_dirs",
]
