"""
constants.py

Project-wide constants for PAER-LLM.

Changed from the original
-------------------------
1. Absolute Windows paths (``r"F:\\PhD\\PhD_Project\\Dataset"``) are gone. They
   made the project unrunnable on any other machine, which is fatal for a
   thesis artefact that a reviewer or collaborator may need to reproduce.
   Paths now come from ``Utils.paths``.
2. Added the RAVDESS filename schema so speaker / statement / repetition
   metadata can be recovered from the filename. This is what makes a
   speaker-independent evaluation possible.
3. Added explicit feature dimensions so the model, the scalers and the
   inference path can never silently disagree.
"""

from Utils.paths import DATASET_DIR, OUTPUTS_DIR

# ---------------------------------------------------------------------------
# Paths (kept as strings for backwards compatibility with existing notebooks)
# ---------------------------------------------------------------------------
DATASET_PATH = str(DATASET_DIR)
OUTPUT_PATH = str(OUTPUTS_DIR)

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050          # used by librosa-based psychoacoustic features
HUBERT_SAMPLE_RATE = 16000   # HuBERT is trained at 16 kHz - do not change
N_FFT = 2048
HOP_LENGTH = 512
N_BARK = 24

# ---------------------------------------------------------------------------
# Machine learning
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20              # fraction of the *training* actors held out for
                             # model selection, so the test set stays untouched
N_MFCC = 40

# Feature block sizes. PSYCHO_DIM must equal PITCH + LOUDNESS + BARK because
# Utils.inference.extract_psycho_features concatenates them in that order and
# the training table was built in that order too.
PITCH_DIM = 5
LOUDNESS_DIM = 5
BARK_DIM = N_BARK
PSYCHO_DIM = PITCH_DIM + LOUDNESS_DIM + BARK_DIM   # 34
HUBERT_DIM = 768
NUM_CLASSES = 8

PITCH_COLUMNS = [
    "mean_pitch", "max_pitch", "min_pitch", "std_pitch", "pitch_range",
]
LOUDNESS_COLUMNS = [
    "mean_loudness", "max_loudness", "min_loudness", "std_loudness",
    "loudness_range",
]
# NOTE: the existing bark_features.csv / psychoacoustic_features.csv use
# 1-based band numbering (bark_1 .. bark_24), not 0-based. Matching the data
# rather than "fixing" the data, because renaming the columns would silently
# invalidate every CSV already in Outputs/.
BARK_COLUMNS = [f"bark_{i}" for i in range(1, N_BARK + 1)]
PSYCHO_COLUMNS = PITCH_COLUMNS + LOUDNESS_COLUMNS + BARK_COLUMNS
HUBERT_COLUMNS = [f"hubert_{i}" for i in range(HUBERT_DIM)]

# ---------------------------------------------------------------------------
# RAVDESS filename schema
# 03-01-05-01-01-01-06.wav
#  |  |  |  |  |  |  |
#  |  |  |  |  |  |  +-- actor        (01..24, odd = male, even = female)
#  |  |  |  |  |  +----- repetition   (01, 02)
#  |  |  |  |  +-------- statement    (01 "kids", 02 "dogs")
#  |  |  |  +----------- intensity    (01 normal, 02 strong)
#  |  |  +-------------- emotion      (01..08)
#  |  +----------------- vocal channel(01 speech, 02 song)
#  +-------------------- modality     (03 = audio-only)
# ---------------------------------------------------------------------------
RAVDESS_FIELDS = (
    "modality",
    "vocal_channel",
    "emotion_id",
    "intensity_id",
    "statement_id",
    "repetition_id",
    "actor_id",
)

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITY_MAP = {"01": "normal", "02": "strong"}
STATEMENT_MAP = {"01": "kids", "02": "dogs"}

# Alphabetical order - this is what sklearn's LabelEncoder produces, and it is
# the order the trained checkpoint's output units correspond to. Kept here only
# as a fallback; at inference time the *saved* label encoder is authoritative.
EMOTION_LABELS = sorted(set(EMOTION_MAP.values()))
