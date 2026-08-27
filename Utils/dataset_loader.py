"""
dataset_loader.py

Discovery and metadata parsing for the RAVDESS corpus.

The bug this file fixes
-----------------------
The original ``get_audio_files()`` walked ``Dataset/`` and returned every
``.wav`` it found. Your ``Dataset/`` folder contains the corpus **twice**:

    Dataset/Actor_01 ... Dataset/Actor_24                     (1440 files)
    Dataset/audio_speech_actors_01-24/Actor_01 ... Actor_24   (1440 files)

so the walk returned 2880 paths for 1440 distinct recordings. Every feature CSV
in Outputs/ therefore has 2880 rows and 1440 unique filenames - each utterance
appears twice with byte-identical features. A random train/test split then puts
one copy of a recording in train and its clone in test, which is why the
reported accuracies are optimistic.

``get_audio_files()`` now de-duplicates by RAVDESS basename, and
``load_dataset_index()`` returns the parsed metadata needed for a
speaker-independent split.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from Utils.constants import EMOTION_MAP, INTENSITY_MAP, STATEMENT_MAP
from Utils.paths import DATASET_DIR


def parse_ravdess_filename(filename: str) -> dict:
    """Parse a RAVDESS basename into its metadata fields.

    Parameters
    ----------
    filename : str
        Either a bare basename or a full path.

    Returns
    -------
    dict
        Keys: file, emotion, emotion_id, intensity, statement, repetition,
        actor, gender.

    Raises
    ------
    ValueError
        If the name does not follow the 7-field RAVDESS schema. Failing loudly
        here is deliberate: a silently mis-parsed actor id would corrupt the
        speaker-independent split without any visible symptom.
    """
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    parts = stem.split("-")

    if len(parts) != 7:
        raise ValueError(
            f"'{base}' is not a RAVDESS filename (expected 7 hyphen-separated "
            f"fields, got {len(parts)})"
        )

    modality, channel, emo, intensity, statement, repetition, actor = parts

    if emo not in EMOTION_MAP:
        raise ValueError(f"'{base}' has unknown emotion id '{emo}'")

    actor_num = int(actor)

    return {
        "file": base,
        "modality": modality,
        "vocal_channel": channel,
        "emotion_id": emo,
        "emotion": EMOTION_MAP[emo],
        "intensity": INTENSITY_MAP.get(intensity, intensity),
        "statement": STATEMENT_MAP.get(statement, statement),
        "repetition": int(repetition),
        "actor": actor_num,
        # RAVDESS convention: odd actor ids are male, even are female.
        "gender": "male" if actor_num % 2 == 1 else "female",
    }


def get_audio_files(
    dataset_dir: str | os.PathLike | None = None,
    *,
    deduplicate: bool = True,
) -> list[str]:
    """Return sorted paths to the dataset's ``.wav`` files.

    Parameters
    ----------
    dataset_dir : path, optional
        Defaults to ``Utils.paths.DATASET_DIR``.
    deduplicate : bool
        When True (default) only the first path found for each RAVDESS
        basename is kept, so a nested duplicate copy of the corpus cannot
        inflate the dataset. Set to False only if you deliberately want raw
        filesystem contents.

    Returns
    -------
    list[str]
    """
    root = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            f"Expected RAVDESS actor folders under {root}."
        )

    paths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".wav"):
                paths.append(os.path.join(dirpath, name))

    paths.sort()

    if not deduplicate:
        return paths

    seen: set[str] = set()
    unique: list[str] = []
    for p in paths:
        base = os.path.basename(p)
        if base in seen:
            continue
        seen.add(base)
        unique.append(p)

    dropped = len(paths) - len(unique)
    if dropped:
        print(
            f"[dataset_loader] {dropped} duplicate recording(s) ignored "
            f"({len(paths)} files on disk -> {len(unique)} unique utterances). "
            f"Check for a nested second copy of the corpus."
        )

    return unique


def load_dataset_index(
    dataset_dir: str | os.PathLike | None = None,
) -> pd.DataFrame:
    """Build a metadata table for every unique recording.

    Returns
    -------
    pandas.DataFrame
        One row per utterance with columns: path, file, emotion, actor,
        gender, intensity, statement, repetition.
    """
    rows = []
    for path in get_audio_files(dataset_dir):
        meta = parse_ravdess_filename(path)
        meta["path"] = path
        rows.append(meta)

    df = pd.DataFrame(rows)
    cols = [
        "path", "file", "emotion", "emotion_id", "actor", "gender",
        "intensity", "statement", "repetition",
    ]
    return df[cols].sort_values("file").reset_index(drop=True)


def deduplicate_feature_table(
    df: pd.DataFrame,
    key: str = "file",
) -> pd.DataFrame:
    """Drop duplicated utterances from an already-extracted feature CSV.

    Lets you reuse the existing 2880-row CSVs in ``Outputs/`` without spending
    hours re-extracting HuBERT embeddings - the duplicate rows are identical,
    so keeping the first occurrence loses nothing.
    """
    if key not in df.columns:
        raise KeyError(f"column '{key}' not found; got {list(df.columns)[:5]}...")

    before = len(df)
    out = df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    if len(out) != before:
        print(
            f"[dataset_loader] de-duplicated feature table: "
            f"{before} rows -> {len(out)} unique utterances"
        )
    return out


_META_COLUMNS = {
    "file", "path", "emotion", "emotion_id", "actor", "gender",
    "intensity", "statement", "repetition", "modality", "vocal_channel",
}


def resolve_feature_columns(
    df: pd.DataFrame,
    expected: list[str],
    expected_dim: int,
    what: str,
) -> list[str]:
    """Return the feature columns of ``df`` in the correct order.

    Tries, in order:

    1. The exact names from ``Utils.constants``.
    2. The same names with the opposite bark/embedding index base - the CSVs in
       ``Outputs/`` number bark bands from 1 (``bark_1..bark_24``) while HuBERT
       dimensions are numbered from 0, and hard-coding one convention broke on
       the other.
    3. Everything that is not a metadata column, in the order the CSV declares
       them.

    Falling back on positional order is safe here because the feature tables
    were written column-by-column in a fixed order (pitch, loudness, bark), and
    that same order is reproduced by ``Utils.inference.extract_psycho_features``.
    """
    if all(c in df.columns for c in expected):
        return expected

    # Try flipping the index base of any trailing numbered block.
    import re

    m = re.match(r"^(.*?)(\d+)$", expected[-1]) if expected else None
    if m:
        prefix = m.group(1)
        numbered = [c for c in expected if c.startswith(prefix)]
        head = [c for c in expected if not c.startswith(prefix)]
        for base in (1, 0):
            candidate = head + [f"{prefix}{i}" for i in range(base, base + len(numbered))]
            if all(c in df.columns for c in candidate):
                print(
                    f"[dataset_loader] {what}: using {prefix}{base}.."
                    f"{prefix}{base + len(numbered) - 1} numbering from the CSV"
                )
                return candidate

    fallback = [c for c in df.columns if c not in _META_COLUMNS]
    if len(fallback) == expected_dim:
        print(
            f"[dataset_loader] {what}: column names did not match "
            f"Utils.constants; falling back to CSV column order "
            f"({fallback[0]} .. {fallback[-1]})"
        )
        return fallback

    missing = [c for c in expected if c not in df.columns]
    raise KeyError(
        f"cannot resolve {what} columns. Expected {expected_dim}, found "
        f"{len(fallback)} non-metadata columns. Missing names include "
        f"{missing[:5]}. CSV columns start: {list(df.columns)[:8]}"
    )


def attach_metadata(df: pd.DataFrame, key: str = "file") -> pd.DataFrame:
    """Add actor / gender / intensity / statement columns to a feature table."""
    meta = pd.DataFrame([parse_ravdess_filename(f) for f in df[key]])
    meta = meta[["file", "actor", "gender", "intensity", "statement", "repetition"]]
    return df.merge(meta, on=key, how="left", validate="one_to_one")


__all__ = [
    "parse_ravdess_filename",
    "get_audio_files",
    "load_dataset_index",
    "deduplicate_feature_table",
    "attach_metadata",
    "resolve_feature_columns",
]
