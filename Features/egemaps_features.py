"""
egemaps_features.py

eGeMAPSv02 extraction (88 functionals) via openSMILE, and a loader that returns
the same five-tuple contract as ``Training.train.load_features``.

Why this exists
---------------
Section II-B of the manuscript cites GeMAPS/eGeMAPS as the standardised
parameter set for perceptually motivated descriptors, and then uses a bespoke
34-descriptor set. The first question a reviewer asks is "why not eGeMAPS-88,
and does your set beat it?". Table XIII answers it on identical folds, which
requires the feature table this module produces.

eGeMAPS also carries the voice-quality measures our 34 descriptors lack
(spectral tilt, harmonics-to-noise ratio, jitter, shimmer). Section VIII-D
predicts that those are what the *happy* class needs, so the eGeMAPS row is
simultaneously a benchmark and a test of that prediction.

Usage
-----
    pip install opensmile                 # brings its own compiled binary
    python -m Features.egemaps_features   # writes Outputs/egemaps_features.csv

The CSV schema matches ``psychoacoustic_features.csv``: a ``file`` column, an
``emotion`` column, then one column per feature. Downstream code therefore
needs no special case -- ``MultimodalEmotionModel`` infers ``psycho_dim`` from
the data (``Training/train.py``, ``psycho_dim=train_ds.psycho_dim``), so an
88-column table trains without any change to the model.

Determinism and integrity
-------------------------
Files are enumerated through ``Utils.dataset_loader.get_audio_files``, which
de-duplicates by RAVDESS basename. That is the same guard as
Algorithm 1 in the paper: if the corpus is unpacked twice, this module still
produces 1440 rows, not 2880.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from Utils.constants import HUBERT_COLUMNS, HUBERT_DIM
from Utils.dataset_loader import get_audio_files, parse_ravdess_filename
from Utils.paths import HUBERT_CSV, OUTPUTS_DIR, ensure_dirs

EGEMAPS_CSV = OUTPUTS_DIR / "egemaps_features.csv"
EGEMAPS_DIM = 88          # eGeMAPSv02 Functionals
FEATURE_SET = "eGeMAPSv02"


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def extract(output_csv: Path = EGEMAPS_CSV, *, overwrite: bool = False) -> Path:
    """Extract eGeMAPSv02 functionals for every unique RAVDESS utterance."""
    try:
        import opensmile
    except ImportError:  # pragma: no cover
        sys.exit(
            "opensmile is not installed. Run:\n"
            "    pip install opensmile\n"
            "It ships its own compiled extractor; no separate openSMILE build "
            "is required."
        )

    output_csv = Path(output_csv)
    if output_csv.exists() and not overwrite:
        print(f"[egemaps] {output_csv} already exists; pass --overwrite to redo")
        return output_csv

    files = get_audio_files()
    if not files:
        sys.exit("[egemaps] no audio files found under Dataset/")
    print(f"[egemaps] {len(files)} unique utterances (de-duplicated by basename)")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers=1,          # deterministic ordering
    )

    rows, names = [], None
    for i, path in enumerate(sorted(map(str, files)), 1):
        meta = parse_ravdess_filename(path)
        try:
            frame = smile.process_file(path)
        except Exception as exc:                      # noqa: BLE001
            print(f"[egemaps] FAILED {meta['file']}: {exc}")
            continue
        values = frame.to_numpy(dtype=np.float64).ravel()
        if names is None:
            names = [c.replace("_", "-") for c in frame.columns]
            if len(names) != EGEMAPS_DIM:
                print(f"[egemaps] warning: expected {EGEMAPS_DIM} features, "
                      f"got {len(names)}; using the actual count")
        rows.append({"file": meta["file"], "emotion": meta["emotion"],
                     **dict(zip(names, values))})
        if i % 100 == 0 or i == len(files):
            print(f"[egemaps]   {i}/{len(files)}")

    df = pd.DataFrame(rows)
    if df["file"].duplicated().any():                 # integrity, Algorithm 1
        raise RuntimeError("[egemaps] duplicate basenames survived extraction")

    ensure_dirs()
    df.to_csv(output_csv, index=False)
    print(f"[egemaps] wrote {output_csv}  ({df.shape[0]} rows x "
          f"{df.shape[1] - 2} features)")
    return output_csv


# ---------------------------------------------------------------------------
# loading, mirroring Training.train.load_features
# ---------------------------------------------------------------------------
def egemaps_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns of an eGeMAPS table: everything except the metadata."""
    meta_cols = {"file", "emotion", "emotion_id", "actor", "gender",
                 "intensity", "statement", "repetition", "modality",
                 "vocal_channel"}
    return [c for c in df.columns if c not in meta_cols]


def load_egemaps_features(egemaps_csv: Path = EGEMAPS_CSV):
    """Return ``(meta, X_ege, X_hubert, y, encoder)``.

    The same five-tuple as ``Training.train.load_features``, so every
    downstream routine that accepts that contract accepts this one. ``X_ege``
    takes the place of ``X_psycho``; the model infers its input width from the
    array, so nothing else needs to know the dimensionality changed.
    """
    egemaps_csv = Path(egemaps_csv)
    if not egemaps_csv.exists():
        sys.exit(f"{egemaps_csv} not found. Run:\n"
                 f"    python -m Features.egemaps_features")

    ege = pd.read_csv(egemaps_csv).drop_duplicates(subset="file", keep="first")
    hub = pd.read_csv(HUBERT_CSV).drop_duplicates(subset="file", keep="first")

    merged = ege.merge(
        hub.drop(columns=[c for c in ("emotion",) if c in hub.columns]),
        on="file", how="inner", validate="one_to_one",
    )
    if len(merged) != len(ege):
        raise RuntimeError(
            f"feature tables do not align: egemaps={len(ege)}, "
            f"hubert={len(hub)}, merged={len(merged)}"
        )

    ege_cols = egemaps_columns(ege)
    hub_cols = [c for c in HUBERT_COLUMNS if c in merged.columns]
    if len(hub_cols) != HUBERT_DIM:
        hub_cols = [c for c in merged.columns if c.startswith("hubert")]

    meta = pd.DataFrame([parse_ravdess_filename(f) for f in merged["file"]])
    meta = meta.reset_index(drop=True)

    X_ege = merged[ege_cols].to_numpy(dtype=np.float32)
    X_hub = merged[hub_cols].to_numpy(dtype=np.float32)

    # Guard against non-finite functionals, which openSMILE can emit for a
    # near-silent frame; a single NaN would silently poison the standardiser.
    n_bad = int((~np.isfinite(X_ege)).sum())
    if n_bad:
        print(f"[egemaps] replacing {n_bad} non-finite values with column means")
        col_mean = np.nanmean(np.where(np.isfinite(X_ege), X_ege, np.nan), axis=0)
        bad = ~np.isfinite(X_ege)
        X_ege[bad] = np.take(col_mean, np.where(bad)[1])

    encoder = LabelEncoder()
    y = encoder.fit_transform(merged["emotion"].to_numpy())

    print(f"[egemaps] loaded {len(merged)} utterances, "
          f"{X_ege.shape[1]} eGeMAPS features + {X_hub.shape[1]} HuBERT dims")
    return meta, X_ege, X_hub, y, encoder


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Extract eGeMAPSv02 functionals")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default=str(EGEMAPS_CSV))
    a = ap.parse_args(argv)
    extract(Path(a.out), overwrite=a.overwrite)


if __name__ == "__main__":
    main()
