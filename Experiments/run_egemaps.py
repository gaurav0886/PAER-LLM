"""
run_egemaps.py

The two eGeMAPS rows of Table XIII, on the folds of Table IV.

    python -m Features.egemaps_features      # once, to build the feature table
    python -m Experiments.run_egemaps        # then this

Runs, under protocol P4 (actor-disjoint five-fold CV, validation-selected):

    * eGeMAPS only          -- modality=psycho with an 88-d perceptual stream
    * eGeMAPS + HuBERT      -- gated fusion of the two

Both use the identical folds, encoder depth, latent width and classification
head as every other row of Table XIII, so the comparison against the
34-descriptor set is controlled rather than merely adjacent.

Writes ``Evaluation/egemaps_results.json`` and prints the two table rows
already formatted as LaTeX.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from Experiments.run_all import _base_args, run_cv, summarise
from Features.egemaps_features import load_egemaps_features
from Models.multimodal_model import MultimodalEmotionModel
from Utils.paths import EVALUATION_DIR, ensure_dirs

DEFAULT_SEEDS = [42, 1, 7, 13, 2024]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--cv", type=int, default=5)
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]

    ensure_dirs()
    (EVALUATION_DIR / "predictions").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta, Xe, Xh, y, enc = load_egemaps_features()
    ege_dim = Xe.shape[1]
    print(f"[egemaps] device={device} seeds={seeds} eGeMAPS dim={ege_dim}")

    base = dict(epochs=a.epochs, patience=a.patience, fusion="gated",
                hidden_dim=128, dropout=0.3, lr=1e-3, weight_decay=1e-2,
                batch_size=32, label_smoothing=0.05)

    results, t0 = {"seeds": seeds, "egemaps_dim": ege_dim,
                   "classes": enc.classes_.tolist()}, time.time()

    for tag, modality in [("egemaps_only", "psycho"), ("egemaps_hubert", "both")]:
        print(f"\n=== {tag} (P4, gated) ===")
        runs, preds = [], []
        for s in seeds:
            print(f"  seed {s}")
            fo, pr = run_cv(Xe, Xh, y, meta,
                            _base_args(**{**base, "modality": modality}),
                            device, s, n_splits=a.cv)
            runs += fo
            preds += pr
        results[tag] = {k: summarise(runs, k)
                        for k in ("accuracy", "macro_f1", "uar")}
        results[tag]["runs"] = runs
        results[tag]["n_params"] = int(
            MultimodalEmotionModel(psycho_dim=ege_dim, fusion="gated",
                                   modality=modality).n_trainable()
        )
        json.dump(preds, open(EVALUATION_DIR / "predictions" / f"p4_{tag}.json", "w"))

    results["elapsed_min"] = round((time.time() - t0) / 60, 1)
    out = EVALUATION_DIR / "egemaps_results.json"
    json.dump(results, open(out, "w"), indent=2)

    # ---- Table XIII rows, ready to paste ----------------------------------
    print("\n" + "=" * 70)
    print("Table XIII rows (paste over the two \\RUN eGeMAPS lines)")
    print("=" * 70)
    label = {"egemaps_only": f"eGeMAPS only             & {ege_dim}",
             "egemaps_hubert": f"eGeMAPS + HuBERT (gated) & {ege_dim}+768"}
    for tag in ("egemaps_only", "egemaps_hubert"):
        b = results[tag]
        print(f"{label[tag]:<32s} & {b['accuracy']['mean']:.2f} "
              f"& {b['macro_f1']['mean']:.2f} & {b['uar']['mean']:.2f} "
              f"& {b['n_params']:,} \\\\".replace(",", "{,}"))
        print(f"%   SD  acc {b['accuracy']['sd']:.2f}  F1 {b['macro_f1']['sd']:.2f}"
              f"  UAR {b['uar']['sd']:.2f}   n={b['accuracy']['n']}")
    print(f"\nWrote {out}  ({results['elapsed_min']} min)")


if __name__ == "__main__":
    main()
