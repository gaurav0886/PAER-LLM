"""
run_all.py

Single-command driver for the complete experimental grid required by the
revised manuscript. Run from the project root:

    python -m Experiments.run_all

It writes one consolidated file, ``Evaluation/all_results.json``, containing
every number the paper's tables need.

What it runs
------------
1. **The six-rung protocol ladder** (Table 1), all with gated fusion, all with
   every seed:

   =====  ==========  ================  ================  ==================
   Rung   Duplicates  Partition         Model selection   Reporting
   =====  ==========  ================  ================  ==================
   P1     present     random 80/20      validation        single split
   P2     removed     random 80/20      validation        single split
   P2t    removed     random 80/20      **test**          single split
   P3     removed     actor-disjoint    validation        single split
   P3t    removed     actor-disjoint    **test**          single split
   P4     removed     actor-disjoint    validation        5-fold mean
   =====  ==========  ================  ================  ==================

   This yields four non-overlapping increments: duplication (P1-P2),
   speaker-dependence (P2-P3), test-set selection (P2t-P2 and P3t-P3), and
   single-split optimism (P3-P4).

2. **The fusion ablation** (Table 2): concat / gated / cross-attention, under
   P4, every seed. All three now emit an identical 128-d representation, so
   the classification head has identical capacity and only the fusion operator
   differs.

3. **The modality ablation** (Table 4): psychoacoustic-only, HuBERT-only and
   fused, under P4, every seed. The HuBERT-only condition doubles as the
   in-harness frozen-HuBERT baseline, so the paper never has to compare
   against a figure obtained under someone else's protocol.

4. **Per-fold, per-seed raw predictions**, written to
   ``Evaluation/predictions/``, from which every table and figure in the paper
   is regenerated. Per-class and per-speaker analyses are computed from these
   pooled cross-validated predictions rather than from a single split.

Cost
----
On cached features this is CPU/GPU-light: roughly
(6 rungs + 3 fusions + 3 modalities) x n_seeds x folds short runs. With the
default five seeds expect 30-60 minutes on a consumer GPU. Use ``--seeds 42``
for a quick smoke test first.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from Models.multimodal_model import MultimodalEmotionModel
from Training.emotion_dataset import EmotionDataset
from Training.train import fit_fold, load_features, parse_args, run_epoch, set_seed
from Utils.constants import NUM_CLASSES, TEST_SIZE, VAL_SIZE
from Utils.paths import EVALUATION_DIR, MODELS_DIR, ensure_dirs
from Utils.splits import actor_kfold, actor_split

DEFAULT_SEEDS = [42, 1, 7, 13, 2024]


# ---------------------------------------------------------------------------
def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "uar": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _predict(model, Xp, Xh, idx, ps, hs, device):
    model.eval()
    with torch.no_grad():
        p = torch.as_tensor(ps.transform(Xp[idx]), dtype=torch.float32).to(device)
        h = torch.as_tensor(hs.transform(Xh[idx]), dtype=torch.float32).to(device)
        logits, _, _ = model(p, h)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def _base_args(**over):
    a = parse_args([])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _random_split(y, seed, dup_note=""):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=seed, stratify=y)
    tr, va = train_test_split(tr, test_size=VAL_SIZE, random_state=seed, stratify=y[tr])
    return tr, va, te


# ---------------------------------------------------------------------------
def run_single(Xp, Xh, y, tr, va, te, args, device, seed, select_on="val"):
    """Train one single-split condition and return test metrics + predictions."""
    select_idx = te if select_on == "test" else va
    model, (ps, hs), _hist, _best = fit_fold(
        Xp, Xh, y, tr, va, args, device, None, select_idx=select_idx, seed=seed
    )
    probs = _predict(model, Xp, Xh, te, ps, hs, device)
    m = _metrics(y[te], probs.argmax(1))
    m["n_test"] = int(len(te))
    m["n_params"] = int(model.n_trainable())
    return m, probs, te


def run_cv(Xp, Xh, y, meta, args, device, seed, n_splits=5, select_on="val"):
    """Actor-disjoint CV for one seed. Returns per-fold metrics and predictions."""
    folds, preds = [], []
    for k, (tr, te) in enumerate(actor_kfold(meta, n_splits=n_splits, random_state=seed), 1):
        sub = meta.iloc[tr].reset_index(drop=True)
        inner = actor_split(sub, test_size=0.0001, val_size=0.2, random_state=seed)
        tr_idx = tr[np.concatenate([inner["train"], inner["test"]])]
        va_idx = tr[inner["val"]]
        select_idx = te if select_on == "test" else va_idx

        model, (ps, hs), _h, _b = fit_fold(
            Xp, Xh, y, tr_idx, va_idx, args, device, None,
            select_idx=select_idx, seed=seed,
        )
        probs = _predict(model, Xp, Xh, te, ps, hs, device)
        m = _metrics(y[te], probs.argmax(1))
        m.update(fold=k, seed=seed, n_test=int(len(te)),
                 test_actors=sorted(int(a) for a in meta.iloc[te]["actor"].unique()))
        folds.append(m)
        preds.append({"fold": k, "seed": seed, "idx": te.tolist(),
                      "probs": probs.tolist(), "y_true": y[te].tolist()})
        print(f"    fold {k}: acc {m['accuracy']*100:5.2f}  F1 {m['macro_f1']*100:5.2f}")
    return folds, preds


def summarise(runs, key):
    v = np.array([r[key] for r in runs]) * 100
    n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    from scipy import stats
    tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    return {
        "mean": float(v.mean()), "sd": sd, "n": n,
        "ci95_of_mean": [float(v.mean() - tcrit * sd / np.sqrt(n)),
                         float(v.mean() + tcrit * sd / np.sqrt(n))],
        "values": v.tolist(),
    }


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--skip-ladder", action="store_true")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]

    ensure_dirs()
    (EVALUATION_DIR / "predictions").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}  seeds={seeds}")

    meta, Xp, Xh, y, enc = load_features()
    meta_d, Xp_d, Xh_d, y_d, _ = load_features(keep_duplicates=True)

    base = dict(epochs=a.epochs, patience=a.patience, fusion="gated",
                modality="both", hidden_dim=128, dropout=0.3, lr=1e-3,
                weight_decay=1e-2, batch_size=32, label_smoothing=0.05)
    results, t0 = {"seeds": seeds, "classes": enc.classes_.tolist()}, time.time()

    # ---- 1. protocol ladder -------------------------------------------------
    if not a.skip_ladder:
        print("\n=== PROTOCOL LADDER (gated fusion) ===")
        ladder = {}
        for rung, dup, part, sel in [
            ("P1", True, "random", "val"),
            ("P2", False, "random", "val"),
            ("P2t", False, "random", "test"),
            ("P3", False, "speaker", "val"),
            ("P3t", False, "speaker", "test"),
        ]:
            print(f"\n-- {rung}: duplicates={dup} partition={part} select_on={sel}")
            runs = []
            for s in seeds:
                XP, XH, Y, MT = (Xp_d, Xh_d, y_d, meta_d) if dup else (Xp, Xh, y, meta)
                if part == "random":
                    tr, va, te = _random_split(Y, s)
                else:
                    sp = actor_split(MT, random_state=s)
                    tr, va, te = sp["train"], sp["val"], sp["test"]
                m, _p, _t = run_single(XP, XH, Y, tr, va, te,
                                       _base_args(**base), device, s, select_on=sel)
                runs.append(m)
                print(f"    seed {s}: acc {m['accuracy']*100:5.2f}  F1 {m['macro_f1']*100:5.2f}")
            ladder[rung] = {k: summarise(runs, k) for k in ("accuracy", "macro_f1", "uar")}
            ladder[rung]["runs"] = runs
        results["ladder"] = ladder

    # ---- 2. fusion ablation + P4 -------------------------------------------
    print("\n=== FUSION ABLATION (P4) ===")
    fusion = {}
    for f in ["concat", "gated", "cross_attention"]:
        print(f"\n-- fusion={f}")
        allruns, allpreds = [], []
        for s in seeds:
            print(f"  seed {s}")
            fo, pr = run_cv(Xp, Xh, y, meta, _base_args(**{**base, "fusion": f}),
                            device, s, n_splits=a.cv)
            allruns += fo
            allpreds += pr
        fusion[f] = {k: summarise(allruns, k) for k in ("accuracy", "macro_f1", "uar")}
        fusion[f]["runs"] = allruns
        fusion[f]["n_params"] = int(MultimodalEmotionModel(fusion=f).n_trainable())
        json.dump(allpreds, open(EVALUATION_DIR / "predictions" / f"p4_{f}.json", "w"))
    results["fusion"] = fusion
    if "P4" not in results.get("ladder", {}):
        results.setdefault("ladder", {})["P4"] = fusion["gated"]

    # ---- 3. modality ablation ----------------------------------------------
    print("\n=== MODALITY ABLATION (P4, gated) ===")
    modality = {}
    for md in ["psycho", "hubert"]:
        print(f"\n-- modality={md}")
        allruns = []
        for s in seeds:
            fo, _ = run_cv(Xp, Xh, y, meta, _base_args(**{**base, "modality": md}),
                           device, s, n_splits=a.cv)
            allruns += fo
        modality[md] = {k: summarise(allruns, k) for k in ("accuracy", "macro_f1", "uar")}
        modality[md]["runs"] = allruns
        modality[md]["n_params"] = int(MultimodalEmotionModel(modality=md).n_trainable())
    modality["both"] = fusion["gated"]
    results["modality"] = modality

    results["elapsed_min"] = round((time.time() - t0) / 60, 1)
    out = EVALUATION_DIR / "all_results.json"
    json.dump(results, open(out, "w"), indent=2)

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 68)
    print("SUMMARY  (mean +/- SD over runs; CI is 95% CI of the mean)")
    print("=" * 68)
    for name, block in [("LADDER", results.get("ladder", {})),
                        ("FUSION", results["fusion"]),
                        ("MODALITY", results["modality"])]:
        print(f"\n{name}")
        for k, v in block.items():
            acc, f1 = v["accuracy"], v["macro_f1"]
            print(f"  {k:16s} acc {acc['mean']:6.2f} +/- {acc['sd']:4.2f} "
                  f"[{acc['ci95_of_mean'][0]:.2f}, {acc['ci95_of_mean'][1]:.2f}]  "
                  f"F1 {f1['mean']:6.2f} +/- {f1['sd']:4.2f}  (n={acc['n']})")
    print(f"\nWrote {out}  ({results['elapsed_min']} min)")


if __name__ == "__main__":
    main()
