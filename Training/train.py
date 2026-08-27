"""
train.py

Reproducible training entry point for PAER-LLM.

Run from the project root:

    python -m Training.train                       # speaker-independent, default
    python -m Training.train --fusion concat       # ablation
    python -m Training.train --split random        # the OLD (leaky) protocol
    python -m Training.train --cv 5                # 5-fold actor-disjoint CV

What this fixes relative to 09_training.ipynb
---------------------------------------------
1. **Duplicate utterances removed.** The source CSVs contain 2880 rows for 1440
   distinct recordings. Every row had an identical twin, so under a random
   split a recording's clone sat in the test set. Rows are de-duplicated on
   ``file`` before anything else happens.

2. **Scaling no longer leaks.** ``07_multimodal_feature_fusion.ipynb`` calls
   ``StandardScaler().fit_transform()`` on the *entire* dataset and writes the
   scaled result to ``multimodal_features.csv``; training then splits that file.
   The test set's mean and variance were therefore baked into the training
   inputs. This script reads the **unscaled** ``psychoacoustic_features.csv``
   and ``hubert_embeddings.csv`` and fits the scalers on the training fold only.
   ``multimodal_features.csv`` should be treated as contaminated and not used
   for any reported result.

3. **Speaker-independent split.** Actors in train, validation and test are
   disjoint (see ``Utils/splits.py``).

4. **Model selection no longer happens on the test set.** The notebook saved
   the checkpoint with the best accuracy on ``test_loader`` and then reported
   that same accuracy as the final result - the classic optimistic bias. There
   is now a separate validation fold for early stopping and checkpointing; the
   test fold is touched exactly once, by ``Evaluation/evaluate.py``.

5. **Class imbalance handled.** RAVDESS "neutral" has half the utterances of
   every other class. Cross-entropy is class-weighted and the reported headline
   metric is macro-F1 / unweighted average recall alongside accuracy.

6. **Reproducibility.** All seeds set, the exact actor assignment and the full
   config are written next to the checkpoint.

Expect lower numbers than the 95.14% in Evaluation/classification_report.txt.
The previous figure measured the model's ability to recognise recordings it had
already seen a duplicate of, from speakers it had already heard.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

from Models.multimodal_model import MultimodalEmotionModel
from Training.emotion_dataset import EmotionDataset
from Utils.constants import (
    HUBERT_COLUMNS,
    HUBERT_DIM,
    NUM_CLASSES,
    PSYCHO_COLUMNS,
    PSYCHO_DIM,
    RANDOM_STATE,
)
from Utils.dataset_loader import (
    attach_metadata,
    deduplicate_feature_table,
    resolve_feature_columns,
)
from Utils.paths import (
    EVALUATION_DIR,
    HUBERT_CSV,
    MODELS_DIR,
    PSYCHO_CSV,
    ensure_dirs,
)
from Utils.splits import actor_kfold, actor_split, save_split


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_features(
    keep_duplicates: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Load, de-duplicate and align the raw (unscaled) feature tables.

    Parameters
    ----------
    keep_duplicates : bool
        If True, re-introduce the duplicated rows after alignment, reproducing
        the contaminated dataset the original notebooks trained on. Combined
        with ``--split random`` this reproduces the original protocol exactly
        and lets you quantify how much of the reported accuracy was leakage.
        Never use this for a reported result.
    """
    psycho_raw = pd.read_csv(PSYCHO_CSV)
    hubert_raw = pd.read_csv(HUBERT_CSV)

    psycho = deduplicate_feature_table(psycho_raw)
    hubert = deduplicate_feature_table(hubert_raw)

    # How many identical copies of the corpus are present on disk.
    dup_factor = max(1, int(round(len(psycho_raw) / max(len(psycho), 1))))

    merged = psycho.merge(
        hubert.drop(columns=[c for c in ("emotion",) if c in hubert.columns]),
        on="file",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(psycho):
        raise RuntimeError(
            f"feature tables do not align: psycho={len(psycho)}, "
            f"hubert={len(hubert)}, merged={len(merged)}"
        )

    # Resolve column names against the actual CSVs before adding metadata
    # columns, so the fallback cannot pick up 'actor', 'gender' etc.
    psycho_cols = resolve_feature_columns(
        psycho, PSYCHO_COLUMNS, PSYCHO_DIM, "psychoacoustic"
    )
    hubert_cols = resolve_feature_columns(
        hubert, HUBERT_COLUMNS, HUBERT_DIM, "hubert"
    )

    merged = attach_metadata(merged)

    X_psycho = merged[psycho_cols].to_numpy(dtype=np.float32)
    X_hubert = merged[hubert_cols].to_numpy(dtype=np.float32)

    encoder = LabelEncoder()
    y = encoder.fit_transform(merged["emotion"].to_numpy())

    if keep_duplicates and dup_factor > 1:
        X_psycho = np.tile(X_psycho, (dup_factor, 1))
        X_hubert = np.tile(X_hubert, (dup_factor, 1))
        y = np.tile(y, dup_factor)
        merged = pd.concat([merged] * dup_factor, ignore_index=True)
        print(
            f"[data] LEAKAGE MODE: duplicated x{dup_factor} -> {len(merged)} rows. "
            "Every utterance now has an identical twin. Results from this run "
            "are contaminated by construction and exist only for comparison."
        )

    print(
        f"[data] {len(merged)} rows | "
        f"{merged['actor'].nunique()} actors | "
        f"psycho {X_psycho.shape} | hubert {X_hubert.shape}"
    )
    print("[data] class counts:")
    print(merged["emotion"].value_counts().sort_index().to_string())

    return merged, X_psycho, X_hubert, y, encoder


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over ``loader``. Trains when ``optimizer`` is given."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    preds: list[int] = []
    truth: list[int] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for psycho, hubert, labels in loader:
            psycho = psycho.to(device, non_blocking=True)
            hubert = hubert.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, _, _ = model(psycho, hubert)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Grad clipping was absent; attention layers on a small dataset
                # can produce occasional large steps.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            truth.extend(labels.cpu().tolist())

    n = len(truth)
    return {
        "loss": total_loss / n,
        "accuracy": float(np.mean(np.array(preds) == np.array(truth))),
        "macro_f1": f1_score(truth, preds, average="macro", zero_division=0),
        "uar": recall_score(truth, preds, average="macro", zero_division=0),
    }


def fit_fold(
    X_psycho, X_hubert, y,
    train_idx, val_idx,
    args,
    device,
    checkpoint_path: Path | None,
    select_idx=None,
    seed: int | None = None,
):
    """Scale, train and early-stop on one fold.

    Parameters
    ----------
    select_idx : array-like, optional
        Indices of the partition used for early stopping and checkpointing.
        Defaults to ``val_idx``. Passing the TEST indices here reproduces the
        undisclosed practice of selecting the model on the test partition
        (protocols P2t / P3t); it is available so the resulting optimism can be
        measured, and must never be used for a reported result.
    """
    if seed is not None:
        set_seed(seed)
    if select_idx is None:
        select_idx = val_idx
    psycho_scaler = StandardScaler().fit(X_psycho[train_idx])
    hubert_scaler = StandardScaler().fit(X_hubert[train_idx])

    def prep(idx):
        return EmotionDataset(
            psycho_scaler.transform(X_psycho[idx]),
            hubert_scaler.transform(X_hubert[idx]),
            y[idx],
        )

    train_ds, val_ds = prep(train_idx), prep(select_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MultimodalEmotionModel(
        psycho_dim=train_ds.psycho_dim,
        hubert_dim=train_ds.hubert_dim,
        hidden_dim=args.hidden_dim,
        num_classes=NUM_CLASSES,
        fusion=args.fusion,
        dropout=args.dropout,
        modality=args.modality,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=train_ds.class_weights(NUM_CLASSES).to(device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_score = -np.inf
    best_state = None
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"  epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
            f"macroF1 {val_metrics['macro_f1']:.4f}"
        )

        # Selection on macro-F1, not accuracy: accuracy rewards the model for
        # ignoring the under-represented neutral class.
        score = val_metrics["macro_f1"]
        if score > best_score + 1e-5:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch} (best macro-F1 {best_score:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if checkpoint_path is not None:
        model.save(checkpoint_path)
        joblib.dump(psycho_scaler, MODELS_DIR / "psycho_scaler.pkl")
        joblib.dump(hubert_scaler, MODELS_DIR / "hubert_scaler.pkl")

    return model, (psycho_scaler, hubert_scaler), history, best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the PAER-LLM multimodal model")
    p.add_argument("--fusion", default="cross_attention",
                   choices=["cross_attention", "gated", "concat"])
    p.add_argument("--split", default="speaker", choices=["speaker", "random"],
                   help="'random' reproduces the original leaky protocol for "
                        "comparison only - do not report it as a result")
    p.add_argument("--cv", type=int, default=0,
                   help="if > 0, run k-fold actor-disjoint cross-validation")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--seeds", default=None,
                   help="comma-separated seed list, e.g. 42,1,7,13,2024. "
                        "Every condition is repeated for each seed and results "
                        "are aggregated over (fold x seed) runs, so that "
                        "initialisation variance is measured rather than "
                        "assumed away.")
    p.add_argument("--select-on", default="val", choices=["val", "test"],
                   help="partition used for early stopping and checkpointing. "
                        "'test' reproduces undisclosed test-set model selection "
                        "(protocols P2t/P3t) - for the decomposition ONLY.")
    p.add_argument("--modality", default="both",
                   choices=["both", "psycho", "hubert"],
                   help="ablate a modality: 'psycho' = 34 psychoacoustic "
                        "features only, 'hubert' = 768-d HuBERT only")
    p.add_argument("--keep-duplicates", action="store_true",
                   help="re-introduce duplicate utterances to reproduce the "
                        "original contaminated protocol - for the leakage "
                        "comparison table ONLY, never for a reported result")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    ensure_dirs()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} fusion={args.fusion} split={args.split} "
          f"modality={args.modality}")

    meta, X_psycho, X_hubert, y, encoder = load_features(
        keep_duplicates=args.keep_duplicates
    )
    joblib.dump(encoder, MODELS_DIR / "label_encoder.pkl")

    tag = f"{args.fusion}_{args.split}"
    if args.modality != "both":
        tag += f"_{args.modality}only"
    if args.keep_duplicates:
        tag += "_dup"
    (MODELS_DIR / "eval_context.json").write_text(
        json.dumps({"keep_duplicates": bool(args.keep_duplicates),
                    "split": args.split, "fusion": args.fusion, "tag": tag},
                   indent=2),
        encoding="utf-8",
    )

    started = time.time()

    if args.cv > 0:
        scores = []
        for fold, (tr, te) in enumerate(actor_kfold(meta, n_splits=args.cv, random_state=args.seed), 1):
            # Carve a validation set out of the training actors only.
            sub = meta.iloc[tr].reset_index(drop=True)
            inner = actor_split(sub, test_size=0.0001, val_size=0.2, random_state=args.seed)
            tr_idx = tr[np.concatenate([inner["train"], inner["test"]])]
            va_idx = tr[inner["val"]]

            print(f"\n[fold {fold}/{args.cv}] train={len(tr_idx)} val={len(va_idx)} test={len(te)}")
            model, scalers, _, _ = fit_fold(
                X_psycho, X_hubert, y, tr_idx, va_idx, args, device, None
            )

            ps, hs = scalers
            test_ds = EmotionDataset(
                ps.transform(X_psycho[te]), hs.transform(X_hubert[te]), y[te]
            )
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            m = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
            print(f"[fold {fold}] test acc {m['accuracy']:.4f} macroF1 {m['macro_f1']:.4f} UAR {m['uar']:.4f}")
            scores.append(m)

        acc = np.array([s["accuracy"] for s in scores])
        f1 = np.array([s["macro_f1"] for s in scores])
        uar = np.array([s["uar"] for s in scores])
        print(f"\n[cv] === {tag} ===")
        print(f"[cv] accuracy {acc.mean()*100:.2f} +/- {acc.std(ddof=1)*100:.2f}")
        print(f"[cv] macro-F1 {f1.mean()*100:.2f} +/- {f1.std(ddof=1)*100:.2f}")
        print(f"[cv] UAR      {uar.mean()*100:.2f} +/- {uar.std(ddof=1)*100:.2f}")

        (EVALUATION_DIR / f"cv_results_{tag}.json").write_text(
            json.dumps({"folds": scores,
                        "accuracy_mean": float(acc.mean()),
                        "accuracy_std": float(acc.std(ddof=1)),
                        "macro_f1_mean": float(f1.mean()),
                        "macro_f1_std": float(f1.std())}, indent=2),
            encoding="utf-8",
        )
        return

    # ---- single split ----
    if args.split == "speaker":
        split = actor_split(meta, random_state=args.seed)
        train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]
        print(f"[split] actors -> {split['actors']}")
        save_split(split, MODELS_DIR / "split_actors.json")
    else:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=args.seed, stratify=y
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, random_state=args.seed, stratify=y[train_idx]
        )
        print("[split] WARNING: random utterance split - speaker-dependent, "
              "for comparison against the old protocol only")
        np.save(MODELS_DIR / "test_idx.npy", test_idx)

    print(f"[split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    model, scalers, history, best_val = fit_fold(
        X_psycho, X_hubert, y, train_idx, val_idx, args, device,
        MODELS_DIR / "best_model.pth",
    )

    np.save(MODELS_DIR / "test_idx.npy", test_idx)

    config = vars(args) | {
        "best_val_macro_f1": float(best_val),
        "n_utterances": int(len(y)),
        "classes": encoder.classes_.tolist(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (MODELS_DIR / f"train_config_{tag}.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (EVALUATION_DIR / f"training_history_{tag}.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    print(f"\n[done] best validation macro-F1 {best_val:.4f}")
    print("[done] run `python -m Evaluation.evaluate` for the held-out test result")


if __name__ == "__main__":
    main()
