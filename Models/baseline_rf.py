"""
baseline_rf.py

MFCC + Random Forest baseline.

Why the 92.36% in Outputs/baseline_results.txt is not usable
------------------------------------------------------------
``02_mfcc_baseline.ipynb`` walked ``Dataset/`` (which holds two copies of the
corpus), extracted MFCCs for all 2880 paths, then split randomly. Every test
recording had an identical twin in the training set, so the Random Forest was
largely being asked to recall memorised vectors. A published MFCC + RF baseline
on speaker-independent RAVDESS is typically in the 45-60% range - 92% is a
signal that something is wrong, not a strong baseline.

``evaluate_baseline`` below runs the same model under the speaker-independent
protocol so the multimodal result has an honest comparison point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from Utils.constants import RANDOM_STATE
from Utils.splits import actor_kfold


def train_random_forest(
    X_train,
    y_train,
    *,
    n_estimators: int = 500,
    random_state: int = RANDOM_STATE,
    class_weight: str | None = "balanced",
) -> RandomForestClassifier:
    """Fit a Random Forest baseline.

    ``class_weight="balanced"`` compensates for RAVDESS's half-sized neutral
    class; the original left it unset.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_baseline(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    n_splits: int = 5,
) -> dict:
    """Speaker-independent cross-validated baseline.

    Parameters
    ----------
    X : (N, D) feature matrix
    y : (N,) integer or string labels
    meta : DataFrame with an ``actor`` column, aligned with ``X``
    """
    accuracies, f1s = [], []

    for fold, (train_idx, test_idx) in enumerate(
        actor_kfold(meta, n_splits=n_splits), 1
    ):
        model = train_random_forest(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        acc = accuracy_score(y[test_idx], pred)
        f1 = f1_score(y[test_idx], pred, average="macro", zero_division=0)
        accuracies.append(acc)
        f1s.append(f1)

        print(f"[baseline fold {fold}] accuracy {acc:.4f} macro-F1 {f1:.4f}")

    acc = np.array(accuracies)
    f1 = np.array(f1s)
    print(f"\n[baseline] accuracy {acc.mean():.4f} +/- {acc.std():.4f}")
    print(f"[baseline] macro-F1 {f1.mean():.4f} +/- {f1.std():.4f}")

    return {
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std()),
        "folds": n_splits,
    }


__all__ = ["train_random_forest", "evaluate_baseline"]
