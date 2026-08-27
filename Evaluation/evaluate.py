"""
evaluate.py

Held-out evaluation for PAER-LLM. Run **once**, after training:

    python -m Evaluation.evaluate

What this fixes relative to 10_evaluation.ipynb
-----------------------------------------------
1. The notebook re-derived the split with ``train_test_split(random_state=42)``
   and evaluated the checkpoint that had itself been selected on that same
   split. The number it produced was a *model-selection* score presented as a
   test score. This script loads the exact test indices saved by training and
   evaluates a checkpoint that never saw them.

2. The notebook reported accuracy only. On RAVDESS, where "neutral" has half
   the utterances of every other class, accuracy overstates performance.
   Macro-F1 and unweighted average recall (UAR) are the standard metrics in
   speech-emotion papers and are reported here alongside a bootstrap
   confidence interval - a single point estimate on 288 test utterances has a
   margin of roughly +/- 5 points, which matters a great deal when comparing
   against published baselines.

3. Adds a per-speaker breakdown and a gender breakdown. If accuracy varies
   wildly across held-out actors, the single-split number is not trustworthy
   and you should report cross-validated results instead
   (``python -m Training.train --cv 5``).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from Models.multimodal_model import MultimodalEmotionModel
from Training.train import load_features
from Utils.paths import (
    BEST_MODEL,
    EVALUATION_DIR,
    MODELS_DIR,
    ensure_dirs,
)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a classification metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main() -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ctx_path = MODELS_DIR / "eval_context.json"
    ctx = json.loads(ctx_path.read_text(encoding="utf-8")) if ctx_path.exists() else {}
    keep_duplicates = bool(ctx.get("keep_duplicates", False))
    if keep_duplicates:
        print(
            "[eval] WARNING: the loaded checkpoint was trained in LEAKAGE MODE "
            "(duplicated utterances). These numbers are contaminated by "
            "construction and belong only in the leakage comparison table."
        )

    meta, X_psycho, X_hubert, y, _ = load_features(keep_duplicates=keep_duplicates)
    encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    psycho_scaler = joblib.load(MODELS_DIR / "psycho_scaler.pkl")
    hubert_scaler = joblib.load(MODELS_DIR / "hubert_scaler.pkl")

    test_idx_path = MODELS_DIR / "test_idx.npy"
    if not test_idx_path.exists():
        raise FileNotFoundError(
            "Models/test_idx.npy not found - run `python -m Training.train` "
            "first. Do not re-derive the split here; that is what made the "
            "original evaluation optimistic."
        )
    test_idx = np.load(test_idx_path)

    model = MultimodalEmotionModel.load(BEST_MODEL, map_location=device).to(device)
    model.eval()

    psycho = torch.as_tensor(
        psycho_scaler.transform(X_psycho[test_idx]), dtype=torch.float32
    ).to(device)
    hubert = torch.as_tensor(
        hubert_scaler.transform(X_hubert[test_idx]), dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        logits, _, _ = model(psycho, hubert)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    y_true = y[test_idx]
    y_pred = probs.argmax(axis=1)
    classes = list(encoder.classes_)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)

    acc_ci = bootstrap_ci(y_true, y_pred, accuracy_score)
    f1_ci = bootstrap_ci(
        y_true, y_pred, lambda a, b: f1_score(a, b, average="macro", zero_division=0)
    )

    print("\n=== Held-out test set ===")
    print(f"utterances : {len(test_idx)}")
    print(f"actors     : {sorted(meta.iloc[test_idx]['actor'].unique().tolist())}")
    print(f"accuracy   : {accuracy*100:.2f}%  (95% CI {acc_ci[0]*100:.2f} - {acc_ci[1]*100:.2f})")
    print(f"macro-F1   : {macro_f1*100:.2f}%  (95% CI {f1_ci[0]*100:.2f} - {f1_ci[1]*100:.2f})")
    print(f"UAR        : {uar*100:.2f}%")
    print(f"chance     : {100/len(classes):.2f}%")

    report = classification_report(
        y_true, y_pred, target_names=classes, zero_division=0, digits=3
    )
    print("\n" + report)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df.to_string())

    # Per-speaker and per-gender breakdown
    test_meta = meta.iloc[test_idx].copy()
    test_meta["correct"] = (y_true == y_pred)
    by_actor = test_meta.groupby("actor")["correct"].mean().round(3)
    by_gender = test_meta.groupby("gender")["correct"].mean().round(3)

    print("\nAccuracy by held-out actor:")
    print(by_actor.to_string())
    print("\nAccuracy by gender:")
    print(by_gender.to_string())

    spread = float(by_actor.max() - by_actor.min())
    if spread > 0.15:
        print(
            f"\nNOTE: accuracy varies by {spread*100:.1f} points across held-out "
            "actors. A single split is not a stable estimate - report 5-fold "
            "actor-disjoint CV (`python -m Training.train --cv 5`) instead."
        )

    # Persist
    tag = ctx.get("tag", "run")
    (EVALUATION_DIR / f"classification_report_{tag}.txt").write_text(report, encoding="utf-8")
    cm_df.to_csv(EVALUATION_DIR / f"confusion_matrix_{tag}.csv")
    (EVALUATION_DIR / f"test_metrics_{tag}.json").write_text(
        json.dumps(
            {
                "n_test": int(len(test_idx)),
                "test_actors": sorted(int(a) for a in test_meta["actor"].unique()),
                "accuracy": float(accuracy),
                "accuracy_ci95": acc_ci,
                "macro_f1": float(macro_f1),
                "macro_f1_ci95": f1_ci,
                "uar": float(uar),
                "per_actor_accuracy": {str(k): float(v) for k, v in by_actor.items()},
                "per_gender_accuracy": {str(k): float(v) for k, v in by_gender.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nSaved to Evaluation/")


if __name__ == "__main__":
    main()
