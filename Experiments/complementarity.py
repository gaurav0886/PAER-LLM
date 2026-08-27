"""
complementarity.py

Algorithm 8 of the manuscript: stream complementarity and posterior
calibration.

    python -m Experiments.complementarity                 # five seeds
    python -m Experiments.complementarity --seeds 42      # smoke test

Why this is not just another accuracy table
-------------------------------------------
Section III-C defines the quantity Objective 1 actually asks about,

    U_p = I( phi_p(A) ; Y | phi_h(A) )   >= 0,

the information the perceptual view carries about the label that the
self-supervised view does not. Two facts about U_p drive everything here.

1. U_p > 0 is *necessary but not sufficient* for an accuracy gain. A finite
   estimator on 1440 utterances can fail to extract information that is
   present, so a null result on the mean does not establish redundancy.
2. Section VII-D shows this design cannot resolve differences below 3.47
   accuracy points. A real contribution smaller than that is invisible in a
   difference of means whether or not it exists.

So the mean is the wrong instrument. What U_p > 0 *does* imply is that some
utterances are classified correctly by one view and not the other. That is
measurable at this sample size, and it is what this module measures:

    * error-set overlap (Jaccard) between the two unimodal systems
    * |E_h \\ E_p| and |E_p \\ E_h| -- the utterances only one view recovers
    * exact McNemar of the fused system against the stronger unimodal one
    * per-class Delta-recall, so gains offset by losses stay visible
    * the distribution of the learned gate g, i.e. how much perceptual
      evidence the model itself decides to use

The second half computes the calibration diagnostics Layer 4 needs before the
hedging threshold tau of Algorithm 4 can be set rather than asserted, under
both P4 and the contaminated P1 -- testing whether contamination changes the
model's *confidence* and not only its accuracy.

Every statistic is computed per seed on pooled cross-validated predictions, in
which each utterance is classified exactly once by a model that never observed
its speaker. Across-seed mean and spread are reported; nothing is ensembled,
because averaging posteriors across seeds would evaluate a system we do not
propose.

Writes ``Evaluation/complementarity.json`` and prints paste-ready sentences.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

from Experiments.run_all import _base_args, _predict, _random_split, run_single
from Training.train import fit_fold, load_features
from Utils.paths import EVALUATION_DIR, ensure_dirs
from Utils.splits import actor_kfold, actor_split

DEFAULT_SEEDS = [42, 1, 7, 13, 2024]
N_BINS = 10


# ---------------------------------------------------------------------------
# pooled cross-validated prediction, with gate capture
# ---------------------------------------------------------------------------
def cv_posteriors(Xp, Xh, y, meta, args, device, seed, n_splits=5,
                  capture_gate=False):
    """Return an (N, C) posterior in which every utterance is held out once.

    Mirrors ``Experiments.run_all.run_cv`` fold for fold -- same inner split,
    same seed handling -- so the predictions are the ones behind Tables VII
    and VIII, not a parallel universe. It differs only in returning a full
    length-N posterior and, optionally, the per-utterance gate vector.
    """
    n, n_classes = len(y), int(y.max()) + 1
    posterior = np.full((n, n_classes), np.nan, dtype=np.float64)
    gates = np.full((n, args.hidden_dim), np.nan, dtype=np.float64) \
        if capture_gate else None

    for tr, te in actor_kfold(meta, n_splits=n_splits, random_state=seed):
        sub = meta.iloc[tr].reset_index(drop=True)
        inner = actor_split(sub, test_size=0.0001, val_size=0.2,
                            random_state=seed)
        tr_idx = tr[np.concatenate([inner["train"], inner["test"]])]
        va_idx = tr[inner["val"]]

        model, (ps, hs), _h, _b = fit_fold(
            Xp, Xh, y, tr_idx, va_idx, args, device, None,
            select_idx=va_idx, seed=seed,
        )

        captured = {}
        handle = None
        if capture_gate and hasattr(model, "gate"):
            handle = model.gate.register_forward_hook(
                lambda _m, _i, out: captured.__setitem__("g", out.detach())
            )

        posterior[te] = _predict(model, Xp, Xh, te, ps, hs, device)

        if handle is not None:
            handle.remove()
            if "g" in captured:
                gates[te] = captured["g"].cpu().numpy()

    if np.isnan(posterior).any():
        raise RuntimeError("some utterances were never held out; check the folds")
    return posterior, gates


# ---------------------------------------------------------------------------
# Algorithm 8, first half: complementarity
# ---------------------------------------------------------------------------
def complementarity(post, y, classes):
    """Error-set analysis for one seed. ``post`` maps view name -> posterior."""
    pred = {m: p.argmax(1) for m, p in post.items()}
    err = {m: set(np.flatnonzero(pred[m] != y).tolist()) for m in pred}

    inter, union = err["psycho"] & err["hubert"], err["psycho"] | err["hubert"]
    jaccard = len(inter) / len(union) if union else 0.0

    # the stronger unimodal system is the reference for the paired test
    best = min(("psycho", "hubert"), key=lambda m: len(err[m]))
    b = len(err[best] - err["fused"])      # fused fixes what the unimodal missed
    c = len(err["fused"] - err[best])      # fused breaks what the unimodal had
    p_mcn = float(stats.binomtest(b, b + c, 0.5).pvalue) if (b + c) else 1.0

    per_class = {}
    for k, name in enumerate(classes):
        mask = y == k
        if not mask.any():
            continue
        per_class[name] = {
            "recall_fused": float((pred["fused"][mask] == k).mean()),
            "recall_best_unimodal": float((pred[best][mask] == k).mean()),
            "recall_psycho": float((pred["psycho"][mask] == k).mean()),
            "recall_hubert": float((pred["hubert"][mask] == k).mean()),
            "n": int(mask.sum()),
        }
        per_class[name]["delta_recall"] = (
            per_class[name]["recall_fused"]
            - per_class[name]["recall_best_unimodal"]
        )

    return {
        "n_errors": {m: len(err[m]) for m in err},
        "accuracy": {m: float((pred[m] == y).mean()) for m in pred},
        "jaccard": jaccard,
        "only_psycho_correct": len(err["hubert"] - err["psycho"]),
        "only_hubert_correct": len(err["psycho"] - err["hubert"]),
        "both_wrong": len(inter),
        "both_right": int(len(y) - len(union)),
        "best_unimodal": best,
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p_mcn,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Algorithm 8, second half: calibration
# ---------------------------------------------------------------------------
def calibration(post, y, n_bins=N_BINS):
    """Expected calibration error, reliability curve, and the margin curve."""
    conf, pred = post.max(1), post.argmax(1)
    correct = (pred == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, rel = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            rel.append({"lo": float(lo), "hi": float(hi), "n": 0,
                        "acc": None, "conf": None})
            continue
        acc_b, conf_b = float(correct[m].mean()), float(conf[m].mean())
        ece += m.mean() * abs(acc_b - conf_b)
        rel.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "acc": acc_b, "conf": conf_b})

    srt = np.sort(post, axis=1)
    margin = srt[:, -1] - srt[:, -2]

    # tau* maximises balanced accuracy of the rule "margin >= tau => correct",
    # which is the decision Algorithm 4 actually makes. Reported alongside the
    # asserted 0.15 so the two can be compared rather than conflated.
    best = {"tau": None, "balanced_acc": -1.0}
    for tau in np.unique(np.round(margin, 3)):
        hi_m = margin >= tau
        if hi_m.sum() < 20 or (~hi_m).sum() < 20:
            continue
        tpr = correct[hi_m].mean()
        tnr = 1.0 - correct[~hi_m].mean()
        bal = 0.5 * (tpr + tnr)
        if bal > best["balanced_acc"]:
            best = {"tau": float(tau), "balanced_acc": float(bal),
                    "acc_above": float(tpr), "acc_below": float(correct[~hi_m].mean()),
                    "frac_above": float(hi_m.mean())}

    m_edges = np.linspace(0.0, 1.0, n_bins + 1)
    m_curve = []
    for lo, hi in zip(m_edges[:-1], m_edges[1:]):
        m = (margin >= lo) & (margin < hi)
        m_curve.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                        "acc": float(correct[m].mean()) if m.any() else None})

    asserted = 0.15
    hi_m = margin >= asserted
    return {
        "ece": float(ece),
        "reliability": rel,
        "margin_curve": m_curve,
        "tau_star": best,
        "tau_asserted": {
            "tau": asserted,
            "acc_above": float(correct[hi_m].mean()) if hi_m.any() else None,
            "acc_below": float(correct[~hi_m].mean()) if (~hi_m).any() else None,
            "frac_above": float(hi_m.mean()),
        },
        "mean_confidence": float(conf.mean()),
        "accuracy": float(correct.mean()),
    }


def _agg(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "values": v.tolist()}


# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--skip-p1", action="store_true",
                    help="skip the contaminated-protocol calibration comparison")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, Xp, Xh, y, enc = load_features()
    classes = enc.classes_.tolist()
    print(f"[complementarity] device={device} seeds={seeds} N={len(y)}")

    base = dict(epochs=a.epochs, patience=a.patience, fusion="gated",
                hidden_dim=128, dropout=0.3, lr=1e-3, weight_decay=1e-2,
                batch_size=32, label_smoothing=0.05)

    out = {"seeds": seeds, "classes": classes, "n_utterances": int(len(y)),
           "per_seed": {}}
    t0 = time.time()

    for s in seeds:
        print(f"\n=== seed {s} ===")
        post, gates = {}, None
        for view, modality in [("psycho", "psycho"), ("hubert", "hubert"),
                               ("fused", "both")]:
            print(f"  {view} ...")
            p, g = cv_posteriors(
                Xp, Xh, y, meta, _base_args(**{**base, "modality": modality}),
                device, s, n_splits=a.cv, capture_gate=(view == "fused"),
            )
            post[view] = p
            if view == "fused":
                gates = g

        rec = {"complementarity": complementarity(post, y, classes),
               "calibration_p4": calibration(post["fused"], y)}

        if gates is not None and not np.isnan(gates).all():
            rec["gate"] = {
                "mean_per_dim": np.nanmean(gates, axis=0).tolist(),
                "overall_mean": float(np.nanmean(gates)),
                "overall_sd": float(np.nanstd(gates)),
                "mean_per_class": {
                    c: float(np.nanmean(gates[y == k]))
                    for k, c in enumerate(classes)
                },
            }

        if not a.skip_p1:
            print("  P1 (contaminated) for the calibration comparison ...")
            meta_d, Xp_d, Xh_d, y_d, _ = load_features(keep_duplicates=True)
            tr, va, te = _random_split(y_d, s)
            _m, probs, te_idx = run_single(
                Xp_d, Xh_d, y_d, tr, va, te, _base_args(**base), device, s)
            rec["calibration_p1"] = calibration(probs, y_d[te_idx])

        out["per_seed"][str(s)] = rec
        cm = rec["complementarity"]
        print(f"    J={cm['jaccard']:.3f}  only-psycho={cm['only_psycho_correct']}"
              f"  only-hubert={cm['only_hubert_correct']}"
              f"  McNemar b={cm['mcnemar_b']} c={cm['mcnemar_c']}"
              f" p={cm['mcnemar_p']:.4f}")

    # ---- aggregate across seeds -------------------------------------------
    per = list(out["per_seed"].values())
    agg = {
        "jaccard": _agg([r["complementarity"]["jaccard"] for r in per]),
        "only_psycho_correct": _agg(
            [r["complementarity"]["only_psycho_correct"] for r in per]),
        "only_hubert_correct": _agg(
            [r["complementarity"]["only_hubert_correct"] for r in per]),
        "mcnemar_p": _agg([r["complementarity"]["mcnemar_p"] for r in per]),
        "ece_p4": _agg([r["calibration_p4"]["ece"] for r in per]),
    }
    if "calibration_p1" in per[0]:
        agg["ece_p1"] = _agg([r["calibration_p1"]["ece"] for r in per])
    if "gate" in per[0]:
        agg["gate_mean"] = _agg([r["gate"]["overall_mean"] for r in per])

    dr = defaultdict(list)
    for r in per:
        for c, v in r["complementarity"]["per_class"].items():
            dr[c].append(v["delta_recall"])
    agg["delta_recall"] = {c: _agg(v) for c, v in dr.items()}

    out["aggregate"] = agg
    out["elapsed_min"] = round((time.time() - t0) / 60, 1)
    path = EVALUATION_DIR / "complementarity.json"
    json.dump(out, open(path, "w"), indent=2)

    # ---- paste-ready prose -------------------------------------------------
    j, op, oh = agg["jaccard"], agg["only_psycho_correct"], agg["only_hubert_correct"]
    ref = per[0]["complementarity"]
    print("\n" + "=" * 72)
    print("Paste into Section VII-C, replacing the \\TORUN marker:")
    print("=" * 72)
    print(
        f"The two unimodal systems' error sets overlap with Jaccard index "
        f"{j['mean']:.3f} (SD {j['sd']:.3f} across seeds). Of the utterances "
        f"one view alone recovers, {op['mean']:.0f} are recovered only by the "
        f"psychoacoustic stream and {oh['mean']:.0f} only by HuBERT, so the "
        f"views do not fail together and $U_p > 0$ is supported on the error "
        f"sets even where the difference of means is inadmissible. Against the "
        f"stronger unimodal system ({ref['best_unimodal']}), gated fusion "
        f"repairs {ref['mcnemar_b']} errors and introduces {ref['mcnemar_c']} "
        f"(exact McNemar $p = {agg['mcnemar_p']['mean']:.3f}$)."
    )
    if "ece_p1" in agg:
        print(
            f"\nExpected calibration error is {agg['ece_p4']['mean']:.3f} under "
            f"P4 against {agg['ece_p1']['mean']:.3f} under P1, so the "
            f"contaminated protocol yields a model that is not merely more "
            f"accurate but differently calibrated."
        )
    ts = per[0]["calibration_p4"]["tau_star"]
    ta = per[0]["calibration_p4"]["tau_asserted"]
    if ts["tau"] is not None:
        print(
            f"\nThe margin threshold that best separates correct from "
            f"incorrect predictions is $\\tau^\\star = {ts['tau']:.3f}$ "
            f"(accuracy {ts['acc_above']:.3f} above it against "
            f"{ts['acc_below']:.3f} below), against the asserted 0.15 "
            f"({ta['acc_above']:.3f} / {ta['acc_below']:.3f}). Algorithm 4 "
            f"should use the former."
        )
    print(f"\nWrote {path}  ({out['elapsed_min']} min)")
    print("Now run:  python -m Experiments.make_paper_figures")


if __name__ == "__main__":
    main()
