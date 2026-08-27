"""
make_paper_figures.py

Produces the three figures the manuscript reserves space for:

    figures/fig2_decomposition_ladder.pdf     <- Evaluation/all_results.json
    figures/fig5_stream_complementarity.pdf   <- Evaluation/complementarity.json
    figures/fig7_reliability.pdf              <- Evaluation/complementarity.json

    python -m Experiments.run_all
    python -m Experiments.complementarity
    python -m Experiments.make_paper_figures --out Paper/IEEE_TASLP/figures

Design notes
------------
Two categorical hues carry every figure: blue for the speaker-disjoint
protocol (P4) and orange for the contaminated one (P1). The pair was checked
rather than eyeballed -- CVD separation dE 24.5 (protan), normal-vision dE
28.3, both well above the dE 8 target, and both inside the lightness and
chroma bands against a paper-white surface. The same two hues serve as the
diverging poles in panel (b) of Fig. 5, with a neutral gray midpoint at zero,
so a reader never has to distinguish red from green.

Marks are thin, grids recessive, and no number is printed on every point --
only on the ones the caption asks the reader to compare.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from Utils.paths import EVALUATION_DIR, PROJECT_ROOT

# --- validated palette ------------------------------------------------------
BLUE = "#3E72CC"      # speaker-disjoint / P4 / positive pole
ORANGE = "#BF5A22"    # contaminated / P1 / negative pole
INK = "#15181E"
MUTED = "#6B7280"
GRID = "#E3E6EA"
NEUTRAL = "#9AA1AC"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def _style(ax, *, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5)


def _load(path: Path, hint: str):
    if not path.exists():
        sys.exit(f"{path} not found.\n  Run: {hint}")
    return json.load(open(path))


# ===========================================================================
# Figure 2 -- decomposition ladder
# ===========================================================================
def fig2(results, out: Path) -> None:
    """The descent P1 -> P2 -> P3 -> P4, with test-selection shown as offsets.

    P2t and P3t are not rungs in the descent: test-set model selection makes a
    result *better*, not worse, so drawing them in the chain would produce a
    zig-zag that misrepresents the decomposition. They are drawn as upward
    offsets from the rungs they modify, which is what Algorithm 7 computes.
    """
    ladder = results.get("ladder", {})
    chain = [r for r in ("P1", "P2", "P3", "P4") if r in ladder]
    if len(chain) < 2:
        print("[fig2] skipped: the protocol ladder has fewer than two rungs")
        return

    acc = [ladder[r]["accuracy"]["mean"] for r in chain]
    sd = [ladder[r]["accuracy"]["sd"] for r in chain]
    x = np.arange(len(chain), dtype=float)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    _style(ax)

    # everything from P2 downward is a choice the published literature makes
    if "P2" in chain:
        i = chain.index("P2")
        ax.axvspan(i - 0.35, len(chain) - 0.55, color=BLUE, alpha=0.055,
                   zorder=0)

    # the P1 -> P2 segment is our own corpus defect, drawn apart from the rest
    if chain[:2] == ["P1", "P2"]:
        ax.plot(x[:2], acc[:2], linestyle=(0, (4, 2)), color=NEUTRAL,
                linewidth=1.4, zorder=3)
        ax.plot(x[1:], acc[1:], "-", color=BLUE, linewidth=1.5, zorder=3)
    else:
        ax.plot(x, acc, "-", color=BLUE, linewidth=1.5, zorder=3)

    ax.errorbar(x, acc, yerr=sd, fmt="o", color=BLUE, markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.7,
                elinewidth=0.9, capsize=2.2, zorder=5)

    # test-set selection: an upward offset from the rung it modifies
    sel_label = {"P2": "P2t", "P3": "P3t"}
    for rung, tname in sel_label.items():
        if rung not in chain or tname not in ladder:
            continue
        i = chain.index(rung)
        base = ladder[rung]["accuracy"]["mean"]
        top = ladder[tname]["accuracy"]["mean"]
        xo = i + 0.30
        ax.plot([xo, xo], [base, top], "-", color=ORANGE, linewidth=1.0,
                zorder=4)
        ax.plot([xo], [top], "o", color="white", markeredgecolor=ORANGE,
                markeredgewidth=1.1, markersize=4.0, zorder=5)
        ax.annotate(f"{tname}\n$+${top - base:.1f}", xy=(xo, top),
                    xytext=(3, 1), textcoords="offset points", ha="left",
                    va="bottom", fontsize=5.9, color=ORANGE, linespacing=0.95)

    # descent increments, alternating side so labels never sit on the line
    for i in range(len(chain) - 1):
        d = acc[i] - acc[i + 1]
        if abs(d) < 0.15:
            continue
        col = NEUTRAL if chain[i] == "P1" else INK
        ax.annotate(f"$-${abs(d):.1f}",
                    xy=(i + 0.5, (acc[i] + acc[i + 1]) / 2),
                    xytext=(-16 if i % 2 == 0 else 4, -4),
                    textcoords="offset points", ha="right" if i % 2 == 0
                    else "left", fontsize=6.5, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(chain)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(-0.55, len(chain) - 0.18)
    floor = min(a - e for a, e in zip(acc, sd))
    lo, hi = floor - 7.5, max(acc) + 7
    ax.set_ylim(lo, hi)

    if "P2" in chain and "P4" in chain:
        span = ladder["P2"]["accuracy"]["mean"] - ladder["P4"]["accuracy"]["mean"]
        ax.text((chain.index("P2") + len(chain) - 1) / 2, lo + 0.9,
                f"attributable to protocol choices: $-${span:.1f} pts",
                ha="center", va="bottom", fontsize=6.1, color=MUTED,
                style="italic")

    ax.annotate(f"{acc[0]:.1f}", (x[0], acc[0]), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=6.9, color=INK,
                fontweight="bold")
    ax.annotate(f"{acc[-1]:.1f}", (x[-1], acc[-1]),
                textcoords="offset points", xytext=(11, -2), ha="left",
                va="center", fontsize=6.9, color=INK, fontweight="bold")
    ax.set_title("One system, six protocols", loc="left", color=INK, pad=6)

    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"[fig2] wrote {out}")


# ===========================================================================
# Figure 5 -- stream complementarity (three panels, double column)
# ===========================================================================
def fig5(comp, out: Path) -> None:
    seeds = comp["seeds"]
    ref = comp["per_seed"][str(seeds[0])]
    cm = ref["complementarity"]
    agg = comp["aggregate"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45),
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.0],
                                          "wspace": 0.42})

    # ---- (a) error-set overlap --------------------------------------------
    ax = axes[0]
    ax.set_aspect("equal")
    ax.axis("off")
    both = cm["both_wrong"]
    only_p_correct = cm["only_psycho_correct"]   # HuBERT wrong, psycho right
    only_h_correct = cm["only_hubert_correct"]   # psycho wrong, HuBERT right

    ax.add_patch(Circle((-0.32, 0), 0.62, facecolor=BLUE, alpha=0.20,
                        edgecolor=BLUE, linewidth=1.0))
    ax.add_patch(Circle((0.32, 0), 0.62, facecolor=ORANGE, alpha=0.20,
                        edgecolor=ORANGE, linewidth=1.0))
    ax.text(-0.62, 0, f"{only_h_correct}", ha="center", va="center",
            fontsize=9, color=INK, fontweight="bold")
    ax.text(0.62, 0, f"{only_p_correct}", ha="center", va="center",
            fontsize=9, color=INK, fontweight="bold")
    ax.text(0, 0, f"{both}", ha="center", va="center", fontsize=9, color=INK,
            fontweight="bold")
    ax.text(-0.62, -0.22, "only HuBERT\nrecovers", ha="center", va="top",
            fontsize=5.9, color=MUTED)
    ax.text(0.62, -0.22, "only psycho.\nrecovers", ha="center", va="top",
            fontsize=5.9, color=MUTED)
    ax.text(0, -0.22, "both\nwrong", ha="center", va="top", fontsize=5.9,
            color=MUTED)
    ax.text(-1.05, 0.60, "psycho.\nerrors", ha="left", va="center",
            fontsize=6.3, color=BLUE, linespacing=1.0)
    ax.text(1.05, 0.60, "HuBERT\nerrors", ha="right", va="center",
            fontsize=6.3, color=ORANGE, linespacing=1.0)
    ax.text(0, -0.92,
            f"Jaccard {agg['jaccard']['mean']:.3f}   "
            f"both right {cm['both_right']}",
            ha="center", fontsize=6.2, color=MUTED)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.10, 0.86)
    # title drawn below, on a common baseline with the other two panels

    # ---- (b) per-class delta recall (diverging) ---------------------------
    ax = axes[1]
    _style(ax, grid_axis="x")
    dr = agg["delta_recall"]
    names = sorted(dr, key=lambda c: dr[c]["mean"])
    vals = np.array([dr[c]["mean"] for c in names]) * 100
    errs = np.array([dr[c]["sd"] for c in names]) * 100
    ypos = np.arange(len(names))
    colors = [BLUE if v >= 0 else ORANGE for v in vals]
    ax.barh(ypos, vals, height=0.62, color=colors, zorder=3)
    ax.errorbar(vals, ypos, xerr=errs, fmt="none", ecolor=NEUTRAL,
                elinewidth=0.6, capsize=1.6, zorder=4)
    ax.axvline(0, color=NEUTRAL, linewidth=0.8, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=6.6)
    ax.set_xlabel("$\\Delta$ recall, fused $-$ best unimodal (pts)",
                  fontsize=6.8)

    lim = max(4.0, float(np.abs(vals).max() + np.abs(errs).max()) * 1.15)
    ax.set_xlim(-lim, lim)

    # ---- (c) gate distribution --------------------------------------------
    ax = axes[2]
    _style(ax)
    gate = ref.get("gate")
    if gate is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "gate not captured\n(non-gated run)", ha="center",
                va="center", fontsize=6.5, color=MUTED)
    else:
        per_dim = np.array(gate["mean_per_dim"])
        ax.hist(per_dim, bins=22, color=BLUE, alpha=0.75, zorder=3,
                edgecolor="white", linewidth=0.4)
        ax.axvline(0.5, color=NEUTRAL, linewidth=0.8, linestyle=(0, (3, 2)),
                   zorder=4)
        ax.axvline(gate["overall_mean"], color=ORANGE, linewidth=1.2, zorder=5)
        top = ax.get_ylim()[1] * 1.30
        ax.set_ylim(0, top)
        ax.annotate(f"mean {gate['overall_mean']:.2f}",
                    xy=(gate["overall_mean"], top * 0.97),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=6.2, color=ORANGE, va="top", ha="left")
        ax.set_xlabel("mean gate $g_i$ per dimension\n"
                      "$0\\rightarrow$ all HuBERT   "
                      "$1\\rightarrow$ all perceptual",
                      fontsize=6.5, linespacing=1.35)
        ax.set_ylabel("dimensions", fontsize=6.8)
        ax.set_xlim(0, 1)


    # Panel (a) has an equal-aspect box, so its axes shrink vertically and a
    # per-axes title would sit lower than the other two. Draw all three at one
    # figure-space baseline instead.
    titles = ["(a) Error sets of the two views",
              "(b) Where fusion helps and hurts",
              "(c) What the model actually uses"]
    ytop = max(a.get_position().y1 for a in axes) + 0.045
    for a_, t in zip(axes, titles):
        fig.text(a_.get_position().x0, ytop, t, ha="left", va="bottom",
                 fontsize=7.5, color=INK)

    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"[fig5] wrote {out}")


# ===========================================================================
# Figure 7 -- posterior calibration
# ===========================================================================
def fig7(comp, out: Path) -> None:
    seeds = comp["seeds"]
    ref = comp["per_seed"][str(seeds[0])]
    c4 = ref["calibration_p4"]
    c1 = ref.get("calibration_p1")
    agg = comp["aggregate"]

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.95),
                             gridspec_kw={"wspace": 0.46})

    # ---- (a) reliability ---------------------------------------------------
    ax = axes[0]
    _style(ax)
    ax.plot([0, 1], [0, 1], linestyle=(0, (3, 2)), color=NEUTRAL,
            linewidth=0.8, zorder=2)

    def _curve(cal, color, label):
        pts = [(b["conf"], b["acc"]) for b in cal["reliability"]
               if b["n"] and b["conf"] is not None]
        if not pts:
            return
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=color, linewidth=1.3, markersize=3,
                markeredgecolor="white", markeredgewidth=0.5, zorder=4,
                label=label)

    if c1 is not None:
        _curve(c1, ORANGE, f"P1  ECE {agg.get('ece_p1', {}).get('mean', c1['ece']):.3f}")
    _curve(c4, BLUE, f"P4  ECE {agg['ece_p4']['mean']:.3f}")

    ax.set_xlabel("confidence", fontsize=6.8)
    ax.set_ylabel("accuracy", fontsize=6.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(frameon=False, loc="upper left", handlelength=1.2,
              borderpad=0.1, labelspacing=0.25, fontsize=5.9)
    ax.set_title("(a) Reliability", loc="left", color=INK, fontsize=7.2, pad=4)

    # ---- (b) accuracy against the top-2 margin -----------------------------
    ax = axes[1]
    _style(ax)
    pts = [((b["lo"] + b["hi"]) / 2, b["acc"]) for b in c4["margin_curve"]
           if b["n"] and b["acc"] is not None]
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=BLUE, linewidth=1.3, markersize=3,
                markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    ts = c4["tau_star"]
    if ts.get("tau") is not None:
        ax.axvline(ts["tau"], color=ORANGE, linewidth=1.1, zorder=3)
        ax.annotate(f"$\\tau^\\star$ {ts['tau']:.2f}", xy=(ts["tau"], 0.06),
                    xytext=(3, 0), textcoords="offset points", fontsize=6.0,
                    color=ORANGE)
    ax.axvline(0.15, color=NEUTRAL, linewidth=0.9, linestyle=(0, (3, 2)),
               zorder=3)
    ax.annotate("asserted\n0.15", xy=(0.15, 0.97), xytext=(-3, 0),
                textcoords="offset points", fontsize=5.8, color=MUTED,
                va="top", ha="right", linespacing=1.0)
    ax.set_xlabel("top-1 $-$ top-2 margin", fontsize=6.8)
    ax.set_ylabel("accuracy", fontsize=6.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title("(b) Margin", loc="left", color=INK, fontsize=7.2, pad=4)

    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"[fig7] wrote {out}")


# ===========================================================================
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "Paper" / "IEEE_TASLP"
                                         / "figures"))
    ap.add_argument("--only", choices=["2", "5", "7"], default=None)
    a = ap.parse_args(argv)

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if a.only in (None, "2"):
        res = _load(EVALUATION_DIR / "all_results.json",
                    "python -m Experiments.run_all")
        fig2(res, outdir / "fig2_decomposition_ladder.pdf")

    if a.only in (None, "5", "7"):
        comp = _load(EVALUATION_DIR / "complementarity.json",
                     "python -m Experiments.complementarity")
        if a.only in (None, "5"):
            fig5(comp, outdir / "fig5_stream_complementarity.pdf")
        if a.only in (None, "7"):
            fig7(comp, outdir / "fig7_reliability.pdf")

    print("\nDone. In paer_llm_ieee.tex replace each \\reservedfig{...}{...}\n"
          "with \\includegraphics, e.g.\n"
          "  \\includegraphics[width=\\columnwidth]"
          "{figures/fig2_decomposition_ladder.pdf}\n"
          "Fig. 5 already sits in a figure* and takes width=0.95\\textwidth.")


if __name__ == "__main__":
    main()
