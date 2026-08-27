#!/usr/bin/env python3
"""
Figure 1 -- PAER-LLM framework block diagram.

Corrections applied relative to the earlier draft diagram:
  * "Cross-Modal Attention-Based Fusion" -> "Gated Fusion" (adopted operator, Sec. 5.2)
  * "Loudness Features"                  -> "Intensity (RMS)" (Sec. 3.2 refuses "loudness")
  * "Multimodal Representation Learning" -> "Unified representation" (the system is
                                            unimodal: two feature views of one signal)
  * Layer 3 collapsed from five boxes to the four stages actually implemented
  * Layers 4-5 drawn dashed / desaturated and labelled "proposed; not evaluated"
  * Every edge annotated with its tensor shape; dual analysis sample rates shown

Output: figures/fig1_framework_ieee.pdf  (vector, greyscale-legible)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import os

C = {
    "L1": "#2E4A7D", "L1b": "#EDF1F8",
    "L2": "#2F6B58", "L2b": "#EAF2EF",
    "L3": "#8A5A12", "L3b": "#F9F2E4",
    "L4": "#5B4B7A", "L4b": "#F1EEF6",
    "L5": "#8C3A3A", "L5b": "#F8EDED",
    "ink": "#15181E", "ink2": "#4B5261", "ink3": "#7A8290", "white": "#FFFFFF",
}

fig = plt.figure(figsize=(7.5, 7.9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

FS_BOX, FS_SUB, FS_EDGE = 7.0, 5.9, 5.6


def band(y0, h, edge, face, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (2.0, y0), 96.0, h, boxstyle="round,pad=0,rounding_size=0.8",
        linewidth=1.0, edgecolor=edge, facecolor=face,
        linestyle=(0, (4, 2.6)) if dashed else "solid", zorder=1))


def tab(y0, h, color, num, name, dashed=False):
    x0, w, tip = 2.0, 13.4, 2.0
    pts = [(x0, y0), (x0 + w - tip, y0), (x0 + w, y0 + h / 2),
           (x0 + w - tip, y0 + h), (x0, y0 + h)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=color,
                         linewidth=0.8, alpha=0.28 if dashed else 1.0, zorder=2))
    tc = color if dashed else C["white"]
    cx, cy = x0 + (w - tip) / 2, y0 + h / 2
    ax.text(cx, cy + 1.9, num, ha="center", va="center", fontsize=9.0,
            fontweight="bold", color=tc, zorder=3)
    for i, line in enumerate(name):
        ax.text(cx, cy - 1.2 - i * 2.0, line, ha="center", va="center",
                fontsize=6.1, color=tc, fontweight="semibold", zorder=3)


def box(x, y, w, h, title, sub=None, edge=C["ink3"], face=C["white"],
        dashed=False, bold=False, fs=FS_BOX, gap=2.25):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
        linewidth=1.6 if bold else 0.9, edgecolor=edge, facecolor=face,
        linestyle=(0, (3.2, 2.2)) if dashed else "solid", zorder=4))
    sub = sub or []
    block = (0 if not sub else gap + (len(sub) - 1) * 1.95)
    ty = y + h / 2 + block / 2
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=fs,
            color=C["ink"], fontweight="bold" if bold else "semibold", zorder=5)
    for i, s in enumerate(sub):
        ax.text(x + w / 2, ty - gap - i * 1.95, s, ha="center", va="center",
                fontsize=FS_SUB, color=C["ink2"], zorder=5)


def arrow(x1, y1, x2, y2, label=None, color=C["ink2"], dashed=False,
          lw=1.0, rad=0.0, lx=None, ly=None):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8.5,
        linewidth=lw, color=color, zorder=6,
        linestyle=(0, (3.2, 2.2)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2,
                ly if ly is not None else (y1 + y2) / 2 + 1.3,
                label, ha="center", va="center", fontsize=FS_EDGE,
                color=C["ink3"], style="italic", zorder=7)


def goal(x, ymid, w, lines, edge, face, dashed=False):
    """Goal panel, height derived from the number of lines so text never spills."""
    h = 4.6 + 1.95 * len(lines)
    y = ymid - h / 2
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
        linewidth=0.9, edgecolor=edge, facecolor=face,
        linestyle=(0, (3.2, 2.2)) if dashed else "solid", zorder=4))
    ax.text(x + w / 2, y + h - 2.1, "Goal", ha="center", va="center",
            fontsize=6.3, color=edge, fontweight="bold", zorder=5)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 4.3 - i * 1.95, line, ha="center",
                va="center", fontsize=5.6, color=C["ink2"], zorder=5)


# ---------------------------------------------------------------- title
ax.text(50, 98.2, "The PAER-LLM Framework", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color=C["ink"])
ax.text(50, 95.6,
        "A unified psychoacoustic–semantic representation for emotion-sensitive interaction",
        ha="center", va="center", fontsize=6.8, color=C["ink2"], style="italic")

XC, WC = 16.6, 63.4      # content column
XG, WG = 81.6, 16.4      # goal column

# ============================================================ LAYER 1
y1, h1 = 79.0, 14.0
band(y1, h1, C["L1"], C["L1b"])
tab(y1, h1, C["L1"], "1", ["DATA", "ACQUISITION"])
ax.text(XC + WC / 2, y1 + h1 - 1.9, "Emotional Speech Corpus and Integrity Control",
        ha="center", va="center", fontsize=7.3, fontweight="bold", color=C["L1"])

by, bh = y1 + 1.7, 8.6
w, g = 13.2, 3.6
xs = [XC + i * (w + g) for i in range(4)]
box(xs[0], by, w, bh, "RAVDESS", ["1440 utterances", "24 actors, 8 classes"], edge=C["L1"])
box(xs[1], by, w, bh, "Integrity check", ["abort if $r>1$", "Algorithm 1"], edge=C["L1"])
box(xs[2], by, w, bh, "Preprocessing", ["22.05 kHz psycho.", "16 kHz HuBERT"], edge=C["L1"])
box(xs[3], by, w, bh, "Verified corpus", ["$r = 1$", "1440 unique"], edge=C["L1"], bold=True)
for i in range(3):
    arrow(xs[i] + w, by + bh / 2, xs[i + 1], by + bh / 2)
goal(XG, by + bh / 2, WG, ["Duplicate-free,", "standardised signals", "at both analysis rates"],
     C["L1"], C["L1b"])

# ============================================================ LAYER 2
y2, h2 = 62.0, 14.4
band(y2, h2, C["L2"], C["L2b"])
tab(y2, h2, C["L2"], "2", ["FEATURE", "REPRESENTATION"])
ax.text(XC + WC / 2, y2 + h2 - 1.9, "Two Feature Views of a Single Acoustic Modality",
        ha="center", va="center", fontsize=7.3, fontweight="bold", color=C["L2"])

by, bh = y2 + 1.7, 9.0
box(XC, by, 29.4, bh, "Psychoacoustic stream",
    ["Pitch $F_0$ (5)    Intensity RMS (5)",
     "Bark-band energies (24), Alg. 2",
     "$\\mathbf{x}_p \\in \\mathbb{R}^{34}$"], edge=C["L2"])
box(XC + 34.0, by, 29.4, bh, "Self-supervised stream",
    ["HuBERT-base, frozen",
     "final state, mean-pooled",
     "$\\mathbf{x}_h \\in \\mathbb{R}^{768}$"], edge=C["L2"])
ax.text(XC + 31.7, by + bh / 2, "+", ha="center", va="center", fontsize=12,
        color=C["L2"], fontweight="bold", zorder=6)
goal(XG, by + bh / 2, WG, ["Perceptual cues and", "deep semantic cues", "as complementary",
                           "views of one signal"], C["L2"], C["L2b"])

# ============================================================ LAYER 3
y3, h3 = 36.0, 23.0
band(y3, h3, C["L3"], C["L3b"])
tab(y3, h3, C["L3"], "3", ["UNIFIED", "REPRESENTATION"])
ax.text(XC + WC / 2, y3 + h3 - 1.9, "Stream Encoding, Gated Fusion and Classification",
        ha="center", va="center", fontsize=7.3, fontweight="bold", color=C["L3"])
ax.text(XC + WC / 2, y3 + 1.7,
        "Also evaluated and not adopted: concatenation (equivalent, Sec. 5.2); "
        "token cross-attention (Sec. 5.6)",
        ha="center", va="center", fontsize=5.6, color=C["ink3"], style="italic", zorder=6)

eyT, eyB, eh, ew = y3 + 12.0, y3 + 4.6, 6.0, 12.6
box(XC, eyT, ew, eh, "$\\mathrm{Enc}_p$", ["$34 \\rightarrow 128$"], edge=C["L3"], gap=1.9)
box(XC, eyB, ew, eh, "$\\mathrm{Enc}_h$", ["$768 \\rightarrow 128$"], edge=C["L3"], gap=1.9)

fx, fw = XC + 19.4, 17.0
fmid = y3 + 11.0
box(fx, fmid - 6.0, fw, 12.0, "Gated fusion",
    ["$\\mathbf{g}=\\mathrm{sigm}(\\mathbf{W}_g[\\mathbf{e}_p;\\mathbf{e}_h]+\\mathbf{b})$",
     "$\\mathbf{z}=\\mathbf{g}\\odot\\mathbf{e}_p+(\\mathbf{1}{-}\\mathbf{g})\\odot\\mathbf{e}_h$",
     "Algorithm 3"], edge=C["L3"], face="#FDF8EE", bold=True, gap=2.5)

cx, cw = XC + 41.0, 11.4
box(cx, fmid - 6.0, cw, 12.0, "Classifier",
    ["$128 \\rightarrow 128 \\rightarrow 8$", "17,544 params"], edge=C["L3"])
px, pw = XC + 55.6, 7.8
box(px, fmid - 6.0, pw, 12.0, "Posterior",
    ["$\\hat{\\mathbf{p}}$", "8 classes"], edge=C["L3"], bold=True)

arrow(XC + ew, eyT + eh / 2, fx, fmid + 3.0, rad=-0.14,
      label="$\\mathbf{e}_p$ (128)", lx=XC + ew + 3.4, ly=eyT + eh / 2 + 2.2)
arrow(XC + ew, eyB + eh / 2, fx, fmid - 3.0, rad=0.14,
      label="$\\mathbf{e}_h$ (128)", lx=XC + ew + 3.4, ly=eyB + eh / 2 - 2.2)
arrow(fx + fw, fmid, cx, fmid, "$\\mathbf{z}$ (128)", ly=fmid + 1.9)
arrow(cx + cw, fmid, px, fmid)
goal(XG, fmid, WG, ["One 128-d vector", "carrying perceptual", "and semantic evidence,",
                    "plus a posterior"], C["L3"], C["L3b"])

# ---------------------------------------------- evaluated / proposed divider
ax.plot([2.0, 98.0], [34.4, 34.4], linestyle=(0, (1.6, 2.0)),
        linewidth=0.8, color=C["ink3"], zorder=8)
ax.text(2.6, 34.4, "  Evaluated in this work  ↑ ", ha="left", va="center",
        fontsize=5.7, color=C["ink3"], fontweight="bold", zorder=9,
        bbox=dict(boxstyle="square,pad=0.22", fc=C["white"], ec="none"))
ax.text(97.4, 34.4, " ↓  Proposed; not evaluated (Sec. 3.4, Limitation 9)  ",
        ha="right", va="center", fontsize=5.7, color=C["ink3"], style="italic",
        zorder=9, bbox=dict(boxstyle="square,pad=0.22", fc=C["white"], ec="none"))

# ============================================================ LAYER 4
y4, h4 = 20.0, 12.4
band(y4, h4, C["L4"], C["L4b"], dashed=True)
tab(y4, h4, C["L4"], "4", ["LANGUAGE", "INTELLIGENCE"], dashed=True)
ax.text(XC + WC / 2, y4 + h4 - 1.8, "Posterior-Conditioned Response Generation",
        ha="center", va="center", fontsize=7.1, fontweight="bold", color=C["L4"])

by, bh = y4 + 1.6, 7.4
w, g = 19.4, 2.6
xs = [XC + i * (w + g) for i in range(3)]
box(xs[0], by, w, bh, "Posterior-conditioned prompt",
    ["top-3 + mass, margin rule"], edge=C["L4"], dashed=True, fs=6.6)
box(xs[1], by, w, bh, "Instruction-tuned LLM",
    ["Phi-3-mini, frozen"], edge=C["L4"], dashed=True, fs=6.6)
box(xs[2], by, w, bh, "Hedged response",
    ["tentative if margin small"], edge=C["L4"], dashed=True, fs=6.6)
for i in range(2):
    arrow(xs[i] + w, by + bh / 2, xs[i + 1], by + bh / 2, dashed=True, color=C["L4"])
goal(XG, by + bh / 2, WG, ["Emotionally", "appropriate replies", "that hedge under",
                           "genuine ambiguity"], C["L4"], C["L4b"], dashed=True)

# ============================================================ LAYER 5
y5, h5 = 6.0, 11.0
band(y5, h5, C["L5"], C["L5b"], dashed=True)
tab(y5, h5, C["L5"], "5", ["APPLICATION"], dashed=True)
ax.text(XC + WC / 2, y5 + h5 - 1.7, "Emotion-Sensitive Human–Computer Interaction",
        ha="center", va="center", fontsize=7.1, fontweight="bold", color=C["L5"])

by, bh = y5 + 1.6, 6.2
w, g = 14.8, 1.4
xs = [XC + i * (w + g) for i in range(4)]
box(xs[0], by, w, bh, "User speech", [], edge=C["L5"], dashed=True, fs=6.9)
box(xs[1], by, w, bh, "Emotion + confidence", [], edge=C["L5"], dashed=True, fs=6.0)
box(xs[2], by, w, bh, "Adaptive reply", [], edge=C["L5"], dashed=True, fs=6.9)
box(xs[3], by, w, bh, "Interaction outcome", [], edge=C["L5"], dashed=True, fs=6.1)
for i in range(3):
    arrow(xs[i] + w, by + bh / 2, xs[i + 1], by + bh / 2, dashed=True, color=C["L5"])
goal(XG, by + bh / 2, WG, ["Empathetic,", "personalised", "interaction"],
     C["L5"], C["L5b"], dashed=True)

# ------------------------------------------------- inter-layer arrows
for ya, yb, dash in [(y1, y2 + h2, False), (y2, y3 + h3, False),
                     (y3, y4 + h4, True), (y4, y5 + h5, True)]:
    arrow(50, ya, 50, yb, color=C["ink3"], lw=1.1, dashed=dash)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig1_framework_ieee.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("figures/fig1_framework_ieee.png", format="png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print("wrote figures/fig1_framework_ieee.pdf and .png")
