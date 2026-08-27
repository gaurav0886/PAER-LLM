# PAER-LLM — status after reading your project files

`paer_llm_ieee.tex` targets **IEEE/ACM T-ASLP** and is now the primary
manuscript. `paer_llm.tex` (Elsevier / CS&L) has been kept numerically correct
but does **not** carry the new sections; treat the IEEE file as canonical.

Both compile clean: 0 errors, 0 undefined references or citations, 0 overfull
boxes. IEEE version is 20 pages.

---

## 1. Two things your project files settled

### 1.1 The P4/gated conflict — resolved, and my earlier fix was backwards

`Models/train_config.json` and `Models/train_config_cross_attention_random_dup.json`
both record `"fusion": "cross_attention"`. So **every rung of the protocol
ladder was run with token cross-attention**, and the original Table 4 numbers
were right — only the caption, which said "gated fusion", was wrong.

I had "corrected" P4 to 66.25 in the last revision. That was the wrong
direction and it is now reverted. Both files carry:

| | Accuracy | Macro-F1 | UAR |
|---|---|---|---|
| P1 duplicated + random | 94.79 | 94.97 | 95.13 |
| P3 actor-disjoint, single | 72.50 | 70.83 | 73.05 |
| P4 actor-disjoint, 5-fold | 64.28 ± 5.30 | 63.42 ± 4.45 | 64.52 ± 4.59 |
| **P1 − P4** | **30.51** | **31.56** | **30.61** |

Ratio to the 2.03-point architecture effect: **15.0×** ("fifteen times" is
correct). The caption now states the anchor operator explicitly, and the text
argues from Eq. (14) that the choice of anchor moves the result by at most the
architecture effect itself — 2.03 points. Re-running the ladder under gated
fusion is queued for symmetry, not for correctness.

### 1.2 The cross-attention parameter count — confirmed, not an error

584,456 reconstructs exactly:

```
4 token expansions   4 × (K·d² + K·d)  = 264,192
2 attention blocks   2 × (4d² + 4d)    = 132,096
2 post-norm LayerN.  2 × 2d            =     512
output projection    2d² + d           =  32,896
                                fusion = 429,696
+ encoders 136,960 + output LN 256 + head 17,544 = 584,456  ✓
ratio to gated/concat = 3.115  →  "3.1×" is correct
```

The gap I flagged came from a direction-shared reading; the expansions are
unshared per direction, so there are four of them, not two. Section III now
states this and the VERIFY marker is gone. This became
Proposition 3 (cost ordering) with a full complexity table.

---

## 2. The eight items from `Outputs/baseline_results.txt`

Your note listed: *Mathematical Formulation / Research methodology with
comparisons / overall discussion / benchmarking techniques / latest techniques /
comparisons / performance matrix / Graph should be explained.* All eight are now
in the IEEE manuscript.

| # | Item | Where it now lives |
|---|---|---|
| 1 | Mathematical formulation | **§III Problem Formulation** — notation (Eq. 1), the unified-representation objective with its capacity constraint (Eq. 2–4), an information-theoretic complementarity condition (Eq. 5–6), protocol as an operator (Eq. 7–10), and design sensitivity (Eq. 11). Plus **Definition 1** (speaker-disjointness), **Definition 2** (admissibility), and **Propositions 1–3 with proofs**. |
| 2 | Research methodology with comparisons | **§V Research Methodology** and **Table II** — ten design decisions, the alternatives considered for each, and the rationale. Three rows are marked ⊖ because they *reduce* our reported performance and were adopted anyway. |
| 3 | Overall discussion | **§VIII-A Overall Discussion** — the three governing numbers (30.5 / 2.03 / 3.47) read together, then what the results do and do not establish about Objective 1. |
| 4 | Benchmarking techniques | **§V-C Benchmarking Methodology** — protocol-stratified external benchmarking, in-harness baselines (chance, MFCC-40 + RF, frozen HuBERT), and an explicit statement of what we do *not* do. |
| 5 | Latest techniques | **§II-F Recent Systems and the State of the Art** — 2024–2026 systems grouped by protocol rather than by score, with the observation that papers reporting *both* protocols are the only usable ones. |
| 6 | Comparisons | Table II (methodology), Table VII (per-fold), Table VIII (pairwise with Holm), Table XI (performance matrix), Table X (fairness across protocols). |
| 7 | Performance matrix | **Table XI** — every system we ran plus every published figure, with accuracy / macro-F1 / UAR / dispersion / parameters / protocol class. Your MFCC-40 + random-forest baseline (92.36%) is in it, and it makes the argument sharper than anything else in the table: a pre-deep-learning baseline under a contaminated protocol outscores every speaker-independent system listed. |
| 8 | Graphs explained | Every figure now has a *"Reading Fig. N"* paragraph saying what to look at, what the axes encode, and what conclusion is and is not licensed. |

---

## 3. New results extracted from your files (no new runs needed)

### 3.1 Contamination conceals the fairness failure — Table X, §VII-F

From `test_metrics.json` and `test_metrics_cross_attention_random_dup.json`,
same architecture, same seed, only the protocol differs:

| | P1 (contaminated) | P3 (speaker-disjoint) | Ratio |
|---|---|---|---|
| Female accuracy | 96.0 | 83.3 | |
| Male accuracy | 93.5 | 61.7 | |
| **Gender gap** | **2.5** | **21.6** | **8.6×** |
| Per-actor spread | 14.3 | 26.7 | 1.9× |
| Worst-class F1 | 0.915 (happy) | 0.511 (happy) | 1.8× |

A disparity audit run under P1 would report near-parity for a system that is
21.6 points worse on male speakers. This is the paper's strongest new finding:
contamination suppresses the fairness audit by a *larger relative factor*
(8.6×) than it inflates headline accuracy (1.5×). It extends the thesis from
"reported numbers are wrong" to "the safety audits built on them are wrong too."

### 3.2 Contamination is not uniform across classes — Table IX, §VII-E

Both classification reports, side by side, with ΔF1:

```
happy    +0.404      fearful  +0.083
neutral  +0.396      disgust  +0.118
sad      +0.348      surprised +0.171
calm     +0.221      angry    +0.189
```

The two classes a speaker-independent system most needs to fix are the two a
contaminated evaluation reports as solved.

### 3.3 UAR column completed everywhere

Computed from the fold files: P4 concat 66.31 ± 5.74, gated 65.78 ± 4.64,
cross-attention 64.52 ± 4.59, plus UAR paired comparisons in Table VIII.

### 3.4 Per-fold table — Table VII

Every mean, SD and paired difference in the paper reproduces from it. Fold 4
(actors 2, 10, 12, 13, 15) is hardest for all three operators, which is the
concrete reason the paired design matters.

### 3.5 Complexity analysis — Table XII, §VII-H

Parameters and MACs per utterance for all five configurations. Cross-attention
costs 3.11× the parameters and 5.28× the operations for the worst accuracy.
The frozen encoder dominates by four to five orders of magnitude (~16 GMAC for
a 3.7 s utterance), so the case for the gate rests on parameter economy and
interpretability, not latency — and the paper says so rather than overclaiming.

### 3.6 Statistics recomputed from the fold files

Holm applied within metric (3 tests each), macro-F1 pre-declared primary:

```
Macro-F1  gated−xattn  Δ+1.97  SD 0.76  p_Holm 0.013  dz 2.59   ← survives
          concat−xattn Δ+1.86  SD 2.52  p_Holm 0.348  dz 0.74
          gated−concat Δ+0.10  SD 3.02  p_Holm 0.942  dz 0.03
Accuracy  gated−xattn  Δ+1.97  SD 1.15  p_Holm 0.055  dz 1.72
          concat−xattn Δ+2.03  SD 2.61  p_Holm 0.313  dz 0.78
          gated−concat Δ−0.07  SD 2.44  p_Holm 0.954  dz −0.03
UAR       gated−xattn  Δ+1.27  SD 1.02  p_Holm 0.150  dz 1.24
          concat−xattn Δ+1.80  SD 2.59  p_Holm 0.390  dz 0.70
          gated−concat Δ−0.53  SD 3.39  p_Holm 0.744  dz −0.16

Design sensitivity: mean SD(Δ)=2.07 → δ_min = 3.47 (n=5), 2.06 (n=10), 1.23 (n=24)
```

---

## 4. What still needs a run

Four new modules are now in the project. Run them in this order:

```
python -m Experiments.run_all                 # ladder + fusion + modality, 5 seeds
python -m Features.egemaps_features           # eGeMAPSv02 via openSMILE  (pip install opensmile)
python -m Experiments.run_egemaps             # the two eGeMAPS rows of Table XIII
python -m Experiments.complementarity         # Algorithm 8
python -m Experiments.make_paper_figures --out Paper/IEEE_TASLP/figures
```

Smoke-test each with `--seeds 42` first. `run_all` is the long one (~30–60 min
on a GPU for five seeds); the rest are shorter.

### What each new module does

**`Features/egemaps_features.py`** — extracts eGeMAPSv02 (88 functionals) and
exposes `load_egemaps_features()`, which returns the same five-tuple as
`Training.train.load_features`. Nothing downstream needs changing:
`fit_fold` passes `psycho_dim=train_ds.psycho_dim`, so an 88-column table
trains as-is. It enumerates through `get_audio_files()`, so it inherits the
Algorithm 1 de-duplication guard and produces 1440 rows even if the corpus is
unpacked twice. Non-finite functionals (openSMILE can emit them for a
near-silent frame) are replaced with column means rather than being allowed to
poison the standardiser.

**`Experiments/run_egemaps.py`** — runs eGeMAPS-only and eGeMAPS+HuBERT under
P4 on the folds of Table IV, and prints the two Table XIII rows already
formatted as LaTeX.

**`Experiments/complementarity.py`** — Algorithm 8. Produces pooled
cross-validated posteriors for psycho-only, HuBERT-only and fused (each
utterance held out exactly once), captures the gate vector via a forward hook
so `Models/multimodal_model.py` needs no edit, and computes: error-set Jaccard,
the two crescents, exact McNemar of fused against the stronger unimodal system,
per-class Δrecall, gate distribution per dimension and per class, ECE and
reliability under both P4 and P1, and the margin curve with a τ* that maximises
balanced accuracy of the rule "margin ≥ τ ⇒ correct". It prints paste-ready
prose for §VII-C. Everything is per seed with across-seed mean and SD; nothing
is ensembled, because averaging posteriors across seeds would evaluate a system
you do not propose.

**`Experiments/make_paper_figures.py`** — Figs. 2, 5 and 7. Two categorical
hues carry all three: blue `#3E72CC` for the speaker-disjoint protocol, orange
`#BF5A22` for the contaminated one. The pair was validated rather than
eyeballed (CVD ΔE 24.5 protan, normal-vision ΔE 28.3, both inside the lightness
and chroma bands), and the same two serve as the diverging poles for per-class
Δrecall with a neutral midpoint — so no reader has to separate red from green.
All three were rendered and inspected against synthetic inputs of the right
shape before delivery, so the layouts are known-good; only the numbers change.

Fig. 2 draws the descent P1 → P2 → P3 → P4 with P2t and P3t as *upward offsets*
rather than rungs, since test-set selection makes a result better, not worse —
putting them in the chain would produce a zig-zag that misreads the
decomposition.

### Remaining markers

| Marker | Filled by |
|---|---|
| Table V rows P2 / P2t / P3t, and `P2 − P4` | `run_all` |
| Table XIII psychoacoustic-only / HuBERT-only | `run_all` |
| Table XIII eGeMAPS rows | `egemaps_features` + `run_egemaps` |
| Table IV seeds | `run_all` |
| §VII-C statistics | `complementarity` |
| Figs. 2, 5, 7 | `make_paper_figures` |
| §VII-E, §VII-F recomputed from pooled P4 | `complementarity` writes the posteriors |
| Affiliation, repo URL, DOI, hardware | you |

### Layout and figure wiring (done)

The manuscript now has a **one-line layout switch** at the top:

```latex
\documentclass[journal,onecolumn,11pt,draftclsnofoot]{IEEEtran}   % review     <- active
%\documentclass[journal]{IEEEtran}                                % camera-ready
```

*review* is one column, 11pt, with generous leading — IEEE's own review-copy
format. Broader measure, more air, easy to annotate. 39 pages.
*camera-ready* is the two-column T-ASLP page. 20 pages. Both compile with 0
errors, 0 undefined references and 0 overfull boxes; everything downstream
adapts automatically via `\ifCLASSOPTIONonecolumn` — figure widths, wide
floats, and table shrink-to-fit.

Two mechanical changes made this safe:

- **`\paperfig{file}{width}{aspect}`** replaces every `\includegraphics` and
  every `\reservedfig`. It draws the graphic the moment the PDF exists on
  disk, and until then reserves an area of exactly the aspect the finished
  figure will have. So running `make_paper_figures.py` *is* the whole of
  "insert the figures" — no LaTeX edit afterwards, no window where the
  document fails to compile, and no page reflow when they arrive.
- **`\adjustbox{max width=\linewidth}`** replaces `\resizebox`, which forced
  an exact width and would have magnified a narrow table in one-column mode.
  Tables now shrink only when they actually overrun.

### The draft switch

`\draftfalse` is now set, so the PDF reads clean. One deliberate departure
from "hide everything":

| Marker | `\drafttrue` | `\draftfalse` |
|---|---|---|
| `\TORUN{…}` | red note with the command that fills it | removed — these sit at the end of sentences that read correctly without them |
| `\TOADD{…}` | red note | **grey `[…]`, still visible** |
| `\RUN` (table cell) | red `--?--` | **grey `pending`** |

`\TOADD` and `\RUN` stay visible on purpose. Blanking `\TOADD` produced
"G. Sharma is with (e-mail: …)" on the title page, and printing `---` in an
unfilled result cell reads as a legitimate "not applicable" — which is the
exact failure mode this paper exists to criticise, reintroduced by its own
template. Grey and unmistakable is the right middle.

Before submitting, confirm nothing is left:

```
grep -n "TORUN\|TOADD\|\\RUN\b" paer_llm_ieee.tex     # currently 30 hits
```

When that returns nothing, the two markers are inert and the switch no longer
matters.

## 5. Files

| File | Purpose |
|---|---|
| `paer_llm_ieee.tex` | **IEEE T-ASLP manuscript** — primary |
| `paer_llm_ieee.pdf` | compiled draft, 20 pp |
| `paer_llm.tex` / `.pdf` | Elsevier version, numerically corrected, no new sections |
| `figures/fig1_framework_ieee.pdf` | Fig. 1, two-column aspect |
| `figures/fig1_framework.pdf` | Fig. 1, single-column aspect (Elsevier) |
| `make_fig1_ieee.py`, `make_fig1.py` | regenerate either — edit here, not in a drawing tool |
| `references.bib` | 94 entries; `hoenig2001abuse` added for the design-sensitivity argument |

Figures 2, 5 and 7 are reserved blocks at the correct height with the target
filename in the caption, so the page layout is already right — replace
`\reservedfig{...}{...}` with `\includegraphics` when the files exist.
