# PAER-LLM

**A unified psychoacoustic–semantic representation for speech emotion recognition, and the evaluation protocol required to assess it.**

Code, evaluation harness and manuscript for a study of speech emotion
recognition (SER) on RAVDESS. Two feature views of one audio signal — 34
perceptually motivated psychoacoustic descriptors and a frozen mean-pooled
HuBERT-base embedding — are encoded to a common 128-dimensional space and
combined by a learned per-dimension gate.

The headline result is not about the architecture.

> Holding architecture, features, optimiser, schedule and seed fixed and varying
> **only how the corpus is partitioned**, accuracy moves from **94.79 %** under a
> contaminated random split to **64.28 % ± 5.30** under speaker-disjoint five-fold
> cross-validation — a spread of **30.5 points**. Under the most defensible
> protocol, the largest difference between three fusion operators is **2.03
> points**, which is below the design's own **3.47-point** minimum detectable
> effect. Evaluation design is worth roughly fifteen times as much as the fusion
> operator, and the entire architectural range sits beneath the noise floor.

Two further results are reported against the authors' own system: single-token
cross-modal attention is proved analytically degenerate (the query stream
receives exactly zero gradient), and the corrected multi-token form is
outperformed by a single sigmoid gate at 3.1× the parameters.

---

## Status

The manuscript targets IEEE/ACM Transactions on Audio, Speech, and Language
Processing. It is **in preparation** — several result cells are still marked
`pending` while the full experimental grid runs. See
`Paper/IEEE_TASLP/RESULTS_TO_FILL.md` for exactly what is outstanding.

---

## Quick start

```bash
git clone https://github.com/<user>/paer-llm.git
cd paer-llm
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then place the corpus (see [Data](#data)) and check it before anything else:

```bash
python main.py check                    # corpus integrity - run this first
```

Build the two feature tables into `Outputs/` using the modules in `Features/`
(`pitch`, `loudness`, `bark`, `hubert`); `Notebook/03`-`06` record the sequence.
Once `Outputs/psychoacoustic_features.csv` and `Outputs/hubert_embeddings.csv`
exist:

```bash
python -m Experiments.run_all           # protocol ladder, fusion and modality ablations
python -m Features.egemaps_features     # eGeMAPSv02 table  (pip install opensmile)
python -m Experiments.run_egemaps       # eGeMAPS rows of Table XIII
python -m Experiments.complementarity   # stream complementarity + calibration
python -m Experiments.make_paper_figures --out Paper/IEEE_TASLP/figures
```

Every experiment accepts `--seeds 42` for a fast smoke test before the full
five-seed grid. `run_all` is the long one: roughly 30-60 minutes on a consumer
GPU for five seeds.

### Using the trained model

```bash
python main.py features sample.wav      # the 34 psychoacoustic descriptors
python main.py predict  sample.wav      # emotion + posterior
python main.py respond  sample.wav      # emotion + LLM reply
streamlit run Streamlit/app.py          # demonstration interface
```

---

## Data

RAVDESS is **not redistributed here**. Download the speech archive from
Zenodo — <https://doi.org/10.5281/zenodo.1188976> — and unpack it so that the
actor folders sit directly under `Dataset/`:

```
Dataset/
    Actor_01/
        03-01-01-01-01-01-01.wav
        ...
    Actor_24/
```

Only the 1440 **speech** utterances are used; the 1012 song recordings are
excluded.

### A warning worth reading before you run anything

The distributed archive nests a second copy of the corpus inside itself. Unpack
it carelessly and a recursive file walk returns **2880 paths for 1440 distinct
recordings** — every utterance appears twice, byte-identically. Nothing about
this is visible in a summary: class balance, feature distributions and file
counts all look correct. It only surfaces when you compare the number of unique
filenames against the number of rows.

Under a random split, one copy of a recording lands in train and its clone in
test. That single defect is worth a large share of the 30-point gap this paper
decomposes.

`Algorithm 1` in the manuscript is the check, implemented in
`Utils/dataset_loader.py`. The harness runs it automatically and **refuses to
proceed on a mismatch**. Do not disable it.

---

## Reproducing the paper

| Manuscript object | Produced by |
|---|---|
| Table V — protocol ladder (P1…P4) | `Experiments/run_all.py` |
| Table VI — actor/fold assignments | `Utils/splits.py`, seed 42 |
| Table VII — per-fold accuracy | `Experiments/run_all.py` |
| Table VIII — pairwise comparisons, Holm | `Experiments/run_all.py` |
| Table XI — performance matrix | `Experiments/run_all.py` + literature |
| Table XII — parameters and MACs | analytic; `Models/multimodal_model.py` |
| Table XIII — stream ablation | `run_all.py`, `run_egemaps.py` |
| Fig. 1 — framework diagram | `Paper/IEEE_TASLP/make_fig1_ieee.py` |
| Fig. 2, 5, 7 | `Experiments/make_paper_figures.py` |
| §VII-C — complementarity, calibration | `Experiments/complementarity.py` |

The manuscript sources are in `Paper/IEEE_TASLP/`. `paer_llm_ieee.tex` carries a
one-line layout switch between a one-column review copy and the two-column
camera-ready page.

### Published fold assignments

Reported so that results can be reproduced exactly rather than approximately —
one of the five reporting practices §VIII-B argues should be a precondition for
publication on acted corpora. Actors are assigned deterministically from seed 42:

| Fold | Held-out actors |
|---|---|
| 1 | 3, 8, 11, 23, 24 |
| 2 | 6, 7, 14, 18, 19 |
| 3 | 1, 9, 16, 21, 22 |
| 4 | 2, 10, 12, 13, 15 |
| 5 | 4, 5, 17, 20 |

RAVDESS numbers actors so that odd identifiers are male and even are female;
folds are drawn within gender so no partition is skewed by the composition of a
random draw.

---

## Evaluation protocols

The six protocols differ **only** in partitioning and in which partition selects
the model. Everything else is held fixed.

| | Duplicates | Partition | Model selection |
|---|---|---|---|
| P1 | present (2880) | random 80/20 | validation |
| P2 | removed (1440) | random 80/20 | validation |
| P2t | removed | random 80/20 | **test** |
| P3 | removed | actor-disjoint | validation |
| P3t | removed | actor-disjoint | **test** |
| **P4** | removed | actor-disjoint | validation, 5-fold |

**P4 is the reported protocol.** P2t and P3t reproduce undisclosed test-set
model selection so its optimism can be measured; they are never used for a
reported result. P1 reproduces the contaminated condition on purpose.

---

## Layout

```
Features/     feature extraction (pitch, intensity, Bark bands, HuBERT, eGeMAPS)
Fusion/       fusion operators
Models/       the dual-stream model
Training/     training loop, leakage-free fold fitting
Evaluation/   metrics, per-fold results and predictions
Experiments/  run_all, complementarity, run_egemaps, make_paper_figures
Utils/        paths, constants, corpus loading, speaker-disjoint splits
Notebook/     exploratory notebooks (development history, not the pipeline)
Streamlit/    demonstration interface
Paper/        manuscript sources
```

`Experiments/` is the entry point for anything the paper reports.
`Notebook/` records how the work developed and is not the route to a result.

---

## Requirements

Python 3.10+. `pip install -r requirements.txt`. Install PyTorch first if you
need a specific CUDA build: <https://pytorch.org/get-started/locally/>.

`Features/egemaps_features.py` additionally needs `pip install opensmile`, which
ships its own compiled extractor.

---

## Citation

```bibtex
@article{sharma2026paerllm,
  author  = {Sharma, Gaurav},
  title   = {A Unified Psychoacoustic--Semantic Representation for Speech
             Emotion Recognition, and the Evaluation Protocol Required to
             Assess It},
  journal = {(in preparation)},
  year    = {2026}
}
```

If you use the corpus, cite RAVDESS as well: Livingstone & Russo (2018),
*PLoS ONE* 13(5): e0196391.

---

## Licence

Code: MIT, see `LICENSE`.
Corpus: not redistributed; RAVDESS remains under CC BY-NC-SA 4.0.
