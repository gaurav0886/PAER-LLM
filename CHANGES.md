# PAER-LLM — Code Review and Changes

Reviewed: every `.py` file plus notebooks 02, 05, 06, 07, 09, 10, 11, 12, 13.
All source files listed below have been rewritten in place. Originals are in
`_backup_original/`.

---

## The three findings that change your results

### 1. The dataset is duplicated — every recording appears twice

Your `Dataset/` folder contains RAVDESS twice:

```
Dataset/Actor_01 … Dataset/Actor_24                     1440 files
Dataset/audio_speech_actors_01-24/Actor_01 … Actor_24   1440 files  ← same corpus
```

`os.walk` returns 2880 paths, so **every feature CSV in `Outputs/` has 2880 rows
and 1440 unique filenames.** Verified directly on your machine:

```
$ find Dataset -name "*.wav" | wc -l          → 2880
$ python: rows 2880, unique 1440              → pitch_features.csv
```

`train_test_split` then puts one copy of a recording in train and its
byte-identical clone in test. The model is graded on samples it memorised.

This alone accounts for most of the gap between your 95.14% and the ~60-75%
that speaker-independent RAVDESS systems typically report. The class supports in
`Evaluation/classification_report.txt` confirm it: 77 per class, 38 for neutral —
exactly 20% of 2×192 and 2×96.

**Fixed:** `Utils/dataset_loader.py` de-duplicates on RAVDESS basename, and
`deduplicate_feature_table()` cleans the existing CSVs so you do **not** need to
re-extract HuBERT embeddings — the duplicate rows are identical, dropping them
loses nothing.

**You should still do:** move `Dataset/audio_speech_actors_01-24/` out of
`Dataset/` (I left your data untouched). Run `python main.py check` to confirm
1440 / 24 actors.

---

### 2. The cross-attention layer is a no-op — the psychoacoustic branch was never used

This is the serious one, because it is the thesis's central contribution.

`Models/multimodal_model.py` unsqueezed both modalities to sequence length 1
before calling `CrossAttention`. With one query token and one key token the
score matrix is `(B, 1, 1)`, and **softmax over a single element is always
exactly 1.0**. So:

```
output = attention @ V = 1.0 × V = value_proj(hubert_embedding)
```

The query — the entire psychoacoustic branch — is computed, multiplied by a
constant, and discarded.

I ran your original code verbatim to confirm it. Multiplying the psychoacoustic
input by 100 while holding HuBERT fixed:

```
attention tensor shape        : (4, 1, 1)
attention values              : [1.0, 1.0, 1.0, 1.0]
max |logits(p1) − logits(p2)| : 0.0          ← identical output
grad w.r.t. psycho input      : 0.0
psycho_encoder.weight.grad    : all zeros    ← never received a gradient
```

`psycho_encoder` was never trained. Its weights are still at random
initialisation in `best_model.pth`.

Two knock-on consequences:

- **Your attention maps are uninformative.** Every value is 1.0 by
  construction. Any attention-weight figure derived from them shows nothing.
- **There are no activations anywhere in the network.** `psycho_encoder`,
  `hubert_encoder`, the three attention projections and `classifier` are all
  bare `nn.Linear`. A composition of linear maps is one linear map — the "deep
  multimodal network" had the expressive power of logistic regression on
  mean-pooled HuBERT features.

**Fixed:** `Fusion/cross_attention.py` now provides `MultiHeadCrossAttention`
(heads, output projection, dropout, masking) and `TokenCrossAttention`, which
expands each pooled vector into K learned tokens so the softmax runs over K > 1
and the mechanism actually selects. `Models/multimodal_model.py` uses
*bidirectional* fusion (psycho→HuBERT and HuBERT→psycho) with LayerNorm, GELU,
dropout and residuals. The old `CrossAttention` name still imports but raises a
clear error if handed length-1 sequences.

---

### 3. Three independent sources of optimistic bias in the evaluation

| Problem | Where | Effect |
|---|---|---|
| Scaler fit on all 2880 rows *before* splitting | `07_multimodal_feature_fusion.ipynb` cell 6 | Test-set mean/variance baked into training inputs. `multimodal_features.csv` is contaminated — do not use it for any reported number. |
| Random utterance split, not speaker-independent | `09_training.ipynb` cell 7 | Same 24 actors speaking the same 2 sentences appear in train and test. The model can win by recognising the speaker. |
| Checkpoint selected on the test set, then reported as the test score | `09_training.ipynb` cell 29 → `10_evaluation.ipynb` | `test_loader` was used as the validation set for `if val_accuracy > best_accuracy: save`. The reported 95.14% is a model-selection score presented as a generalisation score. |

There was also no handling of class imbalance (neutral has 96 utterances, every
other class 192) while reporting plain accuracy, which flatters the number
further.

**Fixed:** `Training/train.py` reads the *unscaled* CSVs, de-duplicates, splits
by actor (`Utils/splits.py`), fits scalers on the training fold only, keeps a
separate validation fold for early stopping, and saves the test indices.
`Evaluation/evaluate.py` loads those exact indices and touches the test set
once, reporting accuracy, macro-F1 and UAR with bootstrap 95% CIs plus a
per-actor breakdown.

---

## Other bugs fixed

| File | Issue |
|---|---|
| `Features/bark.py` | `extract_bark_features` **defined twice** — the first was an empty stub returning `None`, shadowed only by luck of ordering. Filter bank built with a Python double loop (now vectorised, ~100× faster, cached) and un-normalised, so wide high-frequency bands accumulated energy purely by spanning more FFT bins. Unused `os` / `EMOTION_MAP` imports. |
| `Features/spectral.py` | `np.mean(contrast)` collapsed a 7×T spectral-contrast matrix to one scalar, destroying the per-band peak-to-valley information that *is* spectral contrast. Now returns 7 band means. **Also: these features are never used** — `05_feature_fusion.ipynb` loads `spectral_features.csv`, checks its `file` column, then concatenates only pitch + loudness + bark. None of your 34 psychoacoustic dimensions come from it. Wire it in or drop the claim from the write-up. |
| `Features/hubert.py` | Model loaded as an **import-time side effect** — importing anything from `Utils.inference` downloaded and instantiated a 95M-param model. Debug `print` of shapes and first-10 values on **every** file (4320 lines over the corpus). Never moved to GPU. No batching. Now lazy, cached, GPU-aware, with a batched variant that masks padded frames. |
| `Utils/inference.py` | Hard-coded `"../Models/best_model.pth"` — worked from `Notebook/`, crashed from the project root. Two `joblib.load` calls **per prediction**. Hard-coded `emotion_labels` list instead of the saved `LabelEncoder`. Doubly-nested `with torch.no_grad():`. `generate_response()` reloaded the classifier *and* the 3.8B LLM on every call. |
| `LLM/llm_model.py` | `torch_dtype=torch.float16` unconditionally — broken/very slow on CPU. `device_map="auto"` needs `accelerate`, which is in neither requirements file. `trust_remote_code=True` unnecessarily. `do_sample=False` made every reply for a given emotion byte-identical. `max_tokens=50` truncated mid-sentence. |
| `LLM/prompt_builder.py` | Pasted 10 raw floats from the fusion layer into the prompt. An LLM cannot interpret activations from a network it wasn't trained with — they're token noise, and an ablation would show zero effect. If a reviewer asks what the embedding contributes to the response, the honest answer is "nothing". Replaced with the class distribution (genuinely actionable: low confidence → hedge) and the tokenizer's own chat template instead of hand-written `<\|system\|>` tokens. |
| `Streamlit/app.py` | `os.path.abspath("..")` resolved against cwd, so launching from the project root put `F:\PhD` on `sys.path`. Wrote `temp.wav` into the cwd and never cleaned it up (that's the stray `Streamlit/temp.wav` in your repo); concurrent users would clobber each other. LLM loaded eagerly at startup. No error handling. |
| `Features/pitch.py` | Function docstring floating at module level above the imports, so `help()` showed nothing. Praat default ceiling (600 Hz) truncates high-arousal and female F0 — now explicit and documented. |
| `Utils/constants.py` | Absolute Windows paths (`F:\PhD\...`) made the project unrunnable anywhere else — fatal for a thesis artefact a reviewer needs to reproduce. |
| `Training/emotion_dataset.py` | `torch.tensor()` on an existing tensor warns and copies. No length validation — a mismatch surfaced as an opaque indexing error deep in the training loop. |
| `requirements.txt` / `requirement.txt.txt` | Two files that disagree. The first has no `torch`/`transformers` (project can't run); the second has no `streamlit`/`joblib`. **Neither lists `praat-parselmouth`**, which `Features/pitch.py` imports — a clean install from either crashes on the first pitch extraction. Merged and pinned. |

**Empty stub files left in place** (0 bytes, never imported): `Features/feature_fusion.py`,
`Features/opensmile.py`, `Fusion/feature_fusion.py`, `LLM/response_generator.py`,
`LLM/utils.py`. Delete them or fill them in.

---

## Breaking changes

- **`best_model.pth` will not load.** The architecture changed, and the old
  checkpoint was trained on duplicated data with a dead psychoacoustic branch —
  it should not be reused. `MultimodalEmotionModel.load()` raises a clear
  message rather than failing cryptically.
- **`predict()` now returns 4 values**, not 3: `(emotion, confidence, embedding,
  probabilities)`. Notebooks 13 and 14 need updating.
- **`extract_spectral_features()` returns 11 values**, not 5. Nothing depends on
  it (see above), but regenerating `spectral_features.csv` changes its columns.
- **`build_prompt()` takes `probabilities=` instead of `embedding=`.** The old
  `embedding=` kwarg is still accepted and ignored.

---

## How to re-run

```bash
cd F:\PhD\PhD_Project
pip install -r requirements.txt

python main.py check                      # confirm 1440 utterances / 24 actors

python -m Training.train                  # speaker-independent, honest protocol
python -m Evaluation.evaluate             # held-out test, run once

# For the thesis — a single split of 24 actors is noisy:
python -m Training.train --cv 5           # 5-fold actor-disjoint CV, mean ± std

# Ablation table (this is what makes the cross-attention claim credible):
python -m Training.train --fusion concat  --cv 5
python -m Training.train --fusion gated   --cv 5
python -m Training.train --fusion cross_attention --cv 5

# To quantify how much of the old number was leakage:
python -m Training.train --split random   # reproduces the old protocol
```

Everything above was smoke-tested end to end on synthetic RAVDESS-shaped data
(1440 utterances, 24 actors, duplicated rows) in this session: de-duplication,
actor-disjoint splitting, all three fusion modes, k-fold CV, and evaluation
with bootstrap CIs all run clean.

---

## What to expect, and how to write it up

Your new numbers will be **substantially lower**. That is the correct outcome,
not a regression — you are now measuring generalisation to unseen speakers
instead of recall of duplicated recordings. For calibration: MFCC + Random
Forest on speaker-independent RAVDESS is typically 45–60%; strong
self-supervised multimodal systems land in the 70s.

The most defensible framing for the thesis is to report both and explain the
gap — a leakage-vs-clean comparison table is a genuine methodological
contribution, and it inoculates you against a reviewer finding it first.

Three things worth doing next:

1. **Report the ablation.** `concat` vs `gated` vs `cross_attention`, 5-fold. If
   cross-attention doesn't beat concatenation, say so — that is a finding.
2. **Report per-actor variance.** `evaluate.py` prints it. If accuracy swings
   30 points across held-out actors, a single split is not a result.
3. **Decide what the LLM component is actually doing.** Right now it maps a
   label to a template-ish reply. If you want the embedding to influence
   generation, the real options are a soft-prompt/prefix adapter that projects
   the fused vector into the LLM's embedding space, or retrieval over an
   emotion-annotated response bank. Either is a publishable contribution;
   string-formatting floats is not.

Also worth noting: `Docs/project_journal.md` records "Samples: 2880" and
"Accuracy: 92.36%". Both need revising once you re-run.
