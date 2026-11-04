# TRD — UrduNLP Toolkit

Technical design for the toolkit described in PRD.md. One developer builds
this; choices favour boring, reproducible, CPU-survivable technology over
novelty.

## 1. Architecture

```
            ┌─────────────┐     ┌──────────────────┐
 raw text ─▶│ script_id   │────▶│ normalize        │──▶ tokens ──▶ features/model ──▶ label + scores
            └─────────────┘     │ (config-driven)  │                    │
                                └──────────────────┘                    ▼
                                                        change report (for UI/debug)
 Library core (src/urdu_nlp/) is UI-agnostic. Two thin shells use it:
   - demo/app.py        (Gradio; calls predict() in-process)
   - api/               (FastAPI, stretch; same predict(), adds registry + logging)
 Training and evaluation are offline CLI entry points, never part of serving.
```

Design rules:
- Normalization is a **pure function** `(text, config) -> NormalizedResult`
  with no I/O — fully unit-testable with fixtures.
- Models sit behind one interface: `load(model_id)`, `predict(texts)`,
  `predict_proba(texts)`. The demo and API never import sklearn/transformers
  directly.
- Every experiment is a YAML config in `configs/`; the config hash is stored
  with its metrics so a number in the README traces to exactly one config.

## 2. Pipeline design

### 2.1 Script identification
Count characters in Unicode blocks: Arabic block (U+0600–U+06FF) vs Latin.
`urdu_ratio > 0.5` → `urdu_script`; `latin_ratio > 0.9` → `roman_urdu`;
otherwise `mixed`. Digits, emoji, and masked tokens are excluded from the ratio.

### 2.2 Normalization (ordered stages — order is part of the spec)
1. **Mask** URLs, emails, @mentions, #hashtags → placeholder tokens (kept in a
   side table, restored never — models see the placeholders).
2. **Unicode NFC** normalization.
3. **Urdu-script cleanup** (script is urdu/mixed only): Arabic→Urdu character
   mapping (ك→ک U+0643→U+06A9, ي→ی U+064A→U+06CC, etc.), tatweel (U+0640)
   removal, optional diacritics strip (flag `strip_diacritics`, default on for
   classification, off for display), zero-width joiner/non-joiner preserved.
4. **Roman cleanup:** lowercase; elongation compression (any char repeated
   > 2 times collapses to 2, then lexicon step fixes known words, e.g.
   "bohaat"→ stage 5); punctuation spacing normalized.
5. **Canonical lexicon mapping:** versioned YAML (`lexicon/roman_urdu_v1.yaml`)
   mapping variant → canonical form for the top ~500 Roman Urdu words by
   corpus frequency (mein/main/mai/mey → mein; nahi/nai/nhi → nahi;
   bohat/bohut/bht → bohat). Lookup is whole-token only — substrings are never
   rewritten, and English tokens are excluded via a small English stopword
   guard list so "main" (English: primary) in a code-switched sentence is only
   mapped when the sentence is classified Roman/mixed *and* the token is not
   sentence-initial capitalized… — this ambiguity is a **known limitation**,
   logged in the change report whenever the ambiguous token `main` is mapped.
6. **Change report:** every stage appends `(original_token, new_token, rule)`
   entries; the demo's "view changes" toggle renders this.

### 2.3 Tokenization
Whitespace split after normalization, then split trailing/leading punctuation
from tokens (Urdu: ۔ ، ؛ ؟ ' " ; Roman: . , ! ? ' "). Masked placeholders are
atomic. Sentence segmentation is provided (`split_sentences`) but the
classifiers operate at document level for MVP.

### 2.4 Models
| Model | Features / setup | Role |
|---|---|---|
| Majority baseline | — | honesty floor in every results table |
| TF-IDF + Logistic Regression | word n-grams (1,2) + char n-grams (3,5), sublinear TF, `class_weight="balanced"` | interpretable baseline |
| TF-IDF + LinearSVC | same features; probabilities via `CalibratedClassifierCV` | strongest classical baseline |
| XLM-R fine-tune | `xlm-roberta-base`, max_len 128, lr 2e-5, batch 16, ≤ 5 epochs, early stop on dev macro-F1, seed 42 | transformer comparator |
| UrduBERT (optional) | only if a checkpoint with a clear license is accessible at build time | documented substitution, same protocol |

All models train and are evaluated on the **identical** split files in
`data/processed/` — the split is generated once by `make data` with seed 42
and committed as document IDs.

## 3. Data design

- Splits: stratified 80/10/10 train/dev/test per dataset; near-duplicate
  reviews (normalized-text Jaccard ≥ 0.9) are forced into the same split to
  prevent leakage — this is a deliberate, documented choice.
- Label schema: `{positive, neutral, negative}`; datasets with binary labels
  are mapped and the mapping is recorded in the dataset card.
- Request-only corpora (CLE): store document IDs + features only; raw text
  never enters git. `data/README.md` documents the request process.

## 4. Evaluation methodology

- **Primary metric:** macro-F1 (classes are imbalanced; accuracy is reported
  but never headlined).
- Also: per-class precision/recall/F1, confusion matrix (PNG + JSON),
  5-fold cross-validation mean ± std for classical models; transformer
  reported on the fixed test split with the dev set used for early stopping.
- **Error slices** (computed on test predictions, definitions fixed in code):
  - *Negation slice:* test items containing a negation token
    (nahi, nai, na, mat, نہیں, نہ) — hypothesis: classical models underperform
    here because TF-IDF sees the positive word and misses scope.
  - *Sarcasm slice:* items flagged in the source data or by two-pass manual
    tagging of a 200-item sample; if the dataset has no sarcasm labels, this
    slice is built from the manual sample only and labelled as such.
  - *High code-switch slice:* items with 20–80% Latin tokens in Urdu-script
    set, or ≥ 30% English-dictionary tokens in the Roman set.
- **Error analysis write-up:** 15–20 misclassified examples quoted in
  `notebooks/03_error_analysis.ipynb`, each tagged with the taxonomy
  (negation scope, sarcasm, mixed sentiment, spelling OOV, label noise
  suspected). Suspected label noise is reported as a percentage, not hidden.
- **Cross-dataset check (honesty test):** train on product reviews, evaluate
  on a held-out slice of informal/social text if available; the expected drop
  is reported in Limitations, not tuned away.

## 5. Configuration & reproducibility

- `requirements.txt` fully pinned; Python 3.11.
- Seeds: 42 everywhere (split, training, CV folds); recorded in metrics JSON.
- `results/metrics.json` schema: `{config_hash, dataset, model, split,
  macro_f1, per_class, confusion_matrix_path, created_at, git_commit}`.
- Tests: one pytest fixture per normalization rule (input → expected output +
  expected change-report entries); pipeline smoke test on a 10-row fixture.

## 6. Deployment

- **Demo:** Gradio app; runs locally with `make demo`; deployable to a
  Hugging Face Space (CPU) — classical model is default there (fast, small);
  transformer selectable with a visible "slower on CPU" note.
- **Artifacts:** fine-tuned model pushed to HF Hub with a model card generated
  from `results/metrics.json`; large artifacts never in git.
- **API (stretch):** FastAPI in Docker per BACKEND_SCHEMA.md; model loaded
  once at startup from the registry; `/health` reports model version.

## 7. Technical risks

| Risk | Mitigation |
|---|---|
| Roman Urdu canonical lexicon overfits one dataset's spelling habits | build lexicon from corpus frequency across *both* datasets; version it; report vocab-reduction on dev, not train |
| XLM-R underperforms SVM on small data | expected and acceptable — report honestly; the comparison is the contribution |
| Label noise in scraped/Kaggle data | manual audit of a 200-item sample; report estimated noise; never silently "clean" test labels after seeing results |
| GPU access for fine-tuning | free-tier Colab-class GPU is sufficient for XLM-R-base at len 128; training script must also run CPU-slow rather than fail |
