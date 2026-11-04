# PRD — UrduNLP Toolkit

## 1. Problem

Urdu-speaking users produce enormous amounts of text — product reviews,
comments, customer-support chats — in Urdu script and Roman Urdu. Anyone who
wants to analyze that text (a researcher, a developer, a content platform)
hits the same wall:

1. **No normalization layer.** Roman Urdu spelling is unstandardized and Urdu
   script text arrives with mixed Arabic/Urdu character variants, optional
   diacritics, and inconsistent punctuation. Every project re-solves this
   badly, in private.
2. **No trustworthy baseline.** Published Roman Urdu sentiment results use
   different datasets, splits, and metrics, so numbers are not comparable —
   and several widely circulated figures cannot be reproduced at all.
3. **No honest failure documentation.** Models are demoed on clean examples.
   Negation ("*acha nahi hai*"), sarcasm, and heavy English code-switching —
   the cases that dominate real text — are never measured separately.

## 2. Target users

| User | Need |
|---|---|
| **NLP researchers** (low-resource languages) | A documented dataset + split + baseline they can cite and beat, with licensing clarity |
| **Developers** building Urdu-facing products | A pip-installable normalizer and a sentiment model with a stable API, not a notebook |
| **Content platforms / e-commerce teams** | A first-pass sentiment/abuse signal for Roman Urdu reviews and comments, with known error rates |
| **Students** (the author's peers) | A readable, reproducible example of an end-to-end low-resource NLP project |

## 3. Goals

- G1. Ship a normalization module whose effect is *measured* (vocabulary
  reduction and downstream macro-F1 delta), not asserted.
- G2. Publish comparable baselines: majority, TF-IDF + LogReg, TF-IDF + SVM,
  and fine-tuned XLM-R, all on one fixed, documented split.
- G3. Publish an error analysis with a defined taxonomy (negation, sarcasm,
  code-switch density, mixed sentiment) and per-slice scores.
- G4. Make everything reproducible: one command per stage, pinned deps,
  fixed seeds, metrics written to `results/` as JSON.
- G5. Provide a public demo a non-technical person can try in under a minute.

## 4. Non-goals

- No machine translation, summarization, or generative chat.
- No claim of production-grade moderation: outputs are decision *support*,
  and the docs must say so.
- No redistribution of any dataset whose license forbids it (notably CLE
  corpora) — IDs, features, and instructions only.
- No support for other Perso-Arabic languages (Pashto, Sindhi, Punjabi-Shahmukhi)
  in MVP; the architecture should not forbid them later.
- No mobile app.

## 5. User stories

- *As a researcher,* I want a fixed train/dev/test split with a dataset card,
  so my results are comparable to the published baseline.
- *As a developer,* I want `normalize(text)` to return cleaned text plus a
  change report, so I can debug why my model misread an input.
- *As a developer,* I want batch sentiment prediction with per-item confidence,
  so I can route low-confidence items to human review.
- *As a platform analyst,* I want to paste a Roman Urdu review into a demo and
  see the prediction, the normalized form, and *why the model might be wrong*,
  so I can judge whether to trust it for my data.
- *As a student,* I want to run `make baseline` on a laptop and get the
  reported macro-F1 within a small tolerance, so I can learn from a working
  pipeline instead of a screenshot.

## 6. Functional requirements

- **FR-1 Script detection:** label any input as `urdu_script`, `roman_urdu`,
  or `mixed`, with the character-ratio evidence exposed.
- **FR-2 Urdu-script normalization:** Unicode NFC; map Arabic variants to Urdu
  forms (ك→ک, ي→ی, ة handling); optional diacritic (zabar/zer/pesh) stripping
  as a config flag; tatweel removal; normalize Urdu punctuation spacing.
- **FR-3 Roman Urdu normalization:** case folding; elongation compression
  ("bohattt" → "bohat", max run length configurable); canonical spelling map
  for high-frequency variants (mein/main/mai → mein; bohat/bohut → bohat;
  nahi/nai/nhi → nahi) maintained as a versioned YAML lexicon; preserve an
  `original` copy and emit a per-token change list.
- **FR-4 Code-switch preservation:** English words are never "corrected" into
  Urdu spellings; URLs, @mentions, #hashtags are masked to `<URL>`, `<USER>`,
  `<HASHTAG>` tokens (configurable).
- **FR-5 Tokenization:** split on whitespace and Urdu/Roman punctuation;
  keep masked tokens atomic; sentence segmentation on ۔ . ! ? ؟.
- **FR-6 Sentiment models:** train/evaluate per TRD on identical splits;
  expose `predict(text) -> {label, scores, model_version}`.
- **FR-7 Evaluation harness:** single command writes macro-F1, per-class F1,
  confusion matrix, and the three error slices to `results/`.
- **FR-8 Demo:** per UI_UX_BRIEF — playground, model comparison, dataset
  samples, limitations page.
- **FR-9 Dataset/model cards:** every released model ships a Hugging Face
  model card (data, metrics, limitations); every dataset a card (source,
  license, collection method, known biases).

## 7. Success metrics

| Metric | Target | How measured |
|---|---|---|
| Vocabulary reduction from Roman Urdu normalization | ≥ 15% fewer unique tokens on the review corpus | `evaluate.py` data report |
| Baseline quality | SVM/LogReg macro-F1 beats majority baseline by a wide, reported margin; XLM-R ≥ classical baseline on the same split | `results/metrics.json` |
| Error-slice transparency | macro-F1 reported separately for negation, sarcasm, high-code-switch slices (no minimum — the *reporting* is the requirement) | `results/error_slices.json` |
| Reproducibility | Fresh clone + `make baseline` reproduces headline macro-F1 within ±0.01 | CI-style script, fixed seed |
| Demo latency | < 1 s per single-text prediction on CPU (classical model); transformer may be slower and must say so in UI | demo timing log |
| Normalization test coverage | ≥ 90% line coverage on `normalize.py`, with a fixture case per rule | pytest |

## 8. Scope

**MVP (must ship):** FR-1…FR-7, FR-9, Gradio demo (playground + limitations),
one Roman Urdu sentiment dataset + one Urdu-script dataset, both model families.

**Stretch (only after MVP is measured and documented):**
- FastAPI service per BACKEND_SCHEMA.md with API keys and prediction logging.
- Second task head: abusive-language classification using CLE corpora (if the
  data agreement arrives in time).
- Hugging Face Space deployment + ONNX export of the transformer for CPU use.
- Annotation helper: a small page to label 200 fresh examples and measure
  inter-annotator agreement against the existing labels.
