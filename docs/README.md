# UrduNLP Toolkit — NLP for Urdu & Roman Urdu

A small, reproducible, open-source toolkit for the two written forms of Urdu that
mainstream NLP largely ignores: **Urdu script** (Nastaliq) and **Roman Urdu**
(Urdu written in Latin letters, the dominant form in Pakistani social media,
e-commerce reviews, and messaging).

> Status: in development. Every number that eventually appears in this README
> must be reproducible from scripts in this repo — no borrowed benchmark figures.

## Why this exists

- Urdu has ~230 million speakers, yet off-the-shelf NLP tooling (tokenizers,
  sentiment models, normalizers) is built and benchmarked on English.
- Roman Urdu has **no standard orthography**: the same word appears as
  *mein / main / mai / mey*, *bohat / bohut / bht*. Models trained without
  normalization waste capacity re-learning spelling variants.
- Real Pakistani text is **code-switched** — Urdu and English mixed inside one
  sentence ("*delivery bohat slow thi but quality was amazing*"). Pipelines that
  assume one language per document fail on exactly the text people actually write.
- Available Urdu corpora are scattered across university labs (e.g. CLE at UET
  Lahore), Kaggle uploads of varying provenance, and Hugging Face datasets with
  uneven documentation. This toolkit documents what each dataset is, what its
  license allows, and what a model trained on it can and cannot do.

The low-resource angle is the point: careful normalization, honest evaluation,
and documented failure modes matter more here than another fine-tune.

## Features (planned)

| Module | What it does |
|---|---|
| `normalize` | Script detection (Urdu / Roman / mixed), Unicode cleanup for Urdu script, Roman Urdu spelling canonicalization, elongation compression, URL/mention/hashtag masking |
| `tokenize` | Whitespace + punctuation-aware tokenization tuned for Urdu script punctuation (۔ ، ؟) and Roman Urdu clitics |
| `sentiment` | Sentiment classification: TF-IDF + SVM/LogReg baselines vs fine-tuned XLM-R, trained and evaluated on identical splits |
| `evaluate` | One-command evaluation harness: macro-F1, per-class F1, confusion matrix, error slices (negation, sarcasm, code-switch density) |
| Demo | Gradio app: paste text → normalized text, tokens, sentiment + confidence, side-by-side model comparison |
| API (stretch) | FastAPI service exposing `/normalize`, `/tokenize`, `/sentiment` with a model registry |

## Tech stack

- **Language:** Python 3.11+
- **Classical ML:** scikit-learn (TF-IDF word + char n-grams, LinearSVC, Logistic Regression)
- **Transformers:** Hugging Face `transformers`, base model `xlm-roberta-base`
  (multilingual; UrduBERT substituted if licensing/access allows — decided in TRD)
- **Data:** pandas, datasets (HF), YAML configs for every experiment
- **Demo:** Gradio (fallback: Streamlit)
- **API (stretch):** FastAPI + SQLite model/prediction registry
- **Tooling:** pytest for normalization rules, `Makefile` targets
  (`make data`, `make baseline`, `make finetune`, `make evaluate`, `make demo`),
  pinned `requirements.txt`, fixed seeds everywhere

## Datasets (see PRD/TRD for licensing detail)

| Dataset | Variant | Task | Access |
|---|---|---|---|
| CLE Urdu abusive / threat corpora (UET Lahore) | Urdu script | abuse/threat labels reusable for negative-class analysis | Request form + agreement via CLE — do not redistribute raw text |
| Roman Urdu sentiment dataset (Daraz product reviews, Kaggle mirrors) | Roman Urdu | sentiment (pos/neu/neg) | Public; provenance documented in dataset card |
| Author-collected product-review set (existing repo) | Roman Urdu + code-switched | sentiment | Own data; collection method documented |
| HF-hosted Urdu sets (e.g. Urdu fake-news, Urdu news) | Urdu script | auxiliary eval only | Per-card license |

## Planned repo structure

```
urdu-nlp-toolkit/
├── README.md                  # this file
├── PRD.md  TRD.md  FLOW.md    # product / technical / flow docs
├── UI_UX_BRIEF.md             # demo design brief
├── BACKEND_SCHEMA.md          # API + data model (stretch service)
├── IMPLEMENTATION_PLAN.md     # week-by-week build plan
├── configs/                   # one YAML per experiment (seed, model, data)
├── data/
│   ├── raw/                   # git-ignored; download instructions in data/README.md
│   ├── interim/               # cleaned, pre-split
│   └── processed/             # train/dev/test splits (IDs only where license requires)
├── notebooks/                 # EDA + error analysis (numbered, rerun-able)
├── results/                   # metrics JSON + confusion matrices the README cites
├── src/urdu_nlp/
│   ├── normalize.py  tokenize.py  features.py
│   ├── train_baseline.py  train_transformer.py  evaluate.py  predict.py
├── demo/app.py                # Gradio demo
├── api/                       # FastAPI service (stretch)
└── tests/                     # normalization + pipeline tests
```

## Quickstart (target interface)

```bash
pip install -r requirements.txt
make data        # fetch public datasets, print instructions for request-only ones
make baseline    # TF-IDF + SVM/LogReg on the standard split
make finetune    # XLM-R fine-tune (GPU recommended; config in configs/xlmr.yaml)
make evaluate    # writes results/metrics.json + error slices
make demo        # launches the Gradio demo locally
```

## Results

_To be filled only with numbers produced by `make evaluate` in this repo._
Table will report, per dataset and model: macro-F1 (primary), per-class F1,
and the three error slices (negation, sarcasm, high code-switching). Majority
baseline always shown alongside — a model that cannot beat it does not ship.

## CV bullets this project earns when built

- Built and open-sourced an Urdu & Roman Urdu NLP toolkit (normalization, sentiment classification) with reproducible baselines — TF-IDF/SVM vs fine-tuned XLM-R — evaluated with macro-F1 and documented error analysis on negation, sarcasm, and code-switched text.
- Published Hugging Face model and dataset cards documenting provenance, licensing, and honest limitations for low-resource, code-switched Urdu text.
- Shipped a public Gradio demo and API for Urdu/Roman Urdu text normalization and sentiment prediction.

## License

Code: MIT. Datasets are **not** covered by this license — each remains under
its source license; request-only corpora (CLE) are never redistributed here,
only referenced by ID and checksum.
