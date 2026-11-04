# UrduNLP Toolkit — Low-Resource NLP for Urdu & Roman Urdu

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://github.com/python/mypy)
[![Tests: Pytest](https://img.shields.io/badge/tests-18%2F18%20passing-brightgreen.svg)](https://docs.pytest.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.0-F7931E.svg)](https://scikit-learn.org/)

A reproducible, open-source NLP toolkit engineered for the two real written forms of Urdu: **native Arabic script (Nastaliq)** and **Roman Urdu (Latin transliteration)**.

> **Why Low-Resource NLP Engineering Matters:** Roman Urdu has no standardized spelling (*bohat*, *boht*, *bht*, *bohattt*) and frequent code-switching with English. Off-the-shelf models fail because feature capacity is wasted on orthographic noise. UrduNLP eliminates this noise with a proven **15.26% vocabulary reduction** and leak-free evaluation.

---

## 1. System Architecture

```
┌─────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│ Input Raw Text  │ ──▶ │ Script Detector       │ ──▶ │ Multi-Stage Normalizer│
│ (Urdu / Roman)  │     │ (Arabic / Latin / Mix)│     │ (NFC, Diacritics, Map)│
└─────────────────┘     └───────────────────────┘     └───────────┬───────────┘
                                                                  │
                                                                  ▼
┌─────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│ Streamlit App   │ ◀── │ Calibrated Classifier │ ◀── │ Anti-Leakage Splitter │
│ & Triage Demo   │     │ (TF-IDF + LinearSVC)  │     │ (Jaccard ≥ 0.90 Graph)│
└─────────────────┘     └───────────────────────┘     └───────────────────────┘
```

---

## 2. Key Technical Features

- **Multi-Stage Normalizer (`src/urdu_nlp/normalizer.py`)**:
  - Unicode NFC normalization.
  - Arabic-to-Urdu character unification (e.g., Arabic Kaf 'ك' and Yeh 'ي' mapped to Urdu 'ک' and 'ی').
  - Non-essential diacritic stripping (Zabar, Zer, Pesh, Tanween).
  - Vowel/consonant elongation compression (`bohattt` $\rightarrow$ `bohatt`).
  - Canonical phonetic lexicon mapping (`bht` $\rightarrow$ `bohat`, `nhi` $\rightarrow$ `nahi`).
  - **Empirical Vocabulary Reduction**: Eliminates **15.26% redundant tokens** on real social corpora.
- **Atomic Placeholder Tokenizer (`src/urdu_nlp/tokenizer.py`)**: Preserves `<URL>`, `<USER>`, and `<NUM>` tokens intact while handling Urdu punctuation (۔ ، ؟).
- **Jaccard Near-Duplicate Anti-Leakage Clustering (`src/urdu_nlp/dataset.py`)**:
  Builds a pairwise Jaccard token similarity graph ($J(A, B) \ge 0.90$) and groups near-identical tweets into connected clusters. Splitting entire clusters ensures **zero near-duplicate data leakage** between Train, Dev, and Test.
- **Calibrated Sentiment Classification (`src/urdu_nlp/models/classifier.py`)**:
  Features character and word n-gram TF-IDF representations with probability calibration (Platt scaling).
- **Streamlit Interactive Inspector (`demo/app.py`)**:
  Multi-tab application for live text normalization audits, script identification, and confidence-calibrated sentiment triage.

---

## 3. Measured Empirical Results

All metrics measured against held-out test splits in `results/README.md` and `results/metrics.json`:

| Model / Baseline | Macro F1 | Weighted F1 | Accuracy | Status |
|---|---|---|---|---|
| **TF-IDF + Calibrated LinearSVC** | **0.8413** | **0.8410** | **84.15%** | **Best Model** |
| TF-IDF + Logistic Regression | 0.8124 | 0.8119 | 81.25% | Competitive Baseline |
| Majority Class Baseline | 0.1977 | 0.1448 | 29.65% | Minimal Floor |

- **Empirical Vocabulary Reduction**: **15.26%** (satisfies the contractual PRD requirement of $\ge 15\%$).

---

## 4. Quickstart & Verification

```powershell
# 1. Install dependencies
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Run unit & tokenization test suite (18 tests)
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Command "test"

# 3. Static type check (Mypy)
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Command "typecheck"

# 4. Code quality & lint (Ruff)
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Command "lint"

# 5. Run full vocabulary reduction & model evaluation
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Command "evaluate"

# 6. Launch Streamlit interactive demo
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Command "demo"
```

---

## 5. Repository Structure

```
urdunlp-toolkit/
├── README.md                  # System overview and quickstart
├── pyproject.toml             # Ruff, Mypy, and Pytest configuration
├── requirements.txt           # Pinned dependencies
├── run.ps1                    # Standardized PowerShell runner
├── data/                      # Processed datasets and data cards
├── demo/
│   └── app.py                 # Multi-tab Streamlit demo application
├── src/
│   └── urdu_nlp/
│       ├── detector.py        # Script identifier (Urdu, Roman, Mixed)
│       ├── normalizer.py      # Multi-stage text normalizer
│       ├── tokenizer.py       # Punctuation & placeholder aware tokenizer
│       ├── dataset.py         # Jaccard anti-leakage splitting engine
│       ├── evaluate.py        # Benchmark and error slice analyzer
│       ├── lexicon/           # Canonical Roman Urdu mapping dictionary
│       └── models/            # Calibrated sentiment classifiers
├── tests/                     # 18 automated unit and regression tests
└── results/                   # Evaluation artifacts and error slice analyses
```

