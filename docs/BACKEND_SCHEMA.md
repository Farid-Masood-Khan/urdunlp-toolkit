# Backend Schema — UrduNLP Toolkit API (stretch service)

The toolkit is a library first. This FastAPI service is the **stretch** layer
that exposes it over HTTP with a model registry and auditable prediction
logging. Storage: SQLite for MVP-of-the-service (single file, zero ops);
schema is written to be portable to Postgres (no SQLite-only types).

## 1. Data model

### corpora — datasets the project trains/evaluates on
| field | type | notes |
|---|---|---|
| id | INTEGER PK | |
| name | TEXT UNIQUE | e.g. `roman_urdu_reviews_kaggle`, `cle_urdu_abusive` |
| language_variant | TEXT CHECK in (`urdu_script`,`roman_urdu`,`mixed`) | |
| task | TEXT | `sentiment`, `abuse` |
| source_url | TEXT | where it was obtained |
| license | TEXT | e.g. `CC-BY-4.0`, `CLE-research-agreement`, `unknown-do-not-redistribute` |
| license_notes | TEXT | redistribution limits in plain words |
| size_docs | INTEGER | document count at ingest |
| checksum_sha256 | TEXT | of the raw file as downloaded |
| created_at | TIMESTAMP | |

### documents — individual labelled items (public-license corpora only)
| field | type | notes |
|---|---|---|
| id | INTEGER PK | |
| corpus_id | INTEGER FK → corpora.id | |
| external_id | TEXT | ID in the source dataset; UNIQUE per corpus |
| text | TEXT NULL | **NULL for request-only corpora (CLE)** — text never stored, only IDs/features |
| script | TEXT | `urdu_script` / `roman_urdu` / `mixed` (detector output, auditable) |
| split | TEXT CHECK in (`train`,`dev`,`test`) | assigned once by the fixed-seed splitter |
| gold_label | TEXT | `positive` / `neutral` / `negative` |

### annotations — extra human labels (agreement studies, error-analysis tags)
| field | type | notes |
|---|---|---|
| id | INTEGER PK | |
| document_id | INTEGER FK → documents.id | |
| annotator | TEXT | initials or pseudonymous ID |
| label | TEXT | annotator's label |
| tag | TEXT NULL | error taxonomy tag: `negation`, `sarcasm`, `code_switch`, `mixed_sentiment`, `label_noise_suspected` |
| created_at | TIMESTAMP | |

### models — model registry
| field | type | notes |
|---|---|---|
| id | INTEGER PK | |
| name | TEXT | e.g. `tfidf_svm_v1`, `xlmr_finetune_v1` |
| family | TEXT CHECK in (`majority`,`tfidf_logreg`,`tfidf_svm`,`transformer`) | |
| base_model | TEXT NULL | HF checkpoint for transformers, e.g. `xlm-roberta-base` |
| version | TEXT | semver per model name; UNIQUE(name, version) |
| artifact_uri | TEXT | local path or HF Hub repo |
| config_hash | TEXT | hash of the training YAML — ties metrics to one config |
| is_default | BOOLEAN | the model `/sentiment` uses when none is specified |
| created_at | TIMESTAMP | |

### model_metrics — one row per (model, corpus, split) evaluation
| field | type | notes |
|---|---|---|
| id | INTEGER PK | |
| model_id | INTEGER FK → models.id | |
| corpus_id | INTEGER FK → corpora.id | |
| split | TEXT | normally `test` |
| macro_f1 | REAL | primary metric |
| accuracy | REAL | reported, never headlined |
| per_class_json | TEXT (JSON) | `{label: {precision, recall, f1, support}}` |
| slice_json | TEXT (JSON) NULL | `{negation: {n, macro_f1}, sarcasm: {...}, code_switch: {...}}` |
| git_commit | TEXT | code version that produced this row |
| created_at | TIMESTAMP | |

### prediction_logs — API/demo usage (privacy-first)
| field | type | notes |
|---|---|---|
| id | INTEGER PK | |
| model_id | INTEGER FK → models.id | |
| source | TEXT CHECK in (`demo`,`api`) | |
| input_hash | TEXT | SHA-256 of the raw input — **raw text is not stored** |
| input_preview | TEXT NULL | first 80 chars, only when caller explicitly opts in (`log_preview: true`) |
| script | TEXT | detected script |
| predicted_label | TEXT | |
| confidence | REAL | max class probability |
| latency_ms | INTEGER | |
| api_key_id | INTEGER FK → api_keys.id NULL | NULL for demo traffic |
| created_at | TIMESTAMP | |

### api_keys
| field | type | notes |
|---|---|---|
| id | INTEGER PK | |
| key_hash | TEXT UNIQUE | SHA-256 of the key; plaintext shown once at creation |
| owner_email | TEXT | |
| tier | TEXT CHECK in (`free`,`research`) | free: 60 req/min · research: 600 req/min |
| is_active | BOOLEAN | |
| created_at / last_used_at | TIMESTAMP | |

## 2. API endpoints (base path `/v1`)

### POST /v1/normalize
```json
// request
{ "text": "ye product bohattt acha nahi hai!!", "options": { "strip_diacritics": true } }
// 200 response
{
  "normalized": "ye product bohat acha nahi hai",
  "script": "roman_urdu",
  "changes": [
    { "original": "bohattt", "normalized": "bohat", "rule": "elongation" },
    { "original": "!!", "normalized": "", "rule": "punctuation" }
  ]
}
```

### POST /v1/tokenize
```json
// request
{ "text": "delivery slow thi, but quality amazing thi" }
// 200 response
{ "script": "mixed", "tokens": ["delivery", "slow", "thi", ",", "but", "quality", "amazing", "thi"],
  "sentences": ["delivery slow thi, but quality amazing thi"] }
```

### POST /v1/sentiment
```json
// request  (model optional; default = registry default)
{ "text": "camera bohat acha hai lekin battery bakwas hai", "model": "tfidf_svm_v1" }
// 200 response
{
  "label": "negative",
  "scores": { "positive": 0.21, "neutral": 0.17, "negative": 0.62 },
  "model": { "name": "tfidf_svm_v1", "version": "1.0.0" },
  "script": "roman_urdu",
  "normalized_text": "camera bohat acha hai lekin battery bakwas hai",
  "low_confidence": false,
  "latency_ms": 12
}
```
`low_confidence` is `true` when max score < 0.55 — clients should surface it
exactly as the demo does.

### POST /v1/sentiment/batch
Request: `{ "texts": ["...", "..."], "model": "xlmr_finetune_v1" }` (max 100 items).
Response: `{ "results": [ <sentiment response object>, ... ] }` in input order.

### GET /v1/models
```json
{ "models": [ { "name": "tfidf_svm_v1", "version": "1.0.0", "family": "tfidf_svm",
  "is_default": true, "test_macro_f1": 0.0 } ] }
```
(`test_macro_f1` filled from model_metrics once evaluation has run; `0.0`
placeholder is replaced by the generated value, never typed by hand.)

### GET /v1/models/{name}/metrics
Latest model_metrics rows for that model across corpora, including per-class
and slice JSON — this is the machine-readable version of the README table.

### GET /v1/health · GET /v1/version
Health: `{ "status": "ok", "default_model": "tfidf_svm_v1", "models_loaded": 2 }`.
Version: service version + git commit.

### Errors (all endpoints)
```json
{ "error": { "code": "text_too_long", "message": "Text exceeds 2000 characters." } }
```
| HTTP | code | when |
|---|---|---|
| 422 | `empty_text` / `text_too_long` | validation fails |
| 404 | `model_not_found` | unknown model name |
| 429 | `rate_limited` | tier limit exceeded; `Retry-After` header set |
| 503 | `model_unavailable` | model failed to load; classical fallback offered in message |

## 3. Auth model

- **Demo traffic** (from the Gradio app): no key; rate-limited by IP
  (30 req/min); prediction logs have `api_key_id = NULL`.
- **Programmatic API:** `Authorization: Bearer <api_key>` required for
  `/v1/sentiment/batch` and for limits above the demo tier; single-item
  endpoints accept anonymous calls at the free rate.
- **Admin** (create keys, register models): separate `X-Admin-Key` header,
  value from environment only (`URDU_NLP_ADMIN_KEY`), never in code or git.
- Keys are stored hashed; a leaked database does not leak usable keys.
