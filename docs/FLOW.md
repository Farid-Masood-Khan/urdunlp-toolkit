# FLOW — UrduNLP Toolkit

User flows (demo), request/data flow at prediction time, and the offline
training/evaluation flow. Diagrams are Mermaid; they render on GitHub and in
most Markdown previews.

## 1. Demo user flow

```mermaid
flowchart TD
    A[User opens demo] --> B{Lands on Playground tab}
    B --> C[Types/pastes Urdu or Roman Urdu text<br/>or taps an example chip]
    C --> D{Input checks}
    D -- empty --> E[Inline hint: paste text or pick an example]
    D -- over 2000 chars --> F[Error state: trim or split guidance]
    D -- ok --> G[Script badge appears:<br/>Urdu script / Roman Urdu / Mixed]
    G --> H[User picks model:<br/>SVM baseline default · XLM-R optional]
    H --> I[Analyze pressed]
    I --> J[Loading state<br/>spinner + which model is running]
    J --> K[Result card: label + confidence bars<br/>+ low-confidence warning if max prob &lt; 0.55]
    K --> L{User explores}
    L -- View normalization changes --> M[Diff list: token → token, rule applied]
    L -- Compare models --> N[Side-by-side: baseline vs XLM-R<br/>agreements highlighted, disagreements flagged]
    L -- Another example --> C
    L -- Read why it may be wrong --> O[Limitations panel:<br/>negation, sarcasm, domain shift notes]
```

Key points:
- The script badge appears *before* analysis (on text change) — it teaches the
  user that the pipeline treats the two scripts differently.
- Disagreement between models is a feature, not a bug: the comparison view
  exists so visitors see honest uncertainty instead of one confident answer.

## 2. Prediction data flow (library + API)

```mermaid
sequenceDiagram
    participant U as User / Client
    participant D as Demo (Gradio) or API (FastAPI)
    participant N as normalize()
    participant R as Model registry
    participant M as Model (SVM or XLM-R)

    U->>D: text (+ optional model choice)
    D->>D: validate length / non-empty (422 if invalid)
    D->>N: normalize(text, config)
    N-->>D: normalized_text, script, change_report[]
    D->>R: load(model_id) — cached after first load
    R-->>D: model handle + model_version
    D->>M: predict_proba(normalized_text)
    M-->>D: label, scores{positive, neutral, negative}, latency_ms
    D->>D: build response (label, scores, normalized, changes)
    Note over D: API only — log prediction<br/>(input SHA-256 hash, NOT raw text,<br/>label, confidence, latency) if logging enabled
    D-->>U: result JSON / rendered result card
```

## 3. Training & evaluation flow (offline)

```mermaid
flowchart LR
    A[Dataset sources<br/>Kaggle / HF public sets<br/>CLE request-only] --> B[Ingest + provenance log<br/>source, license, date, checksum]
    B --> C[Clean + label mapping<br/>to pos/neu/neg schema]
    C --> D[Dedup check<br/>near-duplicates forced<br/>into same split]
    D --> E[Fixed split seed 42<br/>train 80 / dev 10 / test 10<br/>IDs committed]
    E --> F1[Train classical<br/>TF-IDF + LogReg / SVM]
    E --> F2[Fine-tune XLM-R<br/>early stop on dev macro-F1]
    F1 --> G[Evaluate on test<br/>identical split for all models]
    F2 --> G
    G --> H[results/metrics.json<br/>+ confusion matrices]
    G --> I[Error slices<br/>negation · sarcasm · code-switch]
    I --> J[Error analysis notebook<br/>15-20 examples, tagged taxonomy]
    H --> K[README results table<br/>+ HF model card<br/>generated from metrics.json]
    J --> K
```

Rules enforced by this flow:
- No model is ever evaluated on a split it trained on; the split is generated
  once, before any training run.
- The README table and the Hugging Face model card are *generated from*
  `results/metrics.json` — they cannot drift from the measured numbers.
- Test labels are never edited after results are seen; suspected label noise
  is documented, not silently fixed.

## 4. Dataset acquisition flow (licensing gate)

```mermaid
flowchart TD
    A[Candidate dataset found] --> B{License allows<br/>research use?}
    B -- no / unclear --> C[Do not download.<br/>Record in data/README.md as rejected + reason]
    B -- yes, public --> D[Download to data/raw/<br/>log URL, date, checksum]
    B -- yes, request-only<br/>(e.g. CLE) --> E[Submit request form + agreement]
    E --> F{Approved?}
    F -- no / no reply by Week 2 --> G[Proceed without it;<br/>note as stretch in plan]
    F -- yes --> H[Store raw text OUTSIDE git<br/>only IDs + features committed]
    D --> I[Dataset card drafted:<br/>source, license, size, label schema, known biases]
    H --> I
```
