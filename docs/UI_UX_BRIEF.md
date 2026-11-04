# UI/UX Brief — UrduNLP Toolkit Demo

The demo is the public face of the project. A professor or a developer should
understand what the toolkit does, try it on real text, and see its limits —
in under two minutes. Framework: **Gradio** (fallback Streamlit if Gradio
deployment blocks); this brief is framework-agnostic at the layout level.

## 1. Screens (tabs)

### Tab 1 — Playground (default)
Purpose: try one text end-to-end.

Layout (single column, max width ~900px, responsive):
```
┌──────────────────────────────────────────────────────┐
│ UrduNLP Toolkit — Urdu & Roman Urdu sentiment        │
│ [Script badge: auto-detected]     [Model: SVM ▾]     │
│ ┌──────────────────────────────────────────────────┐ │
│ │ Text input (textarea, 4 rows)                    │ │
│ │ placeholder: یہاں اردو لکھیں یا Roman Urdu…       │ │
│ └──────────────────────────────────────────────────┘ │
│ Examples: [acha nahi hai 😕] [delivery slow thi but  │
│  quality amazing] [kya bakwas service hai, wah] [مزید]│
│ [ Analyze ]        char count: 0 / 2000              │
│ ── Result card (appears after analysis) ──           │
│ Sentiment: Negative ●  confidence bars per class     │
│ ⚠ low-confidence note (only when max prob < 0.55)    │
│ Normalized text: ……  [Show changes ▾]                │
└──────────────────────────────────────────────────────┘
```

Components:
- **Example chips** — preloaded, deliberately hard cases: a negation
  ("*ye product acha nahi hai*"), a code-switched mixed review ("*delivery
  bohat slow thi but quality was amazing*"), a sarcastic one ("*wah, kya
  kamaal ki service hai, teen din se wait kar raha hoon*"), and one Urdu-script
  example. Hard examples are the demo's honesty statement.
- **Model selector** — `SVM baseline (fast)` default; `XLM-R (slower, CPU)`.
  Switching re-runs analysis on the current text.
- **Confidence bars** — one bar per class with numeric %; sentiment is never
  encoded by color alone (label text + icon + bar).
- **Show changes toggle** — expands the normalization change list
  (token → token, rule name), from the pipeline's change report.

### Tab 2 — Compare models
Same input, two result columns (SVM vs XLM-R). Where predictions agree, a
neutral "both models agree" note; where they disagree, the disagreement is
highlighted with one line: "*Disagreement is common on negation and sarcasm —
see Limitations.*" No winner is declared in the UI.

### Tab 3 — Dataset explorer
Read-only table of sample items from the public datasets: text, script,
gold label, source dataset. Filters: dataset, script, label. A banner states
which datasets are *not* shown because their license forbids redistribution
(CLE), with a link to the request process.

### Tab 4 — About & Limitations
Short version of the README "why", the measured results table (from
`results/metrics.json`), and the Limitations list (from
IMPLEMENTATION_PLAN.md §6) in plain language. This tab is mandatory, not
optional polish — it is linked from every low-confidence warning.

## 2. RTL and mixed-direction handling (critical)

Urdu script is right-to-left; the UI chrome is left-to-right. Rules:

- Any element displaying **Urdu-script user text** (input echo, normalized
  output, dataset samples) uses `dir="rtl"` and `text-align: right`, set per
  element based on the detected script of its content — never globally.
- Font stack for Urdu text: `'Noto Nastaliq Urdu', 'Gulzar', serif` with
  `line-height: 2` (Nastaliq needs vertical room; default line-height clips it).
- Roman Urdu and English text stays `dir="ltr"`. In **mixed** text, wrap runs
  in `<bdi>` / use `unicode-bidi: isolate` so embedded English words and
  numbers don't visually jump positions.
- The textarea itself switches direction live: when the detected script is
  Urdu, set the textarea `dir="rtl"`; otherwise LTR. Placeholder text shows
  one Urdu-script line so the expected direction is obvious before typing.
- Numbers, percentages, and model names inside Urdu-context strings are
  wrapped as LTR isolates so "macro-F1 0.71" never renders reversed.

## 3. States

| State | Behaviour |
|---|---|
| Empty input | Analyze disabled; hint text under the box |
| Typing | Script badge + char count update live; result card clears (stale results must not sit under edited text) |
| Loading | Spinner + text naming the model running; XLM-R shows "first run may take ~10s while the model loads" |
| Low confidence | Amber note under result: model unsure; typical causes listed (negation, sarcasm, heavy code-switching); link to Limitations |
| Error — too long | Inline error, 2000-char limit stated, offer to analyze first 2000 chars |
| Error — model unavailable | Graceful message; classical model offered as fallback if the transformer failed to load |
| No-JS / slow network | Server-rendered example results still readable (Gradio default behaviour acceptable) |

## 4. Accessibility & polish

- Contrast ≥ WCAG AA for all text, including the amber warning (dark text on
  light amber, never amber text on white).
- Full keyboard path: chips are buttons, Analyze reachable by tab, results
  region has `aria-live="polite"`.
- Sentiment icons: ▲ positive / ● neutral / ▼ negative *plus* the word —
  never color-only.
- Footer on every tab: "Research demo. Predictions are decision support, not
  ground truth — see Limitations." + link to the GitHub repo and HF model card.
