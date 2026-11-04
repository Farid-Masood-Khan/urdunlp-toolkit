# Implementation Plan — UrduNLP Toolkit

**Effort:** ~3 weeks part-time (≈10–12 focused hours/week, one developer).
**Sequencing rule:** measurement before polish. The demo is built *after* the
evaluation harness, so the first public screenshot already contains real
numbers. Stretch items (API service) start only when the MVP definition of
done is fully met.

## Phase 0 — Scaffolding & data acquisition (Days 1–2)

- [ ] Init repo with the structure in README.md; `Makefile`, pinned
      `requirements.txt`, `.gitignore` for `data/raw/` and model artifacts.
- [ ] Submit the **CLE dataset request form on Day 1** (approval latency is
      the plan's biggest external dependency — start the clock immediately).
- [ ] Download public datasets (Kaggle Roman Urdu sentiment; any HF Urdu set
      passing the license gate in FLOW.md §4); log URL, date, checksum,
      license into `data/README.md`.
- [ ] Draft dataset cards for the two public sets.

**Milestone M0 (end of Day 2):** repo skeleton committed; public data local
with provenance log; CLE request sent (or documented as unavailable).

## Week 1 — Normalization, tokenization, data pipeline

- [ ] `script_id` + `normalize.py` implementing TRD §2 stages 1–6, config-driven.
- [ ] Build `lexicon/roman_urdu_v1.yaml` from corpus frequency (top ~500
      variants); document how each mapping was derived.
- [ ] pytest fixtures: one case per normalization rule, incl. the ambiguous
      English/Roman token `main`, Arabic→Urdu char mapping, elongation,
      diacritics on/off.
- [ ] Splitter: dedup check + fixed-seed 80/10/10 splits; split IDs committed.
- [ ] EDA notebook: class balance, script mix, vocab size before/after
      normalization (first measurable result: vocabulary-reduction %).

**Milestone M1 (end of Week 1):** `make data` works from a fresh clone;
normalization tests pass; EDA notebook runs top to bottom.
**Definition of done:** a stranger can clone, run `make data`, and see the
same split counts and vocab-reduction figure.

## Week 2 — Models & honest evaluation

- [ ] Classical baselines (majority, TF-IDF+LogReg, TF-IDF+SVM) via
      `make baseline`; 5-fold CV for classical models.
- [ ] `evaluate.py`: macro-F1, per-class F1, confusion matrices, and the three
      error slices (negation / sarcasm / code-switch) per TRD §4.
- [ ] XLM-R fine-tune via `make finetune` (Colab-class GPU acceptable;
      early stopping on dev macro-F1; config in `configs/xlmr.yaml`).
- [ ] Manual audit: hand-check 200 random test items; estimate label noise;
      tag the error-analysis sample (15–20 examples) in
      `notebooks/03_error_analysis.ipynb`.
- [ ] Generate `results/metrics.json`; write the README results table and the
      HF model card *from it*.

**Milestone M2 (end of Week 2):** all four models evaluated on the identical
test split; error slices reported; label-noise estimate documented.
**Definition of done:** every number in the README traces to a row in
`results/metrics.json`, which traces to a config hash and a git commit.

## Week 3 — Demo, release, polish

- [ ] Gradio demo per UI_UX_BRIEF.md (Playground + Compare + Dataset explorer
      + About/Limitations), including RTL handling and all states.
- [ ] Deploy demo (HF Space, CPU; SVM default, XLM-R optional with latency note).
- [ ] Push fine-tuned model + model card to HF Hub; link from README.
- [ ] Write the final Limitations section (§6 draft below, updated with real
      findings); final README pass; tag `v1.0.0`.
- [ ] Stretch, only if M2/M3 are fully done: FastAPI service per
      BACKEND_SCHEMA.md (registry + `/sentiment` + prediction logging).

**Milestone M3 (end of Week 3):** public repo + live demo + HF model card,
all citing the same measured numbers.
**Definition of done (project):** clone → `make data && make baseline &&
make evaluate` reproduces the headline macro-F1 within ±0.01; demo live;
Limitations published.

## Risks — dataset licensing & availability

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CLE approval delayed or refused | Medium | Lose Urdu-script abuse corpora; sentiment plan unaffected | Request on Day 1; MVP sentiment uses public data only; CLE is stretch (abuse head) |
| Kaggle dataset provenance/license unclear | Medium | Cannot redistribute; reviewers question it | Dataset card states provenance honestly; store IDs + features if license is murky; prefer sets with explicit licenses |
| Public Roman Urdu datasets are small / noisy | High | Low absolute scores | Pre-committed framing: honest baselines + error analysis are the contribution; audit 200 items and report noise % |
| Roman Urdu spelling variants beyond lexicon | High | OOV errors persist | Char n-gram features in classical models; document OOV slice in error analysis |
| Free GPU limits fine-tuning time | Medium | Fewer epochs | len 128, ≤5 epochs is small for XLM-R-base; cache tokenized sets; CPU fallback runs (slowly) rather than failing |
| Scope creep (adding translation, more tasks) | Medium | Nothing ships | PRD non-goals enforced; new ideas go to a `ROADMAP.md` parking lot |

## 6. Planned Limitations section (draft — publish in README, update with findings)

> **Limitations**
> - **Domain shift.** Models are trained mainly on e-commerce product reviews.
>   Expect a measurable drop on tweets, news comments, and support chats; the
>   cross-dataset check in `results/` quantifies this on the data we have,
>   and it will not be flattering.
> - **Sarcasm and negation** remain the weakest slices (see error analysis).
>   Negation scope in Roman Urdu is often marked only by word order, and
>   sarcasm labels come from a small hand-tagged sample, so slice scores have
>   wide uncertainty.
> - **Roman Urdu has no standard spelling.** The canonical lexicon covers the
>   most frequent variants in our corpora; creative or regional spellings will
>   be out-of-vocabulary. Normalization reduces this problem; it does not
>   solve it.
> - **Label noise.** A manual audit of the training data found an estimated
>   [X]% of labels questionable (mixed-sentiment reviews forced into one
>   class are the largest group). Reported scores should be read with that
>   ceiling in mind.
> - **Not a moderation system.** Sentiment output is a signal for human
>   review, not a basis for automated action against any person's content.
> - **Classical vs transformer.** On datasets of this size the SVM baseline
>   is competitive with (and far cheaper than) the fine-tuned transformer;
>   we publish both so users can choose with real numbers.
