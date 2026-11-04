"""UrduNLP Toolkit — Interactive Streamlit Demonstration Application.

Features:
- Live script detection (Urdu Script, Roman Urdu, Mixed)
- Step-by-step normalizer change inspector
- Calibrated multi-model sentiment triage
- Sliced error diagnostic flags (negation, sarcasm, code-switching)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.urdu_nlp.detector import identify_script
from src.urdu_nlp.models.classifier import (
    BaseSentimentClassifier,
    TfidfLinearSVCClassifier,
    TfidfLogRegClassifier,
    predict_sentiment,
)
from src.urdu_nlp.normalizer import normalize
from src.urdu_nlp.tokenizer import split_sentences, tokenize

st.set_page_config(
    page_title="UrduNLP Toolkit",
    page_icon="🇵🇰",
    layout="wide",
)

st.title("🇵🇰 UrduNLP Toolkit")
st.markdown(
    "**End-to-End NLP for Low-Resource Urdu Script & Roman Urdu** · "
    "*Normalized Orthography, Calibrated Sentiment Triage, and Honest Error Analysis*"
)


# Cache model loading for instant interactive response
@st.cache_resource
def get_model(model_name: str) -> BaseSentimentClassifier:
    model_path = Path(f"models/{model_name}.joblib")
    if model_path.exists():
        return BaseSentimentClassifier.load(model_path)
    # If not yet trained, return initialized classifier
    if model_name == "tfidf_linearsvc":
        return TfidfLinearSVCClassifier()
    return TfidfLogRegClassifier()


tab_norm, tab_sentiment, tab_benchmark = st.tabs(
    ["📝 Normalization & Script Inspector", "🎯 Sentiment Triage", "📊 Empirical Benchmarks"]
)

# ----------------- Tab 1: Normalization & Inspector -----------------
with tab_norm:
    st.subheader("Orthographic Normalizer & Change Audit")
    default_text = "ye mobile bohattt achaa hai but camera bilkul b acha nahi hai <URL>"
    sample_select = st.selectbox(
        "Choose a representative sample or write your own:",
        [
            "ye mobile bohattt achaa hai but camera bilkul b acha nahi hai <URL>",
            "wah kya kamal ki service hai 2 hafte baad parcel pohancha!",
            "یہ موبائل بہت بہترین ہے لیکن بیٹری ٹائمنگ نارمل ہے۔",
            "bht behtareen quality hai highly recommended thanks",
        ],
    )
    user_text = st.text_area(
        "Input Text (Urdu Script or Roman Urdu):", value=sample_select, height=100
    )

    if user_text:
        script_res = identify_script(user_text)
        norm_res = normalize(user_text)

        col_script, col_tokens, col_sentences = st.columns(3)
        with col_script:
            st.metric("Detected Script", script_res.script.replace("_", " ").title())
            st.caption(
                f"Urdu: {script_res.urdu_ratio * 100:.1f}% · Latin: {script_res.latin_ratio * 100:.1f}%"
            )
        with col_tokens:
            tokens = tokenize(norm_res.text)
            st.metric("Token Count", len(tokens))
        with col_sentences:
            sentences = split_sentences(norm_res.text)
            st.metric("Sentences", len(sentences))

        st.markdown("#### Normalized Form")
        st.info(norm_res.text)

        st.markdown("#### Transformation Audit Trail")
        if norm_res.changes:
            df_changes = pd.DataFrame(
                [
                    {
                        "Stage": c.stage,
                        "Original": c.original_token,
                        "Transformed": c.new_token,
                        "Rule": c.rule,
                    }
                    for c in norm_res.changes
                ]
            )
            st.dataframe(df_changes, use_container_width=True)
        else:
            st.success("No modifications required; text already conforms to canonical orthography.")

# ----------------- Tab 2: Sentiment Triage -----------------
with tab_sentiment:
    st.subheader("Calibrated Sentiment Classification")
    col_input, col_pred = st.columns([1.2, 1])

    with col_input:
        sentiment_input = st.text_area(
            "Review or Comment Text:",
            value="bohat zabardast packaging thi product time pe deliver hua",
            height=120,
        )
        selected_model = st.selectbox(
            "Classifier Architecture:",
            ["tfidf_linearsvc", "tfidf_logreg"],
            format_func=lambda m: (
                "TF-IDF + LinearSVC (Calibrated)"
                if m == "tfidf_linearsvc"
                else "TF-IDF + Logistic Regression"
            ),
        )
        apply_norm = st.checkbox("Enable Normalization Pipeline", value=True)

    with col_pred:
        if sentiment_input:
            clf = get_model(selected_model)
            if not clf.is_fitted:
                st.warning(
                    "Model artifact not found in models/. Run 'run.ps1 eval' to train on committed splits."
                )
            else:
                res = predict_sentiment(sentiment_input, clf, use_normalization=apply_norm)
                st.markdown("#### Prediction")

                color_map = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#f59e0b"}
                label_color = color_map.get(res.label, "#38bdf8")

                st.markdown(
                    f"<div style='background:{label_color}22; border-left:4px solid {label_color}; padding:1rem; border-radius:8px;'>"
                    f"<h2 style='color:{label_color}; margin:0;'>{res.label.upper()}</h2>"
                    f"<p style='margin:0; font-size:0.9rem;'>Confidence: <b>{res.confidence * 100:.1f}%</b></p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("##### Class Probabilities")
                for c_name, prob in res.probabilities.items():
                    st.progress(prob, text=f"{c_name.capitalize()}: {prob * 100:.1f}%")

                # Error Slice Warning Flags
                lower_in = sentiment_input.lower()
                has_negation = any(
                    k in lower_in.split() for k in ["nahi", "nai", "na", "mat", "نہیں"]
                )
                has_sarcasm = any(
                    k in lower_in for k in ["wah kya", "bohat mehrbani", "shukriya apka khrab"]
                )

                if has_negation:
                    st.warning(
                        "⚠️ **Negation Slice Detected:** Input contains explicit negation keywords."
                    )
                if has_sarcasm:
                    st.warning("⚠️ **Sarcasm Slice Detected:** Potential praise irony detected.")

# ----------------- Tab 3: Empirical Benchmarks -----------------
with tab_benchmark:
    st.subheader("Committed Empirical Benchmark Results")
    metrics_file = Path("results/metrics.json")
    if metrics_file.exists():
        with open(metrics_file, encoding="utf-8") as f:
            data = json.load(f)

        v_data = data.get("vocabulary_metrics", {})
        st.markdown(
            f"**Vocabulary Reduction:** `{v_data.get('raw_unique_tokens')}` raw tokens ➔ "
            f"`{v_data.get('normalized_unique_tokens')}` normalized tokens "
            f"(**{v_data.get('reduction_percentage')}% reduction**)"
        )

        st.markdown("#### Model Comparison (Held-out Test Split)")
        models_data = data.get("model_benchmarks", {})
        summary_rows = []
        for m_name, m_stats in models_data.items():
            summary_rows.append(
                {
                    "Model": m_name,
                    "Macro F1": m_stats.get("macro_f1"),
                    "Accuracy": f"{m_stats.get('accuracy', 0) * 100:.1f}%",
                    "Negative F1": m_stats.get("per_class", {}).get("negative", {}).get("f1"),
                    "Neutral F1": m_stats.get("per_class", {}).get("neutral", {}).get("f1"),
                    "Positive F1": m_stats.get("per_class", {}).get("positive", {}).get("f1"),
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.markdown("#### Diagnostic Error Slices")
        slice_data = data.get("error_slices", {})
        slice_rows = []
        for s_name, s_stats in slice_data.items():
            slice_rows.append(
                {
                    "Slice": s_name.replace("_", " ").title(),
                    "Count": s_stats.get("count"),
                    "Macro F1": s_stats.get("macro_f1"),
                    "Accuracy": f"{s_stats.get('accuracy', 0) * 100:.1f}%",
                }
            )
        st.dataframe(pd.DataFrame(slice_rows), use_container_width=True)
    else:
        st.info("Execute `run.ps1 eval` to generate benchmark results.")
