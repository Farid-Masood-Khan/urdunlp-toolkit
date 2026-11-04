"""Unit tests for sentiment models, probability calibration, and persistence."""

import tempfile
from pathlib import Path

from src.urdu_nlp.models.classifier import (
    MajorityBaselineClassifier,
    TfidfLinearSVCClassifier,
    TfidfLogRegClassifier,
    predict_sentiment,
)


def get_sample_training_data() -> tuple[list[str], list[str]]:
    """Sample corpus covering positive, negative, and neutral."""
    texts = [
        "bohat zabardast product hai",
        "achi quality hai time pe aya",
        "superb service highly recommended",
        "bakwas tareen cheez hai",
        "bohat khrab hai paise zaya",
        "toota hua deliver kia ghatiya seller",
        "normal product hai guzara hai",
        "theek thaak hai na acha na bura",
        "average quality reasonable price",
    ]
    labels = [
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "neutral",
    ]
    return texts, labels


def test_majority_baseline_classifier():
    """Majority classifier must always predict the modal training class."""
    texts, labels = get_sample_training_data()
    # Add extra positive sample to make it the majority
    texts.append("dil khush ho gaya zabardast")
    labels.append("positive")

    clf = MajorityBaselineClassifier()
    clf.fit(texts, labels)
    preds = clf.predict(["koi b jumla", "another sentence"])
    assert all(p == "positive" for p in preds)


def test_tfidf_logreg_probabilities_sum_to_one():
    """TF-IDF Logistic Regression output probabilities must sum to 1.0."""
    texts, labels = get_sample_training_data()
    clf = TfidfLogRegClassifier()
    clf.fit(texts, labels)

    res = predict_sentiment("bohat zabardast cheez hai", clf)
    assert res.label in ("positive", "negative", "neutral")
    assert 0.0 <= res.confidence <= 1.0
    total_prob = sum(res.probabilities.values())
    assert abs(total_prob - 1.0) < 1e-3


def test_tfidf_linearsvc_calibrated_probabilities():
    """Calibrated LinearSVC must output well-formed probability distributions."""
    texts, labels = get_sample_training_data()
    clf = TfidfLinearSVCClassifier()
    clf.fit(texts, labels)

    res = predict_sentiment("bakwas quality bilkul bura", clf)
    assert res.label in ("positive", "negative", "neutral")
    assert 0.0 <= res.confidence <= 1.0
    total_prob = sum(res.probabilities.values())
    assert abs(total_prob - 1.0) < 1e-3


def test_model_persistence_and_reload():
    """Trained models must serialize and reload with exact prediction consistency."""
    texts, labels = get_sample_training_data()
    clf = TfidfLogRegClassifier()
    clf.fit(texts, labels)

    test_query = "bohat achaa phone hai"
    pred_orig = clf.predict([test_query])

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_file = Path(tmp_dir) / "model.joblib"
        clf.save(save_file)
        assert save_file.exists()

        reloaded_clf = TfidfLogRegClassifier.load(save_file)
        pred_reloaded = reloaded_clf.predict([test_query])
        assert pred_orig == pred_reloaded

