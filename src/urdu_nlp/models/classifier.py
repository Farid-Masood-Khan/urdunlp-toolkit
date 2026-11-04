"""Unified sentiment classifier interface and implementations (Majority, TF-IDF + LogReg, TF-IDF + Calibrated LinearSVC).

References: TRD §2.4 & PRD FR-6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.urdu_nlp.normalizer import normalize

logger = logging.getLogger(__name__)

SENTIMENT_CLASSES = ["negative", "neutral", "positive"]


@dataclass
class SentimentResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    normalized_text: str
    script: str


class BaseSentimentClassifier:
    """Base class defining the contractual API for sentiment prediction."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.classes = SENTIMENT_CLASSES
        self.is_fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> None:
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[str]:
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str) -> BaseSentimentClassifier:
        obj = joblib.load(path)
        return cast(BaseSentimentClassifier, obj)


class MajorityBaselineClassifier(BaseSentimentClassifier):
    """Majority class baseline predictor (the honesty floor)."""

    def __init__(self):
        super().__init__("majority")
        self.clf = DummyClassifier(strategy="most_frequent")

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self.clf.fit(texts, labels)
        self.classes = list(self.clf.classes_)
        self.is_fitted = True

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.clf.predict(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.clf.predict_proba(texts), dtype=np.float64)


def build_tfidf_feature_extractor() -> FeatureUnion:
    """Builds hierarchical feature union: Word (1,2)-grams + Char (3,5)-grams.

    Per TRD §2.4:
    - Word n-grams capture lexical phrases
    - Char n-grams capture morphological root variants and Roman spelling noise
    """
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=15000,
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=3,
        max_features=25000,
    )
    return FeatureUnion([("word_tfidf", word_vec), ("char_tfidf", char_vec)])


class TfidfLogRegClassifier(BaseSentimentClassifier):
    """Interpretable classical baseline: TF-IDF features with balanced Logistic Regression."""

    def __init__(self, c: float = 1.0, max_iter: int = 1000):
        super().__init__("tfidf_logreg")
        features = build_tfidf_feature_extractor()
        classifier = LogisticRegression(
            C=c,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=42,
        )
        self.pipeline = Pipeline([("features", features), ("clf", classifier)])

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self.pipeline.fit(texts, labels)
        self.classes = list(self.pipeline.named_steps["clf"].classes_)
        self.is_fitted = True

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(texts), dtype=np.float64)


class TfidfLinearSVCClassifier(BaseSentimentClassifier):
    """Strong classical baseline: TF-IDF with LinearSVC and Platt probability calibration."""

    def __init__(self, c: float = 1.0):
        super().__init__("tfidf_linearsvc")
        features = build_tfidf_feature_extractor()
        base_svc = LinearSVC(
            C=c,
            class_weight="balanced",
            random_state=42,
            dual="auto",
        )
        # 3-fold cross-validated probability calibration
        calibrated = CalibratedClassifierCV(estimator=base_svc, cv=3)
        self.pipeline = Pipeline([("features", features), ("clf", calibrated)])

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self.pipeline.fit(texts, labels)
        self.classes = list(self.pipeline.named_steps["clf"].classes_)
        self.is_fitted = True

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(texts), dtype=np.float64)


def predict_sentiment(
    text: str,
    classifier: BaseSentimentClassifier,
    use_normalization: bool = True,
) -> SentimentResult:
    """End-to-end prediction helper with normalization and confidence extraction."""
    if use_normalization:
        norm_res = normalize(text)
        cleaned_text = norm_res.text
        script = norm_res.script
    else:
        cleaned_text = text
        script = "unknown"

    probs = classifier.predict_proba([cleaned_text])[0]
    best_idx = int(np.argmax(probs))
    best_label = classifier.classes[best_idx]
    confidence = float(probs[best_idx])

    prob_dict = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(classifier.classes, probs, strict=False)
    }

    return SentimentResult(
        label=best_label,
        confidence=round(confidence, 4),
        probabilities=prob_dict,
        normalized_text=cleaned_text,
        script=script,
    )
